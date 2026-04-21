# DeepSeek-OCR API Service — Full Architecture

## Table of Contents

- [System Overview](#system-overview)
- [Request Lifecycle](#request-lifecycle)
- [Concurrency Model](#concurrency-model)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Inference Engine](#inference-engine)
- [Post-processing Pipeline](#post-processing-pipeline)
- [Quality Scoring System](#quality-scoring-system)
- [Retry Logic](#retry-logic)
- [Pre-flight Detection](#pre-flight-detection)
- [Tesseract Fallback](#tesseract-fallback)
- [External OCR Routing](#external-ocr-routing)
- [Feedback System](#feedback-system)
- [API Endpoints](#api-endpoints)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Key Dependencies](#key-dependencies)
- [Performance Characteristics](#performance-characteristics)

---

## System Overview

DeepSeek-OCR API is a production-ready FastAPI service that extracts text from document images using the DeepSeek-OCR vision-language model. It runs on a single GPU via vLLM's `AsyncLLMEngine` for concurrent request handling.

```
                          External Services (1..N)
                                  │
                          POST /ocr/image
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     FastAPI (uvicorn)     │  Port 8000
                    │     1 worker process      │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
        Pre-flight         Pre-flight          Pre-flight
        Detection          Detection           Detection
        (~1ms)             (~1ms)              (~1ms)
              │                  │                   │
              ▼                  ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │              vLLM AsyncLLMEngine                     │
    │         Continuous Batching Scheduler                │
    │    ┌─────────┐ ┌─────────┐ ┌─────────┐              │
    │    │ Req A   │ │ Req B   │ │ Req C   │  GPU         │
    │    │ prefill │ │ decode  │ │ prefill │  Memory      │
    │    └─────────┘ └─────────┘ └─────────┘  80%         │
    └─────────────────────────────────────────────────────┘
              │                  │                   │
              ▼                  ▼                   ▼
        Post-process       Post-process        Post-process
        + Score            + Score              + Score
              │                  │                   │
              ▼                  ▼                   ▼
        JSON Response      JSON Response       JSON Response
```

---

## Request Lifecycle

A single `/ocr/image` request with `retry=true` follows this path:

```
1. Upload received
   ├── Validate file size (≤ 20 MB)
   └── Decode image (PIL) + EXIF correction

2. Pre-flight checks (~1ms)
   ├── Blank page? → RED flag, skip OCR, return immediately
   └── Low-quality scan? → RED flag, skip OCR, return immediately

3. Retry loop (up to 3 attempts)
   │
   ├── Attempt 1: "adaptive" preset
   │   ├── Enhance image (adaptive contrast/sharpness for grayscale scans)
   │   ├── Tile image into 640×640 crops (2-9 tiles)
   │   ├── Tokenize tiles → vLLM multi-modal input
   │   ├── AsyncLLMEngine.generate() → raw text + token count
   │   ├── Post-process: strip grounding tags, clean tables, collapse
   │   │   repetition, deduplicate sections, normalize whitespace
   │   ├── Score (6 weighted metrics → composite 0.0-1.0)
   │   └── Score ≥ 0.60? → STOP, use this result
   │
   ├── Attempt 2: "none" preset (no enhancement)
   │   └── Same pipeline → score → ≥ 0.60? → STOP
   │
   └── Attempt 3: "strong" preset (contrast 1.5×, sharpness 2×)
       └── Same pipeline → score → select best of all 3

4. Best result selected (highest composite score)

5. Flag assignment
   ├── GREEN (≥ 0.70): Good quality
   ├── YELLOW (0.50-0.69): Spot-check recommended
   └── RED (< 0.50): Manual review required

6. External OCR routing check
   ├── Clean text ≤ 10 chars → needs_external_ocr: true
   │   └── Try Tesseract fallback first
   └── Token limit hit (≥85% max) + >90% hallucinated + low score
       → needs_external_ocr: true (incomplete_extraction)

7. Feedback storage
   └── Score < 0.70 or Tesseract used → save image + metadata to feedback/pending/

8. JSON response returned
```

---

## Concurrency Model

The service uses vLLM's `AsyncLLMEngine` (v0.8.4) for true concurrent GPU inference.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    AsyncLLMEngine                         │
│                                                           │
│  ┌─────────────┐    ┌──────────────────┐                 │
│  │ Request      │    │ Continuous        │                │
│  │ Queue        │───▶│ Batching          │                │
│  │              │    │ Scheduler         │                │
│  │ req_A ──────▶│    │                   │                │
│  │ req_B ──────▶│    │ Interleaves       │   ┌─────────┐ │
│  │ req_C ──────▶│    │ prefill/decode    │──▶│  GPU    │ │
│  │ req_D ──────▶│    │ across requests   │   │ (single)│ │
│  └─────────────┘    └──────────────────┘   └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

- **Single GPU, multiple requests**: The scheduler interleaves prefill (processing a new image) and decode (generating tokens for in-flight requests) within the same GPU.
- **No semaphore needed**: Unlike the sync `LLM` class which crashes on concurrent calls, `AsyncLLMEngine` handles batching internally.
- **`max_num_seqs`**: Controls maximum concurrent sequences on GPU (set to `MAX_CONCURRENCY` from config, default 100).
- **1 uvicorn worker**: Always 1 process to prevent model duplication in GPU memory.
- **CPU parallelism**: Image preprocessing (tiling, tokenization) runs in thread pool via `run_in_executor`.

### Why Not Sync LLM?

vLLM's sync `llm.generate()` triggers `AssertionError` in the scheduler when called concurrently — the prefill logic asserts single-request-at-a-time. This was the original architecture with `Semaphore(1)`, which serialized all requests to one at a time.

### Why Not Separate vLLM Server?

The model's HuggingFace code (`modeling_deepseekv2.py`) imports `LlamaFlashAttention2`, which doesn't exist in the installed transformers version. The in-process `AsyncLLMEngine` bypasses this by using the custom registered `DeepseekOCRForCausalLM` class directly. The separate server (`vllm.entrypoints.openai.api_server`) spawns a subprocess that loads HF code first, hitting the import error.

---

## Image Processing Pipeline

### Tiling System

Images are split into 640×640 tiles before model inference. This is how the model sees high-resolution documents.

```
Config (config.py):
  BASE_SIZE  = 1024    # Base dimension for aspect ratio calculation
  IMAGE_SIZE = 640     # Tile size
  CROP_MODE  = True    # Enable tiling
  MIN_CROPS  = 2       # Minimum tiles
  MAX_CROPS  = 9       # Maximum tiles (3×3 grid)
```

```
Input Image                     Tile Grid
┌─────────────────┐            ┌──────┬──────┬──────┐
│                  │            │ Tile │ Tile │ Tile │
│  1700 × 2200    │   resize   │  1   │  2   │  3   │
│  (original)     │ ────────▶  ├──────┼──────┼──────┤
│                  │            │ Tile │ Tile │ Tile │
│                  │            │  4   │  5   │  6   │
│                  │            └──────┴──────┴──────┘
└─────────────────┘              1280 × 1280
                                 (2×3 grid = 6 tiles)
```

**Optimal resolution: 1280×1920** — maps perfectly to a 2×3 tile grid (6 tiles at 640×640) with no resizing artifacts. Higher resolutions are downscaled to fit the tile grid anyway.

### Tile Calculation Flow

```
dynamic_preprocess() in image_process.py:
  1. Calculate target aspect ratio from available tile configs
  2. Find closest match: (cols × 640, rows × 640)
  3. Resize image to fit the grid
  4. Split into 640×640 tiles
  5. Each tile becomes a separate visual token sequence
```

### Image Enhancement

Three preset strategies applied before tiling:

| Preset | Contrast | Sharpness | When Used |
|--------|----------|-----------|-----------|
| `adaptive` | Auto (target RMS 0.186) | 1.5× | Attempt 1 — adjusts grayscale scans to match known-good reference |
| `none` | 1.0× | 1.0× | Attempt 2 — raw image, no enhancement |
| `strong` | 1.5× | 2.0× | Attempt 3 — aggressive enhancement for difficult scans |

Adaptive enhancement only activates for grayscale images (detected by channel difference < 10). Color images pass through unchanged.

---

## Inference Engine

### Model Details

```
Model:          DeepSeek-OCR (custom architecture)
Class:          DeepseekOCRForCausalLM (registered with vLLM ModelRegistry)
Precision:      bfloat16
VRAM:           ~6.2 GB (at GPU_MEM_UTIL=0.80)
Max context:    8,192 tokens
Temperature:    0.0 (deterministic)
```

### Generation Parameters

```python
SamplingParams(
    temperature=0.0,           # Deterministic output
    max_tokens=8192,           # Max output tokens
    skip_special_tokens=False, # Keep special tokens for post-processing
    include_stop_str_in_output=True,
    logits_processors=[
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,                    # 20-token n-gram window
            window_size=50,                   # Look-back window
            whitelist_token_ids={128821, 128822},  # Allow table tags
            max_consecutive_empty_cells=30,   # Cap empty <td></td> runs
        )
    ],
)
```

### N-gram No-Repeat Processor

Custom logits processor that prevents generation loops by blocking repeated 20-token sequences. Table structure tokens (`<td>`, `</td>`) are whitelisted to allow natural table generation, but empty cell runs are capped at 30 consecutive.

### Prompt Templates

| Key | Prompt | Use Case |
|-----|--------|----------|
| `document` | `<image>\n<\|grounding\|>Convert the document to markdown.` | Default — structured document extraction |
| `ocr` | `<image>\n<\|grounding\|>OCR this image.` | General OCR |
| `free_ocr` | `<image>\nFree OCR.` | Without layout/grounding |
| `figure` | `<image>\nParse the figure.` | Charts, diagrams |
| `describe` | `<image>\nDescribe this image in detail.` | Image description |

---

## Post-processing Pipeline

Raw model output goes through 5 cleanup stages in `postprocess.py`:

```
Raw Model Output
    │
    ▼
1. Strip grounding tags
   Remove <|ref|>...<|/ref|>, <|det|>...<|/det|>, coordinate arrays [[x,y,w,h]]
   Remove end-of-sentence marker
    │
    ▼
2. Table cleanup (_collapse_empty_table_cells)
   ├── Trim empty <td></td> runs (max 15 per row)
   ├── Remove entirely empty rows
   ├── Remove hallucinated numbered empty-row sequences
   ├── Trim bloated tables (>100 empty cells, cap 60 rows)
   ├── Handle unclosed <table> tags
   ├── Collapse repetitive table rows (>80% duplicate)
   ├── Remove diagonal repetition (single value >40% of cells)
   └── Remove empty tables
    │
    ▼
3. Collapse repeating patterns (_collapse_repeating_patterns)
   ├── Incrementing number + digit filler
   ├── Long digit-space runs (20+ or 8+)
   ├── Dot-separated digits (6+)
   ├── Single char repeated with spaces (12+)
   └── Numbered sequences (15+)
    │
    ▼
4. Section deduplication (_deduplicate_sections)
   ├── Split by markdown headers (# ## ###)
   ├── Detect duplicate headers
   ├── Keep longer version (or more columns for expanded table variants)
   └── Track dedup chars separately from hallucination for scoring
    │
    ▼
5. Whitespace normalization
   ├── Collapse 3+ newlines → 2
   └── Collapse multiple spaces → 1
    │
    ▼
Clean Text Output
```

### CleanStats Tracking

Post-processing tracks removal categories separately:
- `dedup_chars_removed` — Characters removed by section deduplication (not hallucination)
- `hallucination_chars_removed` — Characters removed as fabricated content

This distinction is critical for accurate hallucination scoring — dedup removal should not penalize the score.

---

## Quality Scoring System

Six independent metrics, each normalized to 0.0-1.0, combined with fixed weights:

```
Composite = 0.25 × hallucination_ratio
          + 0.20 × self_consistency
          + 0.20 × token_efficiency
          + 0.15 × content_density
          + 0.10 × structural_integrity
          + 0.10 × repetition_density
```

### Metric Details

#### hallucination_ratio (weight: 0.25)

Measures how much raw output survived post-processing.

```
effective_raw = raw_length - grounding_tag_chars - dedup_chars
ratio = clean_length / effective_raw
```

- Grounding tags (`<|ref|>`, `<|det|>`, coordinates) are excluded from the denominator — they are expected format, not fabrication.
- Dedup-removed content is also excluded.
- If `effective_raw ≤ clean_length`: returns 1.0 (everything was tags).

#### self_consistency (weight: 0.20)

Pairwise text similarity between multiple OCR runs of the same image.

- Single-run: returns **1.0** (deterministic at temp=0, cannot measure consistency, should not penalize).
- Multiple runs: average `SequenceMatcher.ratio()` across all pairs.

#### token_efficiency (weight: 0.20)

Penalizes max-token runs with little clean output (stuck generation loops).

```
if tokens < 80% of max → 1.0 (stopped naturally)
if tokens ≥ 80% of max:
  expected_min = tokens × 0.5
  if clean_len ≥ expected_min → 0.9 (hit max but has real content)
  else → smooth curve 0.1 to 0.9
```

#### content_density (weight: 0.15)

Text volume relative to image area.

```
With dimensions:  expected = pixels / 200, ratio = clean_len / expected
Without:          smooth curve: ≥500 chars → 1.0, ≥100 → 0.5-1.0, ≥20 → 0.2-0.5
```

#### structural_integrity (weight: 0.10)

Presence of recognizable patterns (does NOT require any specific structure):

| Signal | Credit |
|--------|--------|
| Markdown headers (`# ##`) | +1.0 |
| Tables with content cells | +1.0 |
| Meaningful text (>30 chars non-markup) | +1.0 |
| Data patterns (dates, amounts, names, emails, phones) | +1.0 |

Scoring: 1 signal = 0.75, 2 = 0.875, 3 = 0.95, 4 = 1.0. Substantial text with no signals = 0.4.

#### repetition_density (weight: 0.10)

Remaining repetitive n-grams after cleanup (strips HTML tags first).

- Checks n-grams at lengths 6, 10, 15, 20
- Only counts sequences appearing 8+ times
- `score = max(0, 1.0 - repetition_ratio × 0.3)`
- Text < 100 chars: returns 1.0 (too short to judge)

### Blank Page Caps

Prevents score inflation on empty output:

```
clean_text ≤ 10 chars → composite capped at 0.10
clean_text ≤ 30 chars → composite capped at 0.30
```

### Flag Assignment

| Flag | Composite | Meaning |
|------|-----------|---------|
| GREEN | ≥ 0.70 | Good quality, use as-is |
| YELLOW | 0.50-0.69 | Spot-check recommended |
| RED | < 0.50 | Manual review required |

**Warning downgrades** (green → yellow):
- `hallucination_ratio < 0.25` — severe hallucination
- `token_efficiency < 0.2` — stuck generation loop

---

## Retry Logic

When `retry=true` (default) and score < 0.60:

```
┌───────────────────────────────────────────────────────────────┐
│ Attempt 1: "adaptive"                                         │
│  ├── Auto contrast/sharpness for grayscale scans              │
│  ├── Score result                                             │
│  └── Score ≥ 0.60? ──── YES ──── ▶ STOP (use this result)    │
│                           NO                                   │
│                           │                                    │
│ Attempt 2: "none"         ▼                                   │
│  ├── No enhancement (raw image)                               │
│  ├── Score result                                             │
│  └── Score ≥ 0.60? ──── YES ──── ▶ STOP (use this result)    │
│                           NO                                   │
│                           │                                    │
│ Attempt 3: "strong"       ▼                                   │
│  ├── Contrast 1.5×, Sharpness 2×                              │
│  ├── Score result                                             │
│  └── Select BEST of all 3 attempts (highest composite)        │
└───────────────────────────────────────────────────────────────┘
```

After all attempts, `select_best_result()` re-scores `self_consistency` using the full result set and picks the highest composite.

---

## Pre-flight Detection

Two instant checks (~1ms) run before any OCR processing:

### Blank Page Detection

```python
def is_blank_page(image):
    gray = to_grayscale(image)
    if gray.std() >= 5.0:        # Has variation → not blank
        return False
    dark_ratio = pixels_below_240 / total_pixels
    return dark_ratio < 0.02     # Almost no dark pixels → blank
```

Result: RED flag, `blank_page` detail, zero text, `ocr_engine: "skipped"`.

### Low-Quality Scan Detection

```python
def is_low_quality_scan(image):
    # Find bounding box of content (rows/cols with >1% dark pixels)
    content_area_ratio = (content_h × content_w) / (page_h × page_w)
    return content_area_ratio < 0.12  # Content too small
```

Catches: faxed documents, thumbnail-quality scans, shrunken copies.
Result: RED flag, `low_quality_scan` detail, zero text, `ocr_engine: "skipped"`.

---

## Tesseract Fallback

When DeepSeek-OCR extracts no meaningful text (≤10 chars) from a non-blank page:

```
DeepSeek result: ≤ 10 chars clean text
        │
        ▼
┌──────────────────────────┐
│  Preprocessing           │
│  Adaptive thresholding:  │
│  ├── Gaussian blur σ=25  │
│  ├── Background subtract │
│  └── Normalize to 0-255  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  pytesseract (eng)       │
│  ├── Success (≥10 chars) │  → Return with ocr_engine: "tesseract"
│  │   flag: YELLOW        │    score: 0.50
│  └── Failure (<10 chars) │  → needs_external_ocr: true
└──────────────────────────┘
```

Disable with: `TESSERACT_FALLBACK=false`

---

## External OCR Routing

The `needs_external_ocr` flag tells callers to route the page to an external OCR service. Two conditions trigger it:

### Condition 1: OCR Failed (empty extraction)

```
clean_text ≤ 10 chars AND composite < 0.60 AND Tesseract fallback also failed
→ needs_external_ocr: true
→ flag_detail: "ocr_failed"
```

### Condition 2: Incomplete Extraction

For dense forms where the model extracts headers but hallucinates table content:

```
num_tokens ≥ MAX_TOKENS × 0.85      (hit token limit)
AND clean_len / raw_len < 0.10      (>90% removed as hallucination)
AND composite < 0.60                 (low quality score)
→ needs_external_ocr: true
→ flag_detail: "incomplete_extraction"
```

Example: A dense HUD compliance form — model generates 7,280 tokens but only 287 chars survive post-processing (1% retained). Headers extracted correctly, but all table content was hallucinated filler.

---

## Feedback System

Automatically stores low-scoring results for future LoRA fine-tuning.

### Storage Flow

```
OCR Result
    │
    ├── Score ≥ 0.70 AND engine != tesseract → NOT saved (no disk wasted)
    │
    └── Score < 0.70 OR engine == tesseract
        │
        ▼
  feedback/pending/
  ├── {timestamp}_{uuid}.png    ← Original image
  └── {timestamp}_{uuid}.json   ← OCR result + metadata
        │
        │  POST /feedback/correct
        │  (human or AI sends corrected text)
        ▼
  feedback/verified/
  ├── {timestamp}_{uuid}.png    ← Same image
  └── {timestamp}_{uuid}.json   ← Metadata + corrected_text
        │
        │  50+ verified pairs accumulated
        ▼
  Ready for LoRA fine-tuning
```

### What Gets Saved

| Condition | Saved? |
|-----------|--------|
| GREEN (≥ 0.70) | No |
| YELLOW (0.50-0.69) | Yes |
| RED (< 0.50) | Yes |
| Tesseract fallback used | Yes |

Estimated ~5% of pages saved. Storage: ~210 KB per entry (image + JSON metadata).

Disable with: `FEEDBACK_ENABLED=false`

---

## API Endpoints

### OCR Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ocr/image` | POST | OCR a single image with scoring + retry |
| `/ocr/image/base64` | POST | Same but accepts base64-encoded image |
| `/ocr/pdf` | POST | OCR all pages of a PDF (concurrent page processing) |
| `/ocr/batch` | POST | OCR multiple images in one request |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model info, GPU info |
| `/` | GET | Service info and links |

### Feedback Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/feedback/stats` | GET | Counts, disk usage, training readiness |
| `/feedback/pending` | GET | List entries awaiting correction |
| `/feedback/correct` | POST | Submit corrected text for an entry |

### Response Format

All OCR endpoints return:

```json
{
  "text": "cleaned markdown output",
  "raw_text": "raw model output with grounding tags",
  "num_tokens": 1024,
  "score": {
    "composite": 0.95,
    "variables": {
      "self_consistency": 1.0,
      "hallucination_ratio": 0.97,
      "token_efficiency": 1.0,
      "structural_integrity": 0.88,
      "repetition_density": 0.95,
      "content_density": 0.85
    },
    "weights": { ... }
  },
  "flag": "green",
  "flag_message": "Good quality (0.95).",
  "flag_details": [],
  "attempts": 1,
  "preset": "adaptive",
  "ocr_engine": "deepseek",
  "needs_external_ocr": false
}
```

### Flag Detail Codes

| Code | Severity | Trigger |
|------|----------|---------|
| `no_content` | critical | No meaningful text (≤10 chars) |
| `ocr_failed` | critical | Page has content but model couldn't read it |
| `incomplete_extraction` | critical | Model hit token limit, >90% hallucinated |
| `blank_page` | critical | Blank page skipped |
| `low_quality_scan` | critical | Content too small, skipped |
| `tesseract_fallback` | warning | Tesseract used, verify accuracy |
| `possible_hallucination` | warning | >75% of output removed |
| `max_tokens_hit` | warning | Stuck generation loop |
| `repetitive_content` | info | Repetitive patterns detected |
| `sparse_content` | info | Very little text vs image size |
| `low_content` | info | Less than 30 chars extracted |

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/DeepSeek-OCR` | Path to model weights |
| `PORT` | `8000` | API port |
| `HOST` | `0.0.0.0` | API bind address |
| `GPU_MEM_UTIL` | `0.80` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Max model context length |
| `MAX_TOKENS` | `8192` | Max output tokens per inference |
| `MAX_IMAGE_SIZE_MB` | `20` | Max upload size per image |
| `MAX_PDF_SIZE_MB` | `100` | Max upload size per PDF |
| `MAX_PDF_PAGES` | `50` | Max pages per PDF |
| `MAX_BATCH_SIZE` | `16` | Max images per batch request |
| `REQUEST_TIMEOUT_S` | `120` | Request timeout (seconds) |
| `SCORE_THRESHOLD` | `0.60` | Score below this triggers retry |
| `MAX_RETRIES` | `3` | Max retry attempts per page |
| `TESSERACT_FALLBACK` | `true` | Enable Tesseract fallback |
| `FEEDBACK_DIR` | `./feedback` | Feedback storage path |
| `FEEDBACK_ENABLED` | `true` | Enable feedback storage |
| `FEEDBACK_SCORE_THRESHOLD` | `0.70` | Save results below this score |

### Forced Environment

```bash
VLLM_USE_V1=0           # Use legacy vLLM engine (required for custom model)
CUDA_VISIBLE_DEVICES=0   # Single GPU
```

---

## Project Structure

```
DeepSeek-OCR-1/
├── api_service.py                          ← Main FastAPI service (v4.0.0)
│   ├── Lifespan: AsyncLLMEngine init
│   ├── Pre-flight: blank page + low-quality scan detection
│   ├── Inference: async generate via vLLM
│   ├── Retry: up to 3 enhancement presets
│   ├── Fallback: Tesseract OCR
│   ├── Feedback: auto-save low-scoring results
│   └── Endpoints: /ocr/image, /ocr/pdf, /ocr/batch, /feedback/*
│
├── start.sh                                ← Entrypoint script
├── requirements.txt                        ← Python dependencies (pinned)
├── README.md                               ← User-facing documentation
├── ARCHITECTURE.md                         ← This file
│
├── images/                                 ← Input images directory
├── feedback/                               ← Feedback storage (auto-created)
│   ├── pending/                            ← Awaiting correction
│   └── verified/                           ← Ready for fine-tuning
│
└── DeepSeek-OCR-master/
    └── DeepSeek-OCR-vllm/
        ├── config.py                       ← Image tiling config (sizes, crops)
        ├── deepseek_ocr.py                 ← Custom vLLM model class
        │   ├── DeepseekOCRForCausalLM      ← Registered with ModelRegistry
        │   ├── DeepseekOCRProcessingInfo
        │   ├── DeepseekOCRMultiModalProcessor
        │   └── DeepseekOCRDummyInputsBuilder
        │
        └── process/
            ├── __init__.py                 ← Public API exports
            ├── image_process.py            ← Image tokenization (DO NOT MODIFY)
            ├── ngram_norepeat.py           ← Logits processor (DO NOT MODIFY)
            ├── postprocess.py              ← Output cleanup + hallucination removal
            ├── enhance.py                  ← Adaptive image enhancement
            └── score.py                    ← Quality scoring system
```

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `vllm` | 0.8.4 | GPU inference engine (AsyncLLMEngine) |
| `torch` | 2.6.0 | PyTorch backend |
| `transformers` | 4.57.6 | Tokenizer + model loading |
| `flash_attn` | 2.8.3 | Flash Attention 2 for fast inference |
| `fastapi` | 0.135.2 | HTTP API framework |
| `uvicorn` | 0.42.0 | ASGI server |
| `pillow` | 12.1.1 | Image loading and manipulation |
| `PyMuPDF` | 1.27.2 | PDF to image conversion |
| `pytesseract` | - | Tesseract OCR fallback |
| `scipy` | 1.17.1 | Adaptive thresholding for Tesseract preprocessing |
| `numpy` | 1.26.4 | Image array operations |

---

## Performance Characteristics

### Benchmarks (54-page housing compliance document set)

| Metric | Value |
|--------|-------|
| Average processing time | ~4.6s per page |
| GREEN (good quality) | 94.4% of pages |
| YELLOW (spot-check) | 3.7% of pages |
| RED (manual review) | 1.9% of pages |
| Average composite score | 0.876 |
| Blank/low-quality skip time | ~1ms |
| Tesseract fallback success | 100% (1/1) |
| Total chars extracted | 131,610 |

### Concurrent Request Throughput

| Scenario | Time | Speedup |
|----------|------|---------|
| 4 sequential requests | ~32.5s | 1× |
| 4 concurrent requests | ~25.3s | 1.28× |

Concurrent requests share GPU compute via continuous batching. Speedup is modest because the GPU is the bottleneck — more requests don't add GPU capacity, they just reduce idle time between requests.

### Memory Usage

| Component | VRAM |
|-----------|------|
| Model weights (bfloat16) | ~6.2 GB |
| KV cache (at 80% util) | Remaining allocation |
| Per-request overhead | Managed by vLLM scheduler |

### Latency Breakdown (single page)

| Stage | Time |
|-------|------|
| Pre-flight detection | ~1ms |
| Image enhancement | ~10ms |
| Tiling + tokenization | ~50ms |
| vLLM inference | ~2-6s (varies with content density) |
| Post-processing | ~5ms |
| Scoring | ~2ms |
| **Total (single attempt)** | **~2-6s** |
| **With retry (3 attempts)** | **~6-18s** |
