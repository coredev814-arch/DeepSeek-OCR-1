# DeepSeek-OCR API Service

Production-ready OCR API powered by DeepSeek-OCR with intelligent quality scoring, automatic retry, Tesseract fallback, and feedback-driven learning.

## Architecture

```
Image Upload
    │
    ▼
┌─────────────────────────┐
│  Pre-flight Detection   │  ~1ms
│  ├─ Blank page?         │  (std < 5, dark < 2%)
│  └─ Low-quality scan?   │  (content area < 12%)
└────────┬────────────────┘
         │ skip → RED flag, no OCR
         ▼
┌─────────────────────────┐
│  DeepSeek-OCR (Primary) │  ~2-6s per page
│  ├─ Adaptive enhance    │
│  ├─ Tokenize + crop     │
│  └─ vLLM inference      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Post-processing        │
│  ├─ Remove grounding    │
│  ├─ Clean tables        │
│  ├─ Remove hallucination│
│  ├─ Dedup sections      │
│  └─ Normalize output    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Quality Scoring        │  6 weighted metrics → composite 0.0-1.0
│  └─ Score < 0.60?       │
│     └─ Retry (up to 3x) │  Different enhancement presets
└────────┬────────────────┘
         │
         ▼ if score still low and clean_text <= 10 chars
┌─────────────────────────┐
│  Tesseract Fallback     │  Adaptive threshold preprocessing
│  └─ Extract text        │  ~0.5s
└────────┬────────────────┘
         │
         ▼ if score < 0.70 or tesseract used
┌─────────────────────────┐
│  Feedback Storage       │  Save (image, text) pair for fine-tuning
│  └─ feedback/pending/   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  JSON Response          │
│  ├─ text, raw_text      │
│  ├─ score + variables   │
│  ├─ flag (green/yellow/ │
│  │       red)           │
│  ├─ ocr_engine          │
│  └─ needs_external_ocr  │
└─────────────────────────┘
```

## Quick Start

```bash
# Set model path (default: /workspace/models/DeepSeek-OCR)
export MODEL_PATH=/workspace/models/DeepSeek-OCR

# Start the service
./start.sh

# Or run directly
python3 api_service.py
```

The service starts on `http://0.0.0.0:8000`. Check readiness:

```bash
curl http://localhost:8000/health
```

## API Endpoints

### OCR Endpoints

#### `POST /ocr/image`

OCR a single image with quality scoring and optional retry.

```bash
# Basic usage
curl -X POST http://localhost:8000/ocr/image \
  -F "file=@document.png"

# With options
curl -X POST http://localhost:8000/ocr/image \
  -F "file=@document.png" \
  -F "prompt=document" \
  -F "raw=false" \
  -F "retry=true"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Image file (JPEG, PNG, etc.) |
| `prompt` | string | `document` | Prompt type: `document`, `ocr`, `free_ocr`, `figure`, `describe` |
| `raw` | bool | `false` | Return raw output with grounding annotations |
| `retry` | bool | `true` | Retry with different enhancements on low scores |

#### `POST /ocr/image/base64`

Same as `/ocr/image` but accepts base64-encoded image data.

```bash
curl -X POST http://localhost:8000/ocr/image/base64 \
  -F "image_base64=$(base64 -w0 document.png)"
```

#### `POST /ocr/pdf`

OCR all pages of a PDF with per-page scoring and retry.

```bash
curl -X POST http://localhost:8000/ocr/pdf \
  -F "file=@document.pdf" \
  -F "dpi=144" \
  -F "retry=true"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | PDF file |
| `dpi` | int | `144` | Resolution for PDF rendering |
| `prompt` | string | `document` | Prompt type |
| `raw` | bool | `false` | Return raw output |
| `retry` | bool | `true` | Retry low-scoring pages |

#### `POST /ocr/batch`

OCR multiple images in a single request.

```bash
curl -X POST http://localhost:8000/ocr/batch \
  -F "files=@page1.png" \
  -F "files=@page2.png" \
  -F "files=@page3.png"
```

### Feedback Endpoints

#### `GET /feedback/stats`

```bash
curl http://localhost:8000/feedback/stats
```

```json
{
  "pending": 5,
  "verified": 12,
  "total": 17,
  "ready_for_training": false,
  "engines": {"deepseek": 4, "tesseract": 1},
  "disk_usage_mb": 3.42
}
```

#### `GET /feedback/pending`

List pages awaiting correction.

```bash
curl http://localhost:8000/feedback/pending
```

```json
{
  "entries": [
    {
      "entry_id": "20260414_133921_4520e5346794",
      "timestamp": "20260414_133921",
      "filename": "page_2.png",
      "score": 0.5,
      "flag": "yellow",
      "ocr_engine": "tesseract",
      "text_length": 2260
    }
  ],
  "total": 1
}
```

#### `POST /feedback/correct`

Submit corrected text for a pending entry.

```bash
curl -X POST http://localhost:8000/feedback/correct \
  -F "entry_id=20260414_133921_4520e5346794" \
  -F "corrected_text=Form RD 3560-8 USDA RURAL HOUSING SERVICE..."
```

## Response Format

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
    }
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

### `ocr_engine` Values

| Value | Meaning |
|-------|---------|
| `deepseek` | Primary model extracted text successfully |
| `tesseract` | DeepSeek failed, Tesseract fallback extracted text |
| `skipped` | Page skipped (blank or low-quality scan) |

### `flag` Values

| Flag | Composite Score | Action |
|------|----------------|--------|
| `green` | >= 0.70 | Good quality, use as-is |
| `yellow` | 0.50 - 0.69 | Spot-check recommended |
| `red` | < 0.50 | Manual review required |

### `flag_details` Codes

| Code | Severity | Meaning |
|------|----------|---------|
| `no_content` | critical | No meaningful text extracted |
| `ocr_failed` | critical | Page has content but model couldn't read it |
| `blank_page` | critical | Blank page detected and skipped |
| `low_quality_scan` | critical | Content too small to read, skipped |
| `tesseract_fallback` | warning | Text extracted by Tesseract, verify accuracy |
| `possible_hallucination` | warning | >75% of output removed as hallucinated |
| `max_tokens_hit` | warning | Model stuck in generation loop |
| `repetitive_content` | info | Output contains repetitive patterns |
| `sparse_content` | info | Very little text relative to image size |
| `low_content` | info | Less than 30 characters extracted |

## Quality Scoring System

Six weighted metrics produce a composite score (0.0-1.0):

### Scoring Variables

| Variable | Weight | What it measures |
|----------|--------|------------------|
| `hallucination_ratio` | 0.25 | Ratio of clean text to raw output. Low = most content was hallucinated and removed |
| `self_consistency` | 0.20 | Text similarity across retry attempts. 1.0 for single runs (deterministic at temp=0) |
| `token_efficiency` | 0.20 | Penalizes hitting max tokens with minimal output (stuck generation loops) |
| `content_density` | 0.15 | Text volume relative to image area. Low = under-extraction |
| `structural_integrity` | 0.10 | Presence of recognizable patterns: headers, tables, dates, amounts |
| `repetition_density` | 0.10 | Remaining repetitive n-grams after cleanup. Low = possible hallucination |

### Composite Calculation

```
composite = 0.25 * hallucination_ratio
          + 0.20 * self_consistency
          + 0.20 * token_efficiency
          + 0.15 * content_density
          + 0.10 * structural_integrity
          + 0.10 * repetition_density
```

**Blank page caps** (prevent score inflation on empty output):
- Clean text <= 10 chars: composite capped at 0.10
- Clean text <= 30 chars: composite capped at 0.30

### Retry Logic

When `retry=true` and score < 0.60:

```
Attempt 1: "adaptive" preset  → auto contrast/sharpness  → score → good enough? stop
Attempt 2: "none" preset      → no enhancement           → score → good enough? stop
Attempt 3: "strong" preset    → contrast 1.5x, sharp 2x  → score → select best result
```

## Pre-flight Page Detection

Before any OCR processing, two instant checks run (~1ms):

### Blank Page Detection

Detects pure white/empty pages.

- Pixel standard deviation < 5.0
- Dark pixel ratio (< 240) < 2%
- Result: RED flag, `blank_page` detail, zero text, no OCR performed

### Low-Quality Scan Detection

Detects scans where content is shrunk to a tiny unreadable area.

- Finds content bounding box (rows/cols with >1% dark pixels)
- If content area < 12% of total page area: skip
- Catches: faxed documents, thumbnail-quality scans, shrunken copies
- Result: RED flag, `low_quality_scan` detail, zero text, no OCR performed

## Tesseract Fallback

When DeepSeek-OCR extracts no meaningful text (<=10 chars) from a non-blank page:

1. Image preprocessed with adaptive thresholding (removes watermarks)
2. Tesseract OCR runs on the preprocessed image
3. If text extracted: returns with `ocr_engine: "tesseract"`, flag YELLOW
4. If Tesseract also fails: returns with `needs_external_ocr: true`

Disable with: `TESSERACT_FALLBACK=false`

## Feedback System

Automatically saves low-scoring OCR results for future model fine-tuning.

### How It Works

```
1. Page scores below 0.70 or uses Tesseract fallback
   → Image + metadata saved to feedback/pending/

2. Downstream AI or human sends corrected text
   → POST /feedback/correct
   → Entry moves to feedback/verified/

3. When 50+ verified pairs accumulated
   → Ready for LoRA fine-tuning
   → Model learns your specific document types
   → Fewer failures over time
```

### Storage Structure

```
feedback/
├── pending/                          ← awaiting correction
│   ├── 20260414_134025_e3e12e1e911b.png   ← original image
│   └── 20260414_134025_e3e12e1e911b.json  ← OCR result + metadata
└── verified/                         ← corrected, ready for training
    ├── 20260414_133921_4520e5346794.png
    └── 20260414_133921_4520e5346794.json
```

### What Gets Saved

- Pages scoring **below 0.70** (YELLOW/RED)
- All **Tesseract fallback** results
- GREEN pages are **not saved** (no disk wasted)
- Estimated storage: ~210 KB per page (image + metadata)

### Disk Usage Estimate

| Volume | Storage |
|--------|---------|
| 100 pages | ~21 MB |
| 1,000 pages | ~210 MB |
| 10,000 pages | ~2.1 GB |

Only ~5% of pages are typically saved (failures only).

Disable with: `FEEDBACK_ENABLED=false`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/DeepSeek-OCR` | Path to model weights |
| `PORT` | `8000` | API port |
| `HOST` | `0.0.0.0` | API host |
| `GPU_MEM_UTIL` | `0.75` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Max model context length |
| `MAX_TOKENS` | `8192` | Max output tokens |
| `MAX_IMAGE_SIZE_MB` | `20` | Max image upload size |
| `MAX_PDF_SIZE_MB` | `100` | Max PDF upload size |
| `MAX_PDF_PAGES` | `50` | Max pages per PDF |
| `MAX_BATCH_SIZE` | `16` | Max images per batch |
| `REQUEST_TIMEOUT_S` | `120` | Request timeout (seconds) |
| `SCORE_THRESHOLD` | `0.60` | Score below this triggers retry |
| `MAX_RETRIES` | `3` | Max retry attempts |
| `TESSERACT_FALLBACK` | `true` | Enable Tesseract fallback |
| `FEEDBACK_DIR` | `/workspace/DeepSeek-OCR-1/feedback` | Feedback storage path |
| `FEEDBACK_ENABLED` | `true` | Enable feedback storage |
| `FEEDBACK_SCORE_THRESHOLD` | `0.70` | Save results below this score |

## Post-processing Pipeline

Raw model output goes through these cleanup stages:

1. **Remove grounding tags** — `<|ref|>...<|/ref|>`, `<|det|>...<|/det|>`, coordinate arrays
2. **Table cleanup** — Trim empty cells (max 15/row), remove bloated tables (>100 empty cells), cap at 60 rows
3. **Collapse repeating patterns** — Remove hallucinated number sequences, digit-space runs, single-char repeats
4. **Section deduplication** — Detect duplicate markdown headers, keep the more complete version
5. **Whitespace normalization** — Collapse 3+ newlines to 2, multiple spaces to 1

## Project Structure

```
DeepSeek-OCR-1/
├── api_service.py                    ← Main FastAPI service
├── start.sh                          ← Entrypoint script
├── requirements.txt                  ← Python dependencies
├── README.md                         ← This file
├── images/                           ← Input images
├── feedback/                         ← Feedback storage (auto-created)
│   ├── pending/                      ← Awaiting correction
│   └── verified/                     ← Ready for fine-tuning
└── DeepSeek-OCR-master/
    └── DeepSeek-OCR-vllm/
        ├── config.py                 ← vLLM configuration
        ├── deepseek_ocr.py           ← Model implementation
        └── process/
            ├── __init__.py           ← Public API exports
            ├── image_process.py      ← Image tokenization (DO NOT MODIFY)
            ├── ngram_norepeat.py     ← Logits processor (DO NOT MODIFY)
            ├── postprocess.py        ← Output cleanup
            ├── enhance.py            ← Image enhancement
            └── score.py              ← Quality scoring system
```

## Performance

Tested on 54-page housing compliance document set:

| Metric | Value |
|--------|-------|
| Average processing time | ~4.6s per page |
| GREEN (good quality) | 94.4% of pages |
| YELLOW (spot-check) | 3.7% of pages |
| RED (manual review) | 1.9% of pages |
| Average composite score | 0.876 |
| Blank/low-quality skip time | ~1ms (instant) |
| Tesseract fallback success | 100% (1/1 pages) |
| Total chars extracted | 131,610 |

## Model Details

- **Model**: DeepSeek-OCR (bfloat16)
- **Size**: ~6.2 GB VRAM
- **Inference**: vLLM v0.8.4, temperature 0 (deterministic)
- **Max context**: 8,192 tokens
- **N-gram no-repeat**: 20-token window prevents generation loops
- **Concurrency**: Single semaphore serializes GPU calls (1 request at a time)
- **Workers**: Always 1 uvicorn worker (prevents model duplication)
