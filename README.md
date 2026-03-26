<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek-OCR" />
</div>

<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>
</div>

<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>Paper Link</b></a> |
  <a href="https://arxiv.org/abs/2510.18234"><b>Arxiv Paper Link</b></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Processing Pipeline](#processing-pipeline)
- [Quality Scoring System](#quality-scoring-system)
- [Flagging System](#flagging-system)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [CLI Usage](#cli-usage)
- [Supported Modes](#supported-modes)
- [Prompt Reference](#prompt-reference)

---

## Overview

DeepSeek-OCR is a production-grade document OCR system built on the DeepSeek-OCR multimodal LLM. It converts scanned documents, PDFs, and images into structured Markdown with HTML tables, complete with a multi-variable quality scoring system that automatically retries low-quality extractions using different image enhancement presets.

The system is deployed as a FastAPI service optimized for RunPod GPU pods, using vLLM for high-throughput inference.

### Key Capabilities

- Single image, batch image, and multi-page PDF processing
- Markdown output with HTML table preservation
- 6-variable weighted quality scoring per page
- Red/yellow/green flagging with detailed diagnostics
- Automatic retry with 3 image enhancement presets
- N-gram repetition prevention to reduce hallucination
- Post-processing cleanup for table artifacts and duplicated content

---

## Architecture

```
DeepSeek-OCR/
├── api_service.py                    # FastAPI REST service (production entry point)
├── start.sh                          # Container startup script
├── Dockerfile                        # RunPod GPU pod container
├── requirements.txt                  # Python dependencies
│
└── DeepSeek-OCR-master/
    ├── DeepSeek-OCR-vllm/            # vLLM-optimized implementation (primary)
    │   ├── config.py                 # Model and processing configuration
    │   ├── deepseek_ocr.py           # vLLM model wrapper
    │   ├── run_dpsk_ocr_image.py     # CLI: single image inference
    │   ├── run_dpsk_ocr_pdf.py       # CLI: PDF batch processing
    │   ├── run_dpsk_ocr_eval_batch.py# CLI: benchmark evaluation
    │   ├── deepencoder/              # Vision encoder (SAM ViT-B, CLIP-L)
    │   └── process/                  # Processing pipeline modules
    │       ├── score.py              # Quality scoring & flagging
    │       ├── postprocess.py        # Output cleanup (tables, dedup)
    │       ├── image_process.py      # Image tokenization & cropping
    │       ├── enhance.py            # Adaptive image enhancement
    │       └── ngram_norepeat.py     # Repetition prevention logits processor
    │
    └── DeepSeek-OCR-hf/              # HuggingFace transformers implementation (alternative)
        └── run_dpsk_ocr.py
```

---

## Processing Pipeline

Every document (image or PDF page) passes through six stages:

```
INPUT (Image / PDF page)
  │
  ├─[1] IMAGE ENHANCEMENT
  │     Adaptive contrast normalization (target RMS: 0.186)
  │     Sharpness enhancement (1.5x default)
  │     Three presets: adaptive, none, strong
  │
  ├─[2] IMAGE PREPROCESSING
  │     EXIF rotation correction
  │     Dynamic cropping into tile grid (2-9 tiles)
  │     Vision encoder tokenization (SAM ViT-B + CLIP)
  │     Base image: 1024x1024 (256 tokens)
  │     Local crops: 640x640 (100 tokens each)
  │
  ├─[3] MODEL INFERENCE (vLLM)
  │     DeepSeek-OCR multimodal LLM
  │     Temperature: 0.0 (deterministic)
  │     Max output: 8192 tokens
  │     N-gram repetition blocking (n=20, window=50)
  │
  ├─[4] POST-PROCESSING
  │     Remove grounding annotations
  │     Collapse empty table cells (max 15 per row)
  │     Remove hallucinated repeating patterns
  │     Deduplicate markdown sections
  │     Normalize whitespace
  │
  ├─[5] QUALITY SCORING
  │     Compute 6 independent metrics (0.0-1.0 each)
  │     Weighted composite score (0.0-1.0)
  │
  └─[6] FLAGGING & RETRY
        Red (<0.60) / Yellow (0.60-0.74) / Green (>=0.75)
        If score < threshold: retry with next enhancement preset
        Up to 3 attempts; return best result
```

---

## Quality Scoring System

Each OCR result is evaluated using six independent metrics combined into a weighted composite score.

### Scoring Weights

| Variable | Weight | What It Measures |
|---|---|---|
| `self_consistency` | **0.30** | Pairwise text similarity across multiple inference runs. Returns 0.5 with a single run. |
| `hallucination_ratio` | **0.25** | Ratio of cleaned text to raw text. Low ratio means post-processing removed a lot (likely hallucination). |
| `token_efficiency` | **0.15** | Penalizes runs that hit max tokens but produce little clean output. |
| `structural_integrity` | **0.15** | Detects expected document structures: headers, tables with content, paragraphs (>50 chars), data patterns (dates, currency, names). |
| `repetition_density` | **0.10** | Detects repeated 3-10 character sequences appearing 5+ times. |
| `content_density` | **0.05** | Ratio of extracted text length to image pixel area. |

### Composite Score Formula

```
composite = 0.30 * self_consistency
          + 0.25 * hallucination_ratio
          + 0.15 * token_efficiency
          + 0.15 * structural_integrity
          + 0.10 * repetition_density
          + 0.05 * content_density
```

All individual scores are clamped to `[0.0, 1.0]`. The composite score ranges from 0.0 (worst) to 1.0 (best).

---

## Flagging System

Each result is assigned a color flag based on the composite score and critical issue detection.

### Flag Thresholds

| Flag | Composite Score | Meaning |
|---|---|---|
| **Green** | >= 0.75 | Good quality. No action needed. |
| **Yellow** | 0.60 - 0.74 | Borderline. Spot-check recommended. |
| **Red** | < 0.60 | Low quality. Manual review required. |

### Critical Issue Overrides

Regardless of composite score, a result is flagged **Red** if:

- Fewer than 10 characters were extracted (no meaningful content)
- `hallucination_ratio` < 0.4 (more than 60% of raw output was removed as hallucinated)
- `token_efficiency` < 0.3 (max tokens hit with very little useful output)

### Flag Detail Codes

| Code | Severity | Trigger |
|---|---|---|
| `no_content` | critical | < 10 characters extracted |
| `possible_hallucination` | critical | hallucination_ratio < 0.4 |
| `max_tokens_hit` | critical | token_efficiency < 0.3 |
| `repetitive_content` | warning | repetition_density < 0.4 |
| `no_structure` | warning | structural_integrity < 0.3 |
| `low_content` | info | < 50 characters extracted |

---

## API Reference

The service exposes a REST API on port 8000 (configurable).

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "/workspace/models/DeepSeek-OCR",
  "gpu": "NVIDIA A100-SXM4-40GB",
  "scoring": {
    "threshold": 0.6,
    "max_retries": 3
  }
}
```

### Single Image OCR

```
POST /ocr/image
Content-Type: multipart/form-data
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Image file (JPEG, PNG) |
| `prompt` | string | `"document"` | Prompt type (see [Prompt Reference](#prompt-reference)) |
| `raw` | boolean | `false` | Return raw output with grounding annotations |
| `retry` | boolean | `true` | Enable retry on low-quality scores |

Response:
```json
{
  "text": "# Document Title\n\nExtracted markdown content...",
  "raw_text": "<|ref|>...<|/ref|>raw model output...",
  "num_tokens": 1245,
  "score": {
    "composite": 0.78,
    "variables": {
      "self_consistency": 0.85,
      "hallucination_ratio": 0.75,
      "token_efficiency": 1.0,
      "structural_integrity": 0.75,
      "repetition_density": 0.70,
      "content_density": 0.35
    },
    "weights": { ... }
  },
  "flag": "green",
  "flag_message": "Good quality score (0.78).",
  "flag_details": [],
  "attempts": 1,
  "preset": "adaptive"
}
```

### Single Image OCR (Base64)

```
POST /ocr/image/base64
Content-Type: application/json
```

Same parameters as `/ocr/image` but accepts `image_base64` (base64-encoded image string) instead of a file upload.

### PDF OCR

```
POST /ocr/pdf
Content-Type: multipart/form-data
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | PDF file (max 100 MB, 50 pages) |
| `prompt` | string | `"document"` | Prompt type |
| `dpi` | integer | `144` | PDF rendering resolution |
| `raw` | boolean | `false` | Return raw output |
| `retry` | boolean | `true` | Enable retry per page |

Response:
```json
{
  "job_id": "f59420d82923",
  "total_pages": 5,
  "pages": [
    {
      "page": 1,
      "text": "# Page content...",
      "image": "output/f59420d82923/processed/page_1.png",
      "flag": "green",
      "flag_message": "Good quality score (0.82).",
      "flag_details": [],
      "score": { "composite": 0.82, "variables": { ... }, "weights": { ... } }
    }
  ],
  "summary": { "green": 3, "yellow": 1, "red": 1 },
  "flagged_pages": [
    { "page": 4, "flag": "yellow", "flag_message": "...", "score": 0.67 }
  ]
}
```

### Batch Image OCR

```
POST /ocr/batch
Content-Type: multipart/form-data
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `files` | file[] | required | Multiple images (max 16) |
| `prompt` | string | `"document"` | Prompt type |
| `raw` | boolean | `false` | Return raw output |
| `retry` | boolean | `true` | Enable retry |

Response:
```json
{
  "results": [ { "index": 0, "filename": "page1.jpg", "text": "...", "score": { ... }, "flag": "green" } ],
  "errors": [],
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "summary": { "green": 2, "yellow": 1, "red": 0 },
  "flagged_results": [ ... ]
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/workspace/models/DeepSeek-OCR` | Path to model weights |
| `PORT` | `8000` | API listening port |
| `GPU_MEM_UTIL` | `0.75` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Model context length |
| `MAX_TOKENS` | `8192` | Max output tokens per inference |
| `MAX_IMAGE_SIZE_MB` | `20` | Max upload size for images |
| `MAX_PDF_SIZE_MB` | `100` | Max upload size for PDFs |
| `MAX_PDF_PAGES` | `50` | Max pages per PDF |
| `MAX_BATCH_SIZE` | `16` | Max images per batch request |
| `REQUEST_TIMEOUT_S` | `120` | Per-request timeout (seconds) |
| `SCORE_THRESHOLD` | `0.6` | Composite score below this triggers retry |
| `MAX_RETRIES` | `3` | Max retry attempts per page |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |

### Image Enhancement Presets

The retry system cycles through these presets when a page scores below the threshold:

| Preset | Contrast | Sharpness | Description |
|---|---|---|---|
| `adaptive` | Auto (target RMS 0.186) | 1.5x | Measures image contrast and adjusts to optimal target |
| `none` | 1.0x (no change) | 1.0x (no change) | Original image, no enhancement |
| `strong` | 1.5x | 2.0x | Aggressive enhancement for faded/low-contrast scans |

### Model Configuration

Set in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

| Setting | Default | Description |
|---|---|---|
| `BASE_SIZE` | `1024` | Base image resolution for global view |
| `IMAGE_SIZE` | `640` | Local crop resolution |
| `CROP_MODE` | `True` | Enable dynamic tiling (Gundam mode) |
| `MIN_CROPS` | `2` | Minimum tile grid |
| `MAX_CROPS` | `9` | Maximum tile grid |
| `MAX_CONCURRENCY` | `100` | Max parallel requests to vLLM |
| `NUM_WORKERS` | `64` | Image preprocessing thread pool size |

---

## Deployment

### Docker (RunPod)

```bash
# Build
docker build -t deepseek-ocr-api .

# Run
docker run --gpus all -p 8000:8000 deepseek-ocr-api
```

The Dockerfile:
1. Starts from `runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
2. Installs Python dependencies, vLLM 0.8.5, FastAPI, and flash-attn
3. Downloads model weights from HuggingFace at build time
4. Exposes port 8000 and runs `start.sh`

### Startup Sequence

1. Register `DeepseekOCRForCausalLM` with vLLM model registry
2. Load model onto GPU with configured memory utilization
3. Initialize `DeepseekOCRProcessor` for image tokenization
4. Create thread pool (64 workers) for image preprocessing
5. Initialize inference semaphore (serializes GPU access)
6. Begin accepting requests on port 8000

### Resource Requirements

- **GPU**: ~30-40 GB VRAM (A100-40G recommended)
- **CPU**: 64 preprocessing threads
- **Disk**: ~15 GB for model weights

### Graceful Shutdown

The service handles shutdown signals by:
- Draining the thread pool executor
- Clearing CUDA cache
- Releasing GPU memory

---

## CLI Usage

For direct inference without the API server:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```

### Single Image (Streaming Output)

```bash
python run_dpsk_ocr_image.py
```

### PDF Batch Processing

```bash
python run_dpsk_ocr_pdf.py
```

### Benchmark Evaluation

```bash
python run_dpsk_ocr_eval_batch.py
```

### HuggingFace Transformers (Alternative)

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='your_image.jpg',
    base_size=1024,
    image_size=640,
    crop_mode=True,
)
```

---

## Supported Modes

### Resolution Options

| Mode | Resolution | Vision Tokens |
|---|---|---|
| Tiny | 512 x 512 | 64 |
| Small | 640 x 640 | 100 |
| **Base** (default) | **1024 x 1024** | **256** |
| Large | 1280 x 1280 | 400 |
| **Gundam** (default dynamic) | **n x 640x640 + 1 x 1024x1024** | **Variable** |

The production API uses **Gundam mode** (dynamic resolution with cropping) by default, providing the best balance of detail and token efficiency.

---

## Prompt Reference

| Key | Prompt String | Best For |
|---|---|---|
| `document` (default) | `<image>\n<\|grounding\|>Convert the document to markdown.` | Scanned documents, forms, certificates |
| `ocr` | `<image>\n<\|grounding\|>OCR this image.` | General text-in-image extraction |
| `free_ocr` | `<image>\nFree OCR.` | Text extraction without layout preservation |
| `figure` | `<image>\nParse the figure.` | Charts, diagrams, figures in documents |
| `describe` | `<image>\nDescribe this image in detail.` | General image description |

---

## Anti-Hallucination Measures

The system uses multiple layers to prevent and detect hallucinated output:

1. **N-gram Logits Processor** — During generation, bans any 20-token n-gram that appeared in the last 50 tokens. Table cell tokens (`<td>`, `</td>`) are whitelisted, but hard-banned after 30+ consecutive empty cells.

2. **Post-Processing Cleanup** — After generation, removes:
   - Excessive empty table cells (>15 per row)
   - Repeating digit/number patterns
   - Duplicated markdown sections
   - Bloated tables (>100 empty cells trimmed to 60 rows max)

3. **Hallucination Ratio Scoring** — Compares raw output length to cleaned output length. If more than 60% was removed, the page is flagged red as `possible_hallucination`.

4. **Retry System** — Low-scoring pages are automatically reprocessed with different image enhancement settings, and the best result is kept.

---

## Release History

- **2026/01/27** — [DeepSeek-OCR2](https://github.com/deepseek-ai/DeepSeek-OCR-2) released
- **2025/10/23** — DeepSeek-OCR officially supported in upstream [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm)
- **2025/10/20** — Initial release of DeepSeek-OCR

## Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).
