"""
DeepSeek-OCR API Service
Production-ready FastAPI service for RunPod deployment.

Uses vLLM AsyncLLMEngine for concurrent inference — multiple requests
are batched on the GPU automatically without semaphore serialization.
"""

import asyncio
import base64
import io
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pytesseract
import torch
from scipy import ndimage

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add the vllm source directory to path so imports work
VLLM_SRC = os.path.join(
    os.path.dirname(__file__), "DeepSeek-OCR-master", "DeepSeek-OCR-vllm"
)
sys.path.insert(0, VLLM_SRC)

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from config import CROP_MODE, MAX_CONCURRENCY, NUM_WORKERS
from deepseek_ocr import DeepseekOCRForCausalLM
from process import (
    clean_output,
    CleanStats,
    enhance_scan,
    enhance_scan_with_preset,
    ENHANCEMENT_PRESETS,
    OCRResult,
    score_result,
    select_best_result,
    needs_retry,
    compute_flags,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_RETRIES,
)
from process.image_process import DeepseekOCRProcessor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("deepseek-ocr")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/DeepSeek-OCR")
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.80"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Safety limits
MAX_IMAGE_SIZE_MB = int(os.environ.get("MAX_IMAGE_SIZE_MB", "20"))
MAX_PDF_SIZE_MB = int(os.environ.get("MAX_PDF_SIZE_MB", "100"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "50"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "120"))

# Scoring / retry
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", str(DEFAULT_THRESHOLD)))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

# Tesseract fallback
TESSERACT_FALLBACK = os.environ.get("TESSERACT_FALLBACK", "true").lower() == "true"

# External OCR routing (returned in response when needs_external_ocr=true)
EXTERNAL_OCR_URL = os.environ.get("EXTERNAL_OCR_URL", "")

# Feedback storage
FEEDBACK_DIR = os.environ.get("FEEDBACK_DIR", os.path.join(os.path.dirname(__file__), "feedback"))
FEEDBACK_ENABLED = os.environ.get("FEEDBACK_ENABLED", "true").lower() == "true"
FEEDBACK_SCORE_THRESHOLD = float(os.environ.get("FEEDBACK_SCORE_THRESHOLD", "0.70"))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\n<|grounding|>OCR this image.",
    "free_ocr": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}
DEFAULT_PROMPT = "document"

# ---------------------------------------------------------------------------
# Global instances (initialized in lifespan)
# ---------------------------------------------------------------------------
engine: Optional[AsyncLLMEngine] = None
sampling_params: Optional[SamplingParams] = None
processor: Optional[DeepseekOCRProcessor] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_blank_page(image: Image.Image, std_threshold: float = 5.0, dark_threshold: float = 0.02) -> bool:
    """Fast pixel-based blank page detection. Returns True if the image is blank."""
    gray = np.array(image.convert("L"))
    if gray.std() >= std_threshold:
        return False
    dark_ratio = (gray < 240).sum() / gray.size
    return dark_ratio < dark_threshold


def is_low_quality_scan(image: Image.Image, content_area_threshold: float = 0.12) -> bool:
    """Detect scans where content is shrunk to a tiny area, making text unreadable.

    Checks the ratio of the content bounding box to the full page area.
    Returns True if content occupies less than ``content_area_threshold`` of the page.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    # Find rows/cols with meaningful dark pixels
    row_dark = (gray < 200).sum(axis=1)
    col_dark = (gray < 200).sum(axis=0)
    row_thresh = w * 0.01
    col_thresh = h * 0.01
    content_rows = np.where(row_dark > row_thresh)[0]
    content_cols = np.where(col_dark > col_thresh)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return True  # no content at all

    content_h = content_rows[-1] - content_rows[0]
    content_w = content_cols[-1] - content_cols[0]
    content_area_ratio = (content_h * content_w) / (h * w)

    return content_area_ratio < content_area_threshold


def _skip_page_result(reason: str, flag_detail: str) -> dict:
    """Return a pre-built result dict for a skipped page (no OCR needed)."""
    return {
        "text": "",
        "raw_text": "",
        "num_tokens": 0,
        "score": {
            "composite": 0.0,
            "variables": {
                "self_consistency": 0.0,
                "hallucination_ratio": 0.0,
                "token_efficiency": 0.0,
                "structural_integrity": 0.0,
                "repetition_density": 0.0,
                "content_density": 0.0,
            },
        },
        "flag": "red",
        "flag_message": reason,
        "flag_details": [flag_detail],
        "attempts": 0,
        "preset": None,
        "needs_external_ocr": False,
        "ocr_engine": "skipped",
    }


def _preprocess_for_tesseract(image: Image.Image) -> Image.Image:
    """Adaptive thresholding to remove watermarks and normalize contrast."""
    gray = np.array(image.convert("L")).astype(float)
    background = ndimage.gaussian_filter(gray, sigma=25)
    normalized = gray - background + 128
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized)


def _run_tesseract_fallback(image: Image.Image) -> Optional[dict]:
    """Try Tesseract OCR as fallback. Returns result dict or None."""
    try:
        preprocessed = _preprocess_for_tesseract(image)
        text = pytesseract.image_to_string(preprocessed, lang="eng")
        text = text.strip()

        if len(text) < 10:
            return None

        return {
            "text": text,
            "raw_text": text,
            "num_tokens": 0,
            "score": {
                "composite": 0.50,
                "variables": {
                    "self_consistency": 1.0,
                    "hallucination_ratio": 1.0,
                    "token_efficiency": 1.0,
                    "structural_integrity": 0.0,
                    "repetition_density": 1.0,
                    "content_density": 0.0,
                },
            },
            "flag": "yellow",
            "flag_message": f"Tesseract fallback — extracted {len(text)} chars. Verify accuracy.",
            "flag_details": [{
                "code": "tesseract_fallback",
                "severity": "warning",
                "message": "Text extracted by Tesseract fallback — accuracy may vary.",
            }],
            "attempts": 0,
            "preset": None,
            "needs_external_ocr": False,
            "ocr_engine": "tesseract",
        }
    except Exception as e:
        logger.warning("Tesseract fallback failed: %s", e)
        return None


def _save_feedback(image: Image.Image, result: dict, filename: str = None):
    """Save low-scoring OCR results for future fine-tuning."""
    if not FEEDBACK_ENABLED:
        return

    score = result.get("score", {}).get("composite", 1.0)
    engine_used = result.get("ocr_engine", "deepseek")

    # Save if score is below threshold or if Tesseract was used
    if score >= FEEDBACK_SCORE_THRESHOLD and engine_used != "tesseract":
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    entry_id = f"{timestamp}_{uuid.uuid4().hex[:12]}"

    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    # Save image
    img_path = os.path.join(pending_dir, f"{entry_id}.png")
    image.save(img_path, format="PNG")

    # Save metadata
    meta = {
        "id": entry_id,
        "timestamp": timestamp,
        "filename": filename,
        "ocr_engine": result.get("ocr_engine"),
        "score": result.get("score", {}).get("composite"),
        "flag": result.get("flag"),
        "text": result.get("text", ""),
        "raw_text": result.get("raw_text", ""),
        "attempts": result.get("attempts", 0),
        "corrected_text": None,  # filled by /feedback/correct
        "status": "pending",
    }
    meta_path = os.path.join(pending_dir, f"{entry_id}.json")
    import json as _json
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    logger.info("Feedback saved: %s (score=%.3f, engine=%s)", entry_id, score, result.get("ocr_engine"))


def _mark_needs_external_ocr(result: dict) -> None:
    """Set external OCR flag and include routing URL if configured."""
    result["needs_external_ocr"] = True
    if EXTERNAL_OCR_URL:
        result["external_ocr_url"] = EXTERNAL_OCR_URL


def _validate_prompt(prompt: str) -> str:
    if prompt not in PROMPTS:
        raise HTTPException(
            400,
            f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}",
        )
    return prompt


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from bytes with EXIF correction and scan enhancement."""
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = enhance_scan(image)
    return image.convert("RGB")


def preprocess_image(image: Image.Image, prompt_key: str = DEFAULT_PROMPT) -> dict:
    """Preprocess a single image into vLLM input format."""
    prompt = PROMPTS.get(prompt_key, PROMPTS[DEFAULT_PROMPT])
    features = processor.tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": features},
    }


def pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(doc) > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(
            400,
            f"PDF has {len(doc)} pages, maximum allowed is {MAX_PDF_PAGES}",
        )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img = enhance_scan(img).convert("RGB")
        images.append(img)
    doc.close()
    return images


async def _run_inference(image: Image.Image, prompt_key: str = DEFAULT_PROMPT) -> dict:
    """Run inference via AsyncLLMEngine. Supports true concurrent requests."""
    # Preprocess image synchronously (CPU work)
    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(None, preprocess_image, image, prompt_key)

    # Generate a unique request ID
    request_id = str(uuid.uuid4())

    # Submit to async engine — this returns an async generator
    results_generator = engine.generate(
        prompt=vllm_input,
        sampling_params=sampling_params,
        request_id=request_id,
    )

    # Collect the final output
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except Exception as e:
        logger.error("Inference failed for request %s: %s", request_id, e, exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")

    if final_output is None:
        raise HTTPException(500, "Inference returned no output")

    text = final_output.outputs[0].text
    num_tokens = len(final_output.outputs[0].token_ids)

    return {"text": text, "num_tokens": num_tokens}


def _format_result(inference_output: dict, raw: bool) -> dict:
    """Build a consistent result dict from inference output."""
    text = inference_output["text"]
    num_tokens = inference_output["num_tokens"]
    stats = CleanStats()
    cleaned = clean_output(text, stats=stats)

    # Score the result
    ocr_result = OCRResult(
        raw_text=text,
        clean_text=cleaned,
        num_tokens=num_tokens,
        max_tokens=MAX_TOKENS,
        clean_stats=stats,
    )
    score = score_result(ocr_result)
    flag_info = compute_flags(ocr_result, SCORE_THRESHOLD)

    result = {
        "text": text if raw else cleaned,
        "raw_text": text,
        "num_tokens": num_tokens,
        "score": score.to_dict(),
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
        "needs_external_ocr": False,
        "ocr_engine": "deepseek",
    }

    # Flag OCR extraction failure: page has content but model couldn't read it
    clean_len = len(cleaned.strip())
    raw_len = len(text.strip())
    if clean_len <= 10 and score.composite < SCORE_THRESHOLD:
        _mark_needs_external_ocr(result)
        if not any(d.get("code") == "ocr_failed" for d in result["flag_details"]):
            result["flag_details"].append({
                "code": "ocr_failed",
                "severity": "critical",
                "message": "OCR extraction failed — page has content but model could not read it. Route to external OCR.",
            })

    # Flag incomplete extraction: model hit max tokens and most output was hallucinated
    if (
        not result["needs_external_ocr"]
        and num_tokens >= MAX_TOKENS * 0.85
        and raw_len > 0
        and clean_len / raw_len < 0.10
        and score.composite < SCORE_THRESHOLD
    ):
        _mark_needs_external_ocr(result)
        result["flag_details"].append({
            "code": "incomplete_extraction",
            "severity": "critical",
            "message": f"Model hit token limit with {clean_len}/{raw_len} chars retained ({clean_len/raw_len*100:.0f}%). Most output was hallucinated. Route to external OCR.",
        })

    return result


async def _run_inference_with_retry(
    image: Image.Image,
    prompt_key: str,
) -> dict:
    """Run OCR with scoring and retry on low-quality results.

    Tries different enhancement presets and returns the best-scoring result.
    """
    results: list[OCRResult] = []

    # Score below which retrying is pointless — the page needs external OCR
    HOPELESS_THRESHOLD = 0.20

    for attempt, preset in enumerate(ENHANCEMENT_PRESETS):
        if attempt > 0 and results:
            last_score = results[-1].score.composite if results[-1].score else 0
            if not needs_retry(results[-1], SCORE_THRESHOLD):
                break  # previous result was good enough
            if last_score < HOPELESS_THRESHOLD:
                logger.info("Score %.3f < %.2f — skipping remaining retries", last_score, HOPELESS_THRESHOLD)
                break  # clearly hopeless, don't waste time
        if attempt >= MAX_RETRIES:
            break

        # Apply enhancement preset
        if preset["contrast"] is None:
            enhanced = enhance_scan(image)
        else:
            enhanced = enhance_scan_with_preset(
                image, preset["contrast"], preset["sharpness"]
            )
        enhanced = enhanced.convert("RGB")

        output = await _run_inference(enhanced, prompt_key)

        text = output["text"]
        num_tokens = output["num_tokens"]
        retry_stats = CleanStats()
        cleaned = clean_output(text, stats=retry_stats)

        ocr_result = OCRResult(
            raw_text=text,
            clean_text=cleaned,
            num_tokens=num_tokens,
            max_tokens=MAX_TOKENS,
            preset_name=preset["name"],
            clean_stats=retry_stats,
        )
        score_result(ocr_result, other_results=results, image_width=image.width, image_height=image.height)
        results.append(ocr_result)

        logger.info(
            "Attempt %d/%d (preset=%s): %d tokens, score=%.3f",
            attempt + 1,
            MAX_RETRIES,
            preset["name"],
            num_tokens,
            ocr_result.score.composite,
        )

        if not needs_retry(ocr_result, SCORE_THRESHOLD):
            break

    best = select_best_result(results)
    flag_info = compute_flags(best, SCORE_THRESHOLD)

    result = {
        "text": best.clean_text,
        "raw_text": best.raw_text,
        "num_tokens": best.num_tokens,
        "score": best.score.to_dict() if best.score else None,
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
        "attempts": len(results),
        "preset": best.preset_name,
        "needs_external_ocr": False,
        "ocr_engine": "deepseek",
    }

    # Flag OCR extraction failure: page has content but model couldn't read it
    clean_len = len(best.clean_text.strip())
    raw_len = len(best.raw_text.strip())
    composite = best.score.composite if best.score else 0
    if clean_len <= 10 and composite < SCORE_THRESHOLD:
        # Try Tesseract fallback before giving up
        if TESSERACT_FALLBACK:
            logger.info("DeepSeek-OCR failed — trying Tesseract fallback")
            fallback = _run_tesseract_fallback(image)
            if fallback:
                logger.info("Tesseract fallback extracted %d chars", len(fallback["text"]))
                fallback["attempts"] = len(results)
                return fallback

        _mark_needs_external_ocr(result)
        if not any(d.get("code") == "ocr_failed" for d in result["flag_details"]):
            result["flag_details"].append({
                "code": "ocr_failed",
                "severity": "critical",
                "message": "OCR extraction failed — page has content but model could not read it. Route to external OCR.",
            })

    # Flag incomplete extraction: model hit max tokens and most output was hallucinated
    if (
        not result["needs_external_ocr"]
        and best.num_tokens >= MAX_TOKENS * 0.85
        and raw_len > 0
        and clean_len / raw_len < 0.10
        and composite < SCORE_THRESHOLD
    ):
        _mark_needs_external_ocr(result)
        result["flag_details"].append({
            "code": "incomplete_extraction",
            "severity": "critical",
            "message": f"Model hit token limit with {clean_len}/{raw_len} chars retained ({clean_len/raw_len*100:.0f}%). Most output was hallucinated. Route to external OCR.",
        })

    return result


def _check_file_size(data: bytes, max_mb: int, label: str = "File"):
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            413, f"{label} is {size_mb:.1f} MB, maximum allowed is {max_mb} MB"
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load AsyncLLMEngine.  Shutdown: release resources."""
    global engine, sampling_params, processor

    # ---- Startup ----
    logger.info("Loading model from %s …", MODEL_PATH)
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        task="generate",
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=128,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        disable_mm_preprocessor_cache=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822},
            max_consecutive_empty_cells=30,
        )
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    processor = DeepseekOCRProcessor()

    logger.info("Model loaded and ready (async engine).")

    yield  # ---- App runs here ----

    # ---- Shutdown ----
    logger.info("Shutting down …")
    if engine:
        engine.shutdown_background_loop()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")


app = FastAPI(
    title="DeepSeek-OCR API",
    description="Production OCR API powered by DeepSeek-OCR",
    version="4.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR API", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if engine is not None else "loading",
        "model": MODEL_PATH,
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        ),
        "scoring": {
            "threshold": SCORE_THRESHOLD,
            "max_retries": MAX_RETRIES,
        },
    }


@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR a single image with quality scoring and optional retry.

    - **file**: Image file (JPEG, PNG, etc.)
    - **prompt**: Prompt type — one of: document, ocr, free_ocr, figure, describe
    - **raw**: If true, return raw output with grounding annotations
    - **retry**: If true, retry with different enhancements on low scores
    """
    _validate_prompt(prompt)

    data = await file.read()
    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")

    # Load original image (without enhancement — retry system handles it)
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Skip blank pages entirely
    if is_blank_page(image):
        logger.info("Blank page detected — skipping OCR")
        return JSONResponse(_skip_page_result("Blank page detected — skipped OCR", "blank_page"))

    # Skip low-quality scans where content is too small to read
    if is_low_quality_scan(image):
        logger.info("Low-quality scan detected — skipping OCR")
        return JSONResponse(_skip_page_result("Low-quality scan — content too small to read", "low_quality_scan"))

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        _save_feedback(image, result, filename=file.filename)
        return JSONResponse(result)
    else:
        enhanced = enhance_scan(image).convert("RGB")
        output = await _run_inference(enhanced, prompt)
        result = _format_result(output, raw)
        _save_feedback(image, result, filename=file.filename)
        return JSONResponse(result)


@app.post("/ocr/image/base64")
async def ocr_image_base64(
    image_base64: str = Form(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """OCR a single image from base64-encoded data."""
    _validate_prompt(prompt)

    try:
        data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")

    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Skip blank pages entirely
    if is_blank_page(image):
        logger.info("Blank page detected — skipping OCR")
        return JSONResponse(_skip_page_result("Blank page detected — skipped OCR", "blank_page"))

    # Skip low-quality scans where content is too small to read
    if is_low_quality_scan(image):
        logger.info("Low-quality scan detected — skipping OCR")
        return JSONResponse(_skip_page_result("Low-quality scan — content too small to read", "low_quality_scan"))

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        _save_feedback(image, result)
        return JSONResponse(result)
    else:
        enhanced = enhance_scan(image).convert("RGB")
        output = await _run_inference(enhanced, prompt)
        result = _format_result(output, raw)
        _save_feedback(image, result)
        return JSONResponse(result)


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    dpi: int = Form(144),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR a PDF document (all pages) with per-page scoring and retry.

    - **file**: PDF file
    - **prompt**: Prompt type
    - **dpi**: Resolution for PDF rendering (default 144)
    - **raw**: If true, return raw output with grounding annotations
    - **retry**: If true, retry low-scoring pages
    """
    _validate_prompt(prompt)

    pdf_bytes = await file.read()
    _check_file_size(pdf_bytes, MAX_PDF_SIZE_MB, "PDF")

    loop = asyncio.get_event_loop()
    images = await loop.run_in_executor(None, pdf_to_images, pdf_bytes, dpi)

    if not images:
        raise HTTPException(400, "Could not extract any pages from the PDF")

    # Detect blank and low-quality pages before OCR
    skip_flags = []  # None = process, str = reason to skip
    for img in images:
        if is_blank_page(img):
            skip_flags.append("blank_page")
        elif is_low_quality_scan(img):
            skip_flags.append("low_quality_scan")
        else:
            skip_flags.append(None)

    skip_count = sum(1 for s in skip_flags if s is not None)
    if skip_count:
        logger.info("Skipping %d page(s) (blank or low-quality) — no OCR needed", skip_count)

    # First pass: batch processable pages only
    pages = [None] * len(images)

    # Fill in skipped pages immediately
    skip_messages = {
        "blank_page": "Blank page detected — skipped OCR",
        "low_quality_scan": "Low-quality scan — content too small to read",
    }
    for i, skip in enumerate(skip_flags):
        if skip:
            result = _skip_page_result(skip_messages[skip], skip)
            result["page"] = i + 1
            pages[i] = result

    # Run OCR on processable pages concurrently
    processable_indices = [i for i, skip in enumerate(skip_flags) if skip is None]
    if processable_indices:
        async def _ocr_page(page_idx: int) -> tuple[int, dict]:
            output = await _run_inference(images[page_idx], prompt)
            result = _format_result(output, raw)
            result["page"] = page_idx + 1
            return page_idx, result

        ocr_tasks = [_ocr_page(i) for i in processable_indices]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        for item in ocr_results:
            if isinstance(item, Exception):
                logger.error("Page OCR failed: %s", item)
                continue
            page_idx, result = item
            pages[page_idx] = result

    retry_indices = []
    for i, result in enumerate(pages):
        if result is not None and skip_flags[i] is None and retry and result["score"]["composite"] < SCORE_THRESHOLD:
            retry_indices.append(i)

    # Retry low-scoring pages individually with different presets
    for idx in retry_indices:
        logger.info("Retrying page %d (score=%.3f < %.3f)", idx + 1, pages[idx]["score"]["composite"], SCORE_THRESHOLD)
        # Get the original (un-enhanced) PDF page image
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        original_img = Image.open(io.BytesIO(img_data))
        doc.close()

        retry_result = await _run_inference_with_retry(original_img, prompt)
        retry_result["page"] = idx + 1
        if raw:
            retry_result["text"] = retry_result["raw_text"]

        # Use retry result if it scored better
        if retry_result.get("score", {}).get("composite", 0) > pages[idx]["score"]["composite"]:
            pages[idx] = retry_result

    # Fill any None pages (from failed OCR)
    for i, p in enumerate(pages):
        if p is None:
            pages[i] = _skip_page_result("OCR failed", "ocr_failed")
            pages[i]["page"] = i + 1

    full_text = "\n\n---\n\n".join(p["text"] for p in pages)

    # Build summary by flag color
    summary = {"green": 0, "yellow": 0, "red": 0}
    flagged_pages = []
    for p in pages:
        color = p.get("flag", "yellow")
        summary[color] = summary.get(color, 0) + 1
        if color in ("yellow", "red"):
            flagged_pages.append({
                "page": p["page"],
                "flag": color,
                "flag_message": p.get("flag_message"),
                "score": p["score"]["composite"] if p.get("score") else None,
            })

    return JSONResponse(
        {
            "num_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
            "total_tokens": sum(p["num_tokens"] for p in pages),
            "summary": summary,
            "flagged_pages": flagged_pages,
        }
    )


@app.post("/ocr/batch")
async def ocr_batch(
    files: list[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
    retry: bool = Form(True),
):
    """
    OCR multiple images in a single batch with scoring.

    - **files**: Multiple image files (max MAX_BATCH_SIZE)
    - **prompt**: Prompt type
    - **raw**: If true, return raw output
    - **retry**: If true, retry low-scoring images
    """
    _validate_prompt(prompt)

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            400,
            f"Too many files ({len(files)}). Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    # Load all images
    raw_images: list[Image.Image] = []
    enhanced_images: list[Image.Image] = []
    valid_indices: list[int] = []
    errors: list[dict] = []

    for i, f in enumerate(files):
        try:
            data = await f.read()
            _check_file_size(data, MAX_IMAGE_SIZE_MB, f"File '{f.filename}'")
            try:
                img = Image.open(io.BytesIO(data))
            except Exception as e:
                raise HTTPException(400, f"Could not decode image: {e}")
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            raw_images.append(img)
            enhanced_images.append(enhance_scan(img).convert("RGB"))
            valid_indices.append(i)
        except HTTPException as e:
            errors.append({"index": i, "filename": f.filename, "error": e.detail})
        except Exception as e:
            errors.append({"index": i, "filename": f.filename, "error": str(e)})

    results: list[dict] = []

    # Separate skippable pages (blank or low-quality) before OCR
    skip_messages = {
        "blank_page": "Blank page detected — skipped OCR",
        "low_quality_scan": "Low-quality scan — content too small to read",
    }
    processable_raw = []
    processable_enhanced = []
    processable_valid = []
    for j, (img_raw, img_enh) in enumerate(zip(raw_images, enhanced_images)):
        original_idx = valid_indices[j]
        if is_blank_page(img_raw):
            skip_type = "blank_page"
        elif is_low_quality_scan(img_raw):
            skip_type = "low_quality_scan"
        else:
            skip_type = None

        if skip_type:
            logger.info("%s — skipping OCR for %s", skip_messages[skip_type], files[original_idx].filename)
            result = _skip_page_result(skip_messages[skip_type], skip_type)
            result["index"] = original_idx
            result["filename"] = files[original_idx].filename
            results.append(result)
        else:
            processable_raw.append(img_raw)
            processable_enhanced.append(img_enh)
            processable_valid.append((j, original_idx))

    if processable_enhanced:
        async def _ocr_batch_item(k: int) -> tuple[int, dict]:
            j, original_idx = processable_valid[k]
            output = await _run_inference(processable_enhanced[k], prompt)
            result = _format_result(output, raw)
            result["index"] = original_idx
            result["filename"] = files[original_idx].filename
            return k, result

        ocr_tasks = [_ocr_batch_item(k) for k in range(len(processable_enhanced))]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

        retry_queue = []  # (result_list_index, raw_image_index, original_file_index)
        for item in ocr_results:
            if isinstance(item, Exception):
                logger.error("Batch item OCR failed: %s", item)
                continue
            k, result = item
            j, original_idx = processable_valid[k]
            result_pos = len(results)
            results.append(result)

            if retry and result["score"]["composite"] < SCORE_THRESHOLD:
                retry_queue.append((result_pos, j, original_idx))

        # Retry low-scoring images
        for result_pos, j, original_idx in retry_queue:
            logger.info(
                "Retrying %s (score=%.3f)",
                files[original_idx].filename,
                results[result_pos]["score"]["composite"],
            )
            retry_result = await _run_inference_with_retry(
                raw_images[j], prompt
            )
            if raw:
                retry_result["text"] = retry_result["raw_text"]
            retry_result["index"] = original_idx
            retry_result["filename"] = files[original_idx].filename

            if retry_result.get("score", {}).get("composite", 0) > results[result_pos]["score"]["composite"]:
                results[result_pos] = retry_result

    # Build summary by flag color
    summary = {"green": 0, "yellow": 0, "red": 0}
    flagged_results = []
    for r in results:
        color = r.get("flag", "yellow")
        summary[color] = summary.get(color, 0) + 1
        if color in ("yellow", "red"):
            flagged_results.append({
                "index": r.get("index"),
                "filename": r.get("filename"),
                "flag": color,
                "flag_message": r.get("flag_message"),
                "score": r["score"]["composite"] if r.get("score") else None,
            })

    return JSONResponse(
        {
            "results": results,
            "errors": errors if errors else None,
            "total": len(files),
            "succeeded": len(results),
            "failed": len(errors),
            "summary": summary,
            "flagged_results": flagged_results,
        }
    )


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------


@app.post("/feedback/correct")
async def feedback_correct(
    entry_id: str = Form(...),
    corrected_text: str = Form(...),
):
    """Submit corrected text for a previously saved feedback entry.

    The downstream AI or human reviewer sends the correct text for a page
    that scored below threshold. These verified pairs are used for fine-tuning.
    """
    import json as _json

    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    verified_dir = os.path.join(FEEDBACK_DIR, "verified")
    os.makedirs(verified_dir, exist_ok=True)

    meta_path = os.path.join(pending_dir, f"{entry_id}.json")
    img_path = os.path.join(pending_dir, f"{entry_id}.png")

    if not os.path.exists(meta_path):
        raise HTTPException(404, f"Feedback entry '{entry_id}' not found")

    with open(meta_path) as f:
        meta = _json.load(f)

    meta["corrected_text"] = corrected_text
    meta["status"] = "verified"

    # Move to verified directory
    verified_meta = os.path.join(verified_dir, f"{entry_id}.json")
    verified_img = os.path.join(verified_dir, f"{entry_id}.png")

    with open(verified_meta, "w") as f:
        _json.dump(meta, f, indent=2)

    if os.path.exists(img_path):
        os.rename(img_path, verified_img)
    os.remove(meta_path)

    logger.info("Feedback verified: %s (%d chars corrected text)", entry_id, len(corrected_text))

    return JSONResponse({
        "status": "verified",
        "entry_id": entry_id,
        "corrected_length": len(corrected_text),
    })


@app.get("/feedback/stats")
async def feedback_stats():
    """Show feedback storage statistics."""
    import json as _json

    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    verified_dir = os.path.join(FEEDBACK_DIR, "verified")

    pending = 0
    verified = 0
    engines = {}

    for d, status in [(pending_dir, "pending"), (verified_dir, "verified")]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".json"):
                continue
            if status == "pending":
                pending += 1
            else:
                verified += 1
            try:
                with open(os.path.join(d, f)) as fp:
                    meta = _json.load(fp)
                eng = meta.get("ocr_engine", "unknown")
                engines[eng] = engines.get(eng, 0) + 1
            except Exception:
                pass

    # Estimate disk usage
    total_bytes = 0
    for d in [pending_dir, verified_dir]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                total_bytes += os.path.getsize(os.path.join(d, f))

    return JSONResponse({
        "pending": pending,
        "verified": verified,
        "total": pending + verified,
        "ready_for_training": verified >= 50,
        "engines": engines,
        "disk_usage_mb": round(total_bytes / (1024 * 1024), 2),
    })


@app.get("/feedback/pending")
async def feedback_pending():
    """List pending feedback entries awaiting correction."""
    import json as _json

    pending_dir = os.path.join(FEEDBACK_DIR, "pending")
    if not os.path.isdir(pending_dir):
        return JSONResponse({"entries": []})

    entries = []
    for f in sorted(os.listdir(pending_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(pending_dir, f)) as fp:
            meta = _json.load(fp)
        entries.append({
            "entry_id": meta["id"],
            "timestamp": meta["timestamp"],
            "filename": meta.get("filename"),
            "score": meta.get("score"),
            "flag": meta.get("flag"),
            "ocr_engine": meta.get("ocr_engine"),
            "text_length": len(meta.get("text", "")),
        })

    return JSONResponse({"entries": entries, "total": len(entries)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_service:app",
        host=HOST,
        port=PORT,
        workers=1,
    )
