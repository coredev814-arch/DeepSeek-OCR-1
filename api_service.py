"""
DeepSeek-OCR API Service
Production-ready FastAPI service for RunPod deployment.

Changes from original:
- asyncio.Semaphore guards all GPU inference (prevents concurrent llm.generate crashes)
- Modern FastAPI lifespan replaces deprecated @app.on_event("startup")
- Graceful shutdown cleans up ThreadPoolExecutor and GPU memory
- File size limits prevent OOM from oversized uploads
- Per-request timeout via asyncio.wait_for
- Batch endpoint returns partial results on per-item failures
- Removed nested ThreadPoolExecutor creation inside requests
- Fixed uvicorn workers locked to 1 (multi-process would duplicate the model)
- Consistent response schema for raw/non-raw modes
- Structured logging replaces print statements
- Modular scoring/retry system for quality assurance
"""

import asyncio
import base64
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import torch

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
from vllm import LLM, SamplingParams
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
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.75"))
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
# Global model instances (initialized in lifespan)
# ---------------------------------------------------------------------------
llm: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None
processor: Optional[DeepseekOCRProcessor] = None
thread_pool: Optional[ThreadPoolExecutor] = None

# Semaphore prevents concurrent llm.generate() calls that would conflict on GPU
_inference_semaphore: Optional[asyncio.Semaphore] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


async def preprocess_images_batch(
    images: list[Image.Image], prompt_key: str
) -> list[dict]:
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(thread_pool, preprocess_image, img, prompt_key)
        for img in images
    ]
    return await asyncio.gather(*futures)


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


async def _run_inference(inputs: list[dict]) -> list:
    """Run llm.generate() with semaphore protection."""
    loop = asyncio.get_event_loop()

    async with _inference_semaphore:
        try:
            outputs = await asyncio.wait_for(
                loop.run_in_executor(
                    None, llm.generate, inputs, sampling_params
                ),
                timeout=REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                504, f"Inference timed out after {REQUEST_TIMEOUT_S}s"
            )
        except Exception as e:
            logger.error("Inference failed: %s", e, exc_info=True)
            raise HTTPException(500, f"Inference error: {e}")

    return outputs


def _format_result(output, raw: bool) -> dict:
    """Build a consistent result dict from a single vLLM output."""
    text = output.outputs[0].text
    stats = CleanStats()
    cleaned = clean_output(text, stats=stats)
    num_tokens = len(output.outputs[0].token_ids)

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

    return {
        "text": text if raw else cleaned,
        "raw_text": text,
        "num_tokens": num_tokens,
        "score": score.to_dict(),
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
    }


async def _run_inference_with_retry(
    image: Image.Image,
    prompt_key: str,
    raw_image_data: Optional[bytes] = None,
) -> dict:
    """Run OCR with scoring and retry on low-quality results.

    Tries different enhancement presets and returns the best-scoring result.
    """
    results: list[OCRResult] = []

    for attempt, preset in enumerate(ENHANCEMENT_PRESETS):
        if attempt > 0 and results and not needs_retry(results[-1], SCORE_THRESHOLD):
            break  # previous result was good enough
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

        loop = asyncio.get_event_loop()
        vllm_input = await loop.run_in_executor(
            thread_pool, preprocess_image, enhanced, prompt_key
        )
        outputs = await _run_inference([vllm_input])

        text = outputs[0].outputs[0].text
        retry_stats = CleanStats()
        cleaned = clean_output(text, stats=retry_stats)
        num_tokens = len(outputs[0].outputs[0].token_ids)

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

    return {
        "text": best.clean_text,
        "raw_text": best.raw_text,
        "num_tokens": best.num_tokens,
        "score": best.score.to_dict() if best.score else None,
        "flag": flag_info["flag"],
        "flag_message": flag_info["message"],
        "flag_details": flag_info["details"],
        "attempts": len(results),
        "preset": best.preset_name,
    }


def _check_file_size(data: bytes, max_mb: int, label: str = "File"):
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            413, f"{label} is {size_mb:.1f} MB, maximum allowed is {max_mb} MB"
        )


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model.  Shutdown: release resources."""
    global llm, sampling_params, processor, thread_pool, _inference_semaphore

    # ---- Startup ----
    logger.info("Loading model from %s …", MODEL_PATH)
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=MODEL_PATH,
        task="generate",
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        swap_space=0,
        max_num_seqs=MAX_CONCURRENCY,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        disable_mm_preprocessor_cache=True,
    )

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
    thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    _inference_semaphore = asyncio.Semaphore(1)

    logger.info("Model loaded and ready.")

    yield  # ---- App runs here ----

    # ---- Shutdown ----
    logger.info("Shutting down …")
    if thread_pool:
        thread_pool.shutdown(wait=False)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")


app = FastAPI(
    title="DeepSeek-OCR API",
    description="Production OCR API powered by DeepSeek-OCR",
    version="3.0.0",
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
        "status": "healthy" if llm is not None else "loading",
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

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        return JSONResponse(result)
    else:
        image = enhance_scan(image).convert("RGB")
        loop = asyncio.get_event_loop()
        vllm_input = await loop.run_in_executor(
            thread_pool, preprocess_image, image, prompt
        )
        outputs = await _run_inference([vllm_input])
        return JSONResponse(_format_result(outputs[0], raw))


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

    if retry:
        result = await _run_inference_with_retry(image, prompt)
        if raw:
            result["text"] = result["raw_text"]
        return JSONResponse(result)
    else:
        image = enhance_scan(image).convert("RGB")
        loop = asyncio.get_event_loop()
        vllm_input = await loop.run_in_executor(
            thread_pool, preprocess_image, image, prompt
        )
        outputs = await _run_inference([vllm_input])
        return JSONResponse(_format_result(outputs[0], raw))


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
    images = await loop.run_in_executor(thread_pool, pdf_to_images, pdf_bytes, dpi)

    if not images:
        raise HTTPException(400, "Could not extract any pages from the PDF")

    # First pass: batch all pages
    batch_inputs = await preprocess_images_batch(images, prompt)
    outputs = await _run_inference(batch_inputs)

    pages = []
    retry_indices = []

    for i, output in enumerate(outputs):
        result = _format_result(output, raw)
        result["page"] = i + 1
        pages.append(result)

        if retry and result["score"]["composite"] < SCORE_THRESHOLD:
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

    if enhanced_images:
        batch_inputs = await preprocess_images_batch(enhanced_images, prompt)
        outputs = await _run_inference(batch_inputs)

        retry_indices = []
        for j, output in enumerate(outputs):
            original_idx = valid_indices[j]
            result = _format_result(output, raw)
            result["index"] = original_idx
            result["filename"] = files[original_idx].filename
            results.append(result)

            if retry and result["score"]["composite"] < SCORE_THRESHOLD:
                retry_indices.append(j)

        # Retry low-scoring images
        for j in retry_indices:
            original_idx = valid_indices[j]
            logger.info(
                "Retrying %s (score=%.3f)",
                files[original_idx].filename,
                results[j]["score"]["composite"],
            )
            retry_result = await _run_inference_with_retry(
                raw_images[j], prompt
            )
            if raw:
                retry_result["text"] = retry_result["raw_text"]
            retry_result["index"] = original_idx
            retry_result["filename"] = files[original_idx].filename

            if retry_result.get("score", {}).get("composite", 0) > results[j]["score"]["composite"]:
                results[j] = retry_result

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
# Main — workers is always 1 to avoid duplicating the model in GPU memory
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_service:app",
        host=HOST,
        port=PORT,
        workers=1,
    )
