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
"""

import asyncio
import base64
import io
import logging
import os
import re
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
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

from config import CROP_MODE, MAX_CONCURRENCY, NUM_WORKERS
from deepseek_ocr import DeepseekOCRForCausalLM
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
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.9"))
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
    """Raise 400 if prompt key is unknown, otherwise return it."""
    if prompt not in PROMPTS:
        raise HTTPException(
            400,
            f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}",
        )
    return prompt


def clean_output(text: str) -> str:
    """Remove grounding annotations and clean up the OCR output."""
    text = text.replace("<｜end▁of▁sentence｜>", "")

    # Remove grounding refs (non-image)
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if "<|ref|>image<|/ref|>" not in match[0]:
            text = text.replace(match[0], "")

    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL Image from bytes with EXIF correction."""
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
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
    """Preprocess a list of images using the module-level thread pool."""
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
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
    doc.close()
    return images


async def _run_inference(inputs: list[dict]) -> list:
    """
    Run llm.generate() with semaphore protection.

    This ensures only one generate() call uses the GPU at a time,
    preventing memory conflicts from concurrent requests.
    """
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
    cleaned = clean_output(text)
    return {
        "text": text if raw else cleaned,
        "raw_text": text,
        "num_tokens": len(output.outputs[0].token_ids),
    }


def _check_file_size(data: bytes, max_mb: int, label: str = "File"):
    """Raise 413 if data exceeds the size limit."""
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

    # Allow up to MAX_CONCURRENCY inference calls to queue, but only 1 runs at a time
    _inference_semaphore = asyncio.Semaphore(1)

    logger.info("Model loaded and ready.")

    yield  # ---- App runs here ----

    # ---- Shutdown ----
    logger.info("Shutting down …")
    if thread_pool:
        thread_pool.shutdown(wait=False)
    # Release GPU memory
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete.")


app = FastAPI(
    title="DeepSeek-OCR API",
    description="Production OCR API powered by DeepSeek-OCR",
    version="2.0.0",
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
    }


@app.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
):
    """
    OCR a single image.

    - **file**: Image file (JPEG, PNG, etc.)
    - **prompt**: Prompt type — one of: document, ocr, free_ocr, figure, describe
    - **raw**: If true, return raw output with grounding annotations
    """
    _validate_prompt(prompt)

    data = await file.read()
    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")
    image = load_image_from_bytes(data)

    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(thread_pool, preprocess_image, image, prompt)

    outputs = await _run_inference([vllm_input])

    return JSONResponse(_format_result(outputs[0], raw))


@app.post("/ocr/image/base64")
async def ocr_image_base64(
    image_base64: str = Form(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
):
    """
    OCR a single image from base64-encoded data.

    - **image_base64**: Base64-encoded image
    - **prompt**: Prompt type
    - **raw**: If true, return raw output with grounding annotations
    """
    _validate_prompt(prompt)

    try:
        data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    _check_file_size(data, MAX_IMAGE_SIZE_MB, "Image")
    image = load_image_from_bytes(data)

    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(thread_pool, preprocess_image, image, prompt)

    outputs = await _run_inference([vllm_input])

    return JSONResponse(_format_result(outputs[0], raw))


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    dpi: int = Form(144),
    raw: bool = Form(False),
):
    """
    OCR a PDF document (all pages).

    - **file**: PDF file
    - **prompt**: Prompt type
    - **dpi**: Resolution for PDF rendering (default 144)
    - **raw**: If true, return raw output with grounding annotations
    """
    _validate_prompt(prompt)

    pdf_bytes = await file.read()
    _check_file_size(pdf_bytes, MAX_PDF_SIZE_MB, "PDF")

    loop = asyncio.get_event_loop()
    images = await loop.run_in_executor(thread_pool, pdf_to_images, pdf_bytes, dpi)

    if not images:
        raise HTTPException(400, "Could not extract any pages from the PDF")

    # Preprocess all pages using the shared thread pool
    batch_inputs = await preprocess_images_batch(images, prompt)

    outputs = await _run_inference(batch_inputs)

    pages = [
        {
            "page": i + 1,
            **_format_result(output, raw),
        }
        for i, output in enumerate(outputs)
    ]

    full_text = "\n\n---\n\n".join(p["text"] for p in pages)

    return JSONResponse(
        {
            "num_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
            "total_tokens": sum(p["num_tokens"] for p in pages),
        }
    )


@app.post("/ocr/batch")
async def ocr_batch(
    files: list[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
):
    """
    OCR multiple images in a single batch.

    - **files**: Multiple image files (max MAX_BATCH_SIZE)
    - **prompt**: Prompt type
    - **raw**: If true, return raw output
    """
    _validate_prompt(prompt)

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            400,
            f"Too many files ({len(files)}). Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    # Load all images, collecting per-file errors
    images: list[Image.Image] = []
    valid_indices: list[int] = []
    errors: list[dict] = []

    for i, f in enumerate(files):
        try:
            data = await f.read()
            _check_file_size(data, MAX_IMAGE_SIZE_MB, f"File '{f.filename}'")
            images.append(load_image_from_bytes(data))
            valid_indices.append(i)
        except HTTPException as e:
            errors.append({"index": i, "filename": f.filename, "error": e.detail})
        except Exception as e:
            errors.append({"index": i, "filename": f.filename, "error": str(e)})

    results: list[dict] = []

    if images:
        batch_inputs = await preprocess_images_batch(images, prompt)

        outputs = await _run_inference(batch_inputs)

        for j, output in enumerate(outputs):
            original_idx = valid_indices[j]
            results.append(
                {
                    "index": original_idx,
                    "filename": files[original_idx].filename,
                    **_format_result(output, raw),
                }
            )

    return JSONResponse(
        {
            "results": results,
            "errors": errors if errors else None,
            "total": len(files),
            "succeeded": len(results),
            "failed": len(errors),
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