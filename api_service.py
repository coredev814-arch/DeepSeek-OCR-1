"""
DeepSeek-OCR API Service
Production-ready FastAPI service for RunPod deployment.
"""

import asyncio
import base64
import io
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add the vllm source directory to path so imports work
VLLM_SRC = os.path.join(os.path.dirname(__file__), "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
sys.path.insert(0, VLLM_SRC)

import fitz  # PyMuPDF
import img2pdf
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
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/DeepSeek-OCR")
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.9"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
WORKERS = int(os.environ.get("WORKERS", "1"))

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
# Global model instances (initialized at startup)
# ---------------------------------------------------------------------------
llm: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None
processor: Optional[DeepseekOCRProcessor] = None
thread_pool: Optional[ThreadPoolExecutor] = None

app = FastAPI(
    title="DeepSeek-OCR API",
    description="Production OCR API powered by DeepSeek-OCR",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_output(text: str) -> str:
    """Remove grounding annotations and clean up the OCR output."""
    if "<｜end▁of▁sentence｜>" in text:
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
    image = Image.open(io.BytesIO(data))
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


def pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
    doc.close()
    return images


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global llm, sampling_params, processor, thread_pool

    print(f"[DeepSeek-OCR] Loading model from {MODEL_PATH} ...")
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
    print("[DeepSeek-OCR] Model loaded and ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
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
    if prompt not in PROMPTS:
        raise HTTPException(400, f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}")

    data = await file.read()
    image = load_image_from_bytes(data)

    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(thread_pool, preprocess_image, image, prompt)

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    result_text = outputs[0].outputs[0].text

    return JSONResponse({
        "text": result_text if raw else clean_output(result_text),
        "raw_text": result_text if not raw else None,
        "num_tokens": len(outputs[0].outputs[0].token_ids),
    })


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
    if prompt not in PROMPTS:
        raise HTTPException(400, f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}")

    try:
        data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    image = load_image_from_bytes(data)

    loop = asyncio.get_event_loop()
    vllm_input = await loop.run_in_executor(thread_pool, preprocess_image, image, prompt)

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    result_text = outputs[0].outputs[0].text

    return JSONResponse({
        "text": result_text if raw else clean_output(result_text),
        "num_tokens": len(outputs[0].outputs[0].token_ids),
    })


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
    if prompt not in PROMPTS:
        raise HTTPException(400, f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}")

    pdf_bytes = await file.read()

    loop = asyncio.get_event_loop()
    images = await loop.run_in_executor(thread_pool, pdf_to_images, pdf_bytes, dpi)

    if not images:
        raise HTTPException(400, "Could not extract any pages from the PDF")

    # Preprocess all pages in parallel
    def preprocess_all():
        from concurrent.futures import ThreadPoolExecutor as TPE
        with TPE(max_workers=min(NUM_WORKERS, len(images))) as pool:
            return list(pool.map(lambda img: preprocess_image(img, prompt), images))

    batch_inputs = await loop.run_in_executor(thread_pool, preprocess_all)

    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    pages = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        if "<｜end▁of▁sentence｜>" in text:
            text = text.replace("<｜end▁of▁sentence｜>", "")
        pages.append({
            "page": i + 1,
            "text": text if raw else clean_output(text),
            "num_tokens": len(output.outputs[0].token_ids),
        })

    full_text = "\n\n---\n\n".join(p["text"] for p in pages)

    return JSONResponse({
        "num_pages": len(pages),
        "pages": pages,
        "full_text": full_text,
        "total_tokens": sum(p["num_tokens"] for p in pages),
    })


@app.post("/ocr/batch")
async def ocr_batch(
    files: list[UploadFile] = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    raw: bool = Form(False),
):
    """
    OCR multiple images in a single batch.

    - **files**: Multiple image files
    - **prompt**: Prompt type
    - **raw**: If true, return raw output
    """
    if prompt not in PROMPTS:
        raise HTTPException(400, f"Unknown prompt type '{prompt}'. Choose from: {list(PROMPTS.keys())}")

    images = []
    for f in files:
        data = await f.read()
        images.append(load_image_from_bytes(data))

    loop = asyncio.get_event_loop()

    def preprocess_all():
        from concurrent.futures import ThreadPoolExecutor as TPE
        with TPE(max_workers=min(NUM_WORKERS, len(images))) as pool:
            return list(pool.map(lambda img: preprocess_image(img, prompt), images))

    batch_inputs = await loop.run_in_executor(thread_pool, preprocess_all)

    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        results.append({
            "index": i,
            "filename": files[i].filename,
            "text": text if raw else clean_output(text),
            "num_tokens": len(output.outputs[0].token_ids),
        })

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host=HOST, port=PORT, workers=WORKERS)
