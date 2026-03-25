import asyncio
import re
import os
import ast
import json
import logging

import torch

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

# ---------------------------------------------------------------------------
# Logging setup — replaces silent `except: pass` blocks
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


# ========================== SAFE PARSING ==========================


def safe_literal_eval(expr: str):
    """Use ast.literal_eval instead of eval() to avoid arbitrary code execution."""
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError) as e:
        logger.warning("safe_literal_eval failed on input: %s — %s", expr[:120], e)
        return None


# ========================== IMAGE LOADING ==========================


def load_image(image_path: str) -> Image.Image | None:
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        logger.error("Failed to load image %s: %s", image_path, e)
        return None


# ========================== REGEX PARSING ==========================


def re_match(text: str):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


# ========================== COORDINATE EXTRACTION ==========================


def extract_coordinates_and_label(ref_text, image_width: int, image_height: int):
    """Extract label type and bounding-box coordinates from a regex match tuple."""
    try:
        label_type = ref_text[1]
        # SAFE: use ast.literal_eval instead of eval
        cor_list = safe_literal_eval(ref_text[2])
        if cor_list is None:
            return None
    except Exception as e:
        logger.warning("extract_coordinates_and_label error: %s", e)
        return None

    return (label_type, cor_list)


# ========================== BOUNDING-BOX DRAWING ==========================

# Pre-defined color palette for consistent, distinguishable colours
_COLOR_PALETTE = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),
    (0, 130, 200),   (245, 130, 48),  (145, 30, 180),
    (70, 240, 240),  (240, 50, 230),  (210, 245, 60),
    (250, 190, 212), (0, 128, 128),   (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0),   (255, 215, 180),
    (0, 0, 128),     (128, 128, 128),
]


def draw_bounding_boxes(image: Image.Image, refs: list) -> Image.Image:
    """Draw labelled bounding boxes on *image* and return the annotated copy."""
    image_width, image_height = image.size

    # FIX: work in RGBA so overlay transparency is handled correctly
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_draw)
    draw_overlay = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result is None:
                continue

            label_type, points_list = result
            color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
            color_a = color + (40,)

            for points in points_list:
                if len(points) != 4:
                    logger.warning("Skipping malformed bbox (expected 4 values): %s", points)
                    continue

                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                # Crop detected image regions
                if label_type == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                    except Exception as e:
                        logger.error("Failed to crop/save image region %d: %s", img_idx, e)
                    img_idx += 1

                # Draw rectangle + label
                width = 4 if label_type == "title" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw_overlay.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                text_x = x1
                text_y = max(0, y1 - 15)
                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                draw.rectangle(
                    [text_x, text_y, text_x + text_w, text_y + text_h],
                    fill=(255, 255, 255, 180),
                )
                draw.text((text_x, text_y), label_type, font=font, fill=color)

        except Exception as e:
            logger.error("Error processing ref %d: %s", i, e)
            continue

    img_draw = Image.alpha_composite(img_draw, overlay)
    return img_draw.convert("RGB")


# ========================== vLLM ENGINE (singleton) ==========================


class OCREngine:
    """Wraps AsyncLLMEngine so the model is loaded once and reused."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.75,
        tensor_parallel_size: int = 1,
    ):
        engine_args = AsyncEngineArgs(
            model=model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=max_model_len,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("Initialising vLLM engine from %s …", model_path)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._request_counter = 0

    async def generate(
        self,
        image=None,
        prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.0,
        ngram_size: int = 30,
        window_size: int = 90,
    ) -> str:
        """Run a single generation request and return the full output text."""
        if not prompt:
            raise ValueError("prompt must not be empty")

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=ngram_size,
                window_size=window_size,
                # whitelist: <td>, </td>
                whitelist_token_ids={128821, 128822},
            )
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        self._request_counter += 1
        request_id = f"request-{self._request_counter}-{int(time.time())}"

        if image and "<image>" in prompt:
            request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        else:
            request = {"prompt": prompt}

        printed_length = 0
        final_output = ""

        async for request_output in self.engine.generate(
            request, sampling_params, request_id
        ):
            if request_output.outputs:
                full_text = request_output.outputs[0].text
                new_text = full_text[printed_length:]
                print(new_text, end="", flush=True)
                printed_length = len(full_text)
                final_output = full_text

        print()  # newline after streaming
        return final_output


# ========================== GEOMETRY RENDERING ==========================


def render_geometry(output_text: str, save_path: str):
    """If the model output contains line_type geometry data, render it."""
    if "line_type" not in output_text:
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    parsed = safe_literal_eval(output_text)
    if parsed is None:
        logger.warning("Could not parse geometry output")
        return

    try:
        lines = parsed["Line"]["line"]
        line_types = parsed["Line"]["line_type"]
        endpoints = parsed["Line"]["line_endpoint"]
    except (KeyError, TypeError) as e:
        logger.warning("Geometry data missing expected keys: %s", e)
        return

    fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    for idx, line in enumerate(lines):
        try:
            parts = line.split(" -- ")
            p0 = safe_literal_eval(parts[0])
            p1 = safe_literal_eval(parts[-1])
            if p0 is None or p1 is None:
                continue

            style = "--" if line_types[idx] == "--" else "-"
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle=style, linewidth=0.8, color="k")
            ax.scatter(p0[0], p0[1], s=5, color="k")
            ax.scatter(p1[0], p1[1], s=5, color="k")
        except Exception as e:
            logger.warning("Skipping line %d: %s", idx, e)

    for endpoint in endpoints:
        try:
            label, coords_str = endpoint.split(": ", 1)
            coords = safe_literal_eval(coords_str)
            if coords is None:
                continue
            ax.annotate(
                label, coords, xytext=(1, 1),
                textcoords="offset points", fontsize=5, fontweight="light",
            )
        except Exception as e:
            logger.warning("Skipping endpoint annotation: %s", e)

    # Circles
    try:
        if "Circle" in parsed:
            centers = parsed["Circle"]["circle_center"]
            radii = parsed["Circle"]["radius"]
            for center_str, r in zip(centers, radii):
                center = safe_literal_eval(center_str.split(": ", 1)[1])
                if center is None:
                    continue
                circle = Circle(center, radius=r, fill=False, edgecolor="black", linewidth=0.8)
                ax.add_patch(circle)
    except Exception as e:
        logger.warning("Circle rendering error: %s", e)

    plt.savefig(save_path)
    plt.close()
    logger.info("Geometry saved to %s", save_path)


# ========================== POST-PROCESSING ==========================


def postprocess_output(
    raw_output: str,
    image: Image.Image,
    output_dir: str,
):
    """Clean up model output, save results, and return processed markdown."""
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    # Save raw output
    raw_path = os.path.join(output_dir, "result_ori.mmd")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    logger.info("Raw output saved to %s", raw_path)

    # Parse references
    matches_ref, matches_images, matches_other = re_match(raw_output)

    # Draw bounding boxes
    result_image = draw_bounding_boxes(image, matches_ref)
    box_path = os.path.join(output_dir, "result_with_boxes.jpg")
    result_image.save(box_path)
    logger.info("Annotated image saved to %s", box_path)

    # Replace image references with markdown image links
    cleaned = raw_output
    for idx, match_img in enumerate(tqdm(matches_images, desc="Replacing image refs")):
        cleaned = cleaned.replace(match_img, f"![](images/{idx}.jpg)\n")

    # Strip non-image references and fix special chars
    for match_other in tqdm(matches_other, desc="Stripping other refs"):
        cleaned = cleaned.replace(match_other, "")
    cleaned = cleaned.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    # Save cleaned output
    clean_path = os.path.join(output_dir, "result.mmd")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    logger.info("Cleaned output saved to %s", clean_path)

    # Geometry rendering (if applicable)
    render_geometry(raw_output, os.path.join(output_dir, "geo.jpg"))

    return cleaned


# ========================== BATCH PROCESSING ==========================


async def process_single(engine: OCREngine, image_path: str, output_dir: str, prompt: str):
    """Process a single image end-to-end."""
    image = load_image(image_path)
    if image is None:
        logger.error("Skipping %s — could not load image", image_path)
        return

    image = image.convert("RGB")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    if "<image>" in prompt:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )
    else:
        image_features = ""

    raw_output = await engine.generate(image=image_features, prompt=prompt)

    if "<image>" in prompt:
        postprocess_output(raw_output, image, output_dir)


async def process_batch(image_paths: list[str], output_root: str, prompt: str):
    """Process multiple images, reusing a single engine instance."""
    engine = OCREngine()
    for i, path in enumerate(image_paths):
        logger.info("Processing image %d/%d: %s", i + 1, len(image_paths), path)
        name = os.path.splitext(os.path.basename(path))[0]
        output_dir = os.path.join(output_root, name)
        await process_single(engine, path, output_dir, prompt)


# ========================== MAIN ==========================


async def main():
    # Single-image mode (matches original behaviour)
    engine = OCREngine()
    await process_single(engine, INPUT_PATH, OUTPUT_PATH, PROMPT)

    # ---- Batch mode example (uncomment to use) ----
    # image_dir = "/path/to/images"
    # paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    # await process_batch(paths, OUTPUT_PATH, PROMPT)


if __name__ == "__main__":
    asyncio.run(main())