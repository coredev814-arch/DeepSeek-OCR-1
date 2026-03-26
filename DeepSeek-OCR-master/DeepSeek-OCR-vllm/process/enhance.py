"""Image enhancement for scanned documents.

Adaptively adjusts contrast and sharpness of grayscale scans
to match the characteristics of known-good OCR reference images.
"""

import numpy as np
from PIL import Image, ImageEnhance

# Target contrast derived from known-good OCR results (hud.png ~ 0.186)
TARGET_CONTRAST = 0.186


def enhance_scan(image: Image.Image) -> Image.Image:
    """Adaptively enhance a scanned/grayscale document for better OCR.

    Measures the image's RMS contrast (std/mean of grayscale pixel values)
    and adjusts to match TARGET_CONTRAST. Only applies to grayscale images;
    colour images are returned unchanged.

    Args:
        image: Input PIL Image (any mode).

    Returns:
        Enhanced RGB image, or the original if no enhancement was needed.
    """
    is_grayscale = _is_grayscale(image)

    if not is_grayscale:
        return image

    image = image.convert("RGB")

    gray = np.array(image.convert("L"))
    mean_val = gray.mean()
    if mean_val == 0:
        return image
    current_contrast = gray.std() / mean_val

    factor = TARGET_CONTRAST / current_contrast if current_contrast > 0 else 1.0
    factor = max(0.8, min(factor, 1.5))  # clamp to safe range

    if abs(factor - 1.0) > 0.05:
        image = ImageEnhance.Contrast(image).enhance(factor)
        image = ImageEnhance.Sharpness(image).enhance(1.5)

    return image


def enhance_scan_with_preset(
    image: Image.Image, contrast: float, sharpness: float
) -> Image.Image:
    """Enhance an image with explicit contrast and sharpness values.

    Used by the scoring/retry system to explore different enhancement
    presets per retry attempt.

    Args:
        image: Input PIL Image.
        contrast: Contrast multiplier (1.0 = unchanged).
        sharpness: Sharpness multiplier (1.0 = unchanged).

    Returns:
        Enhanced RGB image.
    """
    image = image.convert("RGB")
    if abs(contrast - 1.0) > 0.01:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if abs(sharpness - 1.0) > 0.01:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image


def _is_grayscale(image: Image.Image) -> bool:
    """Detect if an image is effectively grayscale."""
    if image.mode == "L":
        return True
    if image.mode in ("RGB", "RGBA"):
        arr = np.array(image.convert("RGB"))
        channel_diff = np.abs(
            arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)
        ).mean()
        return channel_diff < 10
    return False


# Enhancement presets for retry system — each explores a different
# generation path through the model's non-deterministic behavior.
ENHANCEMENT_PRESETS = [
    {"name": "adaptive", "contrast": None, "sharpness": None},  # use adaptive
    {"name": "none", "contrast": 1.0, "sharpness": 1.0},        # no enhancement
    {"name": "strong", "contrast": 1.5, "sharpness": 2.0},      # aggressive
]
