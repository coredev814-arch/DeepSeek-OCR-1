"""DeepSeek-OCR processing pipeline.

Modules:
    image_process  — Image preprocessing and tokenization (DO NOT MODIFY)
    ngram_norepeat — Custom logits processor for vLLM (DO NOT MODIFY)
    postprocess    — Output cleanup and hallucination removal
    enhance        — Adaptive image enhancement for scanned documents
    score          — Weighted multi-variable quality scoring system
"""

from .postprocess import clean_output, CleanStats
from .enhance import enhance_scan, enhance_scan_with_preset, ENHANCEMENT_PRESETS
from .score import (
    OCRResult,
    ScoreBreakdown,
    score_result,
    select_best_result,
    needs_retry,
    compute_flags,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_WEIGHTS,
    FLAG_GREEN_THRESHOLD,
    FLAG_YELLOW_THRESHOLD,
)

__all__ = [
    # Postprocess
    "clean_output",
    "CleanStats",
    # Enhancement
    "enhance_scan",
    "enhance_scan_with_preset",
    "ENHANCEMENT_PRESETS",
    # Scoring & Flagging
    "OCRResult",
    "ScoreBreakdown",
    "score_result",
    "select_best_result",
    "needs_retry",
    "compute_flags",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_WEIGHTS",
]
