"""Weighted multi-variable scoring system for OCR output quality.

Evaluates OCR results using multiple independent metrics, each with
its own weight. The composite score determines whether a result is
acceptable or needs to be retried with different preprocessing.

Design principle: many variables with individual weights ensure that
no single metric dominates — a change in one variable has a bounded
effect on the final score.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "self_consistency": 0.30,
    "hallucination_ratio": 0.25,
    "token_efficiency": 0.15,
    "structural_integrity": 0.15,
    "repetition_density": 0.10,
    "content_density": 0.05,
}

# Quality threshold — results below this are candidates for retry
DEFAULT_THRESHOLD = 0.6

# Max number of retry attempts per page
DEFAULT_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """Detailed breakdown of an OCR quality score."""
    self_consistency: float = 0.0
    hallucination_ratio: float = 0.0
    token_efficiency: float = 0.0
    structural_integrity: float = 0.0
    repetition_density: float = 0.0
    content_density: float = 0.0
    composite: float = 0.0
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def to_dict(self) -> dict:
        return {
            "composite": round(self.composite, 4),
            "variables": {
                "self_consistency": round(self.self_consistency, 4),
                "hallucination_ratio": round(self.hallucination_ratio, 4),
                "token_efficiency": round(self.token_efficiency, 4),
                "structural_integrity": round(self.structural_integrity, 4),
                "repetition_density": round(self.repetition_density, 4),
                "content_density": round(self.content_density, 4),
            },
            "weights": self.weights,
        }


@dataclass
class OCRResult:
    """A single OCR inference result with metadata for scoring."""
    raw_text: str
    clean_text: str
    num_tokens: int
    max_tokens: int
    preset_name: str = "adaptive"
    score: Optional[ScoreBreakdown] = None


# ---------------------------------------------------------------------------
# Individual scoring variables (each returns 0.0 - 1.0)
# ---------------------------------------------------------------------------

def _score_hallucination_ratio(result: OCRResult) -> float:
    """How much of the raw output survived post-processing.

    Score = clean_chars / raw_chars.
    A result where post-processing removed 90% of the text is mostly
    hallucination (score ≈ 0.1). A clean result scores near 1.0.
    """
    raw_len = len(result.raw_text.strip())
    clean_len = len(result.clean_text.strip())
    if raw_len == 0:
        return 0.0
    ratio = clean_len / raw_len
    return min(ratio, 1.0)


def _score_token_efficiency(result: OCRResult) -> float:
    """Penalize max-token runs with little clean output.

    If the model generated max_tokens (e.g., 7280/8192) but the clean
    output is tiny, the model was stuck in a hallucination loop.
    """
    token_ratio = result.num_tokens / result.max_tokens
    clean_len = len(result.clean_text.strip())

    # If tokens < 80% of max, model stopped naturally — likely good
    if token_ratio < 0.8:
        return 1.0

    # Model hit or nearly hit max tokens — check if output is substantial
    # A good max-token result should have proportional clean content
    # Expect at least ~2 clean chars per token for good output
    expected_min_chars = result.num_tokens * 0.5
    if clean_len >= expected_min_chars:
        return 0.8  # hit max but has real content
    if clean_len >= expected_min_chars * 0.3:
        return 0.4  # some content but mostly hallucination
    return 0.1  # max tokens with almost no real content


def _score_structural_integrity(result: OCRResult) -> float:
    """Check for expected structural elements in the output.

    Documents should contain recognizable structure: headers, tables,
    paragraphs, etc. A result with none of these is likely garbage.
    """
    text = result.clean_text
    if not text.strip():
        return 0.0

    signals = 0.0
    checks = 0

    # Has markdown headers?
    checks += 1
    if re.search(r"#{1,3}\s+\S", text):
        signals += 1.0

    # Has table structure?
    checks += 1
    tables = re.findall(r"<table>.*?</table>", text, re.DOTALL)
    if tables:
        # Check tables aren't just empty shells
        for table in tables:
            content_cells = re.findall(r"<td[^>]*>([^<]+)</td>", table)
            if len(content_cells) >= 3:
                signals += 1.0
                break
        else:
            signals += 0.3  # tables exist but sparse

    # Has meaningful paragraphs (>50 chars of non-table text)?
    checks += 1
    non_table = re.sub(r"<table>.*?</table>", "", text, flags=re.DOTALL)
    non_table = re.sub(r"<[^>]+>", "", non_table)
    if len(non_table.strip()) > 50:
        signals += 1.0

    # Has recognizable data patterns (dates, dollar amounts, names)?
    checks += 1
    data_patterns = (
        re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text)  # dates
        or re.search(r"\$[\d,]+\.\d{2}", text)         # dollar amounts
        or re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text) # proper names
    )
    if data_patterns:
        signals += 1.0

    return signals / checks if checks > 0 else 0.0


def _score_repetition_density(result: OCRResult) -> float:
    """Detect remaining repetitive patterns in the clean output.

    Even after post-processing, some subtle repetitions may remain.
    This measures how much of the text is repetitive.

    HTML/XML tags are stripped before analysis because table markup
    (``<td>``, ``<tr>``, etc.) is naturally repetitive and would
    otherwise produce false positives on structured documents.
    """
    # Strip HTML tags so table markup doesn't count as repetition
    text = re.sub(r"<[^>]+>", " ", result.clean_text)
    # Collapse whitespace left by tag removal
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 50:
        return 0.5  # too short to judge

    # Check for repeated short sequences (3-10 chars)
    total_repeated = 0
    for seq_len in [3, 5, 8, 10]:
        seen = Counter()
        for i in range(len(text) - seq_len):
            chunk = text[i:i + seq_len]
            if chunk.strip():
                seen[chunk] += 1
        # Count chars in sequences that appear 5+ times
        for chunk, count in seen.items():
            if count >= 5:
                total_repeated += len(chunk) * (count - 1)

    repetition_ratio = total_repeated / len(text) if text else 0
    # Clamp — some repetition is normal in structured docs
    return max(0.0, 1.0 - repetition_ratio * 0.5)


def _score_content_density(
    result: OCRResult,
    image_width: int = 0,
    image_height: int = 0,
) -> float:
    """Ratio of clean text to image area.

    Very short output from a large image suggests the model failed
    to extract most of the content.
    """
    clean_len = len(result.clean_text.strip())

    if clean_len == 0:
        return 0.0

    # Without image dimensions, use absolute thresholds
    if image_width == 0 or image_height == 0:
        if clean_len >= 1000:
            return 1.0
        if clean_len >= 500:
            return 0.8
        if clean_len >= 200:
            return 0.5
        if clean_len >= 50:
            return 0.3
        return 0.1

    # With dimensions: expect ~1 char per 100 pixels for document images
    image_area = image_width * image_height
    expected_chars = image_area / 100
    ratio = clean_len / expected_chars
    return min(ratio, 1.0)


def _score_self_consistency(
    current: OCRResult,
    others: list[OCRResult],
) -> float:
    """Pairwise similarity between multiple OCR runs.

    If the model produces similar text across different preprocessing
    runs, the result is likely correct. Wildly different outputs
    indicate unreliable generation.
    """
    if not others:
        # Single run — can't measure consistency, return neutral
        return 0.5

    similarities = []
    for other in others:
        ratio = SequenceMatcher(
            None,
            current.clean_text,
            other.clean_text,
        ).ratio()
        similarities.append(ratio)

    if not similarities:
        return 0.5

    return sum(similarities) / len(similarities)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def score_result(
    result: OCRResult,
    other_results: Optional[list[OCRResult]] = None,
    image_width: int = 0,
    image_height: int = 0,
    weights: Optional[dict] = None,
) -> ScoreBreakdown:
    """Compute the weighted composite quality score for an OCR result.

    Args:
        result: The OCR result to score.
        other_results: Other runs of the same image for self-consistency.
        image_width: Original image width (0 if unknown).
        image_height: Original image height (0 if unknown).
        weights: Override default scoring weights.

    Returns:
        ScoreBreakdown with per-variable and composite scores.
    """
    w = weights or DEFAULT_WEIGHTS

    breakdown = ScoreBreakdown(weights=w)

    breakdown.self_consistency = _score_self_consistency(
        result, other_results or []
    )
    breakdown.hallucination_ratio = _score_hallucination_ratio(result)
    breakdown.token_efficiency = _score_token_efficiency(result)
    breakdown.structural_integrity = _score_structural_integrity(result)
    breakdown.repetition_density = _score_repetition_density(result)
    breakdown.content_density = _score_content_density(
        result, image_width, image_height
    )

    breakdown.composite = (
        w["self_consistency"] * breakdown.self_consistency
        + w["hallucination_ratio"] * breakdown.hallucination_ratio
        + w["token_efficiency"] * breakdown.token_efficiency
        + w["structural_integrity"] * breakdown.structural_integrity
        + w["repetition_density"] * breakdown.repetition_density
        + w["content_density"] * breakdown.content_density
    )

    result.score = breakdown
    return breakdown


def select_best_result(results: list[OCRResult]) -> OCRResult:
    """From multiple scored results, return the one with the highest composite score.

    Also re-scores self_consistency using the full set of results.
    """
    if len(results) == 1:
        return results[0]

    # Re-score self_consistency with the full result set
    for i, result in enumerate(results):
        others = [r for j, r in enumerate(results) if j != i]
        if result.score is not None:
            result.score.self_consistency = _score_self_consistency(
                result, others
            )
            w = result.score.weights
            result.score.composite = (
                w["self_consistency"] * result.score.self_consistency
                + w["hallucination_ratio"] * result.score.hallucination_ratio
                + w["token_efficiency"] * result.score.token_efficiency
                + w["structural_integrity"] * result.score.structural_integrity
                + w["repetition_density"] * result.score.repetition_density
                + w["content_density"] * result.score.content_density
            )

    return max(results, key=lambda r: r.score.composite if r.score else 0.0)


def needs_retry(
    result: OCRResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Check if a result's score is below the retry threshold."""
    if result.score is None:
        return True
    return result.score.composite < threshold


# ---------------------------------------------------------------------------
# Flagging — Green / Yellow / Red quality flags
# ---------------------------------------------------------------------------

# Composite score boundaries for color flags
FLAG_GREEN_THRESHOLD = 0.75   # >= 0.75 → green
FLAG_YELLOW_THRESHOLD = 0.60  # >= 0.60 → yellow, below → red

# Individual variable thresholds that can force a downgrade
_VARIABLE_THRESHOLDS = {
    "no_content": 10,           # clean text shorter than this → red
    "low_content": 50,          # clean text shorter than this → informational only
    "hallucination_ratio": 0.4, # below this → critical (forces red)
    "token_efficiency": 0.3,    # below this → critical (forces red)
    "repetition_density": 0.4,  # below this → warning (noted but doesn't force color)
    "structural_integrity": 0.3,# below this → warning (noted but doesn't force color)
}


def compute_flags(
    result: OCRResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute a Green/Yellow/Red quality flag for an OCR result.

    Returns a dict with:
        - flag: "green", "yellow", or "red"
        - message: short summary for the flag color
        - details: list of individual issue dicts (code + message + severity)

    Flag logic (applied in order):
        red    — no content, critical issues, OR composite < 0.60
        yellow — composite between 0.60 and 0.75
        green  — composite >= 0.75 and no critical issues

    Critical issues (force red regardless of composite):
        - hallucination_ratio below threshold
        - token_efficiency below threshold

    Warnings (informational, included in details but don't change color):
        - repetition_density below threshold
        - structural_integrity below threshold
        - low content length

    Args:
        result: A scored OCRResult.
        threshold: Composite score threshold (unused, reserved for future use).

    Returns:
        Flag dict with color, message, and details.
    """
    details: list[dict] = []
    clean_len = len(result.clean_text.strip())
    score = result.score

    # --- No content → always red ---
    if clean_len <= _VARIABLE_THRESHOLDS["no_content"]:
        return {
            "flag": "red",
            "message": "No meaningful text extracted. Manual review required.",
            "details": [{
                "code": "no_content",
                "severity": "critical",
                "message": "No meaningful text was extracted from this page.",
            }],
        }

    # --- Very little content (informational — doesn't affect flag color) ---
    if clean_len < _VARIABLE_THRESHOLDS["low_content"]:
        details.append({
            "code": "low_content",
            "severity": "info",
            "message": f"Very little text extracted ({clean_len} chars). Page may be mostly blank or handwritten.",
        })

    # --- Unscored → yellow ---
    if score is None:
        details.append({
            "code": "unscored",
            "severity": "warning",
            "message": "Page was not scored — quality is unknown.",
        })
        return {
            "flag": "yellow",
            "message": "Quality could not be determined. Spot-check recommended.",
            "details": details,
        }

    # --- Check individual variables for issues ---
    has_critical = False

    if score.hallucination_ratio < _VARIABLE_THRESHOLDS["hallucination_ratio"]:
        pct = (1 - score.hallucination_ratio) * 100
        details.append({
            "code": "possible_hallucination",
            "severity": "critical",
            "message": f"~{pct:.0f}% of raw output was removed as hallucinated content.",
        })
        has_critical = True

    if score.token_efficiency < _VARIABLE_THRESHOLDS["token_efficiency"]:
        details.append({
            "code": "max_tokens_hit",
            "severity": "critical",
            "message": "Model hit token limit with little clean output — likely stuck in a generation loop.",
        })
        has_critical = True

    if score.repetition_density < _VARIABLE_THRESHOLDS["repetition_density"]:
        details.append({
            "code": "repetitive_content",
            "severity": "warning",
            "message": "Output contains repetitive patterns that may indicate hallucination.",
        })

    if score.structural_integrity < _VARIABLE_THRESHOLDS["structural_integrity"]:
        details.append({
            "code": "no_structure",
            "severity": "warning",
            "message": "No recognizable document structure (headers, tables, paragraphs) detected.",
        })

    # --- Determine color from composite + critical issues ---
    composite = score.composite

    if composite < FLAG_YELLOW_THRESHOLD or has_critical:
        flag = "red"
        message = f"Low quality score ({composite:.2f}). Manual review required."
    elif composite < FLAG_GREEN_THRESHOLD:
        flag = "yellow"
        message = f"Borderline quality score ({composite:.2f}). Spot-check recommended."
    else:
        flag = "green"
        message = f"Good quality ({composite:.2f})."

    return {
        "flag": flag,
        "message": message,
        "details": details,
    }
