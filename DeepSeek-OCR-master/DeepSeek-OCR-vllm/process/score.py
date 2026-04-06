"""Weighted multi-variable scoring system for OCR output quality.

General-purpose scoring designed for diverse document types: forms,
letters, reports, invoices, receipts, legal documents, handwritten
notes, certificates, spreadsheets, etc.

Evaluates OCR results using multiple independent metrics, each with
its own weight. The composite score determines whether a result is
acceptable or needs to be retried with different preprocessing.

Design principles:
- No assumption about document structure (headers, tables, etc.)
- Coordinate/grounding tags stripped during cleaning are NOT hallucination
- Natural text repetition (legal boilerplate, form labels) is expected
- Single-run deterministic inference should not be penalized
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .postprocess import CleanStats


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "self_consistency": 0.20,
    "hallucination_ratio": 0.25,
    "token_efficiency": 0.20,
    "structural_integrity": 0.10,
    "repetition_density": 0.10,
    "content_density": 0.15,
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
    clean_stats: Optional["CleanStats"] = None


# ---------------------------------------------------------------------------
# Tag measurement — separates grounding tags from real content
# ---------------------------------------------------------------------------

# Patterns that are expected model output format, not hallucination
_GROUNDING_TAG_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>"
    r"|<\|det\|>.*?<\|/det\|>"
    r"|<\uff5cend\u2581of\u2581sentence\uff5c>"
    r"|\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]"
)


def _measure_grounding_tags(raw_text: str) -> int:
    """Count characters in the raw text that are grounding/coordinate tags.

    These tags are expected output format and their removal during
    cleaning should not be counted as hallucination.
    """
    return sum(len(m.group()) for m in _GROUNDING_TAG_PATTERN.finditer(raw_text))


# ---------------------------------------------------------------------------
# Individual scoring variables (each returns 0.0 - 1.0)
# ---------------------------------------------------------------------------

def _score_hallucination_ratio(result: OCRResult) -> float:
    """How much of the raw output survived post-processing.

    Excludes grounding tags and dedup-removed content from the
    denominator, since neither represents fabricated content.

    For general documents: the ratio measures clean_chars / effective_raw,
    where effective_raw = raw - tags - dedup.
    """
    raw_len = len(result.raw_text.strip())
    clean_len = len(result.clean_text.strip())
    if raw_len == 0:
        return 0.0

    # Subtract characters that are expected format, not hallucination
    tag_chars = _measure_grounding_tags(result.raw_text)
    dedup_removed = 0
    if result.clean_stats is not None:
        dedup_removed = result.clean_stats.dedup_chars_removed

    effective_raw = raw_len - tag_chars - dedup_removed
    # If after removing tags the effective raw is smaller than clean,
    # that means almost everything was tags — score is perfect
    if effective_raw <= clean_len:
        return 1.0

    ratio = clean_len / effective_raw
    return min(ratio, 1.0)


def _score_token_efficiency(result: OCRResult) -> float:
    """Penalize max-token runs with little clean output.

    If the model generated max_tokens but the clean output is tiny,
    the model was stuck in a hallucination loop. Applies a smooth
    curve rather than hard cutoffs.
    """
    token_ratio = result.num_tokens / result.max_tokens
    clean_len = len(result.clean_text.strip())

    # If tokens < 80% of max, model stopped naturally — likely good
    if token_ratio < 0.8:
        return 1.0

    # Model hit or nearly hit max tokens — check if output is substantial
    expected_min_chars = result.num_tokens * 0.5
    if clean_len >= expected_min_chars:
        return 0.9  # hit max but has real content — normal for dense docs
    ratio = clean_len / expected_min_chars if expected_min_chars > 0 else 0
    # Smooth curve: 0.9 at ratio=1.0 down to 0.1 at ratio=0
    return max(0.1, 0.9 * ratio)


def _score_structural_integrity(result: OCRResult) -> float:
    """Check for recognizable content patterns in the output.

    General-purpose: does NOT require any specific structure type.
    Awards credit for ANY recognizable pattern — a plain text letter
    scores just as well as a complex form with tables.
    """
    text = result.clean_text
    if not text.strip():
        return 0.0

    signals = 0.0

    # Has markdown headers?
    if re.search(r"#{1,3}\s+\S", text):
        signals += 1.0

    # Has table structure with real content?
    tables = re.findall(r"<table>.*?</table>", text, re.DOTALL)
    if tables:
        for table in tables:
            content_cells = re.findall(r"<td[^>]*>([^<]+)</td>", table)
            if len(content_cells) >= 2:
                signals += 1.0
                break
        else:
            signals += 0.3

    # Has meaningful text (>30 chars of non-markup text)?
    non_markup = re.sub(r"<[^>]+>", "", text)
    non_markup = re.sub(r"#{1,3}\s+", "", non_markup).strip()
    if len(non_markup) > 30:
        signals += 1.0

    # Has recognizable data patterns (dates, amounts, names, emails, phones)?
    data_patterns = (
        re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text)      # dates
        or re.search(r"\$[\d,]+\.?\d*", text)               # dollar amounts
        or re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text)     # proper names
        or re.search(r"\S+@\S+\.\S+", text)                 # emails
        or re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", text)  # phone numbers
    )
    if data_patterns:
        signals += 1.0

    # Normalize: any 1 signal is enough for a decent score.
    # 1 signal = 0.75, 2 = 0.875, 3 = 0.95, 4 = 1.0
    max_signals = 4.0
    if signals >= max_signals:
        return 1.0
    if signals >= 1.0:
        return 0.5 + 0.5 * (signals / max_signals)
    # No signals at all — but if there's substantial text, still give partial credit
    if len(non_markup) > 100:
        return 0.4
    return 0.25


def _score_repetition_density(result: OCRResult) -> float:
    """Detect remaining repetitive patterns in the clean output.

    Strips HTML/XML tags before analysis since table markup is naturally
    repetitive. Uses longer n-gram windows and a higher repeat threshold
    to avoid false positives on documents with natural repetition
    (legal text, form labels, financial data).
    """
    # Strip HTML tags so table markup doesn't count
    text = re.sub(r"<[^>]+>", " ", result.clean_text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 100:
        return 1.0  # too short to judge — assume OK

    # Only check longer sequences (6+ chars) to avoid matching common
    # short words/phrases that repeat naturally in any document
    total_repeated = 0
    for seq_len in [6, 10, 15, 20]:
        seen = Counter()
        for i in range(len(text) - seq_len):
            chunk = text[i:i + seq_len]
            if chunk.strip():
                seen[chunk] += 1
        # Only count sequences that appear 8+ times (higher bar)
        for chunk, count in seen.items():
            if count >= 8:
                total_repeated += len(chunk) * (count - 1)

    repetition_ratio = total_repeated / len(text) if text else 0
    # More forgiving: scale factor 0.3 instead of 0.5
    return max(0.0, 1.0 - repetition_ratio * 0.3)


def _score_content_density(
    result: OCRResult,
    image_width: int = 0,
    image_height: int = 0,
) -> float:
    """Ratio of clean text to image area.

    Very short output from a large image suggests the model failed
    to extract most of the content. Uses smooth scaling rather than
    hard cutoffs to handle diverse document types — a short receipt
    and a dense legal page are both valid.
    """
    clean_len = len(result.clean_text.strip())

    if clean_len == 0:
        return 0.0

    # With image dimensions: use pixel-to-char ratio
    if image_width > 0 and image_height > 0:
        image_area = image_width * image_height
        # Expect roughly 1 char per 200 pixels for average documents
        # (more forgiving than 1:100 — handles sparse docs like receipts)
        expected_chars = image_area / 200
        ratio = clean_len / expected_chars
        return min(ratio, 1.0)

    # Without dimensions: smooth curve based on absolute char count
    if clean_len >= 500:
        return 1.0
    if clean_len >= 100:
        return 0.5 + 0.5 * ((clean_len - 100) / 400)
    if clean_len >= 20:
        return 0.2 + 0.3 * ((clean_len - 20) / 80)
    return 0.1


def _score_self_consistency(
    current: OCRResult,
    others: list[OCRResult],
) -> float:
    """Pairwise similarity between multiple OCR runs.

    If the model produces similar text across different preprocessing
    runs, the result is likely correct. Wildly different outputs
    indicate unreliable generation.

    Single-run results return 1.0 because a deterministic model at
    temperature 0 will always produce the same output for the same
    input — the inability to measure consistency should not penalize
    the score at all.
    """
    if not others:
        return 1.0

    similarities = []
    for other in others:
        ratio = SequenceMatcher(
            None,
            current.clean_text,
            other.clean_text,
        ).ratio()
        similarities.append(ratio)

    if not similarities:
        return 1.0

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
FLAG_GREEN_THRESHOLD = 0.70   # >= 0.70 → green
FLAG_YELLOW_THRESHOLD = 0.50  # >= 0.50 → yellow, below → red


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
        red    — no content OR composite < 0.50
        yellow — composite between 0.50 and 0.70, OR green demoted by warning
        green  — composite >= 0.70

    Warnings (downgrade by one level: green→yellow, yellow stays yellow):
        - hallucination_ratio below 0.25 (severe — most content may be fabricated)
        - token_efficiency below 0.2 (model stuck in generation loop)

    Informational (included in details but don't change color):
        - repetition_density below threshold
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
    if clean_len <= 10:
        return {
            "flag": "red",
            "message": "No meaningful text extracted. Manual review required.",
            "details": [{
                "code": "no_content",
                "severity": "critical",
                "message": "No meaningful text was extracted from this page.",
            }],
        }

    # --- Very little content (informational) ---
    if clean_len < 30:
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

    # --- Check for warnings (downgrade one level, not force red) ---
    has_warning = False

    # Flag hallucination if severe (>75% removed after excluding tags)
    if score.hallucination_ratio < 0.25:
        pct = (1 - score.hallucination_ratio) * 100
        details.append({
            "code": "possible_hallucination",
            "severity": "warning",
            "message": f"~{pct:.0f}% of output was removed as hallucinated content.",
        })
        has_warning = True

    # Flag token efficiency if extreme
    if score.token_efficiency < 0.2:
        details.append({
            "code": "max_tokens_hit",
            "severity": "warning",
            "message": "Model hit token limit with very little clean output — likely stuck in a generation loop.",
        })
        has_warning = True

    # --- Informational (don't change flag color) ---
    if score.repetition_density < 0.4:
        details.append({
            "code": "repetitive_content",
            "severity": "info",
            "message": "Output contains repetitive patterns that may indicate hallucination.",
        })

    if score.content_density < 0.15:
        details.append({
            "code": "sparse_content",
            "severity": "info",
            "message": "Extracted text is very short relative to image size.",
        })

    # --- Determine color from composite score ---
    composite = score.composite

    if composite < FLAG_YELLOW_THRESHOLD:
        flag = "red"
        message = f"Low quality score ({composite:.2f}). Manual review required."
    elif composite < FLAG_GREEN_THRESHOLD:
        flag = "yellow"
        message = f"Borderline quality score ({composite:.2f}). Spot-check recommended."
    else:
        flag = "green"
        message = f"Good quality ({composite:.2f})."

    # --- Warnings downgrade by one level (green→yellow, yellow stays) ---
    if has_warning:
        if flag == "green":
            flag = "yellow"
            message = f"Score OK ({composite:.2f}) but has warnings. Spot-check recommended."

    return {
        "flag": flag,
        "message": message,
        "details": details,
    }
