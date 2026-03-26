"""Post-processing for DeepSeek-OCR model output.

Cleans up hallucinated content, collapses repetitive patterns,
deduplicates sections, and normalizes the markdown output.
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Table cleanup
# ---------------------------------------------------------------------------

def _collapse_empty_table_cells(text: str) -> str:
    """Collapse runs of excessive empty <td></td> cells in table rows."""
    MAX_EMPTY_CELLS_PER_ROW = 15

    def _trim_row(match: re.Match) -> str:
        row_html = match.group(0)
        empty_cells = re.findall(r"<td></td>", row_html)
        if len(empty_cells) <= MAX_EMPTY_CELLS_PER_ROW:
            return row_html
        parts = re.split(r"(<td></td>)", row_html)
        result = []
        empty_count = 0
        for part in parts:
            if part == "<td></td>":
                empty_count += 1
                if empty_count <= MAX_EMPTY_CELLS_PER_ROW:
                    result.append(part)
            else:
                result.append(part)
        return "".join(result)

    text = re.sub(r"<tr>.*?</tr>", _trim_row, text, flags=re.DOTALL)

    text = re.sub(
        r"(<td></td>){" + str(MAX_EMPTY_CELLS_PER_ROW) + r",}",
        "<td></td>" * MAX_EMPTY_CELLS_PER_ROW,
        text,
    )

    # Remove entirely empty table rows
    text = re.sub(
        r"<tr>(?:\s*<td(?:\s+colspan=\"\d+\")?></td>\s*)+</tr>",
        "",
        text,
    )

    # Remove hallucinated numbered empty-row sequences
    def _clean_numbered_rows(match: re.Match) -> str:
        table_html = match.group(0)
        non_empty = re.findall(r"<td>([^<]+)</td>", table_html)
        number_only = [c for c in non_empty if re.match(r"^\d+\.$", c)]
        if len(number_only) > 15 and len(number_only) / max(len(non_empty), 1) > 0.5:
            cleaned = re.sub(r"(<td>\d+\.</td>(?:<td></td>)*)", "", table_html)
            cleaned = re.sub(r"<tr>\s*</tr>", "", cleaned)
            return cleaned
        return table_html

    text = re.sub(r"<table>.*?</table>", _clean_numbered_rows, text, flags=re.DOTALL)

    # Trim bloated tables (>100 empty cells)
    def _trim_bloated_table(match: re.Match) -> str:
        table_html = match.group(0)
        total_empty = len(re.findall(r"<td></td>", table_html))
        if total_empty <= 100:
            return table_html
        MAX_ROWS = 60
        rows = re.findall(r"<tr>.*?</tr>", table_html, re.DOTALL)
        kept = []
        for row in rows:
            empty = len(re.findall(r"<td></td>", row))
            content = re.findall(r"<td[^>]*>([^<]+)</td>", row)
            meaningful = [
                c for c in content
                if not re.match(r"^\d{1,3}\.$", c)
                and not re.match(r"^&lt;.*&gt;$", c)
                and not re.match(r"^<[^>]+>$", c)
                and not re.match(r"^\$0(\.00)?$", c)
                and len(c.strip()) > 1
            ]
            if len(meaningful) >= 1 or empty <= 5:
                kept.append(row)
                if len(kept) >= MAX_ROWS:
                    break
        if not kept:
            return ""
        return "<table>" + "".join(kept) + "</table>"

    text = re.sub(r"<table>.*?</table>", _trim_bloated_table, text, flags=re.DOTALL)

    # Handle unclosed tables
    open_count = text.count("<table>")
    close_count = text.count("</table>")
    if open_count > close_count:
        last_open = text.rfind("<table>")
        unclosed_content = text[last_open:]
        rows = re.findall(r"<tr>.*?</tr>", unclosed_content, re.DOTALL)

        if rows:
            row_contents = []
            for row in rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                content = tuple(c.strip() for c in cells if c.strip())
                row_contents.append(content)
            content_counts = Counter(row_contents)
            top_n = content_counts.most_common(5)
            top_count = sum(c for _, c in top_n)
            is_repetitive = (
                len(rows) > 20
                and top_count > len(rows) * 0.8
                and top_count > 20
            )
            if is_repetitive:
                seen_counts = Counter()
                kept = []
                for i, row in enumerate(rows):
                    content = row_contents[i]
                    seen_counts[content] += 1
                    if seen_counts[content] <= 2:
                        kept.append(row)
            else:
                total_empty = len(re.findall(r"<td></td>", unclosed_content))
                if total_empty > 100:
                    kept = []
                    for row in rows:
                        empty = len(re.findall(r"<td></td>", row))
                        content = re.findall(r"<td[^>]*>([^<]+)</td>", row)
                        meaningful = [
                            c for c in content
                            if not re.match(r"^\d{1,3}\.$", c)
                            and not re.match(r"^&lt;.*&gt;$", c)
                            and not re.match(r"^<[^>]+>$", c)
                            and not re.match(r"^\$0(\.00)?$", c)
                            and len(c.strip()) > 1
                        ]
                        if len(meaningful) >= 1 or empty <= 5:
                            kept.append(row)
                            if len(kept) >= 60:
                                break
                else:
                    kept = rows[:60]

            if kept:
                text = text[:last_open] + "<table>" + "".join(kept) + "</table>"
            else:
                text = text[:last_open]
        else:
            text = text[:last_open]

    # Collapse repetitive table rows
    def _trim_repetitive_table(match: re.Match) -> str:
        table_html = match.group(0)
        rows = re.findall(r"<tr>.*?</tr>", table_html, re.DOTALL)
        if len(rows) <= 20:
            return table_html
        row_contents = []
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            content = tuple(c.strip() for c in cells if c.strip())
            row_contents.append(content)
        content_counts = Counter(row_contents)
        if not content_counts:
            return table_html
        top_n = content_counts.most_common(5)
        top_count = sum(c for _, c in top_n)
        if top_count > len(rows) * 0.8 and top_count > 20 and len(content_counts) <= max(len(rows) // 5, 10):
            seen_counts = Counter()
            kept = []
            for i, row in enumerate(rows):
                content = row_contents[i]
                seen_counts[content] += 1
                if seen_counts[content] <= 2:
                    kept.append(row)
            if not kept:
                return ""
            return "<table>" + "".join(kept) + "</table>"
        return table_html

    text = re.sub(r"<table>.*?</table>", _trim_repetitive_table, text, flags=re.DOTALL)

    # Remove tables dominated by a single repeated cell value
    # (diagonal repetition: same text appears across different columns in many rows)
    def _trim_diagonal_repetition(match: re.Match) -> str:
        table_html = match.group(0)
        rows = re.findall(r"<tr>.*?</tr>", table_html, re.DOTALL)
        if len(rows) <= 5:
            return table_html

        # Collect all non-empty cell values across the table
        all_cell_values = []
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            for c in cells:
                stripped = c.strip()
                if stripped:
                    all_cell_values.append(stripped)

        if not all_cell_values:
            return table_html

        # Check if a single value dominates >40% of all non-empty cells
        # and appears in more than 5 rows
        value_counts = Counter(all_cell_values)
        most_common_val, most_common_count = value_counts.most_common(1)[0]

        # Count how many rows contain this value
        rows_with_value = 0
        for row in rows:
            if most_common_val in row:
                rows_with_value += 1

        if most_common_count >= 5 and rows_with_value > len(rows) * 0.4:
            # This value is a hallucinated repeat — remove rows that ONLY
            # contain this value (plus empty cells)
            kept = []
            for row in rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                non_empty = [c.strip() for c in cells if c.strip()]
                unique_values = set(non_empty)
                # Keep rows that have content OTHER than the repeated value
                if not non_empty or unique_values - {most_common_val}:
                    kept.append(row)
            if not kept:
                return ""
            return "<table>" + "".join(kept) + "</table>"

        return table_html

    text = re.sub(r"<table>.*?</table>", _trim_diagonal_repetition, text, flags=re.DOTALL)

    # Remove empty tables
    text = re.sub(r"<table>\s*</table>", "", text)

    return text


# ---------------------------------------------------------------------------
# Repeating pattern cleanup
# ---------------------------------------------------------------------------

def _collapse_repeating_patterns(text: str) -> str:
    """Collapse hallucinated repeating character sequences."""
    # Incrementing number + digit filler
    text = re.sub(r"(?:\d{1,3}\s+(?:\d\s+){3,}){3,}", "", text)
    # Long digit-space runs
    text = re.sub(r"(?:\d\s){20,}", "", text)
    # Dot-separated digits
    text = re.sub(r"(?:1\.){6,}", "", text)
    # Digit-space runs (8+)
    text = re.sub(r"(?:\d\s){8,}", "", text)
    # Single char repeated with spaces 12+ times
    text = re.sub(r"(\S)\s(?:\1\s){11,}\1?", r"\1", text)
    # Numbered sequences
    text = re.sub(r"(?:\d+\.\s*){15,}", "", text)
    text = re.sub(r"(?:\d+\.){15,}", "", text)

    return text


# ---------------------------------------------------------------------------
# Section deduplication
# ---------------------------------------------------------------------------

def _deduplicate_sections(text: str) -> str:
    """Remove duplicated markdown sections."""
    parts = re.split(r"((?:^|\n)#{1,3}\s+[^\n]+)", text)
    if len(parts) <= 2:
        return text

    sections = []
    i = 0
    while i < len(parts):
        if re.match(r"\n?#{1,3}\s+", parts[i].lstrip()):
            header = parts[i]
            content = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((header, content))
            i += 2
        else:
            sections.append(("", parts[i]))
            i += 1

    seen_headers = {}
    result = []
    for header, content in sections:
        normalized = header.strip().lower()
        if not normalized or normalized not in seen_headers:
            if normalized:
                seen_headers[normalized] = len(result)
            result.append((header, content))
        else:
            prev_idx = seen_headers[normalized]
            prev_content = result[prev_idx][1]
            if len(content.strip()) > len(prev_content.strip()) * 1.2:
                result[prev_idx] = (header, content)

    return "".join(h + c for h, c in result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_output(text: str) -> str:
    """Remove grounding annotations and clean up the OCR output.

    This is the main entry point for post-processing. It applies all
    cleanup steps in order and returns normalized markdown.
    """
    text = text.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")

    # Remove grounding refs (non-image)
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if "<|ref|>image<|/ref|>" not in match[0]:
            text = text.replace(match[0], "")

    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = _collapse_empty_table_cells(text)
    text = _collapse_repeating_patterns(text)
    text = _deduplicate_sections(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()
