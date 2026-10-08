"""Remove PDF-extraction noise from per-page markdown.

The corpus PDFs carry running headers/footers, letter-spaced banners ("S E C T I O N 0 5"),
highlight tags and orphan bullet glyphs. Left in, they pollute embeddings (every page looks
alike) and waste context tokens. Rules are deliberately conservative: long lines are never
dropped, so real content that happens to mention "Document 6 of 6" survives.
"""

from __future__ import annotations

import re

_MARK_TAG = re.compile(r"</?mark>")
_PAGE_FOOTER = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
_RUNNING_HEADER = re.compile(
    r"(Technical Reference|Document\s+\d+\s+of\s+\d+|Senior AI Engineer Technical Assessment)",
    re.IGNORECASE,
)
_ORPHAN_BULLET = re.compile(r"^\s*(?:[-*•]\s*)+$")
_LEADING_DOUBLE_BULLET = re.compile(r"^(\s*)-\s+•\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MAX_FOOTER_LEN = 140


def _is_letter_spaced(line: str) -> bool:
    """True for banners like 'S E C T I O N 0 5' (mostly single-character tokens)."""
    tokens = line.replace("*", " ").split()
    if len(tokens) < 6:
        return False
    singles = sum(1 for t in tokens if len(t) == 1)
    return singles / len(tokens) > 0.6


def clean_heading_text(text: str) -> str:
    text = _MARK_TAG.sub("", text)
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip(" _*")


def clean_page_markdown(markdown: str) -> str:
    out: list[str] = []
    for raw in markdown.splitlines():
        line = _MARK_TAG.sub("", raw).rstrip()
        stripped = line.strip()
        if _PAGE_FOOTER.match(stripped):
            continue
        if stripped and len(stripped) < _MAX_FOOTER_LEN and _RUNNING_HEADER.search(stripped) \
                and not stripped.startswith("|"):
            continue
        if _is_letter_spaced(stripped):
            continue
        if _ORPHAN_BULLET.match(line):
            continue
        heading = _HEADING.match(stripped)
        if heading:
            line = f"{heading.group(1)} {clean_heading_text(heading.group(2))}"
        line = _LEADING_DOUBLE_BULLET.sub(r"\1- ", line)
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
