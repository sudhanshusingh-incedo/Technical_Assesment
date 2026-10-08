"""Post-generation verification: every citation must point at a passage the model was given.

Hallucinated passage ids are removed, citations are renumbered in order of first use, and an
answer that claims full coverage without a single valid citation is treated as ungrounded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from kassist.domain import RetrievedChunk

_MARKER = re.compile(r"\[(\d{1,3})\]")


class Citation(BaseModel):
    id: int
    document: str
    doc_id: str
    source_file: str
    pages: str
    page_start: int
    page_end: int
    section: str
    chunk_id: str
    snippet: str
    relevance: float


class VerifiedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    dropped_ids: list[int]


_MD_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"\*\*|__|`")


def _snippet(text: str, limit: int = 320) -> str:
    """Plain-text preview: markdown markers removed so clients can't render it as headings/bold."""
    plain = _MD_EMPHASIS.sub("", _MD_HEADING.sub("", text.replace("<br>", " ")))
    flat = re.sub(r"\s+", " ", plain).strip()
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + " ..."


def verify_citations(answer: str, passages: Sequence[RetrievedChunk]) -> VerifiedAnswer:
    valid = range(1, len(passages) + 1)
    order: list[int] = []
    dropped: list[int] = []
    for match in _MARKER.finditer(answer):
        n = int(match.group(1))
        if n in valid:
            if n not in order:
                order.append(n)
        elif n not in dropped:
            dropped.append(n)

    renumber = {old: new for new, old in enumerate(order, start=1)}

    def _sub(match: re.Match[str]) -> str:
        n = int(match.group(1))
        return f"[{renumber[n]}]" if n in renumber else ""

    cleaned = _MARKER.sub(_sub, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip()

    citations = []
    for old in order:
        c = passages[old - 1]
        citations.append(Citation(
            id=renumber[old], document=c.doc_title, doc_id=c.doc_id, source_file=c.source_file,
            pages=c.pages_label, page_start=c.page_start, page_end=c.page_end, section=c.section,
            chunk_id=c.chunk_id, snippet=_snippet(c.text), relevance=round(c.relevance, 4),
        ))
    return VerifiedAnswer(answer=cleaned, citations=citations, dropped_ids=dropped)
