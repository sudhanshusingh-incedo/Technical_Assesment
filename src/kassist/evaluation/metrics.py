"""Evaluation metrics (pure functions so they are unit-testable).

Relevance is judged at (document, page) granularity: a retrieved chunk "hits" a gold source if
it is from that document and its page range covers that page. This is robust to re-chunking,
which would invalidate chunk-id-based gold labels.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

from kassist.domain import RetrievedChunk

Gold = tuple[str, int]  # (doc_id, page)


def chunk_matches(chunk: RetrievedChunk, gold: Gold) -> bool:
    doc_id, page = gold
    return chunk.doc_id == doc_id and chunk.page_start <= page <= chunk.page_end


def first_hit_rank(chunks: Sequence[RetrievedChunk], golds: Sequence[Gold]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk_matches(chunk, g) for g in golds):
            return rank
    return None


def hit_at_k(chunks: Sequence[RetrievedChunk], golds: Sequence[Gold], k: int) -> float:
    rank = first_hit_rank(chunks[:k], golds)
    return 1.0 if rank is not None else 0.0


def recall_at_k(chunks: Sequence[RetrievedChunk], golds: Sequence[Gold], k: int) -> float:
    """Share of gold (doc, page) sources covered by the top-k chunks."""
    if not golds:
        return 0.0
    found = sum(1 for g in golds if any(chunk_matches(c, g) for c in chunks[:k]))
    return found / len(golds)


def reciprocal_rank(chunks: Sequence[RetrievedChunk], golds: Sequence[Gold]) -> float:
    rank = first_hit_rank(chunks, golds)
    return 1.0 / rank if rank else 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "").replace(",", "")).lower()


def fact_coverage(answer: str, key_facts: Sequence[str | Sequence[str]]) -> float:
    """Share of required facts present in the answer. A nested list means any-of spellings."""
    if not key_facts:
        return 1.0
    text = _norm(answer)
    hits = 0
    for fact in key_facts:
        options = [fact] if isinstance(fact, str) else list(fact)
        if any(_norm(o) in text for o in options):
            hits += 1
    return hits / len(key_facts)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0
