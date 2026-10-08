import pytest

from kassist.domain import RetrievedChunk
from kassist.evaluation.metrics import (
    fact_coverage,
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def chunk(doc: str, start: int, end: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"{doc}{start}", doc_id=doc, doc_title=doc, source_file="f.pdf",
                          page_start=start, page_end=end or start, section="s", text="t")


RANKED = [chunk("a", 1), chunk("b", 4, 5), chunk("a", 7)]


def test_hit_and_rank_use_page_ranges():
    gold = [("b", 5)]
    assert hit_at_k(RANKED, gold, 1) == 0.0
    assert hit_at_k(RANKED, gold, 2) == 1.0
    assert reciprocal_rank(RANKED, gold) == pytest.approx(0.5)
    assert reciprocal_rank(RANKED, [("c", 1)]) == 0.0


def test_recall_counts_distinct_gold_sources():
    assert recall_at_k(RANKED, [("a", 1), ("a", 7), ("c", 2)], 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, [], 3) == 0.0


def test_fact_coverage_with_alternatives_and_formatting():
    answer = "pgvector reaches **96.1%** recall; storage is 122,880 MB."
    assert fact_coverage(answer, ["96.1", ["122880", "122.9 GB"], "Weaviate"]) == pytest.approx(2 / 3)
    assert fact_coverage(answer, []) == 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([], 95) == 0.0
