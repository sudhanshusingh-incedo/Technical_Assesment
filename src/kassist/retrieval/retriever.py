"""Query-time retrieval: hybrid (dense + BM25, RRF-fused) -> cross-encoder rerank -> relevance gate.

The cross-encoder's calibrated score doubles as the "is there any evidence at all?" signal: if
even the best candidate is below `min_relevance`, the caller can decline without spending an
LLM call, which is the first of two guards against fabricated answers (the second is the
generator's own sufficiency judgement plus citation verification).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from kassist.config import RetrievalSettings
from kassist.domain import RetrievalResult, RetrievedChunk
from kassist.embeddings.base import DenseEmbedder, Reranker, SparseEmbedder
from kassist.observability import get_logger
from kassist.vectorstore.qdrant_store import QdrantStore, SearchMode

log = get_logger(__name__)


class Retriever:
    def __init__(self, store: QdrantStore, dense: DenseEmbedder, sparse: SparseEmbedder,
                 reranker: Reranker | None, cfg: RetrievalSettings):
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.cfg = cfg

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
        rerank: bool = True,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        top_k = top_k or self.cfg.top_k
        candidates = self.store.search(
            dense_query=self.dense.embed_query(query),
            sparse_query=self.sparse.embed_query(query),
            limit=self.cfg.candidate_k,
            prefetch_k=self.cfg.prefetch_k,
            doc_ids=doc_ids,
            mode=mode,
        )
        if rerank and self.reranker is not None and candidates:
            texts = [_rerank_text(c, self.cfg.rerank_max_chars) for c in candidates]
            scores = self.reranker.score(query, texts)
            for chunk, score in zip(candidates, scores, strict=True):
                chunk.relevance = score
            candidates.sort(key=lambda c: c.relevance, reverse=True)
        else:
            # Without a reranker there is no calibrated score; treat first-stage order as-is.
            for chunk in candidates:
                chunk.relevance = 1.0

        best = candidates[0].relevance if candidates else 0.0
        kept = [c for c in candidates[:top_k] if c.relevance >= self.cfg.keep_relevance]
        result = RetrievalResult(
            query=query,
            chunks=kept or candidates[:1],
            best_relevance=best,
            sufficient=bool(candidates) and best >= self.cfg.min_relevance,
        )
        log.info("retrieval.done", mode=mode, candidates=len(candidates), kept=len(result.chunks),
                 best_relevance=round(best, 4), sufficient=result.sufficient,
                 chunk_ids=[c.chunk_id[:8] for c in result.chunks],
                 ms=round((time.perf_counter() - t0) * 1000, 1))
        return result


def _rerank_text(chunk: RetrievedChunk, max_chars: int | None) -> str:
    # Give the cross-encoder the same context header the embedder saw. Cross-encoder cost grows
    # with sequence length, so long passages can be truncated on CPU-bound deployments.
    text = chunk.text if max_chars is None else chunk.text[:max_chars]
    return f"{chunk.doc_title} | {chunk.section}\n{text}"
