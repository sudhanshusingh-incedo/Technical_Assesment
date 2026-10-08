"""Local ONNX models via fastembed: no API key, no PyTorch, deterministic, data stays local."""

from __future__ import annotations

import math
from collections.abc import Sequence

from kassist.config import EmbeddingSettings
from kassist.embeddings.base import SparseVec


class FastEmbedDense:
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self._model.query_embed("dimension probe"))))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # passage_embed/query_embed apply the model-specific prefixes (asymmetric retrieval)
        return [v.tolist() for v in self._model.passage_embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class FastEmbedSparse:
    """BM25 term weights; IDF is applied by Qdrant (Modifier.IDF) so the index stays incremental."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [
            SparseVec(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.passage_embed(list(texts), batch_size=self._batch_size)
        ]

    def embed_query(self, text: str) -> SparseVec:
        e = next(iter(self._model.query_embed(text)))
        return SparseVec(indices=e.indices.tolist(), values=e.values.tolist())


class FastEmbedReranker:
    """Cross-encoder: scores (query, passage) jointly; logits mapped to [0, 1] with a sigmoid."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        logits = list(self._model.rerank(query, list(texts)))
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def build_models(cfg: EmbeddingSettings) -> tuple[FastEmbedDense, FastEmbedSparse]:
    return (
        FastEmbedDense(cfg.dense_model, cfg.cache_dir, cfg.batch_size),
        FastEmbedSparse(cfg.sparse_model, cfg.cache_dir, cfg.batch_size),
    )
