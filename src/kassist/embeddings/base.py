"""Interfaces for embedding and reranking models (swappable: local ONNX, OpenAI, test fakes)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class DenseEmbedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1]."""
        ...
