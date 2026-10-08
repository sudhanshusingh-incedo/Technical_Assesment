"""Deterministic test doubles: no model downloads, no network, millisecond runtime."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from kassist.embeddings.base import SparseVec
from kassist.llm.client import LLMUnavailableError

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "for", "what", "which", "does",
         "how", "at", "on", "with", "are", "do", "i", "should", "it", "by", "be", "that"}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9@.%]+", text.lower()) if t not in _STOP]


def _bucket(token: str, size: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % size  # noqa: S324 (not security)


class FakeDense:
    model_name = "fake-dense"
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in tokens(text):
            v[_bucket(t, self.dim)] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    model_name = "fake-sparse"

    def _vec(self, text: str) -> SparseVec:
        counts = Counter(_bucket(t, 100_000) for t in tokens(text))
        idx = sorted(counts)
        return SparseVec(indices=idx, values=[float(counts[i]) for i in idx])

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> SparseVec:
        return self._vec(text)


class FakeReranker:
    """Relevance = share of query terms present in the passage."""

    model_name = "fake-reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(tokens(query))
        return [len(q & set(tokens(t))) / len(q) if q else 0.0 for t in texts]


Handler = Callable[[list[BaseMessage]], Any]


class FakeLLM:
    """Scripted LLM: one handler per output schema. A handler may return an object, raise, or
    return an Exception instance (which is raised)."""

    def __init__(self, handlers: dict[type[BaseModel], Handler] | None = None):
        self.handlers: dict[type[BaseModel], Handler] = handlers or {}
        self.calls: list[tuple[str, str]] = []
        self.last_messages: dict[str, list[BaseMessage]] = {}

    def structured(self, schema: type[BaseModel], messages: list[BaseMessage], tier: str,
                   config: Any = None) -> Any:
        self.calls.append((schema.__name__, tier))
        self.last_messages[schema.__name__] = messages
        handler = self.handlers.get(schema)
        if handler is None:
            raise LLMUnavailableError(f"no handler for {schema.__name__}")
        out = handler(messages)
        if isinstance(out, Exception):
            raise out
        return out

    def stream_text(self, messages: list[BaseMessage], tier: str, config: Any = None):
        """Streams the GroundedAnswer handler's result in the answer protocol, in small chunks so the
        incremental parser is exercised. A handler may also return raw protocol text (str)."""
        from kassist.llm.schemas import GroundedAnswer

        self.calls.append(("GroundedAnswer", tier))
        self.last_messages["GroundedAnswer"] = messages
        self.stream_closed = False
        handler = self.handlers.get(GroundedAnswer)
        if handler is None:
            raise LLMUnavailableError("no handler for GroundedAnswer")
        out = handler(messages)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else to_protocol(out)
        try:
            for i in range(0, len(text), self.chunk_size):
                yield text[i:i + self.chunk_size]
        finally:
            self.stream_closed = True

    chunk_size = 7

    def describe(self) -> dict[str, str]:
        return {"fast": "fake", "strong": "fake"}

    def count(self, schema_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == schema_name)


def to_protocol(answer: Any) -> str:
    """Render a GroundedAnswer the way the real answer model writes it."""
    conf = 0.8 if answer.confidence is None else answer.confidence
    follow = "\n".join(f"- {q}" for q in answer.follow_ups)
    return f"COVERAGE: {answer.coverage}\nCONFIDENCE: {conf}\nANSWER:\n{answer.answer}\nFOLLOW_UPS:\n{follow}\n"
