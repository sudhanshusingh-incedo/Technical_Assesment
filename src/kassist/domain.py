"""Core domain objects shared by ingestion, retrieval and the agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ContentType = Literal["text", "table", "mixed"]


class Chunk(BaseModel):
    """The atomic unit of retrieval, with everything needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    page_start: int
    page_end: int
    section: str
    chunk_index: int
    content_type: ContentType = "text"
    text: str  # what the LLM and the user see
    embed_text: str  # what gets embedded (contextual header + text)
    token_estimate: int = 0

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


class IndexedChunkMeta(BaseModel):
    """Payload stored next to every vector (lineage for idempotency and compatibility checks)."""

    file_sha256: str
    embedding_model: str
    chunker_version: str
    indexed_at: float = 0.0  # unix time of the ingestion run that wrote this document


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    page_start: int
    page_end: int
    section: str
    content_type: ContentType = "text"
    text: str
    fusion_score: float = 0.0  # RRF / first-stage score
    relevance: float = 0.0  # sigmoid(cross-encoder logit), 0..1

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> RetrievedChunk:
        return cls(**{k: payload[k] for k in cls.model_fields if k in payload}, fusion_score=score)

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    best_relevance: float = 0.0
    sufficient: bool = False  # best_relevance >= configured threshold


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source_file: str
    pages: int
    chunks: int
    sections: list[str] = Field(default_factory=list)
    indexed_at: float | None = None
    embedding_model: str | None = None
    chunker_version: str | None = None
