"""Public request/response contract (also rendered in the OpenAPI docs at /docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kassist.agent.citations import Citation
from kassist.agent.confidence import Confidence
from kassist.agent.state import Status, Step
from kassist.conversations import ConversationTurn

Mode = Literal["auto", "rag", "agent"]
CONVERSATION_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"  # server-issued UUID4


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"question": "What recall@10 does pgvector HNSW achieve at 1M vectors?"},
        {"question": "Which vector database suits a cost-conscious startup that needs hybrid search "
                     "and ACID compliance?", "mode": "auto"},
    ]})

    question: str = Field(min_length=1, max_length=4000, description="Natural-language question")
    mode: Mode = Field(default="auto", description=(
        "auto: the router decides; rag: force single-retrieval RAG; agent: force the agentic workflow"))
    top_k: int | None = Field(default=None, ge=1, le=20, description="Passages per retrieval")
    doc_ids: list[str] | None = Field(
        default=None, max_length=10, description="Restrict retrieval to these document ids")
    include_trace: bool = Field(default=True, description="Return the step-by-step workflow trace")
    conversation_id: str | None = Field(
        default=None, pattern=CONVERSATION_ID_PATTERN,
        description="Continue a conversation (value returned by a previous response). Omit to start a new one.")


class WorkflowInfo(BaseModel):
    route: str = Field(description="simple | agentic | clarify | out_of_scope | refused")
    reason: str
    escalated: bool = Field(description="True if simple RAG escalated to the agentic workflow")
    rewritten_question: str | None = Field(
        default=None, description="Standalone form of a follow-up question, resolved from the conversation")
    iterations: int
    tool_calls: int
    llm_calls: int
    queries: list[str] = Field(description="Searches/calculations executed")
    steps: list[Step] | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    request_id: str
    conversation_id: str = Field(description="Send back on the next request to ask follow-up questions")
    status: Status = Field(description=(
        "answered | partial | insufficient_context | clarification_needed | out_of_scope | refused"))
    answer: str
    citations: list[Citation]
    workflow: WorkflowInfo
    usage: Usage
    latency_ms: float
    confidence: Confidence | None = Field(
        default=None, description="Blended confidence (evidence relevance, coverage, citation validity, "
                                  "model self-rating) for answered / partial responses")
    follow_ups: list[str] = Field(
        default_factory=list, description="Suggested next questions; omitted when confidence is very low")
    warnings: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    conversation_id: str
    turns: list[ConversationTurn]


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_file: str
    pages: int
    chunks: int
    indexed_at: datetime | None = Field(default=None, description="When ingestion last indexed it (UTC)")
    embedding_model: str | None = None
    chunker_version: str | None = Field(default=None, description="Chunking logic version + settings")


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int = 0


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
