"""Typed graph state. List fields use reducers so parallel branches merge instead of overwrite."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from kassist.agent.citations import Citation
from kassist.agent.confidence import Confidence
from kassist.domain import RetrievedChunk
from kassist.llm.schemas import GroundedAnswer, SubTask

Status = Literal["answered", "partial", "insufficient_context", "clarification_needed",
                 "out_of_scope", "refused"]
RouteName = Literal["simple", "agentic", "clarify", "out_of_scope", "refused"]


class Step(BaseModel):
    node: str
    detail: str
    ms: float = 0.0


class ToolOutcome(BaseModel):
    tool: str
    input: str
    success: bool
    output: str
    chunk_ids: list[str] = []
    best_relevance: float = 0.0


def merge_evidence(left: list[RetrievedChunk], right: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Union by chunk_id, keeping the highest relevance seen (the same chunk can be found by
    several sub-queries)."""
    merged = {c.chunk_id: c for c in left}
    for c in right:
        if c.chunk_id not in merged or c.relevance > merged[c.chunk_id].relevance:
            merged[c.chunk_id] = c
    return list(merged.values())


class AgentState(TypedDict, total=False):
    # input
    question: str  # standalone question (rewritten from a follow-up when history is given)
    original_question: str
    history: list[tuple[str, str]]  # recent (question, answer) turns, oldest first
    rewritten: bool
    mode: Literal["auto", "rag", "agent"]
    top_k: int
    doc_ids: list[str] | None
    # routing
    route: RouteName
    route_reason: str
    clarification: str | None
    escalated: bool
    # agentic loop
    plan: list[SubTask]
    executed: Annotated[list[str], operator.add]
    evidence: Annotated[list[RetrievedChunk], merge_evidence]
    tool_results: Annotated[list[ToolOutcome], operator.add]
    tool_calls: Annotated[int, operator.add]
    iterations: int
    best_relevance: float
    llm_calls: Annotated[int, operator.add]
    # output
    context: list[RetrievedChunk]  # exact passages (and order) shown to the answer model
    draft: GroundedAnswer | None
    answer: str
    citations: list[Citation]
    status: Status
    confidence: Confidence | None
    follow_ups: list[str]
    steps: Annotated[list[Step], operator.add]
    errors: Annotated[list[str], operator.add]
