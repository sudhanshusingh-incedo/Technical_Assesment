"""Structured-output contracts between the orchestrator and the LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["simple", "agentic", "clarify", "out_of_scope"]
Coverage = Literal["full", "partial", "none"]


class RouteDecision(BaseModel):
    route: Route = Field(description="Which workflow should handle the question")
    reason: str = Field(description="One short sentence justifying the route")
    clarification_question: str | None = Field(
        default=None, description="Only for route=clarify: the question to ask the user")


class SubTask(BaseModel):
    tool: Literal["search", "calculator"]
    query: str | None = Field(default=None, description="search: standalone search query")
    doc_id: str | None = Field(default=None, description="search: optional document id filter")
    expression: str | None = Field(default=None, description="calculator: arithmetic expression")
    purpose: str = Field(default="", description="What this step contributes to the answer")


class Plan(BaseModel):
    subtasks: list[SubTask] = Field(description="Ordered, self-contained steps")


class Reflection(BaseModel):
    complete: bool = Field(description="True if the evidence is enough to answer (or clearly absent)")
    reason: str
    followups: list[SubTask] = Field(default_factory=list)


class Rewrite(BaseModel):
    standalone_question: str = Field(description="The question rewritten to be understandable on its own")
    is_follow_up: bool = Field(description="True if the message depended on the conversation")


class GroundedAnswer(BaseModel):
    """The answer model's output, parsed from its streamed text protocol (see agent/answer_stream.py)."""

    answer: str = Field(description="Markdown answer with [n] citations after each factual sentence")
    coverage: Coverage = Field(description="How completely the context answers the question")
    cited_ids: list[int] = Field(default_factory=list, description="Valid passage ids cited in the answer")
    confidence: float | None = Field(default=None, description="Model's self-rated confidence, 0-1")
    follow_ups: list[str] = Field(default_factory=list, description="Suggested next questions")
    format_ok: bool = Field(default=True, description="False if the model ignored the output protocol")
