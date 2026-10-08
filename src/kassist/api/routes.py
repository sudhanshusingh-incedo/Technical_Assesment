"""HTTP endpoints."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request, Response
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from kassist.agent.graph import AgentResult
from kassist.api.schemas import (
    CONVERSATION_ID_PATTERN,
    AskRequest,
    AskResponse,
    ConversationResponse,
    DocumentsResponse,
    DocumentSummary,
    ErrorResponse,
    Usage,
    WorkflowInfo,
)
from kassist.config import get_settings
from kassist.conversations import ConversationTurn
from kassist.llm.client import LLMUnavailableError
from kassist.observability import fingerprint, get_logger
from kassist.services import Services

log = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    422: {"model": ErrorResponse, "description": "Invalid request"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Service not ready or LLM unavailable"},
}


def get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Service is not ready (index or models not loaded)")
    services.refresh_if_index_changed()  # cheap, rate-limited: picks up re-ingestion without a restart
    if services.index_error:
        raise HTTPException(status_code=503, detail=f"not ready: {services.index_error}")
    return services


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return  # auth disabled (local/dev)
    if not x_api_key or not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def _rate_limit() -> str:
    return get_settings().api.rate_limit


@router.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", tags=["ops"], summary="Readiness probe (index loaded, models warm)")
def ready(request: Request) -> dict[str, object]:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        reason = getattr(request.app.state, "startup_error", "starting")
        raise HTTPException(status_code=503, detail=f"not ready: {reason}")
    services.refresh_if_index_changed()
    if services.index_error:
        raise HTTPException(status_code=503, detail=f"not ready: {services.index_error}")
    return {"status": "ready", "documents": len(services.documents), "chunks": services.store.count(),
            "llm": services.llm.describe()}


@router.get("/v1/documents", tags=["knowledge base"], response_model=DocumentsResponse,
            dependencies=[Depends(require_api_key)], responses=_ERRORS)
def list_documents(services: Services = Depends(get_services)) -> DocumentsResponse:
    return DocumentsResponse(
        documents=[
            DocumentSummary(
                doc_id=d.doc_id, title=d.title, source_file=d.source_file, pages=d.pages, chunks=d.chunks,
                indexed_at=datetime.fromtimestamp(d.indexed_at, tz=UTC) if d.indexed_at else None,
                embedding_model=d.embedding_model, chunker_version=d.chunker_version,
            )
            for d in services.documents
        ],
        total_chunks=sum(d.chunks for d in services.documents),
    )


def _prepare(body: AskRequest, services: Services) -> list[tuple[str, str]]:
    """Validate the request and load conversation history. Raises HTTP errors *before* any
    streaming starts, so the stream endpoint can still answer with a proper status code."""
    settings = services.settings
    if len(body.question) > settings.api.max_question_chars:
        raise HTTPException(status_code=422,
                            detail=f"question exceeds {settings.api.max_question_chars} characters")
    unknown = set(body.doc_ids or []) - {d.doc_id for d in services.documents}
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown doc_ids: {sorted(unknown)}")
    if not body.conversation_id:
        return []
    if not services.conversations.exists(body.conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found or expired")
    past = services.conversations.turns(body.conversation_id, limit=settings.agent.max_history_turns)
    # Earlier turns in their standalone form, so the rewriter sees self-contained questions.
    return [(t.rewritten_question or t.question, t.answer) for t in past]


def _complete(request: Request, body: AskRequest, services: Services, result: AgentResult,
              history: list[tuple[str, str]], t0: float) -> AskResponse:
    """Persist the turn, log, and build the response (shared by /v1/ask and /v1/ask/stream)."""
    settings = services.settings
    conversations = services.conversations
    # Created only after a successful answer, so failed requests leave no empty conversations.
    conversation_id = body.conversation_id or conversations.create()
    if result.status != "refused":  # blocked injection attempts are not kept as context
        conversations.append(conversation_id, ConversationTurn(
            question=result.original_question, answer=result.answer, status=result.status,
            rewritten_question=result.rewritten_question))
    latency = round((time.perf_counter() - t0) * 1000, 1)
    log.info("ask.done", status=result.status, route=result.route, escalated=result.escalated,
             conversation_id=conversation_id, history_turns=len(history),
             rewritten=result.rewritten_question is not None,
             confidence=result.confidence.score if result.confidence else None,
             follow_ups=len(result.follow_ups),
             question_fp=fingerprint(body.question), question_chars=len(body.question),
             question=body.question if settings.log_questions else None,
             citations=[c.chunk_id[:8] for c in result.citations], best_relevance=result.best_relevance,
             llm_calls=result.llm_calls, tool_calls=result.tool_calls,
             input_tokens=result.usage.get("input_tokens"), output_tokens=result.usage.get("output_tokens"),
             latency_ms=latency)
    return AskResponse(
        request_id=request.state.request_id,
        conversation_id=conversation_id,
        status=result.status,  # type: ignore[arg-type]
        answer=result.answer,
        citations=result.citations,
        workflow=WorkflowInfo(
            route=result.route, reason=result.route_reason, escalated=result.escalated,
            rewritten_question=result.rewritten_question,
            iterations=result.iterations, tool_calls=result.tool_calls, llm_calls=result.llm_calls,
            queries=result.queries, steps=result.steps if body.include_trace else None,
        ),
        usage=Usage(**result.usage),
        latency_ms=latency,
        confidence=result.confidence,
        follow_ups=result.follow_ups,
        warnings=result.errors,
    )


@router.post("/v1/ask", tags=["qa"], response_model=AskResponse, response_model_exclude_none=True,
             dependencies=[Depends(require_api_key)], responses=_ERRORS,
             summary="Ask a question; the router picks simple RAG or the agentic workflow")
@limiter.limit(_rate_limit)
def ask(request: Request, body: AskRequest, services: Services = Depends(get_services)) -> AskResponse:
    t0 = time.perf_counter()
    history = _prepare(body, services)
    result = services.agent.run(body.question, mode=body.mode, top_k=body.top_k, doc_ids=body.doc_ids,
                                history=history)
    return _complete(request, body, services, result, history, t0)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/v1/ask/stream", tags=["qa"], dependencies=[Depends(require_api_key)], responses=_ERRORS,
    summary="Same as /v1/ask, streamed as Server-Sent Events",
    description=(
        "Events, in order: `step` (a workflow step finished: {node, detail, ms}), `status` "
        "(e.g. writing answer), `token` (verified answer text, append in order), then exactly one of "
        "`final` (the complete AskResponse; replaces the streamed text) or `error` ({code, message, "
        "request_id}). Streamed citation numbers are already final, and answers the knowledge base "
        "does not support are never streamed. Validation errors (422/404/401/429) are returned as "
        "normal HTTP errors before the stream starts."),
)
@limiter.limit(_rate_limit)
def ask_stream(request: Request, body: AskRequest, services: Services = Depends(get_services)) -> Response:
    t0 = time.perf_counter()
    history = _prepare(body, services)
    request_id = request.state.request_id

    def events() -> Iterator[str]:
        structlog.contextvars.bind_contextvars(request_id=request_id)  # runs after the middleware returns
        try:
            for event in services.agent.stream(body.question, mode=body.mode, top_k=body.top_k,
                                               doc_ids=body.doc_ids, history=history):
                kind = event["type"]
                if kind == "step":
                    yield _sse("step", event["step"].model_dump())
                elif kind in ("status", "token"):
                    yield _sse(kind, {k: v for k, v in event.items() if k != "type"})
                elif kind == "result":
                    response = _complete(request, body, services, event["result"], history, t0)
                    yield _sse("final", response.model_dump(mode="json", exclude_none=True))
        except LLMUnavailableError as exc:
            log.error("ask.llm_unavailable", error=str(exc))
            yield _sse("error", {"code": "llm_unavailable", "request_id": request_id,
                                 "message": "The language model is temporarily unavailable. Please retry shortly."})
        except Exception:
            log.exception("ask_stream.unhandled_error")
            yield _sse("error", {"code": "internal_error", "message": "An internal error occurred.",
                                 "request_id": request_id})
        finally:
            structlog.contextvars.clear_contextvars()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_ConversationId = Path(pattern=CONVERSATION_ID_PATTERN, description="Conversation id from /v1/ask")


@router.get("/v1/conversations/{conversation_id}", tags=["conversations"], response_model=ConversationResponse,
            dependencies=[Depends(require_api_key)], responses={**_ERRORS, 404: {"model": ErrorResponse}},
            summary="Turns of a conversation, oldest first")
def get_conversation(conversation_id: str = _ConversationId,
                     services: Services = Depends(get_services)) -> ConversationResponse:
    if not services.conversations.exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found or expired")
    return ConversationResponse(conversation_id=conversation_id,
                                turns=services.conversations.turns(conversation_id))


@router.delete("/v1/conversations/{conversation_id}", tags=["conversations"], status_code=204,
               dependencies=[Depends(require_api_key)], responses={**_ERRORS, 404: {"model": ErrorResponse}},
               summary="Delete a conversation and its history (right to erasure)")
def delete_conversation(conversation_id: str = _ConversationId,
                        services: Services = Depends(get_services)) -> Response:
    if not services.conversations.delete(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found or expired")
    return Response(status_code=204)
