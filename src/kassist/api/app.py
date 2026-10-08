"""FastAPI application factory: lifespan (model loading), middleware, safe error envelopes."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from kassist import __version__
from kassist.api.routes import limiter, router
from kassist.config import Settings, get_settings
from kassist.llm.client import LLMUnavailableError
from kassist.observability import configure_logging, get_logger
from kassist.services import Services, build_services

log = get_logger(__name__)


def _error(status: int, code: str, message: str, request: Request,
           details: list[dict[str, Any]] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message,
                            "request_id": getattr(request.state, "request_id", None)}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content={"error": body})


def create_app(settings: Settings | None = None, services: Services | None = None,
               services_factory: Callable[[Settings], Services] = build_services) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.services is None:
            try:
                # Load models + check index once at startup (blocking work off the event loop).
                app.state.services = await run_in_threadpool(services_factory, settings)
            except Exception as exc:
                # Stay up and report via /ready instead of crash-looping the container.
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
                log.error("startup.failed", error=app.state.startup_error)
        yield

    app = FastAPI(
        title="Intelligent Knowledge Assistant",
        version=__version__,
        description="Hybrid RAG + agentic workflow over the assessment document corpus. "
                    "Answers are grounded in retrieved passages and cite document/page.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = services
    app.state.limiter = limiter
    limiter.enabled = settings.app_env != "test"

    if settings.api.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.api.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]) -> Any:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.isprintable() else str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Handled here (not only by an exception handler) so even 500s carry the request id.
            # Details go to the server log; the client gets a generic message.
            log.exception("unhandled_error", path=request.url.path)
            response = _error(500, "internal_error", "An internal error occurred.", request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        log.info("http.request", method=request.method, path=request.url.path, status=response.status_code,
                 ms=round((time.perf_counter() - t0) * 1000, 1), request_id=request_id)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Echo field locations and messages, never the submitted values.
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return _error(422, "invalid_request", "Request validation failed", request, details)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        codes = {401: "unauthorized", 404: "not_found", 422: "invalid_request", 503: "not_ready"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail), request)

    @app.exception_handler(RateLimitExceeded)
    async def _rate(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return _error(429, "rate_limited", "Too many requests, slow down", request)

    @app.exception_handler(LLMUnavailableError)
    async def _llm(request: Request, exc: LLMUnavailableError) -> JSONResponse:
        log.error("ask.llm_unavailable", error=str(exc))
        return _error(503, "llm_unavailable", "The language model is temporarily unavailable. "
                                              "Please retry shortly.", request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", path=request.url.path)
        return _error(500, "internal_error", "An internal error occurred.", request)

    app.include_router(router)
    return app
