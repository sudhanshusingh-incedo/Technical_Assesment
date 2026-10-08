"""Structured logging (structlog) with request-scoped context and secret redaction."""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_REDACT_KEYS = {"api_key", "authorization", "openai_api_key", "x-api-key", "password", "token"}


def _redact(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***"
    return event_dict


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def fingerprint(text: str) -> str:
    """Stable, non-reversible id for a piece of user text (log correlation without content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
