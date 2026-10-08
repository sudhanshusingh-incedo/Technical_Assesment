"""Streamlit UI, rendered headlessly (streamlit.testing.AppTest) against a mocked API."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest import mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "src" / "kassist" / "ui" / "streamlit_app.py")
CID = "11111111-1111-4111-8111-111111111111"
DOCS = {"documents": [{"doc_id": "rag", "title": "RAG Architecture Patterns", "source_file": "r.pdf", "pages": 10,
                       "chunks": 56, "indexed_at": "2026-10-09T09:20:00+00:00",
                       "embedding_model": "BAAI/bge-small-en-v1.5", "chunker_version": "1.2-380/60/50"}],
        "total_chunks": 56}


class _Resp:
    def __init__(self, data: dict, status: int = 200):
        self._d, self.status_code, self.headers = data, status, {"content-type": "application/json"}

    def json(self) -> dict:
        return self._d

    def raise_for_status(self) -> None:
        pass


def _final(question: str) -> dict:
    return {
        "request_id": "r1", "conversation_id": CID, "status": "answered", "answer": f"Answer to: {question} [1].",
        "citations": [{"id": 1, "document": "RAG Architecture Patterns", "doc_id": "rag", "source_file": "r.pdf",
                       "pages": "p. 6", "page_start": 6, "page_end": 6, "section": "RAG Architecture Patterns > 4.3",
                       "chunk_id": "abcdef12-x", "snippet": "## Heading **bold** text", "relevance": 0.91}],
        "workflow": {"route": "simple", "reason": "single fact", "escalated": False, "iterations": 0,
                     "tool_calls": 0, "llm_calls": 2, "queries": [], "steps": []},
        "usage": {"input_tokens": 10, "output_tokens": 5, "by_model": {}}, "latency_ms": 900.0, "warnings": [],
        "confidence": {"score": 0.867, "level": "high",
                       "components": {"evidence": 0.9, "coverage": 1.0, "citations": 1.0, "model": 0.9}},
        "follow_ups": ["How does Weaviate compare?", "What is RRF?"],
    }


def _sse(event: str, data: dict) -> list[str]:
    return [f"event: {event}", f"data: {json.dumps(data)}", ""]


@pytest.fixture
def api():
    """Mocked API: records (question, conversation_id) per streamed request."""
    sent: list[tuple[str, str | None]] = []
    script: dict[str, list[str] | None] = {"lines": None}

    def stream(method, url, json=None, **kwargs):
        sent.append((json["question"], json.get("conversation_id")))
        lines = script["lines"] or (_sse("step", {"node": "router", "detail": "simple", "ms": 5})
                                    + _sse("token", {"text": "Answer to: "})
                                    + _sse("final", _final(json["question"])))
        resp = mock.MagicMock(status_code=200)
        resp.iter_lines.return_value = iter(lines)
        return contextlib.nullcontext(resp)

    with mock.patch.object(httpx, "get", return_value=_Resp(DOCS)), \
         mock.patch.object(httpx, "stream", side_effect=stream), \
         mock.patch.object(httpx, "delete") as delete:
        yield {"sent": sent, "delete": delete, "script": script}


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_layout_and_sidebar(api):
    at = _app()
    assert [t.value for t in at.title] == ["Knowledge Assistant"]
    captions = [c.value for c in at.sidebar.caption]
    assert any("RAG Architecture Patterns: 10 pages, 56 chunks · indexed" in c for c in captions)
    assert any("BAAI/bge-small-en-v1.5" in c for c in captions)
    assert [(e.label, e.proto.expanded) for e in at.sidebar.expander] == [("Index details", False),
                                                                           ("Advanced options", False)]
    assert len(at.main.button_group) == 6  # example-question groups
    assert not at.exception


def test_streamed_answer_renders_sources_confidence_and_follow_ups(api):
    at = _app()
    at.chat_input[0].set_value("What is RRF?").run()
    assert any(m.value == "Answer to: What is RRF? [1]." for m in at.main.markdown)
    assert any("Page 6" in e.label for e in at.main.expander)
    # the snippet is escaped, so markdown inside it renders as plain text (not a giant heading)
    assert r"\#\# Heading \*\*bold\*\* text" in [m.value for m in at.main.markdown]
    assert "🟢 Confidence 87% (high)" in [c.value for c in at.main.caption]
    follow = [bg for bg in at.main.button_group if bg.label == "Suggested follow-ups"]
    assert follow and follow[0].options == ["How does Weaviate compare?", "What is RRF?"]
    assert any(c.value.endswith("2 LLM / 0 tool calls") for c in at.main.caption)  # status line
    assert not at.exception


def test_follow_up_click_continues_the_conversation(api):
    at = _app()
    at.chat_input[0].set_value("first question").run()
    follow = [bg for bg in at.main.button_group if bg.label == "Suggested follow-ups"][0]
    follow.set_value("How does Weaviate compare?").run()
    assert api["sent"] == [("first question", None), ("How does Weaviate compare?", CID)]
    assert [m.markdown[0].value for m in at.chat_message if m.name == "user"] == [
        "first question", "How does Weaviate compare?"]  # chronological


def test_example_tag_asks_the_question(api):
    at = _app()
    at.main.button_group[0].set_value("What chunk size and overlap are recommended for "
                                      "RecursiveCharacterTextSplitter?").run()
    assert api["sent"][0][0].startswith("What chunk size")


def test_clear_conversation_erases_it_server_side(api):
    at = _app()
    at.chat_input[0].set_value("q").run()
    at.sidebar.button[0].click().run()
    assert api["delete"].call_args.args[0].endswith(f"/v1/conversations/{CID}")
    assert at.session_state.history == [] and at.session_state.conversation_id is None


def test_stream_error_event_is_shown(api):
    api["script"]["lines"] = _sse("error", {"code": "llm_unavailable", "message": "The language model is down"})
    at = _app()
    at.chat_input[0].set_value("q").run()
    assert [e.value for e in at.main.error] == ["The language model is down"]
