"""Safe incremental parsing of the answer model's output, so answers can stream token by token
without ever showing unverified content.

The answer model writes a small plain-text protocol (JSON can't be streamed as readable text):

    COVERAGE: full | partial | none
    CONFIDENCE: 0.0-1.0
    ANSWER:
    <markdown answer with [n] citations>
    FOLLOW_UPS:
    - <question>
    - <question>

Safety properties, enforced while streaming:
* Coverage is known *before* any answer text, so a "not covered" answer is never streamed and a
  simple-RAG answer that will be escalated can be abandoned before it is shown.
* Citations are validated as soon as a marker is complete: markers pointing at passages the model was
  not given are dropped, valid ones are renumbered by first use (same rule as the final verifier), so
  streamed citation numbers never change afterwards.
* First-citation gate: text is held back until the first valid citation appears. An answer that never
  cites anything is never shown (it is replaced by the "could not ground" message).
* The FOLLOW_UPS section is split off and never streamed as answer text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from kassist.agent.guard import detect_injection, normalise_question
from kassist.llm.schemas import Coverage, GroundedAnswer

_MARKER = re.compile(r"\[(\d{1,3})\]")
_PARTIAL_MARKER = re.compile(r"\[\d{0,3}$")
_ANSWER_LINE = re.compile(r"^[ \t]*\**ANSWER\**[ \t]*:\**[ \t]*", re.IGNORECASE | re.MULTILINE)
_FOLLOW_UPS_LINE = re.compile(r"(?:^|\n)[ \t]*\**FOLLOW[ _-]?UPS\**[ \t]*:\**", re.IGNORECASE)
_COVERAGE = re.compile(r"COVERAGE\**\s*:\s*\**\s*(full|partial|none)", re.IGNORECASE)
_CONFIDENCE = re.compile(r"CONFIDENCE\**\s*:\s*\**\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)
_FOLLOW_UP_MARKER_WORDS = ("follow_ups:", "follow-ups:", "follow ups:", "followups:")
_MAX_HEADER_CHARS = 400


class CitationStreamFilter:
    """Validates and renumbers [n] citation markers in streamed text, with a first-citation gate."""

    def __init__(self, n_passages: int):
        self.valid = range(1, n_passages + 1)
        self.renumber: dict[int, int] = {}
        self.dropped: list[int] = []
        self._pending = ""  # may end with an incomplete marker such as "[1"
        self._held = ""  # processed text waiting for the first valid citation
        self.gate_open = False

    def _process(self, text: str) -> tuple[str, bool]:
        saw_valid = False

        def sub(match: re.Match[str]) -> str:
            nonlocal saw_valid
            n = int(match.group(1))
            if n not in self.valid:
                if n not in self.dropped:
                    self.dropped.append(n)
                return ""
            saw_valid = True
            self.renumber.setdefault(n, len(self.renumber) + 1)
            return f"[{self.renumber[n]}]"

        return _MARKER.sub(sub, text), saw_valid

    def _release(self, processed: str, saw_valid: bool) -> str:
        if self.gate_open:
            return processed
        self._held += processed
        if saw_valid:
            self.gate_open = True
            out, self._held = self._held, ""
            return out
        return ""

    def feed(self, text: str) -> str:
        """Returns the text that is safe to show now (possibly empty)."""
        self._pending += text
        partial = _PARTIAL_MARKER.search(self._pending)
        cut = partial.start() if partial else len(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return self._release(*self._process(ready))

    def finish(self) -> str:
        """Flush what is left. Returns "" if the gate never opened (nothing grounded to show)."""
        out = self._release(*self._process(self._pending))
        self._pending = ""
        return out


@dataclass
class StreamEvent:
    kind: Literal["header", "text"]
    text: str = ""
    coverage: Coverage | None = None
    confidence: float | None = None


@dataclass
class AnswerStreamParser:
    """Splits the protocol into header / answer body / follow-ups while streaming."""

    n_passages: int
    phase: Literal["header", "body", "follow_ups"] = "header"
    header_ok: bool = True
    coverage: Coverage | None = None
    confidence: float | None = None
    raw: str = ""  # everything received
    _header: str = ""
    _body_pending: str = ""  # body text not yet released (may hold a partial FOLLOW_UPS marker)
    body: str = ""  # answer body as written by the model (raw markers)
    _follow_ups: str = ""
    citations: CitationStreamFilter = field(init=False)

    def __post_init__(self) -> None:
        self.citations = CitationStreamFilter(self.n_passages)

    def feed(self, chunk: str) -> Iterator[StreamEvent]:
        self.raw += chunk
        if self.phase == "header":
            self._header += chunk
            match = _ANSWER_LINE.search(self._header)
            if not match:
                if len(self._header) > _MAX_HEADER_CHARS:  # model ignored the format
                    self.header_ok = False
                    self.phase = "body"
                    rest, self._header = self._header, ""
                    yield from self._feed_body(rest, stream=False)
                return
            head, rest = self._header[:match.start()], self._header[match.end():]
            cov, conf = _COVERAGE.search(head), _CONFIDENCE.search(head)
            self.coverage = cov.group(1).lower() if cov else None  # type: ignore[assignment]
            self.confidence = min(1.0, float(conf.group(1))) if conf else None
            self.header_ok = cov is not None
            self.phase = "body"
            yield StreamEvent("header", coverage=self.coverage, confidence=self.confidence)
            yield from self._feed_body(rest.lstrip("\n"), stream=self.header_ok)
        elif self.phase == "body":
            yield from self._feed_body(chunk, stream=self.header_ok)
        else:
            self._follow_ups += chunk

    def _feed_body(self, text: str, stream: bool = True) -> Iterator[StreamEvent]:
        self._body_pending += text
        marker = _FOLLOW_UPS_LINE.search(self._body_pending)
        if marker:
            ready = self._body_pending[:marker.start()]
            self._follow_ups += self._body_pending[marker.end():]
            self._body_pending = ""
            self.phase = "follow_ups"
        else:
            # Hold back the current (unfinished) line if it could become the FOLLOW_UPS marker.
            last_nl = self._body_pending.rfind("\n")
            tail = self._body_pending[last_nl + 1:].strip().lstrip("*").lower()
            if tail and any(word.startswith(tail) for word in _FOLLOW_UP_MARKER_WORDS):
                ready, self._body_pending = self._body_pending[:last_nl + 1], self._body_pending[last_nl + 1:]
            else:
                ready, self._body_pending = self._body_pending, ""
        self.body += ready
        safe = self.citations.feed(ready)
        if stream and safe:
            yield StreamEvent("text", text=safe)

    def finish(self) -> Iterator[StreamEvent]:
        if self.phase == "header":  # stream ended before an ANSWER: line
            self.header_ok = False
            self.phase = "body"
            rest, self._header = self._header, ""
            yield from self._feed_body(rest, stream=False)
        if self._body_pending:
            self.body += self._body_pending
            ready, self._body_pending = self._body_pending, ""
            safe = self.citations.feed(ready)
            if self.header_ok and safe:
                yield StreamEvent("text", text=safe)
        tail = self.citations.finish()
        if self.header_ok and tail:
            yield StreamEvent("text", text=tail)

    def result(self, max_follow_ups: int = 3) -> GroundedAnswer:
        coverage: Coverage = self.coverage or "partial"  # unknown format: be conservative
        return GroundedAnswer(
            answer=self.body.strip(), coverage=coverage, confidence=self.confidence,
            follow_ups=parse_follow_ups(self._follow_ups, max_follow_ups),
            cited_ids=sorted(self.citations.renumber), format_ok=self.header_ok,
        )


def parse_follow_ups(text: str, limit: int = 3, current_question: str = "") -> list[str]:
    """Bullet/numbered lines -> clean, deduplicated, safe questions."""
    seen: set[str] = {current_question.strip().lower()} if current_question else set()
    out: list[str] = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        q = normalise_question(re.sub(r"\[\d{1,3}\]", "", q))
        if not q or len(q) > 200 or detect_injection(q) or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
        if len(out) == limit:
            break
    return out


def parse_complete(text: str, n_passages: int, max_follow_ups: int = 3) -> GroundedAnswer:
    """Parse a fully received output (non-streaming callers and tests)."""
    parser = AnswerStreamParser(n_passages)
    for _ in parser.feed(text):
        pass
    for _ in parser.finish():
        pass
    return parser.result(max_follow_ups)
