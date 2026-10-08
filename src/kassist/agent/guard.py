"""Input guard: normalisation and a lightweight prompt-injection screen.

This is a first line of defence, not the only one: prompts also fence user input and retrieved
text as untrusted data, the model can only call read-only tools, and answers must cite the
corpus. Patterns are deliberately narrow (direct instruction-override attempts) to keep false
positives low on legitimate questions *about* prompt injection, which this corpus covers.
"""

from __future__ import annotations

import re
import unicodedata

_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b"
               r".{0,20}\b(instructions?|prompts?|rules|guidelines|directives)\b", re.I | re.S),
    re.compile(r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}"
               r"\b(system prompt|hidden prompt|initial instructions|your instructions|developer message)\b",
               re.I | re.S),
    re.compile(r"\byou are now\b.{0,30}\b(DAN|jailbroken|unrestricted|unfiltered)\b", re.I | re.S),
    re.compile(r"</?(system|assistant|passage|context)>", re.I),
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalise_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_injection(text: str) -> str | None:
    """Return the name of the matched rule, or None."""
    for i, pattern in enumerate(_INJECTION_PATTERNS):
        if pattern.search(text):
            return f"rule_{i}"
    return None
