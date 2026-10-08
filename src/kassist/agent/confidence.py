"""Blended answer confidence.

LLM self-ratings alone are poorly calibrated (models tend to report ~0.9 regardless), so the score
blends the model's own rating with signals the system measured independently:

    evidence   mean cross-encoder relevance of the passages the answer actually cites   (0-1)
    coverage   full = 1.0, partial = 0.5                                                 (model-reported,
               but it also drives the answered/partial status, so it is checked downstream)
    citations  share of the answer's citation markers that pointed at real passages     (0-1)
    model      the answer model's self-rated confidence (0.5 if it gave none)           (0-1)

Weights favour the measured signals. The score is a ranking signal for the UI ("how much should I
trust this?"), not a probability; its calibration is checked offline against the eval results.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from kassist.llm.schemas import Coverage

WEIGHTS = {"evidence": 0.35, "coverage": 0.25, "citations": 0.15, "model": 0.25}
_COVERAGE_SCORE = {"full": 1.0, "partial": 0.5, "none": 0.0}

Level = Literal["high", "medium", "low"]


class Confidence(BaseModel):
    score: float
    level: Level
    components: dict[str, float]


def level_of(score: float) -> Level:
    if score >= 0.75:
        return "high"
    return "medium" if score >= 0.5 else "low"


def blended_confidence(coverage: Coverage, model_confidence: float | None,
                       cited_relevances: Sequence[float], n_valid_markers: int,
                       n_dropped_markers: int) -> Confidence:
    total_markers = n_valid_markers + n_dropped_markers
    components = {
        "evidence": sum(cited_relevances) / len(cited_relevances) if cited_relevances else 0.0,
        "coverage": _COVERAGE_SCORE[coverage],
        "citations": n_valid_markers / total_markers if total_markers else 0.0,
        "model": min(1.0, max(0.0, model_confidence)) if model_confidence is not None else 0.5,
    }
    score = round(sum(WEIGHTS[k] * v for k, v in components.items()), 3)
    return Confidence(score=score, level=level_of(score),
                      components={k: round(v, 3) for k, v in components.items()})
