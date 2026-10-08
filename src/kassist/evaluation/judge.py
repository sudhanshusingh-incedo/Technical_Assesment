"""LLM-as-judge for answer faithfulness (claim-level, RAGAS-style).

The judge decomposes the answer into atomic factual claims and checks each against the passages
the answer cites. Meta statements ("the documents do not cover X") are excluded, so a correct
refusal is not penalised. Prefer a different/stronger model than the generator to limit
self-preference bias; scores are indicative, not ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kassist.domain import RetrievedChunk
from kassist.llm.client import LLMClient
from kassist.llm.prompts import format_passages

JUDGE_SYSTEM = """You are a strict evaluator of answer faithfulness.
1. Split the ANSWER into atomic factual claims (one fact each). Ignore citation markers, and ignore
   meta statements about what the documents do or do not cover.
2. For each claim decide whether it is directly supported by the PASSAGES. Paraphrase is fine;
   added facts, numbers or conclusions not stated in the passages are NOT supported. Simple
   arithmetic on numbers from the passages counts as supported.
Return the counts and list the unsupported claims verbatim."""


class FaithfulnessVerdict(BaseModel):
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    unsupported: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return min(1.0, self.supported_claims / self.total_claims)


def judge_faithfulness(llm: LLMClient, answer: str, passages: Sequence[RetrievedChunk]) -> FaithfulnessVerdict:
    return llm.structured(FaithfulnessVerdict, [
        SystemMessage(JUDGE_SYSTEM),
        HumanMessage(f"{format_passages(passages)}\n\n<answer>\n{answer}\n</answer>"),
    ], tier="strong")
