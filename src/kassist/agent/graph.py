"""LangGraph orchestration: guard -> router -> {simple RAG | agentic plan-execute-reflect | clarify | decline}.

                  ┌────────── refused ──────────────────────────────────────────────┐
  START → guard → router ─ simple ─→ retrieve → generate ─ full / no evidence ─→ finalize → END
                    │                               └─ partial (auto mode) ─→ escalate ─┐
                    ├─ agentic ─→ plan ──Send×N──→ run_subtask ─→ reflect ─┬─ follow-ups ─→ run_subtask
                    │               ▲─────────────────────────────────────┘ └─ done ─→ synthesize → finalize
                    ├─ clarify ──→ clarify → END
                    └─ out_of_scope → decline → END

Why plan-and-execute (not open-ended ReAct): the steps are known up front for comparison /
multi-constraint questions, sub-queries run in parallel, and cost is bounded by explicit budgets
(max sub-tasks, iterations, tool calls). Every routing decision is plain code and is returned to
the caller as a trace.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Sequence
from contextlib import closing
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from kassist.agent.answer_stream import AnswerStreamParser
from kassist.agent.calculator import CalculatorError, evaluate, format_number
from kassist.agent.citations import Citation, verify_citations
from kassist.agent.confidence import Confidence, blended_confidence
from kassist.agent.guard import detect_injection, normalise_question
from kassist.agent.state import AgentState, Step, ToolOutcome
from kassist.config import AgentSettings, RetrievalSettings
from kassist.domain import RetrievedChunk
from kassist.llm import prompts
from kassist.llm.client import LLMClient, LLMUnavailableError
from kassist.llm.schemas import GroundedAnswer, Plan, Reflection, Rewrite, RouteDecision, SubTask
from kassist.observability import get_logger
from kassist.retrieval.retriever import Retriever

log = get_logger(__name__)

NOT_COVERED = ("I could not find information about this in the knowledge base, so I can't answer "
               "it without guessing.")
UNGROUNDED = ("I found related passages but could not produce an answer that is supported by them, "
              "so I am not answering to avoid giving unverified information.")
REFUSED = ("I can't help with that request. I can answer questions about the documents in the "
           "knowledge base.")
DEFAULT_CLARIFY = "Could you clarify what you are referring to, so I can search the right documents?"

_AGENTIC_HINTS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differences|trade-?offs?|pros and cons|"
    r"which\b.{0,60}\b(should|best|suits?|recommend|choose)|how much|calculate|estimate)\b", re.I)


class AgentResult(BaseModel):
    question: str  # the question actually answered (standalone form)
    original_question: str
    rewritten_question: str | None  # set when a follow-up was rewritten using the conversation
    answer: str
    status: str
    citations: list[Citation]
    route: str
    route_reason: str
    escalated: bool
    iterations: int
    tool_calls: int
    llm_calls: int
    queries: list[str]
    steps: list[Step]
    errors: list[str]
    usage: dict[str, Any]
    best_relevance: float
    confidence: Confidence | None = None  # blended score, only for answered / partial
    follow_ups: list[str] = []  # suggested next questions (skipped when confidence is very low)


def heuristic_route(question: str) -> RouteDecision:
    """Fallback router when the LLM router is unavailable."""
    if _AGENTIC_HINTS.search(question) or question.count("?") > 1:
        return RouteDecision(route="agentic", reason="heuristic: comparison/multi-part/calculation cues")
    return RouteDecision(route="simple", reason="heuristic: single focused question")


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)


class KnowledgeAgent:
    def __init__(self, retriever: Retriever, llm: LLMClient, agent_cfg: AgentSettings,
                 retrieval_cfg: RetrievalSettings, catalog: str, known_doc_ids: Sequence[str],
                 doc_titles: Sequence[str]):
        self.retriever = retriever
        self.llm = llm
        self.cfg = agent_cfg
        self.rcfg = retrieval_cfg
        self.catalog = catalog
        self.known_doc_ids = set(known_doc_ids)
        self.doc_titles = list(doc_titles)
        self.graph = self._build()

    def update_catalog(self, catalog: str, known_doc_ids: Sequence[str], doc_titles: Sequence[str]) -> None:
        """Swap in a new document catalogue after re-ingestion (used by the router and planner)."""
        self.catalog = catalog
        self.known_doc_ids = set(known_doc_ids)
        self.doc_titles = list(doc_titles)

    # ------------------------------------------------------------------ graph
    def _build(self) -> Any:
        g = StateGraph(AgentState)
        nodes: list[tuple[str, Any]] = [
            ("guard", self.guard), ("router", self.router), ("retrieve", self.retrieve),
            ("generate", self.generate), ("escalate", self.escalate), ("plan", self.plan),
            ("run_subtask", self.run_subtask), ("reflect", self.reflect),
            ("synthesize", self.synthesize), ("finalize", self.finalize),
            ("clarify", self.clarify), ("decline", self.decline), ("rewrite", self.rewrite),
        ]
        for name, fn in nodes:
            g.add_node(name, fn)
        g.add_edge(START, "guard")
        g.add_conditional_edges("guard", self._after_guard, ["rewrite", "router", END])
        g.add_edge("rewrite", "router")
        g.add_conditional_edges("router", lambda s: s["route"], {
            "simple": "retrieve", "agentic": "plan", "clarify": "clarify", "out_of_scope": "decline"})
        g.add_edge("retrieve", "generate")
        g.add_conditional_edges("generate", self._after_generate, ["escalate", "finalize"])
        g.add_edge("escalate", "plan")
        g.add_conditional_edges("plan", self._dispatch, ["run_subtask", "synthesize"])
        g.add_edge("run_subtask", "reflect")
        g.add_conditional_edges("reflect", self._dispatch, ["run_subtask", "synthesize"])
        g.add_edge("synthesize", "finalize")
        for terminal in ("finalize", "clarify", "decline"):
            g.add_edge(terminal, END)
        return g.compile()

    def run(self, question: str, mode: str = "auto", top_k: int | None = None,
            doc_ids: list[str] | None = None,
            history: Sequence[tuple[str, str]] | None = None) -> AgentResult:
        """history: previous (question, answer) turns of the conversation, oldest first."""
        usage = UsageMetadataCallbackHandler()
        state = self._initial_state(question, mode, top_k, doc_ids, history)
        final = self.graph.invoke(state, config={"callbacks": [usage], "recursion_limit": 40})
        return self._result(final, usage)

    def stream(self, question: str, mode: str = "auto", top_k: int | None = None,
               doc_ids: list[str] | None = None,
               history: Sequence[tuple[str, str]] | None = None) -> Iterator[dict[str, Any]]:
        """Same as run(), but yields events as the workflow progresses:
        {"type": "step", "step": Step}             a graph node finished
        {"type": "status", "detail": str}          e.g. "writing answer"
        {"type": "token", "text": str}             verified answer text (see answer_stream.py)
        {"type": "result", "result": AgentResult}  always last"""
        usage = UsageMetadataCallbackHandler()
        state = self._initial_state(question, mode, top_k, doc_ids, history)
        final: dict[str, Any] = dict(state)
        for kind, chunk in self.graph.stream(state, config={"callbacks": [usage], "recursion_limit": 40},
                                             stream_mode=["updates", "custom", "values"]):
            if kind == "custom":
                yield chunk
            elif kind == "updates":
                for update in chunk.values():
                    for step in (update or {}).get("steps", []):
                        yield {"type": "step", "step": step}
            else:
                final = chunk
        yield {"type": "result", "result": self._result(final, usage)}

    def _initial_state(self, question: str, mode: str, top_k: int | None, doc_ids: list[str] | None,
                       history: Sequence[tuple[str, str]] | None) -> AgentState:
        q = normalise_question(question)
        recent = list(history or [])[-self.cfg.max_history_turns:] if self.cfg.max_history_turns > 0 else []
        return {
            "question": q, "original_question": q, "history": recent, "rewritten": False,
            "mode": mode,  # type: ignore[typeddict-item]
            "top_k": top_k or self.rcfg.top_k, "doc_ids": doc_ids or None,
            "iterations": 0, "escalated": False, "best_relevance": 0.0, "context": [],
        }

    def _result(self, final: dict[str, Any], usage: UsageMetadataCallbackHandler) -> AgentResult:
        return AgentResult(
            question=final["question"],
            original_question=final["original_question"],
            rewritten_question=final["question"] if final.get("rewritten") else None,
            answer=final.get("answer", ""),
            status=final.get("status", "insufficient_context"),
            citations=final.get("citations", []),
            route=final.get("route", "simple"),
            route_reason=final.get("route_reason", ""),
            escalated=final.get("escalated", False),
            iterations=final.get("iterations", 0),
            tool_calls=final.get("tool_calls", 0),
            llm_calls=final.get("llm_calls", 0),
            queries=final.get("executed", []),
            steps=final.get("steps", []),
            errors=final.get("errors", []),
            usage=_summarise_usage(usage.usage_metadata),
            best_relevance=round(final.get("best_relevance", 0.0), 4),
            confidence=final.get("confidence"),
            follow_ups=final.get("follow_ups", []),
        )

    # ------------------------------------------------------------------ nodes
    def guard(self, state: AgentState) -> dict[str, Any]:
        t = _Timer()
        rule = detect_injection(state["question"])
        if rule:
            log.warning("guard.blocked", rule=rule)
            return {"route": "refused", "route_reason": "prompt-injection pattern detected",
                    "status": "refused", "answer": REFUSED, "citations": [],
                    "steps": [Step(node="guard", detail=f"blocked ({rule})", ms=t.ms)]}
        return {"steps": [Step(node="guard", detail="input accepted", ms=t.ms)]}

    def rewrite(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Resolve a follow-up ("what about its latency?") into a standalone question using the
        conversation, so routing and retrieval work on a self-contained query."""
        t = _Timer()
        try:
            result = self.llm.structured(Rewrite, [
                SystemMessage(prompts.REWRITE_SYSTEM),
                HumanMessage(f"{prompts.format_history(state['history'])}\n\n"
                             f"<new_message>{state['question']}</new_message>"),
            ], tier="fast", config=config)
        except LLMUnavailableError:
            return {"errors": ["rewrite: LLM unavailable, used the question as asked"],
                    "steps": [Step(node="rewrite", detail="skipped (LLM unavailable)", ms=t.ms)]}
        standalone = normalise_question(result.standalone_question)[:2000]
        changed = result.is_follow_up and bool(standalone) and standalone != state["question"]
        if changed and detect_injection(standalone):  # never let a rewrite smuggle in an override
            changed = False
        if not changed:
            return {"llm_calls": 1, "steps": [Step(node="rewrite", detail="standalone question, unchanged",
                                                   ms=t.ms)]}
        return {"question": standalone, "rewritten": True, "llm_calls": 1,
                "steps": [Step(node="rewrite", detail=f"follow-up rewritten as: {standalone}", ms=t.ms)]}

    def router(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        t = _Timer()
        mode = state.get("mode", "auto")
        errors: list[str] = []
        llm_calls = 0
        if mode == "rag":
            decision = RouteDecision(route="simple", reason="forced by request (mode=rag)")
        elif mode == "agent":
            decision = RouteDecision(route="agentic", reason="forced by request (mode=agent)")
        else:
            try:
                decision = self.llm.structured(RouteDecision, [
                    SystemMessage(prompts.ROUTER_SYSTEM.format(catalog=self.catalog)),
                    HumanMessage(f"<question>{state['question']}</question>"),
                ], tier="fast", config=config)
                llm_calls = 1
            except LLMUnavailableError:
                decision = heuristic_route(state["question"])
                errors.append("router: LLM unavailable, used heuristic routing")
        return {"route": decision.route, "route_reason": decision.reason,
                "clarification": decision.clarification_question, "llm_calls": llm_calls,
                "errors": errors,
                "steps": [Step(node="router", detail=f"{decision.route}: {decision.reason}", ms=t.ms)]}

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        t = _Timer()
        res = self.retriever.retrieve(state["question"], top_k=state["top_k"], doc_ids=state.get("doc_ids"))
        return {"evidence": res.chunks, "best_relevance": res.best_relevance,
                "executed": [state["question"]],
                "steps": [Step(node="retrieve", detail=f"{len(res.chunks)} passages, best relevance "
                                                       f"{res.best_relevance:.2f}", ms=t.ms)]}

    def generate(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        t = _Timer()
        context = _rank(state.get("evidence", []))[: state["top_k"]]
        if state.get("best_relevance", 0.0) < self.rcfg.min_relevance:
            draft = GroundedAnswer(answer=NOT_COVERED, coverage="none")
            return {"draft": draft, "context": context, "steps": [Step(
                node="generate", detail="no relevant evidence: declined without calling the LLM", ms=t.ms)]}
        draft = self._answer(state["question"], context, [], config,
                             abandon_unless_full=self._can_escalate(state))
        detail = f"coverage={draft.coverage}" + ("" if draft.answer else " (answer not shown: escalating)")
        return {"draft": draft, "context": context, "llm_calls": 1,
                "steps": [Step(node="generate", detail=detail, ms=t.ms)]}

    def escalate(self, state: AgentState) -> dict[str, Any]:
        draft = state.get("draft")
        coverage = draft.coverage if draft else "none"
        return {"escalated": True, "route": "agentic", "steps": [Step(
            node="escalate", detail=f"single retrieval gave coverage={coverage}; switching to agentic")]}

    def plan(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        t = _Timer()
        errors: list[str] = []
        llm_calls = 0
        try:
            plan = self.llm.structured(Plan, [
                SystemMessage(prompts.PLANNER_SYSTEM.format(catalog=self.catalog,
                                                            max_subtasks=self.cfg.max_subtasks)),
                HumanMessage(f"<question>{state['question']}</question>"),
            ], tier="fast", config=config)
            llm_calls = 1
            subtasks = self._sanitise(plan.subtasks, state)[: self.cfg.max_subtasks]
        except LLMUnavailableError:
            subtasks = []
            errors.append("planner: LLM unavailable, fell back to a single search")
        if not subtasks:
            subtasks = [SubTask(tool="search", query=state["question"], purpose="direct search")]
        desc = "; ".join(_describe(s) for s in subtasks)
        return {"plan": subtasks, "llm_calls": llm_calls, "errors": errors,
                "steps": [Step(node="plan", detail=f"{len(subtasks)} step(s): {desc}", ms=t.ms)]}

    def run_subtask(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Executes one planned step. Receives a Send payload (not the full state)."""
        t = _Timer()
        task: SubTask = payload["subtask"]
        if task.tool == "calculator":
            expr = task.expression or ""
            try:
                outcome = ToolOutcome(tool="calculator", input=expr, success=True,
                                      output=f"{expr} = {format_number(evaluate(expr))}")
            except CalculatorError as exc:
                outcome = ToolOutcome(tool="calculator", input=expr, success=False, output=f"error: {exc}")
            return {"tool_results": [outcome], "tool_calls": 1, "executed": [f"calc: {expr}"],
                    "steps": [Step(node="tool:calculator", detail=outcome.output, ms=t.ms)]}

        query = task.query or ""
        request_docs = payload.get("doc_ids")
        doc_filter = [task.doc_id] if task.doc_id else request_docs
        try:
            res = self.retriever.retrieve(query, top_k=payload["top_k"], doc_ids=doc_filter)
            if task.doc_id and not request_docs and not res.sufficient:
                # The planner may have guessed the wrong document: retry across the corpus.
                wide = self.retriever.retrieve(query, top_k=payload["top_k"])
                if wide.best_relevance > res.best_relevance:
                    res = wide
        except Exception as exc:  # degrade: a failed search is reported, not fatal
            log.exception("tool.search_failed")
            outcome = ToolOutcome(tool="search", input=query, success=False, output=f"error: {type(exc).__name__}")
            return {"tool_results": [outcome], "tool_calls": 1, "executed": [query],
                    "errors": [f"search failed for '{query}'"],
                    "steps": [Step(node="tool:search", detail=f"'{query}' failed", ms=t.ms)]}
        outcome = ToolOutcome(tool="search", input=query, success=True,
                              output=f"{len(res.chunks)} passages, best relevance {res.best_relevance:.2f}",
                              chunk_ids=[c.chunk_id for c in res.chunks], best_relevance=res.best_relevance)
        evidence = [c for c in res.chunks if c.relevance >= self.rcfg.keep_relevance]
        return {"evidence": evidence, "tool_results": [outcome], "tool_calls": 1, "executed": [query],
                "steps": [Step(node="tool:search", detail=f"'{query}' -> {outcome.output}", ms=t.ms)]}

    def reflect(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        t = _Timer()
        iteration = state.get("iterations", 0) + 1
        budget = self.cfg.max_tool_calls - state.get("tool_calls", 0)
        if iteration >= self.cfg.max_iterations or budget <= 0:
            return {"iterations": iteration, "plan": [], "steps": [Step(
                node="reflect", detail=f"stop: iteration/tool budget reached ({iteration}, {budget} calls left)",
                ms=t.ms)]}
        try:
            refl = self.llm.structured(Reflection, [
                SystemMessage(prompts.REFLECT_SYSTEM.format(max_followups=min(2, budget))),
                HumanMessage(self._reflect_prompt(state)),
            ], tier="fast", config=config)
        except LLMUnavailableError:
            return {"iterations": iteration, "plan": [], "errors": ["reflect: LLM unavailable, skipped"],
                    "steps": [Step(node="reflect", detail="skipped (LLM unavailable)", ms=t.ms)]}
        done = {q.lower().strip() for q in state.get("executed", [])}
        followups = [s for s in self._sanitise(refl.followups, state)
                     if _describe_key(s) not in done][: min(2, budget)]
        if refl.complete:
            followups = []
        detail = "complete" if not followups else "follow-ups: " + "; ".join(_describe(s) for s in followups)
        return {"iterations": iteration, "plan": followups, "llm_calls": 1,
                "steps": [Step(node="reflect", detail=f"{detail} ({refl.reason})", ms=t.ms)]}

    def synthesize(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        t = _Timer()
        ranked = [c for c in _rank(state.get("evidence", [])) if c.relevance >= self.rcfg.keep_relevance]
        context = ranked[: self.cfg.max_evidence_chunks]
        best = max((c.relevance for c in state.get("evidence", [])), default=0.0)
        calcs = [r.output for r in state.get("tool_results", []) if r.tool == "calculator" and r.success]
        if best < self.rcfg.min_relevance and not calcs:
            return {"draft": GroundedAnswer(answer=NOT_COVERED, coverage="none"), "context": context,
                    "best_relevance": best, "steps": [Step(
                        node="synthesize", detail="no relevant evidence from any step: declined", ms=t.ms)]}
        draft = self._answer(state["question"], context, calcs, config)
        return {"draft": draft, "context": context, "best_relevance": best, "llm_calls": 1,
                "steps": [Step(node="synthesize", detail=f"{len(context)} passages, coverage={draft.coverage}",
                               ms=t.ms)]}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        t = _Timer()
        draft = state.get("draft")
        errors: list[str] = []
        if draft is None or draft.coverage == "none":
            answer = draft.answer if draft and draft.answer else NOT_COVERED
            answer = re.sub(r"\s*\[\d{1,3}\]", "", answer)  # a decline carries no citations
            return {"answer": answer, "citations": [], "status": "insufficient_context",
                    "steps": [Step(node="finalize", detail="status=insufficient_context", ms=t.ms)]}
        verified = verify_citations(draft.answer, state.get("context", []))
        if verified.dropped_ids:
            errors.append(f"removed citations to non-existent passages: {verified.dropped_ids}")
        if not verified.citations:
            status, answer, citations = "insufficient_context", UNGROUNDED, []
        else:
            status = "answered" if draft.coverage == "full" else "partial"
            answer, citations = verified.answer, verified.citations
        if not citations:
            return {"answer": answer, "citations": citations, "status": status, "errors": errors,
                    "steps": [Step(node="finalize", detail=f"status={status}, 0 citation(s)", ms=t.ms)]}
        confidence = blended_confidence(
            draft.coverage, draft.confidence, [c.relevance for c in citations],
            n_valid_markers=len(citations), n_dropped_markers=len(verified.dropped_ids))
        asked = {state["question"].lower(), state.get("original_question", "").lower()}
        follow_ups = [q for q in draft.follow_ups if q.lower() not in asked][: self.cfg.follow_up_count]
        if confidence.score < self.cfg.follow_up_min_confidence:
            follow_ups = []  # too uncertain to steer the user further
        detail = (f"status={status}, {len(citations)} citation(s), confidence {confidence.score:.2f} "
                  f"({confidence.level}), {len(follow_ups)} follow-up(s)")
        return {"answer": answer, "citations": citations, "status": status, "errors": errors,
                "confidence": confidence, "follow_ups": follow_ups,
                "steps": [Step(node="finalize", detail=detail, ms=t.ms)]}

    def clarify(self, state: AgentState) -> dict[str, Any]:
        return {"answer": state.get("clarification") or DEFAULT_CLARIFY, "citations": [],
                "status": "clarification_needed",
                "steps": [Step(node="clarify", detail="asked the user to clarify")]}

    def decline(self, state: AgentState) -> dict[str, Any]:
        covered = ", ".join(self.doc_titles)
        answer = (f"That question is outside the scope of this knowledge base, which covers: {covered}. "
                  "Please ask about those topics.")
        return {"answer": answer, "citations": [], "status": "out_of_scope",
                "steps": [Step(node="decline", detail="out of scope")]}

    # ------------------------------------------------------------------ edges & helpers
    def _after_guard(self, state: AgentState) -> str:
        if state.get("status") == "refused":
            return END
        return "rewrite" if state.get("history") else "router"

    def _can_escalate(self, state: AgentState) -> bool:
        return (state.get("mode", "auto") == "auto" and not state.get("escalated")
                and state.get("best_relevance", 0.0) >= self.rcfg.min_relevance)

    def _after_generate(self, state: AgentState) -> str:
        draft = state.get("draft")
        if self._can_escalate(state) and draft is not None and draft.coverage != "full":
            return "escalate"  # evidence exists but one retrieval wasn't enough
        return "finalize"

    def _dispatch(self, state: AgentState) -> list[Send] | str:
        plan = state.get("plan") or []
        if not plan:
            return "synthesize"
        return [Send("run_subtask", {"subtask": task, "top_k": state["top_k"],
                                     "doc_ids": state.get("doc_ids")}) for task in plan]

    def _answer(self, question: str, context: list[RetrievedChunk], calcs: list[str],
                config: RunnableConfig, abandon_unless_full: bool = False) -> GroundedAnswer:
        """Generate the grounded answer as a stream. Verified answer text is emitted as `token` events
        (a no-op when the graph is not being streamed). With abandon_unless_full, generation stops as
        soon as the model reports less than full coverage: that answer would be escalated, never shown."""
        writer = get_stream_writer()
        writer({"type": "status", "detail": "writing answer"})
        parser = AnswerStreamParser(n_passages=len(context))
        show = False
        messages = [SystemMessage(prompts.ANSWER_SYSTEM),
                    HumanMessage(prompts.answer_user_prompt(question, context, calcs))]
        with closing(self.llm.stream_text(messages, tier="strong", config=config)) as chunks:  # type: ignore[type-var]
            for chunk in chunks:
                for event in parser.feed(chunk):
                    if event.kind == "header":
                        if abandon_unless_full and event.coverage != "full":
                            return GroundedAnswer(answer="", coverage=event.coverage or "partial",
                                                  confidence=event.confidence)
                        show = event.coverage in ("full", "partial")  # a decline is never streamed
                    elif show:
                        writer({"type": "token", "text": event.text})
        for event in parser.finish():
            if show and event.kind == "text":
                writer({"type": "token", "text": event.text})
        return parser.result(self.cfg.follow_up_count)

    def _sanitise(self, tasks: Sequence[SubTask], state: AgentState) -> list[SubTask]:
        clean: list[SubTask] = []
        allowed = set(state.get("doc_ids") or self.known_doc_ids)
        for task in tasks:
            if task.tool == "search" and task.query and task.query.strip():
                doc_id = task.doc_id if task.doc_id in allowed else None
                clean.append(task.model_copy(update={"query": task.query.strip()[:300], "doc_id": doc_id}))
            elif task.tool == "calculator" and task.expression and task.expression.strip():
                clean.append(task.model_copy(update={"expression": task.expression.strip()[:200]}))
        return clean

    def _reflect_prompt(self, state: AgentState) -> str:
        executed = "\n".join(f"- {q}" for q in state.get("executed", []))
        top = _rank(state.get("evidence", []))[:8]
        calcs = [r.output for r in state.get("tool_results", []) if r.tool == "calculator"]
        return (f"<question>{state['question']}</question>\n\n<executed>\n{executed}\n</executed>\n\n"
                f"{prompts.format_passages(top, max_chars=500)}\n\n"
                f"<calculations>\n" + "\n".join(calcs) + "\n</calculations>")


def _rank(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=lambda c: c.relevance, reverse=True)


def _describe(task: SubTask) -> str:
    if task.tool == "calculator":
        return f"calc({task.expression})"
    return f"search('{task.query}'{', doc=' + task.doc_id if task.doc_id else ''})"


def _describe_key(task: SubTask) -> str:
    return (f"calc: {task.expression}" if task.tool == "calculator" else (task.query or "")).lower().strip()


def _summarise_usage(by_model: dict[str, Any]) -> dict[str, Any]:
    total_in = sum(int(u.get("input_tokens", 0)) for u in by_model.values())
    total_out = sum(int(u.get("output_tokens", 0)) for u in by_model.values())
    return {"input_tokens": total_in, "output_tokens": total_out,
            "by_model": {m: {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
                         for m, u in by_model.items()}}
