"""Evaluation harness.

  python scripts/run_eval.py retrieval      # no LLM needed: retriever ablation + relevance-gate calibration
  python scripts/run_eval.py e2e [--no-judge] [--ids S1,M1]   # full system, needs an LLM

Outputs markdown + JSON under docs/eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kassist.config import PROJECT_ROOT, RetrievalSettings, get_settings
from kassist.evaluation.judge import judge_faithfulness
from kassist.evaluation.metrics import (
    chunk_matches,
    fact_coverage,
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from kassist.ingestion.chunker import chunker_signature, estimate_tokens
from kassist.observability import configure_logging
from kassist.retrieval.retriever import Retriever
from kassist.services import Services, build_services

OUT_DIR = PROJECT_ROOT / "docs" / "eval"
DECLINE_STATUSES = {"insufficient_context", "out_of_scope", "refused"}


def load_questions(ids: set[str] | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_questions.yaml").read_text(encoding="utf-8"))
    qs = data["questions"]
    for q in qs:
        q["gold"] = [tuple(g) for g in q.get("gold", [])]
        q.setdefault("should_decline", False)
    return [q for q in qs if not ids or q["id"] in ids]


# --------------------------------------------------------------------------- retrieval
@dataclass
class Variant:
    name: str
    mode: str
    rerank: bool
    overrides: dict[str, Any]


VARIANTS = [
    Variant("dense only", "dense", False, {}),
    Variant("BM25 only", "sparse", False, {}),
    Variant("hybrid (RRF)", "hybrid", False, {}),
    Variant("hybrid + rerank (20 cand.)", "hybrid", True, {}),
    Variant("hybrid + rerank (12 cand.)", "hybrid", True, {"candidate_k": 12}),
    Variant("hybrid + rerank (20 cand., 700 chars)", "hybrid", True, {"rerank_max_chars": 700}),
    Variant("hybrid + rerank (12 cand., 700 chars)", "hybrid", True, {"candidate_k": 12, "rerank_max_chars": 700}),
]


def run_retrieval(services: Services, questions: list[dict[str, Any]],
                  variants: list[Variant] = VARIANTS) -> dict[str, Any]:
    base = services.retriever
    k = services.settings.retrieval.top_k
    answerable = [q for q in questions if q["gold"]]
    declinable = [q for q in questions if q["should_decline"]]
    rows, per_question = [], {}
    for v in variants:
        cfg: RetrievalSettings = base.cfg.model_copy(update=v.overrides)
        r = Retriever(base.store, base.dense, base.sparse, base.reranker, cfg)
        hits1, hits3, hitsk, recalls, rrs, lat, ctx = [], [], [], [], [], [], []
        for q in answerable:
            t0 = time.perf_counter()
            res = r.retrieve(q["question"], top_k=k, mode=v.mode, rerank=v.rerank)  # type: ignore[arg-type]
            lat.append((time.perf_counter() - t0) * 1000)
            ctx.append(float(sum(estimate_tokens(c.text) for c in res.chunks)))
            hits1.append(hit_at_k(res.chunks, q["gold"], 1))
            hits3.append(hit_at_k(res.chunks, q["gold"], 3))
            hitsk.append(hit_at_k(res.chunks, q["gold"], k))
            recalls.append(recall_at_k(res.chunks, q["gold"], k))
            rrs.append(reciprocal_rank(res.chunks, q["gold"]))
            per_question.setdefault(q["id"], {})[v.name] = {
                "hit@k": hitsk[-1], "rr": round(rrs[-1], 3), "best_relevance": round(res.best_relevance, 4)}
        gate = {}
        if v.rerank:
            # Relevance gate: answerable questions should pass, unanswerable ones should not.
            ans_pass = [r.retrieve(q["question"], top_k=k).sufficient for q in answerable]
            dec_block = [not r.retrieve(q["question"], top_k=k).sufficient for q in declinable]
            gate = {"answerable_pass_rate": mean([float(x) for x in ans_pass]),
                    "unanswerable_block_rate": mean([float(x) for x in dec_block])}
        rows.append({"variant": v.name, "hit@1": mean(hits1), "hit@3": mean(hits3), f"hit@{k}": mean(hitsk),
                     f"recall@{k}": mean(recalls), "mrr": mean(rrs), "latency_ms_mean": mean(lat),
                     "latency_ms_p95": percentile(lat, 95), "context_tokens": mean(ctx), **gate})
        print(f"  {v.name:42s} hit@{k}={mean(hitsk):.2f} mrr={mean(rrs):.2f} "
              f"lat={mean(lat):.0f}ms ctx={mean(ctx):.0f}tok {gate}", flush=True)

    # Score distribution behind the min_relevance threshold (default reranker config)
    calibration = []
    for q in questions:
        if q["category"] in ("adversarial", "ambiguous"):
            continue
        res = base.retrieve(q["question"], top_k=k)
        calibration.append({"id": q["id"], "category": q["category"], "should_decline": q["should_decline"],
                            "best_relevance": round(res.best_relevance, 4)})
    return {"k": k, "chunking": chunker_signature(services.settings.chunking),
            "n_answerable": len(answerable), "variants": rows, "per_question": per_question,
            "calibration": calibration, "min_relevance": base.cfg.min_relevance}


def retrieval_markdown(res: dict[str, Any]) -> str:
    k = res["k"]
    lines = [
        "# Retrieval evaluation", "",
        f"{res['n_answerable']} answerable questions with hand-labelled gold (document, page) sources. "
        f"A retrieved chunk is a hit if it comes from the gold document and its page range covers the gold page. "
        f"Multi-step questions are retrieved with the raw question here (the agent decomposes them at runtime), "
        f"so these numbers are a lower bound for that category. Index chunking: `{res.get('chunking')}`. "
        f"*Context tokens* = estimated size of the passages handed to the LLM.", "",
        f"| Variant | Hit@1 | Hit@3 | Hit@{k} | Recall@{k} | MRR | Context tokens | Mean latency | p95 latency "
        "| Gate: answerable pass | Gate: unanswerable blocked |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["variants"]:
        gate_a = f"{r['answerable_pass_rate']:.0%}" if "answerable_pass_rate" in r else "n/a"
        gate_u = f"{r['unanswerable_block_rate']:.0%}" if "unanswerable_block_rate" in r else "n/a"
        lines.append(f"| {r['variant']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r[f'hit@{k}']:.2f} | "
                     f"{r[f'recall@{k}']:.2f} | {r['mrr']:.2f} | {r.get('context_tokens', 0):.0f} | "
                     f"{r['latency_ms_mean']:.0f} ms | "
                     f"{r['latency_ms_p95']:.0f} ms | {gate_a} | {gate_u} |")
    lines += ["", f"## Relevance-gate calibration (min_relevance = {res['min_relevance']})", "",
              "Best cross-encoder relevance (sigmoid) per question. Questions that must be declined should "
              "fall below the threshold; answerable ones above it.", "",
              "| Id | Category | Should decline | Best relevance |", "|---|---|---|---|"]
    for c in sorted(res["calibration"], key=lambda c: c["best_relevance"]):
        lines.append(f"| {c['id']} | {c['category']} | {'yes' if c['should_decline'] else 'no'} | "
                     f"{c['best_relevance']:.4f} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- end-to-end
def run_e2e(services: Services, questions: list[dict[str, Any]], judge: bool) -> dict[str, Any]:
    records = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            history = [(h["question"], h["answer"]) for h in q.get("history", [])]
            result = services.agent.run(q["question"], history=history)
        except Exception as exc:
            records.append({"id": q["id"], "category": q["category"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {q['id']}: ERROR {exc}", flush=True)
            continue
        latency = (time.perf_counter() - t0) * 1000
        declined = result.status in DECLINE_STATUSES
        cited_gold = any(chunk_matches(c, g) for c in result.citations for g in q["gold"]) if q["gold"] else None
        rec: dict[str, Any] = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "route": result.route, "escalated": result.escalated, "status": result.status,
            "rewritten_question": result.rewritten_question,
            "confidence": result.confidence.score if result.confidence else None,
            "n_follow_ups": len(result.follow_ups),
            "route_ok": result.route in q["expected_route"] or (result.escalated and "agentic" in q["expected_route"]),
            "status_ok": result.status in q["expected_status"],
            "declined": declined, "should_decline": q["should_decline"],
            "fact_coverage": fact_coverage(result.answer, q["key_facts"]) if q["key_facts"] else None,
            "cited_gold": cited_gold, "n_citations": len(result.citations),
            "tool_calls": result.tool_calls, "llm_calls": result.llm_calls, "latency_ms": round(latency),
            "input_tokens": result.usage.get("input_tokens", 0), "output_tokens": result.usage.get("output_tokens", 0),
            "answer": result.answer, "queries": result.queries, "warnings": result.errors,
        }
        if judge and result.citations and not declined:
            passages = list(services.store.get_chunks([c.chunk_id for c in result.citations]).values())
            try:
                verdict = judge_faithfulness(services.llm, result.answer, passages)
                rec["faithfulness"] = round(verdict.score, 3)
                rec["unsupported_claims"] = verdict.unsupported
            except Exception as exc:
                rec["faithfulness_error"] = type(exc).__name__
        records.append(rec)
        print(f"  {q['id']:3s} route={result.route:9s} status={result.status:22s} "
              f"facts={rec['fact_coverage']} faith={rec.get('faithfulness')} {latency:.0f}ms", flush=True)
    return {"records": records, "summary": summarise(records)}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    answerable = [r for r in ok if not r["should_decline"] and r["category"] not in ("ambiguous", "partial")]
    must_decline = [r for r in ok if r["should_decline"]]
    by_route: dict[str, list[float]] = {}
    for r in ok:
        by_route.setdefault(r["route"], []).append(r["latency_ms"])
    return {
        "n": len(records), "errors": len(records) - len(ok),
        "route_accuracy": mean([float(r["route_ok"]) for r in ok]),
        "status_accuracy": mean([float(r["status_ok"]) for r in ok]),
        "decline_recall": mean([float(r["declined"]) for r in must_decline]),
        "false_decline_rate": mean([float(r["declined"]) for r in answerable]),
        "fact_coverage": mean([r["fact_coverage"] for r in ok if r["fact_coverage"] is not None]),
        "citation_gold_hit": mean([float(r["cited_gold"]) for r in answerable if r["cited_gold"] is not None]),
        "faithfulness": mean([r["faithfulness"] for r in ok if "faithfulness" in r]),
        "latency_ms": {route: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
                       for route, v in by_route.items()},
        "tokens_per_question": mean([float(r["input_tokens"] + r["output_tokens"]) for r in ok]),
        # Calibration checks: confidence should be higher where answers are complete / correct.
        # None (shown as n/a) when a group is empty, e.g. no answer missed a key fact.
        "confidence_when_facts_correct": _mean_or_none([r["confidence"] for r in ok if r.get("confidence") is not None
                                                        and r["fact_coverage"] == 1.0]),
        "confidence_when_facts_missing": _mean_or_none([r["confidence"] for r in ok if r.get("confidence") is not None
                                                        and r["fact_coverage"] is not None
                                                        and r["fact_coverage"] < 1.0]),
        "confidence_answered": _mean_or_none([r["confidence"] for r in ok if r.get("confidence") is not None
                                              and r["status"] == "answered"]),
        "confidence_partial": _mean_or_none([r["confidence"] for r in ok if r.get("confidence") is not None
                                             and r["status"] == "partial"]),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def e2e_markdown(res: dict[str, Any], llm_desc: dict[str, str]) -> str:
    s = res["summary"]
    lines = ["# End-to-end evaluation", "", f"LLM: `{llm_desc}`", "",
             "| Metric | Value |", "|---|---|",
             f"| Questions | {s['n']} (errors: {s['errors']}) |",
             f"| Route accuracy | {s['route_accuracy']:.0%} |",
             f"| Status accuracy | {s['status_accuracy']:.0%} |",
             f"| Decline recall (unanswerable / out-of-scope / adversarial declined) | {s['decline_recall']:.0%} |",
             f"| False-decline rate (answerable questions refused) | {s['false_decline_rate']:.0%} |",
             f"| Key-fact coverage | {s['fact_coverage']:.0%} |",
             f"| Answers citing a gold source | {s['citation_gold_hit']:.0%} |",
             f"| Faithfulness (LLM judge, claim level) | {s['faithfulness']:.2f} |",
             f"| Tokens per question (mean) | {s['tokens_per_question']:.0f} |",
             f"| Mean confidence: answers with all key facts | {_fmt(s['confidence_when_facts_correct'])} |",
             f"| Mean confidence: answers missing key facts | {_fmt(s['confidence_when_facts_missing'])} |",
             f"| Mean confidence: status answered (full coverage) | {_fmt(s.get('confidence_answered'))} |",
             f"| Mean confidence: status partial | {_fmt(s.get('confidence_partial'))} |", "",
             "| Route | p50 latency | p95 latency | n |", "|---|---|---|---|"]
    for route, v in s["latency_ms"].items():
        lines.append(f"| {route} | {v['p50'] / 1000:.1f} s | {v['p95'] / 1000:.1f} s | {v['n']} |")
    lines += ["", "## Per question", "",
              "| Id | Category | Route | Status | Route ok | Status ok | Facts | Gold cited | Faithfulness "
              "| Tools | Latency |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["records"]:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['category']} | error: {r['error']} |||||||||")
            continue
        route = r["route"] + (" (escalated)" if r["escalated"] else "")
        facts = "-" if r["fact_coverage"] is None else f"{r['fact_coverage']:.0%}"
        gold = "-" if r["cited_gold"] is None else ("yes" if r["cited_gold"] else "no")
        lines.append(f"| {r['id']} | {r['category']} | {route} | {r['status']} | {'✓' if r['route_ok'] else '✗'} | "
                     f"{'✓' if r['status_ok'] else '✗'} | {facts} | {gold} | {r.get('faithfulness', '-')} | "
                     f"{r['tool_calls']} | {r['latency_ms'] / 1000:.1f} s |")
    lines += ["", "## Answers", ""]
    for r in res["records"]:
        if "error" not in r:
            lines += [f"### {r['id']}: {r['question']}", "", r["answer"], ""]
            if r.get("unsupported_claims"):
                lines += ["Unsupported claims (judge): " + "; ".join(r["unsupported_claims"]), ""]
    return "\n".join(lines) + "\n"


def _llm_from_report(path: Path) -> Any:
    """LLM description from an older report that predates storing it in e2e.json."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM: `"):
                return line[len("LLM: `"):-1]
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["retrieval", "e2e", "report"],
                        help="report: rebuild docs/eval/e2e.md from the saved e2e.json (no LLM calls)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--variants", help="retrieval: comma-separated variant indexes (default: all)")
    parser.add_argument("--out-suffix", default="", help="suffix for output file names")
    args = parser.parse_args()

    if args.suite == "report":
        saved = json.loads((OUT_DIR / "e2e.json").read_text(encoding="utf-8"))
        saved["summary"] = summarise(saved["records"])
        llm = saved.get("llm") or _llm_from_report(OUT_DIR / "e2e.md")
        (OUT_DIR / "e2e.json").write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / "e2e.md").write_text(e2e_markdown(saved, llm), encoding="utf-8")
        print(json.dumps(saved["summary"], indent=2))
        return 0

    settings = get_settings()
    configure_logging("WARNING", json=False)
    services = build_services(settings)
    questions = load_questions(set(args.ids.split(",")) if args.ids else None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.suite == "retrieval":
        chosen = [VARIANTS[int(i)] for i in args.variants.split(",")] if args.variants else VARIANTS
        res = run_retrieval(services, questions, chosen)
        (OUT_DIR / f"retrieval{args.out_suffix}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (OUT_DIR / f"retrieval{args.out_suffix}.md").write_text(retrieval_markdown(res), encoding="utf-8")
    else:
        res = run_e2e(services, questions, judge=not args.no_judge)
        res["llm"] = services.llm.describe()
        (OUT_DIR / "e2e.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / "e2e.md").write_text(e2e_markdown(res, services.llm.describe()), encoding="utf-8")
        print(json.dumps(res["summary"], indent=2))
    print(f"wrote results to {OUT_DIR}")
    services.store.client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
