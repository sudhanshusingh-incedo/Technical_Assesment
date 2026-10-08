"""Render the architecture diagram to docs/architecture.png and docs/architecture.svg.

    python scripts/render_architecture.py

Drawn with matplotlib (no Graphviz / Node needed) so the diagram is reproducible from code and
stays next to the implementation it describes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs"

INK = "#1f2933"
MUTED = "#52606d"
C = {
    "ingest_band": "#eaf2fb", "ingest": "#cfe3f7",
    "query_band": "#edf7ef", "graph_band": "#f3effa", "node": "#ddd3f2", "decision": "#f6e7c1",
    "store": "#fbe1c8", "llm": "#f9d9d9", "client": "#d7efe0", "retriever": "#d3ecec",
}


def panel(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=color, ec="#c3ccd5", lw=1.0, zorder=0))
    ax.text(x + 0.8, y + h - 1.4, label, fontsize=11, fontweight="bold", color=MUTED, va="top", zorder=1)


def box(ax, x, y, w, h, title, sub="", color="#ffffff", title_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=color, ec="#8a99a8", lw=1.0, zorder=2))
    cy = y + h / 2
    if sub:
        # Centre the title + subtitle block vertically, whatever the number of subtitle lines.
        line_h, title_h, gap = 0.7, 0.8, 0.55
        block = title_h + gap + line_h * (sub.count("\n") + 1)
        top = cy + block / 2
        ax.text(x + w / 2, top - title_h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, top - title_h - gap, sub, ha="center", va="top", fontsize=7.4, color=MUTED,
                zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)


def arrow(ax, p1, p2, label="", dashed=False, rad=0.0, color="#3e4c59", lw=1.3, label_xy=None, both=False):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11, lw=lw, color=color,
        linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=4,
        shrinkA=1, shrinkB=1))
    if label:
        lx, ly = label_xy or ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(lx, ly, label, fontsize=7.2, color=color, ha="center", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})


def polyline(ax, points, label="", dashed=False, color="#3e4c59", label_xy=None):
    """Right-angled connector; arrowhead on the last segment."""
    for a, b in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.3, zorder=4,
                linestyle=(0, (4, 3)) if dashed else "-")
    arrow(ax, points[-2], points[-1], dashed=dashed, color=color)
    if label:
        lx, ly = label_xy
        ax.text(lx, ly, label, fontsize=7.2, color=color, ha="center", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9})


def render() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(19, 12.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    ax.text(1, 65, "Knowledge Assistant: architecture", fontsize=16, fontweight="bold", color=INK, va="top")
    ax.text(1, 62.6, "Offline ingestion job populates the index; the API answers through a LangGraph workflow "
            "that routes between simple RAG and a multi-step agent, streams verified answers and keeps "
            "conversation history.", fontsize=9, color=MUTED, va="top")

    # ---------------------------------------------------------------- offline ingestion
    panel(ax, 1, 48.5, 98, 11.5, "OFFLINE  ·  ingestion job  (python -m kassist.ingestion · docker compose run ingest)",
          C["ingest_band"])
    steps = [
        ("PDF corpus", "data/corpus/*.pdf"),
        ("Layout-aware parse", "pymupdf4llm → markdown\ntables + headings\n(cached by file hash)"),
        ("Clean", "running headers/footers,\nbanners, artefacts"),
        ("Structure-aware chunker",
         "section boundaries · ≤380 tok\ntables split with header\n+ Document/Section header"),
        ("Embed (local ONNX)", "bge-small dense (384-d)\n+ BM25 sparse"),
        ("Idempotent upsert", "index_version = file hash\n+ chunker + model\nskip / re-index / delete"),
    ]
    xs = [2.5, 18.5, 34.5, 50.5, 66.5, 82.5]
    for (title, sub), x in zip(steps, xs, strict=True):
        box(ax, x, 50, 14.5, 7.4, title, sub, C["ingest"])
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, (a + 14.5, 53.7), (b, 53.7))

    # ---------------------------------------------------------------- storage between the bands
    box(ax, 82.5, 39.2, 14.5, 7.2, "Qdrant", "dense + BM25 vectors\n+ citation metadata", C["store"])
    arrow(ax, (89.75, 50), (89.75, 46.4))

    # ---------------------------------------------------------------- query time
    panel(ax, 1, 1, 98, 36.5, "QUERY TIME  ·  FastAPI service", C["query_band"])
    box(ax, 2.5, 27, 13, 6.5, "Streamlit UI", "examples · follow-up tags\nlive steps + streaming", C["client"])
    box(ax, 2.5, 14.5, 13, 9.5, "FastAPI", "/v1/ask · /v1/ask/stream (SSE)\n/v1/documents · /v1/conversations\n"
        "auth · validation · rate limit\nrequest id · safe errors", C["client"])
    box(ax, 2.5, 3, 13, 7.5, "Conversation store", "SQLite, keyed by\nconversation_id\nTTL 24 h · 20 turns", C["store"])
    arrow(ax, (9, 27), (9, 24), both=True)
    arrow(ax, (9, 14.5), (9, 10.5), both=True, label="history", label_xy=(11.6, 12.5))

    # LangGraph panel
    panel(ax, 18, 2.2, 61.5, 33.2, "LangGraph workflow (typed state, bounded loops)", C["graph_band"])
    box(ax, 19.5, 26, 9.5, 5.5, "guard", "injection screen", C["node"])
    box(ax, 31, 26, 10.5, 5.5, "rewrite", "follow-up →\nstandalone question", C["node"])
    box(ax, 44, 26, 10, 5.5, "router", "fast LLM", C["decision"])
    box(ax, 59, 26, 18.5, 5.5, "clarify · out of scope · refused", "canned reply, no retrieval", "#ece8f4",
        title_size=8.5)
    arrow(ax, (29, 28.75), (31, 28.75))
    arrow(ax, (41.5, 28.75), (44, 28.75))
    arrow(ax, (54, 28.75), (59, 28.75))

    # simple path
    box(ax, 19.5, 16.5, 13.5, 5.8, "retrieve", "one hybrid search", C["node"])
    box(ax, 19.5, 9.3, 13.5, 5.8, "generate", "strong LLM, streamed", C["node"])
    arrow(ax, (47, 26), (28, 22.3), label="simple", rad=0.0, label_xy=(36.5, 24.6))
    arrow(ax, (26.25, 16.5), (26.25, 15.1))

    # agentic path
    box(ax, 38.5, 16.5, 10.5, 5.8, "plan", "≤4 sub-tasks", C["node"])
    box(ax, 51.5, 16.5, 12.5, 5.8, "parallel tools", "search · calculator\n(LangGraph Send)", C["node"])
    box(ax, 66.5, 16.5, 11, 5.8, "reflect", "follow-ups?\n≤2 rounds, ≤8 calls", C["node"])
    box(ax, 51.5, 9.3, 12.5, 5.8, "synthesize", "strong LLM, streamed", C["node"])
    arrow(ax, (50.5, 26), (45, 22.3), label="agentic", label_xy=(50.6, 24.0))
    arrow(ax, (49, 19.4), (51.5, 19.4))
    arrow(ax, (64, 19.4), (66.5, 19.4))
    arrow(ax, (72, 22.3), (60, 22.3), rad=0.45, dashed=True, label="more evidence", label_xy=(66, 24.4))
    arrow(ax, (70, 16.5), (64, 13.0))
    arrow(ax, (33, 12.2), (38.5, 18.0), dashed=True, label="partial coverage\n→ escalate", rad=-0.25,
          label_xy=(36.2, 12.7))

    # finalize
    box(ax, 32, 3, 27, 5, "finalize", "citation verification · status · blended confidence · ≤3 follow-ups",
        "#c9e9d2")
    arrow(ax, (26.25, 9.3), (34, 8.0))
    arrow(ax, (57.75, 9.3), (55, 8.0))

    # API <-> graph, streaming back
    arrow(ax, (15.5, 21.5), (19.5, 28.0), label="question", label_xy=(16.9, 25.4))
    polyline(ax, [(32, 4.6), (17, 4.6), (17, 15.5), (15.5, 15.5)], dashed=True, color="#2f7d4a",
             label="SSE: steps · verified tokens · final", label_xy=(24.5, 5.6))

    # retriever + LLM on the right
    box(ax, 82.5, 22.5, 14.5, 11, "Hybrid retriever", "dense ANN + BM25\n→ RRF fusion (k=60)\n→ cross-encoder rerank\n"
        "→ relevance gate\n(decline if nothing relevant)", C["retriever"])
    arrow(ax, (89.75, 33.5), (89.75, 39.2), both=True)
    # retrieve -> retriever: along the panel's left margin and under the panel title, clear of all nodes
    polyline(ax, [(19.5, 19.4), (18.7, 19.4), (18.7, 32.7), (81.5, 32.7), (82.5, 31.8)], color="#2d6a6a",
             label="retrieval", label_xy=(70, 32.7))
    arrow(ax, (64, 21.5), (82.5, 25.5), color="#2d6a6a", label="search", label_xy=(76.5, 25.4))

    box(ax, 82.5, 4, 14.5, 14, "LLM providers", "Azure OpenAI / OpenAI\nfast tier: rewrite, router,\nplan, reflect\n"
        "strong tier: answers\n\nper-call fallback → Ollama", C["llm"])
    arrow(ax, (79.5, 11), (82.5, 11), dashed=True, label="structured /\nstreamed calls", label_xy=(80.9, 14.6))

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig = render()
    fig.savefig(OUT / "architecture.png", dpi=160)
    fig.savefig(OUT / "architecture.svg")
    print(f"wrote {OUT / 'architecture.png'} and {OUT / 'architecture.svg'}")


if __name__ == "__main__":
    main()
