"""Demo UI. A thin client of the REST API (no business logic here).

Run: streamlit run src/kassist/ui/streamlit_app.py   (API_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterator
from datetime import datetime

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Example questions by type, shown as clickable tags (one click asks the question).
EXAMPLES: dict[str, list[str]] = {
    "Simple RAG": [
        "What recall@10 does pgvector HNSW achieve at 1M vectors?",
        "What chunk size and overlap are recommended for RecursiveCharacterTextSplitter?",
        "Which backends does LangGraph support for checkpointing?",
    ],
    "Multi-step (agentic)": [
        "Which vector database suits a cost-conscious startup that needs both hybrid search and ACID compliance?",
        "Compare Pinecone, Weaviate and pgvector for hybrid search, metadata filtering and scale.",
        "What security considerations should I address when deploying a RAG system at scale?",
    ],
    "Tool use (calculator)": [
        "How much float32 storage would 10 million text-embedding-3-large vectors need, "
        "and how much with int8 quantisation?",
        "How much storage do 5 million 1536-dimensional float32 vectors need?",
    ],
    "Ambiguous": [
        "Which one is best?",
        "How does hybrid work?",
    ],
    "Partially covered": [
        "What is Qdrant's p99 query latency at 10M vectors?",
    ],
    "Unanswerable / out of scope": [
        "What index types does Milvus support?",
        "How does Semantic Kernel implement planners?",
        "What's the weather going to be like in Paris tomorrow?",
    ],
}
STATUS_BADGE = {
    "answered": ("✅", "Answered"), "partial": ("🟡", "Partially answered"),
    "insufficient_context": ("⛔", "Not in knowledge base"), "clarification_needed": ("❓", "Needs clarification"),
    "out_of_scope": ("🚫", "Out of scope"), "refused": ("🛡️", "Refused"),
}

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_data(ttl=60)
def fetch_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API_URL}/v1/documents", headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()["documents"]  # each: title, pages, chunks, indexed_at, embedding_model, ...
    except Exception:
        return []


def ask_stream(question: str, mode: str, top_k: int, doc_ids: list[str]) -> Iterator[tuple[str, dict]]:
    """Ask within the current conversation via /v1/ask/stream; yields (event, data) pairs.
    The server keeps the history and resolves follow-ups; text arrives already verified."""
    payload = {"question": question, "mode": mode, "top_k": top_k, "doc_ids": doc_ids or None,
               "conversation_id": st.session_state.get("conversation_id")}
    with httpx.stream("POST", f"{API_URL}/v1/ask/stream", json=payload, headers=HEADERS, timeout=180) as r:
        if r.status_code == 404 and payload["conversation_id"]:
            # Conversation expired or was deleted server-side: start a fresh one transparently.
            st.session_state.conversation_id = None
            yield from ask_stream(question, mode, top_k, doc_ids)
            return
        if r.status_code != 200:
            r.read()
            ctype = r.headers.get("content-type", "")
            err = r.json().get("error", {}) if ctype.startswith("application/json") else {}
            raise RuntimeError(f"{r.status_code}: {err.get('message', r.text[:200])}")
        event = "message"
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event == "final":
                    st.session_state.conversation_id = data["conversation_id"]
                yield event, data


def clear_conversation() -> None:
    """Forget the conversation locally and erase it on the server."""
    cid = st.session_state.get("conversation_id")
    if cid:
        # Best effort: if the API is unreachable the conversation still expires server-side.
        with contextlib.suppress(httpx.HTTPError):
            httpx.delete(f"{API_URL}/v1/conversations/{cid}", headers=HEADERS, timeout=10)
    st.session_state.conversation_id = None
    st.session_state.history = []


def _when(iso: str) -> str:
    """ISO timestamp from the API -> short local time for display."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %H:%M")
    except ValueError:
        return iso


def _page_label(c: dict) -> str:
    start, end = c["page_start"], c["page_end"]
    return f"Page {start}" if start == end else f"Pages {start}–{end}"


def _section(c: dict) -> str:
    """Section path without the document title (already shown in the label)."""
    parts = [p for p in c["section"].split(" > ") if p != c["document"]]
    return " > ".join(parts) or c["section"]


def _plain(text: str) -> str:
    """Escape markdown so a source snippet always renders as normal-size plain text."""
    return re.sub(r"([\\`*_#|<>\[\]])", r"\\\1", text)


def _pick(key: str) -> None:
    """Tag clicked: queue its question and clear the selection so the tag can be clicked again."""
    st.session_state.pending_question = st.session_state[key]
    st.session_state[key] = None


STEP_LABELS = {
    "guard": "Checked the question", "rewrite": "Resolved the follow-up", "router": "Chose a workflow",
    "retrieve": "Searched the documents", "tool:search": "Searched", "tool:calculator": "Calculated",
    "plan": "Planned the steps", "reflect": "Reviewed the evidence", "escalate": "Switched to multi-step",
    "generate": "Drafted the answer", "synthesize": "Drafted the answer", "finalize": "Verified citations",
    "clarify": "Needs clarification", "decline": "Out of scope",
}
CONFIDENCE_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def render(resp: dict, idx: int) -> None:
    icon, label = STATUS_BADGE.get(resp["status"], ("ℹ️", resp["status"]))
    wf = resp["workflow"]
    route = wf["route"] + (" (escalated from simple)" if wf.get("escalated") else "")
    if wf.get("rewritten_question"):
        st.caption(f"Interpreted as: {wf['rewritten_question']}")
    st.markdown(resp["answer"])
    if resp["citations"]:
        st.markdown("**Sources**")
        for c in resp["citations"]:
            with st.expander(f"[{c['id']}] {c['document']} · {_page_label(c)} · {_section(c)}"):
                st.caption(f"{c['source_file']} · {_page_label(c)} · relevance {c['relevance']:.2f} · "
                           f"chunk {c['chunk_id'][:8]}")
                st.markdown(_plain(c["snippet"]))
    conf = resp.get("confidence")
    if conf:
        st.caption(f"{CONFIDENCE_ICON.get(conf['level'], '')} Confidence {conf['score']:.0%} ({conf['level']})",
                   help="Blend of evidence relevance, coverage, citation validity and the model's own rating")
    if resp.get("follow_ups"):
        key = f"follow_ups_{idx}"
        st.pills("Suggested follow-ups", resp["follow_ups"], key=key, on_change=_pick, args=(key,))
    with st.expander("Workflow trace"):
        st.caption(f"Router: {wf['reason']}")
        for step in wf.get("steps") or []:
            st.markdown(f"- **{step['node']}** ({step['ms']:.0f} ms): {step['detail']}")
        if resp.get("warnings"):
            st.warning("; ".join(resp["warnings"]))
        if conf:
            st.caption("confidence components: " + ", ".join(f"{k} {v:.0%}" for k, v in conf["components"].items()))
        st.caption(f"tokens in/out: {resp['usage']['input_tokens']}/{resp['usage']['output_tokens']} · "
                   f"request {resp['request_id']}")
    # Compact status line at the bottom of the answer
    st.caption(f"{icon} {label} · workflow: {route} · {resp['latency_ms'] / 1000:.1f} s · "
               f"{wf['llm_calls']} LLM / {wf['tool_calls']} tool calls")


with st.sidebar:
    docs = fetch_documents()
    st.subheader("Knowledge base")
    for d in docs:
        indexed = f" · indexed {_when(d['indexed_at'])}" if d.get("indexed_at") else ""
        st.caption(f"📄 {d['title']}: {d['pages']} pages, {d['chunks']} chunks{indexed}")
    if docs:
        with st.expander("Index details", expanded=False):
            models = sorted({d["embedding_model"] for d in docs if d.get("embedding_model")})
            chunkers = sorted({d["chunker_version"] for d in docs if d.get("chunker_version")})
            n = len(docs)
            st.caption(f"{n} document{'s' if n != 1 else ''} · {sum(d['chunks'] for d in docs)} chunks")
            st.caption(f"Embedding model: {', '.join(models) or 'unknown'}")
            st.caption(f"Chunker: {', '.join(chunkers) or 'unknown'}")
            st.caption("Ingestion runs as a separate job (`python -m kassist.ingestion`); unchanged "
                       "documents are skipped and changes appear here without restarting.")
    else:
        st.error(f"API not reachable at {API_URL}")
    if st.button("Clear conversation"):
        clear_conversation()
    st.divider()
    # Optional API parameters for power users and demos; the defaults are what users get.
    with st.expander("Advanced options", expanded=False):
        mode = st.radio("Workflow", ["auto", "rag", "agent"], horizontal=True,
                        help="auto: the router decides (default) · rag: force simple RAG · "
                             "agent: force the agentic workflow")
        top_k = st.slider("Passages per retrieval", 2, 12, 6,
                          help="How many document passages are given to the LLM per search")
        doc_ids = st.multiselect("Restrict to documents", [d["doc_id"] for d in docs],
                                 format_func=lambda i: next((d["title"] for d in docs if d["doc_id"] == i), i))

# Layout tweaks: less empty space above the title, chat input sits closer to the bottom edge.
st.markdown("""<style>
[data-testid="stMainBlockContainer"] { padding-top: 2rem; }
[data-testid="stBottomBlockContainer"] { padding-bottom: 1rem; }
</style>""", unsafe_allow_html=True)


def _scroll_to_bottom() -> None:
    """Scroll the page to the latest message. Streamlit has no scroll API, so a tiny same-origin iframe
    runs a fixed (never user-supplied) script. The counter makes each call a new element so it reruns."""
    st.session_state.scroll_n = st.session_state.get("scroll_n", 0) + 1
    st.iframe(
        f"""<script>/* {st.session_state.scroll_n} */
        const main = window.parent.document.querySelector('[data-testid="stMain"]');
        if (main) {{ main.scrollTo({{top: main.scrollHeight, behavior: "smooth"}}); }}
        </script>""",
        height=1,
    )


st.title("Knowledge Assistant")

st.session_state.setdefault("history", [])


# Collapsed once a conversation exists; a fixed-height area keeps it compact and scrollable.
with st.expander("Example questions", expanded=not st.session_state.history), st.container(height=230):
    for i, (category, questions) in enumerate(EXAMPLES.items()):
        key = f"examples_{i}"
        st.pills(category, questions, key=key, on_change=_pick, args=(key,))

# Conversation in chronological order (newest at the bottom, next to the input box).
for i, (q, resp, error) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        if error:
            st.error(error)
        else:
            render(resp, i)

question = st.chat_input("Ask a question about the documents") or st.session_state.pop("pending_question", None)
if question:
    # Show the question immediately and jump to it, so it is obvious the request was sent.
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        _scroll_to_bottom()
        resp, error, text = None, None, ""
        progress = st.status("Working…", expanded=True)
        answer_box = st.empty()
        try:
            for event, data in ask_stream(question, mode, top_k, doc_ids):
                if event == "step":
                    progress.markdown(f"✓ **{STEP_LABELS.get(data['node'], data['node'])}**: {data['detail']}")
                elif event == "status":
                    progress.update(label=f"{data['detail'].capitalize()}…")
                elif event == "token":
                    text += data["text"]
                    answer_box.markdown(text + " ▌")
                elif event == "final":
                    resp = data
                elif event == "error":
                    error = data.get("message", "Request failed")
        except Exception as exc:  # show API errors inline instead of crashing the page
            error = str(exc)
        if resp is None and error is None:
            error = "The answer stream ended unexpectedly."
        progress.update(label="Done" if resp else "Failed", state="complete" if resp else "error", expanded=False)
        answer_box.empty()  # the verified final answer replaces the streamed text
        if error:
            st.error(error)
        else:
            render(resp, len(st.session_state.history))
    st.session_state.history.append((question, resp, error))
    _scroll_to_bottom()
