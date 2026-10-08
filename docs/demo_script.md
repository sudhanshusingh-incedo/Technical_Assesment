# Demo recording script (≤ 5 minutes)

Setup before recording: `docker compose up` already running and healthy, the UI open at
http://localhost:8501 with "Workflow trace" expanders ready, and the README architecture diagram open in
another tab. Run each question once beforehand so nothing is cold.

## 0:00-1:00 Architecture and key decisions

Show the README diagram.

> "Two pipelines. Offline, an idempotent ingestion job parses the PDFs with a layout-aware parser,
> because most facts in this corpus are in tables. It chunks along section boundaries and stores dense
> and BM25 vectors in Qdrant. Re-running it skips unchanged files.
>
> Online, FastAPI fronts a LangGraph workflow. A router sends each question to simple RAG or to an
> agentic plan, execute, reflect loop with parallel searches and a calculator.
>
> Three decisions I'd highlight. First, hybrid retrieval with a cross-encoder whose calibrated score also
> acts as a gate, so unanswerable questions are declined before any generation. Second, LangGraph
> plan-and-execute instead of an open-ended ReAct agent, so cost and latency are bounded and every routing
> decision is code. Third, every citation is verified against the passages actually shown to the model."

## 1:00-4:00 Live questions (≈45 s each)

1. **Simple RAG**: under *Simple RAG*, click the tag ("What recall@10 does pgvector HNSW achieve at 1M vectors?").
   Point at: status Answered, route `simple`, 2 LLM calls, the source *Vector Database Comparison p. 5,
   section 8.2*, and the snippet showing the table row.

2. **Multi-step agentic**: under *Multi-step (agentic)*, click the first tag (startup + hybrid search + ACID).
   Open the trace: router reason, the plan's parallel searches, reflect, synthesise. Point at citations
   from several sections. Mention: "Single-shot RAG fails here. The cover page quotes this exact question,
   so it wins retrieval but contains no answer. Decomposition finds the comparison matrix instead."

3. **Tool use**: under *Tool use (calculator)*, click the first tag (storage for 10M text-embedding-3-large vectors).
   Show the trace: search finds 3072 dimensions and the storage formula, reflect issues
   `calc(3072*4*10000000/1e9)`, and the answer is ~122.9 GB float32 / ~30.7 GB int8.

4. **Unanswerable**: under *Unanswerable / out of scope*, click the tag (Milvus).
   Point at: status "Not in knowledge base", 1 LLM call (the router only), the trace line "no relevant
   evidence: declined without calling the LLM". Optionally click the *Partially covered* tag (Qdrant), which
   answers what the corpus says and states what it doesn't.

## 4:00-5:00 What I'd improve with more time

> "A larger evaluation set run in CI as a regression gate. Right now it's 18 hand-labelled questions,
> which is enough to find failure modes but not to measure small changes. I'd generate candidates
> synthetically, review them by hand, and gate merges on decline recall and faithfulness. Second,
> streaming plus semantic caching, because the cross-encoder and the strong-model call dominate
> latency."
