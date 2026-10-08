"""Dump indexed chunks to markdown for manual inspection (python scripts/inspect_chunks.py [out.md])."""

from __future__ import annotations

import sys
from pathlib import Path

from kassist.config import get_settings
from kassist.vectorstore.qdrant_store import QdrantStore


def main(out: Path) -> None:
    s = get_settings()
    store = QdrantStore.from_settings(s.qdrant_url, s.qdrant_path, None, s.collection)
    rows = store._scroll_payloads(["doc_id", "chunk_index", "page_start", "page_end", "section",
                                   "content_type", "token_estimate", "text"])
    rows.sort(key=lambda r: (r["doc_id"], r["chunk_index"]))
    lines = [f"# Indexed chunks ({len(rows)})\n"]
    for r in rows:
        lines.append(f"## {r['doc_id']} #{r['chunk_index']} | pp.{r['page_start']}-{r['page_end']} | "
                     f"{r['content_type']} | ~{r['token_estimate']} tok\n**Section:** {r['section']}\n")
        lines.append(r["text"] + "\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} chunks to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/chunk_samples.md"))
