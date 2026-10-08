"""Pre-fetch the local ONNX models into the configured cache dir.

Used at Docker build time so containers start without network access to the model hub,
and locally to warm the cache once.
"""

from __future__ import annotations

from pathlib import Path

from kassist.config import get_settings
from kassist.embeddings.fastembed_models import FastEmbedReranker, build_models


def _mark_bm25_cached(cache_dir: str | None) -> None:
    """fastembed's BM25 entry declares files that do not exist in the upstream repo (a
    placeholder 'mock.file' and some stop-word lists), so its cache check never succeeds and it
    calls the hub on every start. Creating the missing (unused) files lets the runtime run fully
    offline with HF_HUB_OFFLINE=1."""
    from fastembed.sparse.bm25 import supported_bm25_models

    if not cache_dir:
        return
    declared = [f for m in supported_bm25_models for f in (m.model_file, *m.additional_files)]
    for snapshot in Path(cache_dir).glob("models--Qdrant--bm25/snapshots/*"):
        for name in declared:
            (snapshot / name).touch(exist_ok=True)


def main() -> None:
    cfg = get_settings().embeddings
    dense, sparse = build_models(cfg)
    reranker = FastEmbedReranker(cfg.reranker_model, cfg.cache_dir)
    _mark_bm25_cached(cfg.cache_dir)
    print(f"dense={dense.model_name} (dim={dense.dim}) sparse={sparse.model_name} "
          f"reranker={reranker.model_name} cache_dir={cfg.cache_dir}")


if __name__ == "__main__":
    main()
