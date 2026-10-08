"""CLI: python -m kassist.ingestion [--source DIR] [--force]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kassist.config import get_settings
from kassist.embeddings.fastembed_models import build_models
from kassist.ingestion.pipeline import IngestionPipeline
from kassist.observability import configure_logging
from kassist.vectorstore.qdrant_store import QdrantStore


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index the PDF corpus into the vector store.")
    parser.add_argument("--source", type=Path, default=Path(settings.corpus_dir))
    parser.add_argument("--force", action="store_true", help="re-index even unchanged documents")
    args = parser.parse_args(argv)

    configure_logging(settings.log_level, settings.log_json)
    store = QdrantStore.from_settings(
        settings.qdrant_url, settings.qdrant_path,
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        settings.collection,
    )
    dense, sparse = build_models(settings.embeddings)
    cache = Path(settings.parse_cache_dir) if settings.parse_cache_dir else None
    pipeline = IngestionPipeline(store, dense, sparse, settings.chunking, cache)
    report = pipeline.run(args.source, args.force)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
