"""Offline ingestion: PDFs -> chunks -> dense+sparse vectors -> Qdrant.

Idempotent and incremental: each chunk carries an `index_version` (file hash + chunker version +
embedding model). Unchanged documents are skipped; changed ones are re-indexed with the new
points written *before* stale points are deleted; documents removed from the corpus folder are
deleted from the index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from kassist.config import ChunkingSettings
from kassist.domain import IndexedChunkMeta
from kassist.embeddings.base import DenseEmbedder, SparseEmbedder
from kassist.ingestion.chunker import chunk_document, chunker_signature
from kassist.ingestion.loader import file_sha256, load_pdf, slugify
from kassist.observability import get_logger
from kassist.vectorstore.qdrant_store import QdrantStore, index_version_of

log = get_logger(__name__)


@dataclass
class IngestionReport:
    indexed: dict[str, int] = field(default_factory=dict)  # doc_id -> chunks
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed


class IngestionPipeline:
    def __init__(self, store: QdrantStore, dense: DenseEmbedder, sparse: SparseEmbedder,
                 chunking: ChunkingSettings, parse_cache_dir: Path | None = None):
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.chunking = chunking
        self.parse_cache_dir = parse_cache_dir

    def _meta(self, sha: str) -> IndexedChunkMeta:
        return IndexedChunkMeta(file_sha256=sha, embedding_model=self.dense.model_name,
                                chunker_version=chunker_signature(self.chunking), indexed_at=time.time())

    def run(self, source_dir: Path, force: bool = False) -> IngestionReport:
        started = time.perf_counter()
        report = IngestionReport()
        files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() == ".pdf")
        if not files:
            raise FileNotFoundError(f"No PDF files found in {source_dir}")

        self.store.ensure_collection(self.dense.dim)
        existing = self.store.indexed_versions()
        seen: set[str] = set()

        for path in files:
            doc_id = slugify(path.stem)
            seen.add(doc_id)
            try:
                meta = self._meta(file_sha256(path))
                version = index_version_of(meta)
                if not force and existing.get(doc_id) == {version}:
                    report.skipped.append(doc_id)
                    log.info("ingest.skip_unchanged", doc_id=doc_id)
                    continue
                report.indexed[doc_id] = self._index_file(path, meta)
                self.store.delete_stale(doc_id, keep_index_version=version)
            except Exception as exc:  # one bad file must not abort the whole corpus
                log.exception("ingest.failed", doc_id=doc_id, file=path.name)
                report.failed[doc_id] = f"{type(exc).__name__}: {exc}"

        for doc_id in sorted(set(existing) - seen):
            self.store.delete_document(doc_id)
            report.removed.append(doc_id)
            log.info("ingest.removed", doc_id=doc_id)

        report.seconds = round(time.perf_counter() - started, 2)
        log.info("ingest.done", indexed=report.indexed, skipped=report.skipped,
                 removed=report.removed, failed=list(report.failed), seconds=report.seconds,
                 total_points=self.store.count())
        return report

    def _index_file(self, path: Path, meta: IndexedChunkMeta) -> int:
        t0 = time.perf_counter()
        doc = load_pdf(path, self.parse_cache_dir)
        chunks = chunk_document(doc, self.chunking)
        texts = [c.embed_text for c in chunks]
        dense = self.dense.embed_documents(texts)
        sparse = self.sparse.embed_documents(texts)
        self.store.upsert(chunks, dense, sparse, meta)
        log.info("ingest.indexed", doc_id=doc.doc_id, title=doc.title, pages=len(doc.pages),
                 chunks=len(chunks), tables=sum(c.content_type != "text" for c in chunks),
                 seconds=round(time.perf_counter() - t0, 2))
        return len(chunks)
