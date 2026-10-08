"""Incremental ingestion: unchanged files are skipped, changed ones re-indexed without leftovers,
deleted ones removed. PDF parsing is stubbed so the test exercises pipeline logic only."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from kassist.config import ChunkingSettings
from kassist.ingestion import pipeline as pipeline_mod
from kassist.ingestion.loader import PageMarkdown, ParsedDocument, file_sha256, slugify
from kassist.ingestion.pipeline import IngestionPipeline
from kassist.vectorstore.qdrant_store import QdrantStore
from tests.fakes import FakeDense, FakeSparse


def fake_load_pdf(path: Path, cache_dir=None) -> ParsedDocument:
    content = path.read_text(encoding="utf-8")
    if "BROKEN" in content:
        raise ValueError("corrupt pdf")
    return ParsedDocument(doc_id=slugify(path.stem), title=path.stem, source_file=path.name,
                          file_sha256=file_sha256(path), pages=[PageMarkdown(1, content)])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "load_pdf", fake_load_pdf)
    store = QdrantStore(QdrantClient(":memory:"), "kb")
    pipe = IngestionPipeline(store, FakeDense(), FakeSparse(), ChunkingSettings(max_tokens=50, min_tokens=5))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return pipe, store, corpus


def _write(corpus: Path, name: str, text: str) -> None:
    (corpus / name).write_text(text, encoding="utf-8")


def test_incremental_reindexing(setup):
    pipe, store, corpus = setup
    _write(corpus, "alpha.pdf", "## A\n\nalpha content about vectors.\n\n## B\n\nmore alpha content.")
    _write(corpus, "beta.pdf", "## Beta\n\nbeta content about agents.")

    first = pipe.run(corpus)
    assert set(first.indexed) == {"alpha", "beta"} and first.ok
    total = store.count()

    second = pipe.run(corpus)
    assert second.indexed == {} and sorted(second.skipped) == ["alpha", "beta"]
    assert store.count() == total

    _write(corpus, "alpha.pdf", "## A\n\nalpha content rewritten completely.")
    third = pipe.run(corpus)
    assert list(third.indexed) == ["alpha"] and third.skipped == ["beta"]
    assert len(store.indexed_versions()["alpha"]) == 1  # stale points removed

    (corpus / "beta.pdf").unlink()
    fourth = pipe.run(corpus)
    assert fourth.removed == ["beta"]
    assert set(store.indexed_versions()) == {"alpha"}


def test_one_bad_file_does_not_abort_the_run(setup):
    pipe, store, corpus = setup
    _write(corpus, "good.pdf", "## Good\n\nfine content here.")
    _write(corpus, "bad.pdf", "BROKEN")
    report = pipe.run(corpus)
    assert "good" in report.indexed and "bad" in report.failed and not report.ok


def test_empty_folder_raises(setup):
    pipe, _, corpus = setup
    with pytest.raises(FileNotFoundError):
        pipe.run(corpus)


def test_documents_report_when_and_how_they_were_indexed(setup):
    import time

    pipe, store, corpus = setup
    _write(corpus, "alpha.pdf", "## A\n\nalpha content about vectors.")
    before = time.time()
    pipe.run(corpus)
    (doc,) = store.list_documents()
    assert doc.indexed_at is not None and doc.indexed_at >= before
    assert doc.embedding_model == "fake-dense" and doc.chunker_version.startswith("1.2-")
