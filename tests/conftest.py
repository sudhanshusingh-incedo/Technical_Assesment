from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from kassist.config import Settings
from kassist.domain import Chunk, IndexedChunkMeta
from kassist.ingestion.chunker import chunker_signature
from kassist.retrieval.retriever import Retriever
from kassist.vectorstore.qdrant_store import QdrantStore
from tests.fakes import FakeDense, FakeReranker, FakeSparse

CORPUS = [
    ("vector_database_comparison", "Vector Database Comparison", 5, "Performance Benchmarks",
     "pgvector HNSW (m=16) achieves 96.1% recall@10 at 1M vectors with 22 ms p50 latency."),
    ("vector_database_comparison", "Vector Database Comparison", 4, "pgvector — PostgreSQL Extension",
     "pgvector gives full ACID compliance inside PostgreSQL and partial hybrid search via tsvector."),
    ("vector_database_comparison", "Vector Database Comparison", 3, "Weaviate",
     "Weaviate offers native hybrid search fusing BM25 and vector scores with RRF, plus multi-tenancy."),
    ("rag_architecture_patterns", "RAG Architecture Patterns", 4, "2.4 Recursive Character Text Splitting",
     "RecursiveCharacterTextSplitter recommended configuration: chunk_size=512, chunk_overlap=64."),
    ("rag_architecture_patterns", "RAG Architecture Patterns", 5, "3.4 Dimensionality and Storage",
     "Storage is dimensions x 4 bytes x vectors; text-embedding-3-large has 3072 dimensions."),
    ("agentic_ai_frameworks", "Agentic AI Frameworks", 2, "LangGraph",
     "LangGraph checkpointing persists state snapshots in SQLite, PostgreSQL or Redis."),
]


def make_chunk(doc_id: str, title: str, page: int, section: str, text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}")),
        doc_id=doc_id, doc_title=title, source_file=f"{doc_id}.pdf", page_start=page, page_end=page,
        section=section, chunk_index=index, text=text,
        embed_text=f"Document: {title}\nSection: {section}\n\n{text}", token_estimate=len(text.split()),
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, app_env="test", api_key=None, openai_api_key=None, log_json=False,
                    qdrant_url=None, conversation_db_path=str(tmp_path / "conversations.sqlite"))


@pytest.fixture
def store(settings: Settings) -> QdrantStore:
    s = QdrantStore(QdrantClient(":memory:"), "test_kb")
    dense, sparse = FakeDense(), FakeSparse()
    s.ensure_collection(dense.dim)
    by_doc: dict[str, list[Chunk]] = {}
    for i, row in enumerate(CORPUS):
        by_doc.setdefault(row[0], []).append(make_chunk(*row, index=i))
    for doc_id, chunks in by_doc.items():
        texts = [c.embed_text for c in chunks]
        s.upsert(chunks, dense.embed_documents(texts), sparse.embed_documents(texts),
                 IndexedChunkMeta(file_sha256=f"sha-{doc_id}", embedding_model=dense.model_name,
                                  chunker_version=chunker_signature(settings.chunking)))
    return s


@pytest.fixture
def retriever(store: QdrantStore, settings: Settings) -> Retriever:
    return Retriever(store, FakeDense(), FakeSparse(), FakeReranker(), settings.retrieval)
