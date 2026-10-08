"""Composition root: builds the long-lived services once per process (models are loaded at
startup, not per request) and checks that the index matches the configured models.

The ingestion job runs separately from the API, so the API also notices when the index changes
(documents added, re-indexed or removed): at most every `api.index_refresh_s` seconds it compares a
cheap fingerprint of the stored index versions and, if it changed, re-validates the index and reloads
the document list and the router/planner catalogue. No restart is needed after re-ingestion.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from kassist.agent.graph import KnowledgeAgent
from kassist.config import ChunkingSettings, Settings
from kassist.conversations import ConversationStore, SqliteConversationStore
from kassist.domain import DocumentInfo
from kassist.embeddings.base import DenseEmbedder, Reranker, SparseEmbedder
from kassist.ingestion.chunker import chunker_signature
from kassist.llm.client import LangChainLLM, LLMClient
from kassist.llm.prompts import format_catalog
from kassist.observability import get_logger
from kassist.retrieval.retriever import Retriever
from kassist.vectorstore.qdrant_store import IndexIncompatibleError, QdrantStore

log = get_logger(__name__)

IndexFingerprint = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass
class Services:
    settings: Settings
    store: QdrantStore
    retriever: Retriever
    llm: LLMClient
    agent: KnowledgeAgent
    documents: list[DocumentInfo]
    conversations: ConversationStore
    dense: DenseEmbedder | None = None
    index_error: str | None = None  # set when a re-ingested index no longer matches the config
    _fingerprint: IndexFingerprint = ()
    _checked_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def refresh_if_index_changed(self) -> bool:
        """Reload documents + catalogue if ingestion changed the index. Returns True if reloaded."""
        interval = self.settings.api.index_refresh_s
        if time.monotonic() - self._checked_at < interval:
            return False
        with self._lock:
            if time.monotonic() - self._checked_at < interval:  # another request just checked
                return False
            self._checked_at = time.monotonic()
            fingerprint = index_fingerprint(self.store)
            if fingerprint == self._fingerprint:
                return False
            try:
                if self.dense is not None:
                    check_index(self.store, self.dense, self.settings.chunking)
            except IndexIncompatibleError as exc:
                self.index_error = str(exc)
                self._fingerprint = fingerprint
                log.error("index.incompatible_after_reingest", error=self.index_error)
                return False
            self.documents = self.store.list_documents()
            self.agent.update_catalog(*catalog_for(self.documents))
            self.index_error = None
            self._fingerprint = fingerprint
            log.info("index.reloaded", documents=[d.doc_id for d in self.documents], chunks=self.store.count())
            return True


def index_fingerprint(store: QdrantStore) -> IndexFingerprint:
    """Which documents are indexed, at which version (file hash + chunker + model)."""
    return tuple(sorted((doc_id, tuple(sorted(versions))) for doc_id, versions in store.indexed_versions().items()))


def topic_list(doc: DocumentInfo) -> list[str]:
    """Top-level section names of a document, for the router/planner catalog."""
    topics: list[str] = []
    for section in doc.sections:
        parts = [p for p in section.split(" > ") if p != doc.title]
        if parts and parts[0] not in topics:
            topics.append(parts[0])
    return topics


def catalog_for(documents: list[DocumentInfo]) -> tuple[str, list[str], list[str]]:
    """(catalog text, doc ids, titles) used by the router and planner prompts."""
    catalog = format_catalog([(d.doc_id, d.title, topic_list(d)) for d in documents])
    return catalog, [d.doc_id for d in documents], [d.title for d in documents]


def check_index(store: QdrantStore, dense: DenseEmbedder, chunking: ChunkingSettings) -> None:
    """Fail fast if the index does not match the configuration: vectors from a different
    embedding model silently return garbage, and chunks built with other chunking settings mean
    config changes were never applied."""
    if not store.exists() or store.count() == 0:
        raise IndexIncompatibleError(
            f"Collection '{store.collection}' is missing or empty: run ingestion first "
            "(python -m kassist.ingestion).")
    models = store.embedding_models()
    if models != {dense.model_name}:
        raise IndexIncompatibleError(
            f"Index built with {sorted(models)} but queries use '{dense.model_name}': re-run ingestion.")
    expected = chunker_signature(chunking)
    versions = store.chunker_versions()
    if versions != {expected}:
        raise IndexIncompatibleError(
            f"Index built with chunking {sorted(versions)} but config is '{expected}': re-run ingestion.")


def build_services(settings: Settings, *, store: QdrantStore | None = None,
                   dense: DenseEmbedder | None = None, sparse: SparseEmbedder | None = None,
                   reranker: Reranker | None = None, llm: LLMClient | None = None,
                   conversations: ConversationStore | None = None) -> Services:
    if store is None:
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        store = QdrantStore.from_settings(settings.qdrant_url, settings.qdrant_path, key, settings.collection)
    if dense is None or sparse is None or reranker is None:
        from kassist.embeddings.fastembed_models import FastEmbedReranker, build_models

        d, s = build_models(settings.embeddings)
        dense, sparse = dense or d, sparse or s
        reranker = reranker or FastEmbedReranker(settings.embeddings.reranker_model,
                                                 settings.embeddings.cache_dir)
    check_index(store, dense, settings.chunking)

    documents = store.list_documents()
    catalog, doc_ids, titles = catalog_for(documents)
    retriever = Retriever(store, dense, sparse, reranker, settings.retrieval)
    retriever.retrieve("warm-up query: initialise ONNX sessions and index caches")  # first call is slow
    llm = llm or LangChainLLM(settings)
    agent = KnowledgeAgent(retriever, llm, settings.agent, settings.retrieval, catalog,
                           known_doc_ids=doc_ids, doc_titles=titles)
    log.info("services.ready", documents=doc_ids, chunks=store.count(), llm=llm.describe())
    conversations = conversations or SqliteConversationStore(
        settings.conversation_db_path, settings.conversations.ttl_hours, settings.conversations.max_turns)
    return Services(settings, store, retriever, llm, agent, documents, conversations, dense=dense,
                    _fingerprint=index_fingerprint(store), _checked_at=time.monotonic())
