import pytest

from kassist.retrieval.retriever import Retriever
from kassist.services import check_index
from kassist.vectorstore.qdrant_store import IndexIncompatibleError, QdrantStore
from tests.fakes import FakeDense


def test_relevant_chunk_ranks_first(retriever: Retriever):
    res = retriever.retrieve("pgvector HNSW recall@10 at 1M vectors")
    assert res.sufficient
    assert res.chunks[0].section == "Performance Benchmarks"
    assert res.chunks[0].relevance >= res.chunks[-1].relevance


def test_unrelated_question_is_insufficient(retriever: Retriever):
    res = retriever.retrieve("weather forecast Paris tomorrow")
    assert not res.sufficient
    assert res.best_relevance < retriever.cfg.min_relevance


def test_doc_filter_restricts_results(retriever: Retriever):
    res = retriever.retrieve("hybrid search", doc_ids=["agentic_ai_frameworks"])
    assert {c.doc_id for c in res.chunks} == {"agentic_ai_frameworks"}


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_modes(retriever: Retriever, mode):
    res = retriever.retrieve("LangGraph checkpointing SQLite", mode=mode)
    assert res.chunks[0].doc_id == "agentic_ai_frameworks"


def test_list_documents(store: QdrantStore):
    docs = {d.doc_id: d for d in store.list_documents()}
    assert set(docs) == {"vector_database_comparison", "rag_architecture_patterns", "agentic_ai_frameworks"}
    assert docs["vector_database_comparison"].chunks == 3


def test_index_check_rejects_model_mismatch(store: QdrantStore, settings):
    check_index(store, FakeDense(), settings.chunking)

    class OtherModel(FakeDense):
        model_name = "other-model"

    with pytest.raises(IndexIncompatibleError, match="other-model"):
        check_index(store, OtherModel(), settings.chunking)


def test_index_check_rejects_chunking_change_without_reingest(store: QdrantStore, settings):
    other = settings.chunking.model_copy(update={"max_tokens": 200})
    with pytest.raises(IndexIncompatibleError, match="chunking"):
        check_index(store, FakeDense(), other)

