"""Qdrant-backed vector store: dense + BM25 sparse vectors per chunk, fused with RRF server-side.

Runs against a Qdrant server (docker-compose) or in embedded mode (local dev/tests) with the
same code path.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from qdrant_client import QdrantClient, models

from kassist.domain import Chunk, DocumentInfo, IndexedChunkMeta, RetrievedChunk
from kassist.embeddings.base import SparseVec

DENSE = "dense"
SPARSE = "bm25"
SearchMode = Literal["hybrid", "dense", "sparse"]


class IndexIncompatibleError(RuntimeError):
    """The stored vectors were produced by a different embedding model / chunker."""


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    @classmethod
    def from_settings(cls, url: str | None, path: str, api_key: str | None, collection: str) -> QdrantStore:
        client = QdrantClient(url=url, api_key=api_key, timeout=30) if url else QdrantClient(path=path)
        return cls(client, collection)

    # ---------- schema ----------
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, dense_dim: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # embedded mode ignores payload indexes (and says so)
            warnings.simplefilter("ignore", UserWarning)
            for field in ("doc_id", "index_version"):
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    # ---------- writes ----------
    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVec],
        meta: IndexedChunkMeta,
        batch_size: int = 64,
    ) -> None:
        index_version = index_version_of(meta)
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                payload={**chunk.model_dump(exclude={"embed_text"}), **meta.model_dump(),
                         "index_version": index_version},
            )
            for chunk, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[start : start + batch_size], wait=True)

    def delete_stale(self, doc_id: str, keep_index_version: str) -> None:
        """Remove a document's points from older index versions (called after the new upsert,
        so readers never observe the document missing)."""
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
                must_not=[models.FieldCondition(key="index_version",
                                                match=models.MatchValue(value=keep_index_version))],
            )),
            wait=True,
        )

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_doc_filter([doc_id])),
            wait=True,
        )

    # ---------- reads ----------
    def _scroll_payloads(self, fields: list[str]) -> list[dict[str, Any]]:
        if not self.exists():
            return []
        payloads: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=fields, with_vectors=False
            )
            payloads.extend(p.payload or {} for p in points)
            if offset is None:
                return payloads

    def indexed_versions(self) -> dict[str, set[str]]:
        """doc_id -> set of index_versions currently stored."""
        versions: dict[str, set[str]] = defaultdict(set)
        for p in self._scroll_payloads(["doc_id", "index_version"]):
            versions[p["doc_id"]].add(p["index_version"])
        return dict(versions)

    def embedding_models(self) -> set[str]:
        return {p["embedding_model"] for p in self._scroll_payloads(["embedding_model"])}

    def chunker_versions(self) -> set[str]:
        return {p["chunker_version"] for p in self._scroll_payloads(["chunker_version"])}

    def list_documents(self) -> list[DocumentInfo]:
        # Fine for a small corpus; at scale this becomes a separate document registry table.
        fields = ["doc_id", "doc_title", "source_file", "page_end", "section", "chunk_index",
                  "indexed_at", "embedding_model", "chunker_version"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in self._scroll_payloads(fields):
            grouped[p["doc_id"]].append(p)
        docs = []
        for doc_id, rows in sorted(grouped.items()):
            rows.sort(key=lambda r: r["chunk_index"])
            sections = list(dict.fromkeys(r["section"] for r in rows))
            docs.append(DocumentInfo(
                doc_id=doc_id, title=rows[0]["doc_title"], source_file=rows[0]["source_file"],
                pages=max(r["page_end"] for r in rows), chunks=len(rows), sections=sections,
                indexed_at=max((r.get("indexed_at") or 0.0 for r in rows), default=0.0) or None,
                embedding_model=rows[0].get("embedding_model"), chunker_version=rows[0].get("chunker_version"),
            ))
        return docs

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedChunk]:
        points = self.client.retrieve(self.collection, ids=list(chunk_ids), with_payload=True)
        return {str(p.id): RetrievedChunk.from_payload(p.payload or {}, 0.0) for p in points}

    def search(
        self,
        dense_query: list[float],
        sparse_query: SparseVec,
        limit: int,
        prefetch_k: int,
        doc_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[RetrievedChunk]:
        flt = _doc_filter(doc_ids) if doc_ids else None
        sparse = models.SparseVector(indices=sparse_query.indices, values=sparse_query.values)
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dense_query, using=DENSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        elif mode == "sparse":
            res = self.client.query_points(self.collection, query=sparse, using=SPARSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense_query, using=DENSE, limit=prefetch_k, filter=flt),
                    models.Prefetch(query=sparse, using=SPARSE, limit=prefetch_k, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [RetrievedChunk.from_payload(p.payload or {}, p.score) for p in res.points]


def index_version_of(meta: IndexedChunkMeta) -> str:
    return f"{meta.file_sha256[:16]}:{meta.chunker_version}:{meta.embedding_model}"


def _doc_filter(doc_ids: Sequence[str]) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))])
