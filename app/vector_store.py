"""Local embeddings and a persistent Chroma vector store."""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion import Document


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "document_chunks"
DEFAULT_PERSIST_DIR = "./chroma_db"
GET_PAGE_SIZE = 1000


# ============================================================
# Retrieved document model
# ============================================================

@dataclass(frozen=True)
class RetrievedDocument:
    """A retrieved chunk with scores and structural metadata."""

    text: str
    source: str
    page: int | None
    distance: float
    similarity: float
    section_type: str = "body"
    section_title: str | None = None


# ============================================================
# Model and Chroma initialization
# ============================================================

@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = Path(
        os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    )
    persist_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(persist_dir))


def _get_collection() -> chromadb.Collection:
    return _get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# Embeddings
# ============================================================

def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _get_embedding_model()

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return vectors.tolist()


# ============================================================
# Metadata helpers
# ============================================================

def _metadata_from_chunk(
    chunk: Document,
) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "source": chunk.source,
        "section_type": chunk.section_type,
    }

    if chunk.page is not None:
        metadata["page"] = chunk.page

    if chunk.section_title:
        metadata["section_title"] = chunk.section_title

    return metadata


def _page_from_metadata(
    metadata: dict | None,
) -> int | None:
    if not metadata or "page" not in metadata:
        return None

    return int(metadata["page"])


# ============================================================
# Index status and content hash
# ============================================================

def get_indexed_chunk_count() -> int:
    """Return the total number of chunks stored in Chroma."""
    return _get_collection().count()


def get_indexed_source_hash(source: str) -> str | None:
    """Return the stored SHA-256 hash for an indexed source."""

    collection = _get_collection()

    result = collection.get(
        where={"source": source},
        limit=1,
        include=["metadatas"],
    )

    metadatas = result.get("metadatas") or []

    if not metadatas:
        return None

    return metadatas[0].get("content_hash")


# ============================================================
# Build or update vector store
# ============================================================

def build_vector_store(
    chunks: list[Document],
    content_hashes: dict[str, str] | None = None,
) -> None:
    """
    Add or replace chunks for the supplied sources.

    Stable chunk IDs prevent duplicate records during re-indexing.
    Stale chunks are removed only for sources being updated.
    Content hashes are stored in metadata for future duplicate checks.
    """

    if not chunks:
        return

    # Validate before modifying the collection.
    for chunk in chunks:
        if not chunk.source.strip():
            raise ValueError(
                "Every chunk must have a source filename."
            )

        if not chunk.text.strip():
            raise ValueError(
                "Chunks must not contain empty text."
            )

    # Assign stable IDs based on source and chunk position.
    source_indices: dict[str, int] = defaultdict(int)
    ids: list[str] = []

    for chunk in chunks:
        index = source_indices[chunk.source]
        source_indices[chunk.source] += 1

        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"document-rag:{chunk.source}:{index}",
            )
        )

        ids.append(chunk_id)

    texts = [chunk.text for chunk in chunks]
    metadatas = [
        _metadata_from_chunk(chunk)
        for chunk in chunks
    ]

    # Save the same content hash on each chunk from its source.
    if content_hashes:
        for chunk, metadata in zip(chunks, metadatas):
            content_hash = content_hashes.get(chunk.source)

            if content_hash:
                metadata["content_hash"] = content_hash

    # Generate embeddings only for chunks passed to this function.
    embeddings = _embed_texts(texts)

    collection = _get_collection()

    # Upsert updates existing IDs instead of creating duplicates.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Track which IDs belong to each source in this indexing batch.
    ids_by_source: dict[str, set[str]] = defaultdict(set)

    for chunk, chunk_id in zip(chunks, ids):
        ids_by_source[chunk.source].add(chunk_id)

    # Remove outdated chunks for updated sources only.
    for source, current_ids in ids_by_source.items():
        stored = collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        stale_ids = [
            stored_id
            for stored_id in stored["ids"]
            if stored_id not in current_ids
        ]

        if stale_ids:
            collection.delete(ids=stale_ids)


# ============================================================
# Load all indexed documents
# ============================================================

def get_all_documents(
    page_size: int = GET_PAGE_SIZE,
) -> list[Document]:
    """Load all indexed chunks for lexical retrieval."""

    if page_size <= 0:
        raise ValueError("page_size must be greater than 0.")

    collection = _get_collection()
    total = collection.count()

    documents: list[Document] = []

    for offset in range(0, total, page_size):
        batch = collection.get(
            limit=min(page_size, total - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )

        texts = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []

        for text, metadata in zip(texts, metadatas):
            metadata = metadata or {}

            documents.append(
                Document(
                    text=text or "",
                    source=str(metadata.get("source", "")),
                    page=_page_from_metadata(metadata),
                    section_type=str(
                        metadata.get("section_type", "body")
                    ),
                    section_title=metadata.get("section_title"),
                )
            )

    return documents


# ============================================================
# Vector retrieval
# ============================================================

def retrieve_documents(
    query: str,
    k: int = 5,
    where: dict | None = None,
) -> list[RetrievedDocument]:
    """Return top-k vector results with optional metadata filtering."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty.")

    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_kwargs = {
        "query_embeddings": _embed_texts([query]),
        "n_results": k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    texts = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[RetrievedDocument] = []

    for text, metadata, distance in zip(
        texts,
        metadatas,
        distances,
    ):
        metadata = metadata or {}
        distance_value = float(distance)

        results.append(
            RetrievedDocument(
                text=text or "",
                source=str(metadata.get("source", "")),
                page=_page_from_metadata(metadata),
                distance=distance_value,
                similarity=1.0 - distance_value,
                section_type=str(
                    metadata.get("section_type", "body")
                ),
                section_title=metadata.get("section_title"),
            )
        )

    return results
