"""Local embeddings and a persistent Chroma vector store."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "document_chunks"
DEFAULT_PERSIST_DIR = "./chroma_db"


@dataclass(frozen=True)
class RetrievedDocument:
    """A retrieved chunk plus Chroma similarity scores."""

    text: str
    source: str
    page: int | None
    distance: float
    similarity: float


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformer once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR))
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _get_collection(*, reset: bool = False) -> chromadb.Collection:
    client = _get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def _metadata_from_chunk(chunk: Document) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {"source": chunk.source}
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _page_from_metadata(metadata: dict | None) -> int | None:
    if not metadata or "page" not in metadata:
        return None
    return int(metadata["page"])


def build_vector_store(chunks: list[Document]) -> None:
    """Embed chunks and persist them in a Chroma collection.

    Rebuilds the collection so repeated ingestion does not duplicate vectors.
    Each chunk is stored with its text, ``source`` / ``page`` metadata, and a
    unique ID.

    Args:
        chunks: Output of :func:`app.ingestion.chunk_documents`.
    """
    collection = _get_collection(reset=True)
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=texts,
        embeddings=_embed_texts(texts),
        metadatas=[_metadata_from_chunk(chunk) for chunk in chunks],
    )


def retrieve_documents(query: str, k: int = 5) -> list[RetrievedDocument]:
    """Embed ``query`` and return the top-k nearest chunks from Chroma.

    Args:
        query: Natural-language question or search text.
        k: Maximum number of chunks to return.

    Returns:
        Ranked results with ``text``, ``source``, ``page``, cosine
        ``distance``, and ``similarity`` (``1 - distance``).
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty.")

    collection = _get_collection()
    n_results = min(k, collection.count())
    if n_results == 0:
        return []

    raw = collection.query(
        query_embeddings=_embed_texts([query]),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[RetrievedDocument] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        distance_value = float(distance)
        results.append(
            RetrievedDocument(
                text=text or "",
                source=str((metadata or {}).get("source", "")),
                page=_page_from_metadata(metadata),
                distance=distance_value,
                similarity=1.0 - distance_value,
            )
        )
    return results
