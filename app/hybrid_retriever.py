"""Hybrid retrieval using Vector Search, BM25, and weighted RRF."""

from __future__ import annotations

from dataclasses import dataclass

from app.bm25_retriever import BM25Retriever
from app.ingestion import Document
from app.vector_store import get_all_documents, retrieve_documents


# ============================================================
# Configuration
# ============================================================

RRF_K = 10

VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0

DEFAULT_VECTOR_K = 20
DEFAULT_BM25_K = 20
DEFAULT_FINAL_K = 5


# ============================================================
# Result model
# ============================================================

@dataclass(frozen=True)
class HybridResult:
    """A retrieved chunk with fused and original retrieval scores."""

    text: str
    source: str
    page: int | None
    score: float
    section_type: str = "body"
    section_title: str | None = None
    vector_similarity: float | None = None
    bm25_score: float | None = None


# ============================================================
# Helpers
# ============================================================

def _document_key(
    text: str,
    source: str,
    page: int | None,
) -> tuple[str, str, int | None]:
    """Identify the same chunk across both retrieval methods."""

    return text, source, page


# ============================================================
# Hybrid retrieval
# ============================================================

def hybrid_retrieve(
    query: str,
    chunks: list[Document] | None = None,
    vector_k: int = DEFAULT_VECTOR_K,
    bm25_k: int = DEFAULT_BM25_K,
    final_k: int = DEFAULT_FINAL_K,
) -> list[HybridResult]:
    """Retrieve candidates with vectors and BM25, then fuse rankings.

    If chunks are explicitly supplied, BM25 searches those chunks.
    Otherwise, BM25 searches all documents stored in Chroma.

    No section_type metadata filter is applied.
    """

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty.")

    if min(vector_k, bm25_k, final_k) <= 0:
        raise ValueError(
            "All retrieval k values must be greater than 0."
        )

    # Use the complete indexed corpus for normal multi-document retrieval.
    corpus = (
        list(chunks)
        if chunks is not None
        else get_all_documents()
    )

    if not corpus:
        return []

    # --------------------------------------------------------
    # 1. Vector retrieval
    # --------------------------------------------------------

    vector_results = retrieve_documents(
        query,
        k=vector_k,
    )

    # --------------------------------------------------------
    # 2. BM25 lexical retrieval
    # Ignore results with no lexical match.
    # --------------------------------------------------------

    bm25_retriever = BM25Retriever(corpus)

    bm25_results = [
        result
        for result in bm25_retriever.retrieve(
            query,
            k=bm25_k,
        )
        if result.score > 0
    ]


    print("\n--- VECTOR TOP RESULTS ---")
    for i, doc in enumerate(vector_results[:10], 1):
        print(i, doc.page, doc.section_title, doc.text[:150])

    print("\n--- BM25 TOP RESULTS ---")
    for i, doc in enumerate(bm25_results[:10], 1):
        print(i, doc.chunk.page, doc.chunk.section_title, doc.chunk.text[:150])

    # --------------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # --------------------------------------------------------

    candidates: dict[
        tuple[str, str, int | None],
        HybridResult,
    ] = {}

    fused_scores: dict[
        tuple[str, str, int | None],
        float,
    ] = {}

    vector_similarities: dict[
        tuple[str, str, int | None],
        float,
    ] = {}

    bm25_scores: dict[
        tuple[str, str, int | None],
        float,
    ] = {}

    # Vector Search contribution.
    for rank, result in enumerate(vector_results, start=1):
        key = _document_key(
            result.text,
            result.source,
            result.page,
        )

        if key not in candidates:
            candidates[key] = HybridResult(
                text=result.text,
                source=result.source,
                page=result.page,
                score=0.0,
                section_type=result.section_type,
                section_title=result.section_title,
            )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            VECTOR_WEIGHT / (RRF_K + rank)
        )

        vector_similarities[key] = result.similarity

    # BM25 contribution.
    for rank, result in enumerate(bm25_results, start=1):
        chunk = result.chunk

        key = _document_key(
            chunk.text,
            chunk.source,
            chunk.page,
        )

        if key not in candidates:
            candidates[key] = HybridResult(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=0.0,
                section_type=chunk.section_type,
                section_title=chunk.section_title,
            )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            BM25_WEIGHT / (RRF_K + rank)
        )

        bm25_scores[key] = result.score

    # --------------------------------------------------------
    # 4. Rank unique candidates by fused score
    # --------------------------------------------------------

    ranked_keys = sorted(
        candidates,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    print("\n--- TOP 10 FUSED RESULTS ---")
    for rank, key in enumerate(ranked_keys[:10], 1):
        print(
        rank,
        "score:", round(fused_scores[key], 4),
        "page:", key[2],
        "text:", key[0][:120]
        )



    return [
        HybridResult(
            text=candidates[key].text,
            source=candidates[key].source,
            page=candidates[key].page,
            score=fused_scores[key],
            section_type=candidates[key].section_type,
            section_title=candidates[key].section_title,
            vector_similarity=vector_similarities.get(key),
            bm25_score=bm25_scores.get(key),
        )
        for key in ranked_keys[:final_k]
    ]