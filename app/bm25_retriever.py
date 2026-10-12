"""BM25 lexical retrieval for document chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.ingestion import Document


@dataclass(frozen=True)
class BM25Result:
    """A document chunk returned by BM25."""

    chunk: Document
    score: float


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word tokens."""

    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """BM25 retriever built from document chunks."""

    def __init__(self, chunks: Sequence[Document]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk.")

        self.chunks = list(chunks)

        tokenized_chunks = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[BM25Result]:
        """Return the highest-ranked lexical matches."""

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            BM25Result(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]