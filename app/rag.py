"""Document RAG workflow using hybrid retrieval and Ollama."""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.hybrid_retriever import HybridResult, hybrid_retrieve


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

VECTOR_K = 20
BM25_K = 20
FINAL_K = 5
CONTEXT_K = 5

# Heuristic only; evaluate against the golden dataset.
RELEVANCE_SIMILARITY_THRESHOLD = 0.35

NO_RELEVANT_CONTEXT_ANSWER = (
    "I couldn't find this information in the uploaded documents."
)

EMPTY_QUERY_ANSWER = (
    "Please enter a question so I can search the uploaded documents."
)

GROUNDING_SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question ONLY using the provided document context.

Do not use your general knowledge.
Do not invent or infer unsupported facts.
If the answer cannot be found in the context, say:
'I couldn't find this information in the uploaded documents.'
Do not include source numbers like [1], [2], or [3] in your answer.
Answer the question directly and concisely.
Keep the answer clear and concise."""


# ============================================================
# State model
# ============================================================

class SourceRef(TypedDict):
    source: str
    page: int | None


class RAGState(TypedDict):
    query: str
    documents: list[HybridResult]
    answer: str
    sources: list[SourceRef]
    is_relevant: bool


# ============================================================
# Query validation
# ============================================================

def query_guardrail(query: str) -> str:
    """Reject empty questions before retrieval."""

    query = (query or "").strip()

    if not query:
        raise ValueError(EMPTY_QUERY_ANSWER)

    return query


# ============================================================
# Hybrid retrieval
# ============================================================

def retrieve(query: str) -> list[HybridResult]:
    """Retrieve using Vector Search + BM25 + weighted RRF."""

    return hybrid_retrieve(
        query,
        vector_k=VECTOR_K,
        bm25_k=BM25_K,
        final_k=FINAL_K,
    )


# ============================================================
# Relevance check
# ============================================================

def relevance_check(
    documents: list[HybridResult],
) -> list[HybridResult]:
    """Keep candidates with a positive lexical or vector signal."""

    return [
        doc
        for doc in documents
        if (
            doc.bm25_score is not None
            and doc.bm25_score > 0
        )
        or (
            doc.vector_similarity is not None
            and doc.vector_similarity
            >= RELEVANCE_SIMILARITY_THRESHOLD
        )
    ]


# ============================================================
# Source formatting
# ============================================================

def format_source_line(
    source: str,
    page: int | None,
) -> str:
    """Format the source citation."""

    if page is None:
        return f"Source: {source}"

    return f"Source: {source}, Page {page}"


def unique_sources(
    documents: list[HybridResult],
) -> list[SourceRef]:
    """Return unique source/page pairs."""

    seen: set[tuple[str, int | None]] = set()
    sources: list[SourceRef] = []

    for document in documents:
        key = (document.source, document.page)

        if key in seen:
            continue

        seen.add(key)
        sources.append({
            "source": document.source,
            "page": document.page,
        })

    return sources


def format_context(
    documents: list[HybridResult],
) -> str:
    """Format chunks and available structural metadata."""

    blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        details = [
            format_source_line(
                document.source,
                document.page,
            )
        ]

        if document.section_type:
            details.append(
                f"Section type: {document.section_type}"
            )

        if document.section_title:
            details.append(
                f"Section: {document.section_title}"
            )

        blocks.append(
            f"[{index}] {' | '.join(details)}\n{document.text}"
        )

    return "\n\n".join(blocks)


# ============================================================
# Ollama answer generation
# ============================================================

@lru_cache(maxsize=1)
def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        num_ctx=4096,
        num_predict=256,
    )


def call_ollama(
    query: str,
    documents: list[HybridResult],
) -> str:
    """Generate a grounded answer from retrieved chunks."""

    context_documents = documents[:CONTEXT_K]
    context = format_context(context_documents)

    print("\n--- CONTEXT PREVIEW ---")
    print(context[:2000])
    print(
        "Context chunk sizes:",
        [len(doc.text) for doc in context_documents],
    )
    print("Context characters:", len(context))

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    print("[RAG] Sending context to Ollama...")
    print("[RAG] Calling model:", OLLAMA_MODEL)
    print("[RAG] Context characters:", len(context))

    response = _get_llm().invoke(
        [
            SystemMessage(content=GROUNDING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ============================================================
# Main RAG workflow — no LangGraph
# ============================================================

def run_rag(query: str) -> RAGState:
    """Validate, retrieve, check relevance, and generate an answer."""

    print("[RAG] Validating query...")
    query = query_guardrail(query)

    try:
        # 1. Hybrid retrieval: Vector Top-20 + BM25 Top-20 -> RRF Top-10.
        print("[RAG] Running hybrid retrieval...")
        retrieved_documents = retrieve(query)

        print(
            f"[RAG] Retrieved {len(retrieved_documents)} chunks."
        )

        # Temporary diagnostic output for retrieval ranking.
        print("\n--- FINAL HYBRID RESULTS ---")
        for index, doc in enumerate(retrieved_documents, start=1):
            print(
                index,
                f"page={doc.page}",
                f"section={doc.section_title}",
                doc.text[:120].replace("\n", " "),
            )

        # 2. Basic relevance check.
        print("[RAG] Checking relevance...")
        relevant_documents = relevance_check(retrieved_documents)
        print(
            f"[RAG] Relevant chunks: {len(relevant_documents)}"
        )

        if not relevant_documents:
            return {
                "query": query,
                "documents": [],
                "answer": NO_RELEVANT_CONTEXT_ANSWER,
                "sources": [],
                "is_relevant": False,
            }

        # 3. Preserve citations for the returned chunks.
        sources = unique_sources(relevant_documents)

        # 4. Generate a grounded answer.
        print("[RAG] Calling Ollama...")
        answer = call_ollama(
            query,
            relevant_documents,
        )
        print("[RAG] Ollama response received.")

      
        # 5. Return the answer and sources separately.
        return {
            "query": query,
            "documents": relevant_documents,
            "answer": answer,
            "sources": sources,
            "is_relevant": True,
        }

    except Exception:
        print("[RAG] Workflow failed. Full traceback:")
        traceback.print_exc()
        raise
