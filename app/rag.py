"""LangGraph RAG workflow: retrieve relevant chunks, then generate with Ollama."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.vector_store import RetrievedDocument, retrieve_documents

# Tune during testing against real MiniLM + Chroma scores. This is not a
# universal cutoff; it only decides whether retrieved chunks are usable.
RELEVANCE_SIMILARITY_THRESHOLD = 0.35

NO_RELEVANT_CONTEXT_ANSWER = (
    "I couldn't find this information in the uploaded documents."
)
EMPTY_QUERY_ANSWER = "Please enter a question so I can search the uploaded documents."
OLLAMA_MODEL = "llama3.2:3b"
RETRIEVAL_K = 5

GROUNDING_SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question ONLY using the provided document context.

Do not use your general knowledge.
Do not make up or infer facts that are not supported by the context.
If the answer cannot be found in the context, say:
'I couldn't find this information in the uploaded documents.'
Do not use source numbers like [1], [2], or [3] in your answer.
Answer the question directly and concisely.
Keep the answer concise and clear."""


class SourceRef(TypedDict):
    source: str
    page: int | None


class RAGState(TypedDict):
    query: str
    documents: list[RetrievedDocument]
    answer: str
    sources: list[SourceRef]
    is_relevant: bool


def query_guardrail(state: RAGState) -> dict:
    """Reject empty or whitespace-only queries without calling an LLM."""
    query = (state.get("query") or "").strip()
    if not query:
        return {
            "query": query,
            "answer": EMPTY_QUERY_ANSWER,
            "documents": [],
            "sources": [],
            "is_relevant": False,
        }
    return {
        "query": query,
        "answer": "",
        "documents": [],
        "sources": [],
        "is_relevant": False,
    }


def retrieve(state: RAGState) -> dict:
    """Fetch the top-k nearest chunks for the guarded query."""
    documents = retrieve_documents(state["query"], k=RETRIEVAL_K)
    return {"documents": documents}


def relevance_check(state: RAGState) -> dict:
    """Keep chunks above the similarity threshold; otherwise refuse to answer."""
    relevant = [
        doc
        for doc in state.get("documents") or []
        if doc.similarity >= RELEVANCE_SIMILARITY_THRESHOLD
    ]
    if not relevant:
        return {
            "documents": [],
            "sources": [],
            "is_relevant": False,
            "answer": NO_RELEVANT_CONTEXT_ANSWER,
        }

    sources = _unique_sources(relevant)
    return {
        "documents": relevant,
        "sources": sources,
        "is_relevant": True,
        "answer": "",
    }


def generate_answer(state: RAGState) -> dict:
    """Call Ollama only when retrieved context is relevant."""
    if not state.get("is_relevant"):
        return {}

    documents = state.get("documents") or []
    generated = _call_ollama(state["query"], documents)
    source_lines = "\n".join(
        format_source_line(item["source"], item["page"])
        for item in state.get("sources") or []
    )
    answer = generated.strip()
    if source_lines:
        answer = f"{answer}\n\n{source_lines}"
    return {"answer": answer}


def format_source_line(source: str, page: int | None) -> str:
    """Format a source citation; omit page when it is None (DOCX/TXT)."""
    if page is None:
        return f"Source: {source}"
    return f"Source: {source}, Page {page}"


def _unique_sources(documents: list[RetrievedDocument]) -> list[SourceRef]:
    seen: set[tuple[str, int | None]] = set()
    sources: list[SourceRef] = []
    for document in documents:
        key = (document.source, document.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": document.source, "page": document.page})
    return sources


def _format_context(documents: list[RetrievedDocument]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        citation = format_source_line(document.source, document.page)
        blocks.append(f"[{index}] {citation}\n{document.text}")
    return "\n\n".join(blocks)


def _call_ollama(query: str, documents: list[RetrievedDocument]) -> str:
    """Send only the relevant chunks plus the question to the local Ollama model."""
    user_prompt = (
        f"Context:\n{_format_context(documents)}\n\n"
        f"Question: {query}"
    )
    response = _get_llm().invoke(
        [
            SystemMessage(content=GROUNDING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


@lru_cache(maxsize=1)
def _get_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


def _route_after_guardrail(state: RAGState) -> Literal["retrieve", "__end__"]:
    if state.get("answer"):
        return "__end__"
    return "retrieve"


@lru_cache(maxsize=1)
def build_rag_graph():
    """Compile START → guardrail → retrieve → relevance → generate → END."""
    graph = StateGraph(RAGState)
    graph.add_node("query_guardrail", query_guardrail)
    graph.add_node("retrieve", retrieve)
    graph.add_node("relevance_check", relevance_check)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "query_guardrail")
    graph.add_conditional_edges(
        "query_guardrail",
        _route_after_guardrail,
        {"retrieve": "retrieve", "__end__": END},
    )
    graph.add_edge("retrieve", "relevance_check")
    graph.add_edge("relevance_check", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def run_rag(query: str) -> RAGState:
    """Execute the RAG graph for a single user question."""
    return build_rag_graph().invoke(
        {
            "query": query,
            "documents": [],
            "answer": "",
            "sources": [],
            "is_relevant": False,
        }
    )
