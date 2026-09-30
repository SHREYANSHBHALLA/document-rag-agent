"""End-to-end RAG pipeline check (ingestion → Chroma → LangGraph → Ollama).

Run from the project root:

    python tests/test_end_to_end.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion import chunk_documents, load_document
from app.rag import run_rag
from app import rag as rag_module
from app.vector_store import build_vector_store

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_employee_handbook.txt"
RELEVANT_QUESTION = "How many days of annual leave do employees receive?"
UNRELATED_QUESTION = "What is the capital of France?"


def _wrap_ollama_counter() -> list[int]:
    """Count real Ollama invocations without changing graph structure."""
    calls = [0]
    original = rag_module._call_ollama

    def tracked(query: str, documents):
        calls[0] += 1
        return original(query, documents)

    rag_module._call_ollama = tracked  # type: ignore[method-assign]
    return calls


def _print_result(title: str, state: dict, ollama_calls_before: int, ollama_calls_after: int) -> None:
    print(f"\n=== {title} ===")
    print(f"Question: {state['query']}")
    print(f"Relevant: {state['is_relevant']}")
    print(f"Ollama called: {ollama_calls_after > ollama_calls_before}")
    print("Answer:")
    print(state["answer"])
    if state["sources"]:
        print("Sources:")
        for source in state["sources"]:
            page = source.get("page")
            if page is None:
                print(f"  - {source['source']}")
            else:
                print(f"  - {source['source']}, page {page}")


def main() -> None:
    print(f"Loading {SAMPLE_PATH.name}...")
    documents = load_document(str(SAMPLE_PATH))
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} document unit(s), {len(chunks)} chunk(s).")

    print("Building Chroma vector store...")
    build_vector_store(chunks)

    ollama_calls = _wrap_ollama_counter()

    before = ollama_calls[0]
    relevant_state = run_rag(RELEVANT_QUESTION)
    _print_result("Relevant question", relevant_state, before, ollama_calls[0])

    before = ollama_calls[0]
    unrelated_state = run_rag(UNRELATED_QUESTION)
    _print_result("Unrelated question", unrelated_state, before, ollama_calls[0])

    if unrelated_state["is_relevant"] or ollama_calls[0] > 1:
        print(
            "\nWarning: the unrelated question was not fully rejected. "
            "Ollama should not be called when relevance_check fails."
        )


if __name__ == "__main__":
    main()
