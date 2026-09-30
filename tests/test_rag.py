"""Tests for the LangGraph RAG workflow."""

from unittest.mock import patch

from app.rag import (
    EMPTY_QUERY_ANSWER,
    NO_RELEVANT_CONTEXT_ANSWER,
    format_source_line,
    run_rag,
)
from app.vector_store import RetrievedDocument


def test_relevant_context_calls_ollama_and_includes_sources() -> None:
    hits = [
        RetrievedDocument(
            text="Vacation policy: employees receive 20 days of paid leave.",
            source="employee_handbook.pdf",
            page=12,
            distance=0.12,
            similarity=0.88,
        )
    ]
    with (
        patch("app.rag.retrieve_documents", return_value=hits),
        patch("app.rag._call_ollama", return_value="Employees receive 20 days of paid leave.") as ollama,
    ):
        state = run_rag("How many vacation days do employees get?")

    ollama.assert_called_once()
    _, documents = ollama.call_args.args
    assert documents == hits
    assert state["is_relevant"] is True
    assert "Employees receive 20 days of paid leave." in state["answer"]
    assert "Source: employee_handbook.pdf, Page 12" in state["answer"]


def test_irrelevant_query_does_not_call_ollama() -> None:
    misses = [
        RetrievedDocument(
            text="Vacation policy: employees receive 20 days of paid leave.",
            source="employee_handbook.pdf",
            page=12,
            distance=0.82,
            similarity=0.18,
        )
    ]
    with (
        patch("app.rag.retrieve_documents", return_value=misses),
        patch("app.rag._call_ollama") as ollama,
    ):
        state = run_rag("What is the capital of France?")

    ollama.assert_not_called()
    assert state["is_relevant"] is False
    assert state["answer"] == NO_RELEVANT_CONTEXT_ANSWER


def test_empty_query_does_not_call_ollama() -> None:
    with (
        patch("app.rag.retrieve_documents") as retrieve,
        patch("app.rag._call_ollama") as ollama,
    ):
        state = run_rag("   ")

    retrieve.assert_not_called()
    ollama.assert_not_called()
    assert state["answer"] == EMPTY_QUERY_ANSWER


def test_docx_source_omits_page() -> None:
    hits = [
        RetrievedDocument(
            text="Office hours are 9am to 5pm.",
            source="policy.docx",
            page=None,
            distance=0.10,
            similarity=0.90,
        )
    ]
    with (
        patch("app.rag.retrieve_documents", return_value=hits),
        patch("app.rag._call_ollama", return_value="Office hours are 9am to 5pm."),
    ):
        state = run_rag("What are the office hours?")

    assert "Source: policy.docx" in state["answer"]
    assert "Page" not in state["answer"]
    assert format_source_line("policy.docx", None) == "Source: policy.docx"
