"""Streamlit UI for the Document RAG Agent."""

import hashlib
import os
import sys
import tempfile

import streamlit as st

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ingestion import Document, load_document, chunk_documents
from app.vector_store import (
    build_vector_store,
    get_indexed_source_hash,
    get_indexed_chunk_count,
)
from app.rag import run_rag


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Document RAG Agent",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document RAG Agent")

st.write(
    "Upload PDF, DOCX, or TXT documents and ask questions "
    "based on their content."
)


# ============================================================
# Document Upload and Indexing
# ============================================================

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Index Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        all_documents = []
        content_hashes = {}
        skipped_sources = []
        failed_sources = []

        with st.spinner("Checking and processing documents..."):

            for uploaded_file in uploaded_files:

                source = uploaded_file.name
                file_bytes = uploaded_file.getvalue()

                # Calculate SHA-256 of the original file.
                content_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                # Skip files whose content has not changed.
                existing_hash = get_indexed_source_hash(source)

                if existing_hash == content_hash:
                    skipped_sources.append(source)
                    continue

                suffix = os.path.splitext(source)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                try:
                    documents = load_document(temp_path)

                    if not documents:
                        failed_sources.append(source)
                        continue

                    content_hashes[source] = content_hash

                    # Preserve extracted section and page metadata.
                    for document in documents:
                        all_documents.append(
                            Document(
                                text=document.text,
                                source=source,
                                page=document.page,
                                section_type=document.section_type,
                                section_title=document.section_title,
                            )
                        )

                except Exception as exc:
                    failed_sources.append(source)
                    print(f"Failed to process {source}: {exc}")

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Chunk and index only files that changed or are new.
            chunks = chunk_documents(all_documents)

            if chunks:
                build_vector_store(chunks, content_hashes)

        # ----------------------------------------------------
        # Indexing results
        # ----------------------------------------------------

        if chunks:
            st.success(
                f"Indexed {len(content_hashes)} changed/new "
                f"document(s), creating {len(chunks)} chunk(s)."
            )

        if skipped_sources:
            st.info(
                "Skipped unchanged documents: "
                + ", ".join(skipped_sources)
            )

        if failed_sources:
            st.warning(
                "Could not process: "
                + ", ".join(failed_sources)
            )

        if not chunks and not failed_sources:
            if skipped_sources:
                st.success(
                    "All uploaded documents are already indexed. "
                    "No new chunks were created."
                )
            else:
                st.warning("No chunks were generated.")


# ============================================================
# Question Answering
# ============================================================

st.header("2. Ask a Question")

query = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: How many days of annual leave "
        "do employees receive?"
    ),
)

if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")

    elif get_indexed_chunk_count() == 0:
        st.warning("Please upload and index documents first.")

    else:
        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):
                result = run_rag(query)

            st.subheader("Answer")
            st.write(result["answer"])

            sources = result.get("sources", [])

            if sources:
                st.subheader("Sources")

                for source in sources:
                    if isinstance(source, dict):
                        source_name = source.get(
                            "source", "Unknown"
                        )
                        page = source.get("page")

                        if page is not None:
                            st.write(
                                f"- {source_name}, Page {page}"
                            )
                        else:
                            st.write(f"- {source_name}")
                    else:
                        st.write(f"- {source}")

        except Exception as exc:
            st.error(
                "Unable to generate an answer. "
                "Check the terminal for the detailed error."
            )
            print(f"RAG error: {exc}")