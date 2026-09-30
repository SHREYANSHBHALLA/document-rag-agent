import os
import sys
import tempfile

import streamlit as st

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ingestion import Document, load_document, chunk_documents
from app.vector_store import build_vector_store
from app.rag import run_rag


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


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

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

        with st.spinner("Processing documents..."):

            for uploaded_file in uploaded_files:

                suffix = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                try:
                    documents = load_document(temp_path)

                    # Re-create documents using the actual uploaded filename.
                    # Document is a frozen dataclass, so we cannot modify
                    # document.source directly.
                    for document in documents:
                        all_documents.append(
                            Document(
                                text=document.text,
                                source=uploaded_file.name,
                                page=document.page,
                            )
                        )

                finally:
                    os.remove(temp_path)

            # Chunk all uploaded documents
            chunks = chunk_documents(all_documents)

            # Store chunks in Chroma
            build_vector_store(chunks)

        st.session_state["documents_indexed"] = True
        st.session_state["document_count"] = len(uploaded_files)
        st.session_state["chunk_count"] = len(chunks)

        st.success(
            f"Successfully indexed {len(uploaded_files)} document(s) "
            f"and {len(chunks)} chunk(s)."
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

st.header("2. Ask a Question")

query = st.text_input(
    "Enter your question",
    placeholder="Example: How many days of annual leave do employees receive?"
)


if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")

    elif not st.session_state.get("documents_indexed", False):
        st.warning("Please upload and index documents first.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            result = run_rag(query)

        st.subheader("Answer")
        st.write(result["answer"])

        sources = result.get("sources", [])

        if sources:
            st.subheader("Sources")

            for source in sources:

                if isinstance(source, dict):

                    source_name = source.get(
                        "source",
                        "Unknown"
                    )

                    page = source.get("page")

                    if page:
                        st.write(
                            f"- {source_name}, Page {page}"
                        )
                    else:
                        st.write(
                            f"- {source_name}"
                        )

                else:
                    st.write(f"- {source}")

