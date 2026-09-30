# Document RAG Agent

A simple, production-oriented document Q&A system for a 2-hour machine test.

Users upload PDF, DOCX, or TXT files. The backend chunks and embeds them into Chroma, then a LangGraph RAG workflow retrieves relevant passages and asks a local Ollama model (`llama3.2:3b`) for a grounded answer with source references.

## Architecture

- **Frontend:** Streamlit (`frontend/streamlit_app.py`) — upload documents, ask questions, show answers and sources.
- **API:** FastAPI (`app/main.py`) — ingestion and query endpoints.
- **Ingestion:** `app/ingestion.py` — parse PDF/DOCX/TXT, chunk, embed, persist in Chroma.
- **RAG:** `app/rag.py` — LangGraph retrieve → generate flow.
- **Models:** `app/models.py` — request/response schemas.

Out of scope: Docker, Kubernetes, reranking, authentication, multi-agent workflows, and fine-tuning.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Ensure Ollama is running locally with `llama3.2:3b` pulled. Copy `.env.example` to `.env` if you need to override `CHROMA_PERSIST_DIR`.

3. Run the API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

4. Run the UI:

```bash
streamlit run frontend/streamlit_app.py
```

## Status

Scaffold only. Ingestion, retrieval, and generation are not implemented yet.
