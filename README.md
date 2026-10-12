# Document RAG Agent

A document question-answering application using Retrieval-Augmented Generation (RAG), hybrid retrieval, and a locally running LLM.

## Features

- Upload and query PDF, DOCX, and TXT documents.
- Parse and chunk documents for retrieval.
- Generate embeddings using `all-MiniLM-L6-v2`.
- Hybrid retrieval using ChromaDB vector search and BM25.
- Weighted Reciprocal Rank Fusion (RRF) to combine retrieval results.
- Generate answers with source references using Ollama.
- Return a fallback response when relevant information is not found.
- Evaluate retrieval performance using Recall@K, Precision@K, and MRR.
- Experiment with Docling and different chunking strategies.

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Retrieval:** Vector Search, BM25, Weighted RRF
- **LLM:** Ollama (`llama3.2:3b`)
- **Document Processing:** PyMuPDF, python-docx, Docling experiments

## Architecture

```text
Upload Documents
       ↓
Parse & Chunk
       ↓
MiniLM Embeddings
       ↓
Chroma Vector DB

User Query
       ↓
Hybrid Retrieval
(Vector Search + BM25)
       ↓
Weighted RRF
       ↓
Relevant Context
       ↓
Ollama LLM
       ↓
Answer + Source References
```

## Setup

```bash
git clone https://github.com/SHREYANSHBHALLA/document-rag-agent.git
cd document-rag-agent

python -m venv .venv
```

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start Ollama:

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run frontend/streamlit_app.py
```

## Evaluation

The `docling-test/` directory contains experiments comparing document parsing, chunking strategies, and retrieval performance using a test dataset.

## Author

Shreyansh Bhalla
[GitHub](https://github.com/SHREYANSHBHALLA)
