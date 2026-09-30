# Document RAG Agent

A **4-hour AI machine test assignment** to build a document Q&A agent.

Users can upload one or more **PDF, DOCX, or TXT** files and ask questions based only on their document content.

## Tech Stack

* **UI:** Streamlit
* **Workflow:** LangGraph
* **Vector DB:** Chroma
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM:** Ollama + `llama3.2:3b`
* **Document Processing:** PyMuPDF, python-docx
* **Language:** Python

## Architecture

```text id="tie9e6"
Upload Documents
       ↓
Parse & Chunk
       ↓
MiniLM Embeddings
       ↓
Chroma Vector DB
       ↓
LangGraph
       ↓
Query Validation (Guardrail)
       ↓
Retrieve
       ↓
Relevance Check (Guardrail)
   ↙                    ↘
Not Relevant          Relevant
    ↓                    ↓
 Not Found          Generate Answer
                         ↓
                        END
```

## RAG Flow

1. Upload one or more documents.
2. Documents are parsed and split into chunks.
3. Chunks are converted into embeddings and stored in Chroma.
4. The user's question is converted into an embedding.
5. Chroma retrieves the most relevant chunks.
6. A relevance check verifies whether useful information was found.
7. Only relevant chunks are sent to the local LLM.
8. The answer and source references are displayed.

## Guardrails

The application uses lightweight guardrails:

* Empty questions are rejected.
* Retrieved content is checked for relevance.
* If relevant information is not found, the system returns:

```text id="6kbook"
I couldn't find this information in the uploaded documents.
```

In this case, the LLM is not called.

The LLM is also instructed to answer only from the retrieved document context and not use outside knowledge.

## Project Structure

```text id="c4ycp9"
document-rag-agent/
├── app/
│   ├── ingestion.py
│   ├── rag.py
│   ├── vector_store.py
│   └── models.py
├── frontend/
│   └── streamlit_app.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash id="5ya9s9"
git clone https://github.com/SHREYANSHBHALLA/document-rag-agent.git
cd document-rag-agent

python -m venv .venv
.venv\Scripts\activate

python -m pip install -r requirements.txt
```

Start Ollama:

```bash id="n7ycz7"
ollama run llama3.2:3b
```

Start the application:

```bash id="x7u0rr"
python -m streamlit run frontend/streamlit_app.py
```

Open:

```text id="bdfs33"
http://localhost:8501
```

## Testing

```bash id="e4sjcv"
python tests/test_end_to_end.py
```

Sample documents for testing are included in the tests/ folder. You can download them from the repository and upload them to the application to test document ingestion, retrieval, grounded answers, source references, and questions where the information is not present in the documents.
