"""FastAPI entrypoint for document ingestion and RAG query APIs."""

from fastapi import FastAPI

app = FastAPI(title="Document RAG Agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
