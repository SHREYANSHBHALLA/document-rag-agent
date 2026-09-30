"""Pydantic request and response models for the API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceReference(BaseModel):
    document_id: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
