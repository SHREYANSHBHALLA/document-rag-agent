"""Load PDF, DOCX, and TXT files and split them into text chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fitz
from docx import Document as DocxDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# Prefer larger semantic boundaries first, then fall back to finer splits.
CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Document:
    """A unit of extracted document text with source metadata."""

    text: str
    source: str
    page: int | None


def load_document(file_path: str) -> list[Document]:
    """Detect the file type, extract text, and return page-aware documents.

    PDF pages keep a 1-based ``page`` number. DOCX and TXT files use
    ``page=None``. Empty extracted regions are skipped.

    Args:
        file_path: Path to a PDF, DOCX, or TXT file.

    Returns:
        Extracted documents with ``text``, ``source``, and ``page``.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension or path.name}'. "
            f"Supported types: {supported}."
        )

    source = path.name
    if extension == ".pdf":
        return _load_pdf(path, source)
    if extension == ".docx":
        return _load_docx(path, source)
    return _load_txt(path, source)


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping text chunks.

    Uses recursive, separator-based splitting: paragraphs, then newlines,
    then sentences, then spaces, then hard character cuts.

    Args:
        documents: Loaded documents from :func:`load_document`.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        Chunks that each include ``text``, ``source``, and ``page``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[Document] = []
    for document in documents:
        for text in _chunk_text(document.text, chunk_size, chunk_overlap):
            chunks.append(
                Document(text=text, source=document.source, page=document.page)
            )
    return chunks


def _load_pdf(path: Path, source: str) -> list[Document]:
    documents: list[Document] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append(Document(text=text, source=source, page=page_number))
    return documents


def _load_docx(path: Path, source: str) -> list[Document]:
    docx = DocxDocument(path)
    paragraphs = [paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip()]
    text = "\n".join(paragraphs)
    if not text:
        return []
    return [Document(text=text, source=source, page=None)]


def _load_txt(path: Path, source: str) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Document(text=text, source=source, page=None)]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces = _split_recursively(text.strip(), chunk_size, CHUNK_SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _split_recursively(
    text: str,
    chunk_size: int,
    separators: Sequence[str],
) -> list[str]:
    """Recursively split ``text`` until every piece is within ``chunk_size``."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator, *remaining = separators if separators else ("",)
    parts = list(text) if separator == "" else text.split(separator)

    results: list[str] = []
    buffer: list[str] = []
    buffer_length = 0

    for part in parts:
        extra = len(separator) if buffer else 0
        if buffer_length + extra + len(part) <= chunk_size:
            buffer.append(part)
            buffer_length += extra + len(part)
            continue

        if buffer:
            results.append(separator.join(buffer))
            buffer = []
            buffer_length = 0

        if len(part) > chunk_size:
            results.extend(_split_recursively(part, chunk_size, remaining))
        else:
            buffer = [part]
            buffer_length = len(part)

    if buffer:
        results.append(separator.join(buffer))

    return [piece.strip() for piece in results if piece.strip()]


def _merge_with_overlap(
    pieces: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Greedily merge small pieces, then prefix each next chunk with overlap."""
    if not pieces:
        return []

    merged: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        candidate = f"{current}\n{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        merged.append(current)
        if chunk_overlap:
            overlap_text = current[-chunk_overlap:].lstrip()
            current = f"{overlap_text}\n{piece}" if overlap_text else piece
            if len(current) > chunk_size:
                current = piece
        else:
            current = piece

    merged.append(current)
    return merged
