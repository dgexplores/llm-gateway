"""Ingestion: documents in, chunks out.

Deliberately simple. Chunking strategy is the thing most likely to be wrong, and the
only way to find out is to measure retrieval against a labelled set, so the strategy
is configuration rather than architecture.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .retriever import Chunk

# Markdown heading boundaries, plus blank lines as a fallback. A paragraph-aware
# split is enough to evaluate retrieval honestly; a semantic chunker would confound
# the measurement with the thing being measured.
_SPLIT_PATTERN = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    title: str
    text: str
    path: str = ""


def load_documents(root: Path, pattern: str = "**/*.md") -> list[Document]:
    """Load markdown files, deriving a stable ``doc_id`` from the relative path.

    Deriving the id from the path rather than a counter or a hash of the contents
    keeps ids stable when unrelated files are added, which is what makes evaluation
    runs comparable over time.
    """
    docs: list[Document] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _first_heading(text) or path.stem
        docs.append(
            Document(
                doc_id=rel,
                title=title,
                text=text,
                path=str(path),
            )
        )
    return docs


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def chunk_document(
    doc: Document,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> list[Chunk]:
    """Split a document into overlapping paragraph-aligned chunks.

    Overlap exists so a fact spanning a boundary remains retrievable from at least
    one chunk. Chunks shorter than ``min_chars`` are merged into the previous one
    rather than emitted, because tiny fragments retrieve noise.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than target_chars")

    paragraphs = [p.strip() for p in _SPLIT_PATTERN.split(doc.text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer = ""
    index = 0

    def flush(text: str) -> None:
        nonlocal index
        text = text.strip()
        if len(text) < min_chars and chunks:
            previous = chunks[-1]
            merged = f"{previous.text}\n\n{text}"
            chunks[-1] = Chunk(
                chunk_id=previous.chunk_id,
                doc_id=previous.doc_id,
                text=merged,
                title=previous.title,
            )
            return
        if not text:
            return
        index += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{index:04d}",
                doc_id=doc.doc_id,
                text=text,
                title=doc.title,
                metadata={"source": doc.path},
            )
        )

    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= target_chars:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            continue

        flush(buffer)
        tail = buffer[-overlap_chars:] if overlap_chars else ""
        buffer = f"{tail}\n\n{paragraph}" if tail else paragraph

    flush(buffer)
    return chunks


def stable_chunk_hash(text: str) -> str:
    """Content hash, for detecting corpus drift between evaluation runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_corpus(documents: Iterable[Document], **kwargs: object) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, **kwargs))  # type: ignore[arg-type]
    return chunks
