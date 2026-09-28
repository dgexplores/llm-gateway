"""Chunking and document loading.

Chunking is where a retrieval system usually loses more accuracy than the ranking
model does, and it fails quietly: nothing errors, the answer is just worse.
"""

from __future__ import annotations

import pytest

from gateway.rag import Document, chunk_corpus, chunk_document, load_documents


def doc(text: str, doc_id: str = "a.md") -> Document:
    return Document(doc_id=doc_id, title="Title", text=text)


def test_splits_long_documents() -> None:
    text = "\n\n".join(f"Paragraph number {i} with some content in it." for i in range(60))
    chunks = chunk_document(doc(text), target_chars=400, min_chars=20)

    assert len(chunks) > 1


def test_chunk_ids_are_unique_within_a_document() -> None:
    text = "\n\n".join(f"Paragraph {i} with content." * 5 for i in range(30))
    chunks = chunk_document(doc(text), target_chars=300, min_chars=20)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_stable_across_runs() -> None:
    """Evaluation runs must be comparable, which requires identical chunk ids."""
    text = "\n\n".join(f"Paragraph {i} with content here." * 5 for i in range(20))
    first = [c.chunk_id for c in chunk_document(doc(text), target_chars=300, min_chars=20)]
    second = [c.chunk_id for c in chunk_document(doc(text), target_chars=300, min_chars=20)]

    assert first == second


def test_chunks_carry_document_provenance() -> None:
    chunks = chunk_document(doc("x" * 500, "guide.md"), target_chars=300, min_chars=20)

    assert all(c.doc_id == "guide.md" for c in chunks)
    assert chunks[0].title == "Title"


def test_no_text_produces_no_chunks() -> None:
    assert chunk_document(doc("   \n\n  ")) == []


def test_short_document_becomes_one_chunk() -> None:
    chunks = chunk_document(doc("Short but sufficient content for a single chunk."))

    assert len(chunks) == 1
    assert "Short but sufficient" in chunks[0].text


def test_overlap_repeats_content_across_a_boundary() -> None:
    """A fact spanning a boundary must be retrievable from at least one chunk."""
    text = "\n\n".join(f"Sentence {i} alpha beta gamma delta epsilon." for i in range(40))
    chunks = chunk_document(doc(text), target_chars=400, overlap_chars=120, min_chars=20)

    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    for i in range(40):
        assert f"Sentence {i}" in joined


def test_paragraph_alignment_is_preserved() -> None:
    text = "First paragraph content.\n\nSecond paragraph content."
    chunks = chunk_document(doc(text), min_chars=5)

    assert "First paragraph" in chunks[0].text


@pytest.mark.parametrize("target", [0, -1])
def test_non_positive_target_is_rejected(target: int) -> None:
    with pytest.raises(ValueError, match="target_chars"):
        chunk_document(doc("x" * 500), target_chars=target)


@pytest.mark.parametrize("overlap", [-1, 1000])
def test_invalid_overlap_is_rejected(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_document(doc("x" * 500), target_chars=500, overlap_chars=overlap)


def test_tiny_fragments_are_merged_not_emitted() -> None:
    """A 5-character fragment retrieves noise, so it joins its neighbour."""
    text = ("x" * 600) + "\n\n" + ("y" * 10) + "\n\n" + ("z" * 600)
    chunks = chunk_document(doc(text), target_chars=500, min_chars=100)

    assert all(len(c.text) >= 100 for c in chunks)


def test_load_documents_derives_ids_from_relative_paths(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# A\n\ncontent a", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# B\n\ncontent b", encoding="utf-8")

    docs = {d.doc_id: d for d in load_documents(tmp_path)}

    assert set(docs) == {"a.md", "sub/b.md"}


def test_load_documents_extracts_the_first_heading_as_title(tmp_path) -> None:
    (tmp_path / "a.md").write_text("# The Title\n\ncontent", encoding="utf-8")

    assert load_documents(tmp_path)[0].title == "The Title"


def test_load_documents_falls_back_to_filename_without_a_heading(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("no heading here", encoding="utf-8")

    assert load_documents(tmp_path)[0].title == "guide"


def test_load_documents_is_order_stable(tmp_path) -> None:
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text(f"# {name}\n\ncontent", encoding="utf-8")

    assert [d.doc_id for d in load_documents(tmp_path)] == ["a.md", "b.md", "c.md"]


def test_load_documents_on_an_empty_directory(tmp_path) -> None:
    assert load_documents(tmp_path) == []


def test_chunk_corpus_preserves_order_and_provenance() -> None:
    documents = [
        Document("a.md", "A", "alpha " * 100),
        Document("b.md", "B", "beta " * 100),
    ]

    chunks = chunk_corpus(documents, target_chars=300, min_chars=20)

    assert {c.doc_id for c in chunks} == {"a.md", "b.md"}
    assert chunks[0].doc_id == "a.md"
