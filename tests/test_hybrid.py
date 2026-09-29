"""Hybrid fusion and the ablation harness.

The fusion tests are written against the failure that matters: two retrievers that
disagree, where naive score blending produces a worse ranking than either member
alone. That case is why RRF uses rank instead of score.
"""

from __future__ import annotations

import pytest

from gateway.rag import BM25Retriever, Chunk, HybridRetriever, ScoredChunk
from gateway.rag.hybrid import explain_fusion


def chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, text=text, title=doc)


def scored(cid: str, doc: str, score: float, text: str = "x") -> ScoredChunk:
    return ScoredChunk(chunk=chunk(cid, doc, text), score=score)


class Scripted:
    """Returns a fixed ranking, so fusion can be reasoned about exactly."""

    def __init__(self, name: str, order: list[str], doc_of: dict[str, str]):
        self.name = name
        self._order = order
        self._doc_of = doc_of

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return [
            scored(cid, self._doc_of[cid], float(len(self._order) - i))
            for i, cid in enumerate(self._order[:k])
        ]


def test_needs_at_least_one_member() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        HybridRetriever([])


def test_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        HybridRetriever([Scripted("a", [], {})], k=0)


def test_combines_two_disagreeing_retrievers() -> None:
    """A chunk ranked poorly by both should not outrank one ranked well by either."""
    docs = {"a": "a.md", "b": "b.md", "c": "c.md"}
    a = Scripted("lexical", ["a", "b"], docs)
    b = Scripted("dense", ["b", "c"], docs)

    results = HybridRetriever([a, b]).retrieve("q", k=3)

    assert results[0].chunk_id == "b", "b is ranked by both, so it should win"


def test_a_chunk_only_one_retriever_found_still_appears() -> None:
    docs = {"a": "a.md", "c": "c.md"}
    a = Scripted("lexical", ["a"], docs)
    b = Scripted("dense", ["c"], docs)

    results = HybridRetriever([a, b]).retrieve("q", k=5)

    assert {r.chunk_id for r in results} == {"a", "c"}


def test_a_chunk_missed_by_every_retriever_does_not_appear() -> None:
    docs = {"a": "a.md"}
    a = Scripted("lexical", ["a"], docs)
    b = Scripted("dense", [], docs)

    results = HybridRetriever([a, b]).retrieve("q", k=5)

    assert [r.chunk_id for r in results] == ["a"]


def test_asks_each_member_for_more_than_k() -> None:
    """Asking only for k makes fusion a no-op past k, so the depth must exceed it."""
    seen: list[int] = []

    class Recording:
        name = "recording"

        def retrieve(self, query: str, k: int = 10):
            seen.append(k)
            return []

    HybridRetriever([Recording()]).retrieve("q", k=5)

    assert seen[0] > 5


def test_name_records_its_members() -> None:
    """A results table needs a label that identifies exactly what produced the row."""
    docs = {"a": "a.md"}
    hybrid = HybridRetriever(
        [Scripted("lexical", ["a"], docs), Scripted("dense", ["a"], docs)]
    )

    assert hybrid.name == "lexical+dense"


def test_weights_change_the_ranking() -> None:
    docs = {"a": "a.md", "b": "b.md"}
    a = Scripted("lexical", ["a", "b"], docs)
    b = Scripted("dense", ["b", "a"], docs)

    heavy_b = HybridRetriever([a, b], weights={"dense": 5.0}).retrieve("q", k=2)

    assert heavy_b[0].chunk_id == "b", "the heavier member should win the tie"


def test_results_are_ordered_and_bounded() -> None:
    docs = {f"c{i}": f"d{i}.md" for i in range(6)}
    a = Scripted("lexical", list(docs), docs)
    b = Scripted("dense", list(reversed(list(docs))), docs)

    results = HybridRetriever([a, b]).retrieve("q", k=4)

    assert len(results) == 4
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically() -> None:
    docs = {"z": "z.md", "a": "a.md"}
    a = Scripted("lexical", ["z", "a"], docs)
    b = Scripted("dense", ["a", "z"], docs)
    hybrid = HybridRetriever([a, b])

    assert [r.chunk_id for r in hybrid.retrieve("q", k=2)] == [
        r.chunk_id for r in hybrid.retrieve("q", k=2)
    ]


def test_no_results_when_nothing_matches() -> None:
    assert HybridRetriever([Scripted("lexical", [], {})]).retrieve("q") == []


def test_fusion_over_a_real_bm25_index() -> None:
    chunks = [
        chunk("c1", "status.md", "You set the status code with the status_code parameter."),
        chunk("c2", "deps.md", "Dependency injection declares dependencies for a route."),
        chunk("c3", "body.md", "The request body is parsed from JSON."),
    ]
    hybrid = HybridRetriever([BM25Retriever.build(chunks), BM25Retriever.build(chunks)])

    results = hybrid.retrieve("status_code parameter", k=2)

    assert results
    assert results[0].doc_id == "status.md"


def test_explain_fusion_shows_each_members_contribution() -> None:
    """A hybrid result is unauditable without this: a chunk can surface for reasons no
    reader can reconstruct."""
    docs = {"a": "a.md", "b": "b.md"}
    a = Scripted("lexical", ["a", "b"], docs)
    b = Scripted("dense", ["b", "a"], docs)
    hybrid = HybridRetriever([a, b])

    rows = explain_fusion("q", hybrid, "b")

    assert {r["retriever"] for r in rows} == {"lexical", "dense"}
    assert all(r["contribution"] > 0 for r in rows)
    assert all("raw_score" in r for r in rows)


def test_explain_fusion_reports_a_miss() -> None:
    docs = {"a": "a.md"}
    hybrid = HybridRetriever([Scripted("lexical", ["a"], docs), Scripted("dense", [], docs)])

    rows = {r["retriever"]: r for r in explain_fusion("q", hybrid, "a")}

    assert rows["dense"]["rank"] is None
    assert rows["dense"]["contribution"] == 0.0


# --- the ablation harness


def test_ablation_variants_are_named_and_described() -> None:
    """A row in a results table must say what produced it."""
    from gateway.rag import ablation as module

    found = module.variants()
    names = [v.name for v in found]

    assert "bm25" in names
    assert "none" in names, "a control is needed to show retrieval is doing anything"
    assert len(set(names)) == len(names)
    assert all(v.description for v in found)


def test_ablation_refuses_without_reviewed_goldens(tmp_path) -> None:
    """The harness must not produce a table from unverified questions."""
    from gateway.rag import ablation as module

    module.GOLDENS_PATH = tmp_path / "absent.jsonl"

    assert module.main() == 1


def test_ablation_runs_over_a_constructed_set(tmp_path) -> None:
    """End to end on a small corpus, so the harness is proven rather than assumed."""
    from gateway.rag import ablation as module

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, body in {
        "status.md": "# Status\n\n"
        + ("The status code is set with the status_code parameter. " * 40),
        "deps.md": "# Dependencies\n\n"
        + ("Dependency injection declares the dependencies a route needs. " * 40),
    }.items():
        (corpus / name).write_text(body, encoding="utf-8")

    goldens = corpus.parent / "g.jsonl"
    goldens.write_text(
        '{"id":"q1","question":"status code parameter",'
        '"answer":"a","relevant_doc_ids":["status.md"]}\n'
        '{"id":"q2","question":"dependency injection",'
        '"answer":"a","relevant_doc_ids":["deps.md"]}\n',
        encoding="utf-8",
    )

    module.CORPUS_DIR = corpus
    module.GOLDENS_PATH = goldens
    module.OUT_DIR = corpus.parent / "out"

    assert module.main() == 0

    written = module.OUT_DIR / "ablation.json"
    assert written.exists()

    import json

    payload = json.loads(written.read_text())
    names = [r["retriever"] for r in payload["results"]]
    assert "bm25" in names and "none" in names
    assert (module.OUT_DIR / "ablation.md").exists()

    # The control must score zero: with no retrieval there is nothing to find.
    by_name = {r["retriever"]: r for r in payload["results"]}
    assert by_name["none"]["ndcg@10"] == 0.0
    assert by_name["bm25"]["ndcg@10"] > 0.0, "the baseline must beat the control"
