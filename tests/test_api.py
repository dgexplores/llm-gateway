"""The HTTP boundary.

These are the tests that would catch a defect a user actually sees: a 500 where a 422
belongs, a traceback leaking to the client, or a bare 200 from a service that is
actually in demo mode answering differently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.api import create_app
from gateway.pipeline import MAX_TOP_K


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A service with no corpus, so it runs in demo mode.

    This is the configuration a stranger cloning the repo actually gets on a clean
    machine, so it is the one that must not be broken.
    """
    return TestClient(create_app(corpus_dir=tmp_path / "absent"))


def test_health_reports_ok(client: TestClient) -> None:
    res = client.get("/api/health")

    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_health_reports_mode_and_providers(client: TestClient) -> None:
    """A bare 200 hides the fact that demo mode and rag mode answer differently."""
    body = client.get("/api/health").json()

    assert "mode" in body
    assert "retriever" in body
    assert isinstance(body["providers"], list)


def test_health_reports_no_corpus_honestly(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert body["chunks"] == 0
    assert body["retriever"] == "null"


def test_ask_returns_an_answer(client: TestClient) -> None:
    res = client.post("/api/ask", json={"question": "how does caching work", "top_k": 3})

    assert res.status_code == 200
    body = res.json()
    assert body["text"]
    assert "usage" in body


def test_ask_in_demo_mode_is_labelled(tmp_path) -> None:
    """The user must never mistake extractive demo output for a model answer.

    A corpus is needed so retrieval succeeds and the generator is actually reached.
    Without one the pipeline takes the no-sources refusal path, which is a different
    case and correctly reports ``is_demo`` false: no model produced anything.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "deps.md").write_text(
        "# Dependency Injection\n\n"
        "Dependency injection is a technique where a route declares the dependencies "
        "it needs and the framework resolves them for it. " * 20,
        encoding="utf-8",
    )

    client = TestClient(create_app(corpus_dir=corpus))
    body = client.post("/api/ask", json={"question": "dependency injection"}).json()

    assert body["sources"], "expected the corpus to retrieve something"
    assert body["is_demo"] is True


def test_refusal_path_does_not_claim_a_model_produced_anything(tmp_path) -> None:
    """No sources means nothing was generated, so the demo label must stay false."""
    client = TestClient(create_app(corpus_dir=tmp_path / "none"))
    body = client.post("/api/ask", json={"question": "anything"}).json()

    assert body["sources"] == []
    assert body["is_demo"] is False


def test_ask_without_a_corpus_refuses_rather_than_invents(client: TestClient) -> None:
    body = client.post("/api/ask", json={"question": "anything"}).json()

    assert body["sources"] == []
    lowered = body["text"].lower()
    assert "no matching passages" in lowered or "nothing to ground" in lowered


def test_empty_question_is_rejected(client: TestClient) -> None:
    res = client.post("/api/ask", json={"question": "   "})

    assert res.status_code == 422


def test_missing_question_field_is_rejected(client: TestClient) -> None:
    assert client.post("/api/ask", json={}).status_code == 422


def test_oversized_question_is_rejected(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "x" * 5000}).status_code == 422


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1])
def test_out_of_range_top_k_is_rejected(client: TestClient, top_k: int) -> None:
    res = client.post("/api/ask", json={"question": "test", "top_k": top_k})

    assert res.status_code == 422


def test_top_k_defaults_when_absent(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "test"}).status_code == 200


def test_sources_endpoint_returns_a_list(client: TestClient) -> None:
    res = client.get("/api/sources", params={"q": "caching", "k": 3})

    assert res.status_code == 200
    assert isinstance(res.json()["results"], list)


def test_sources_requires_a_query(client: TestClient) -> None:
    assert client.get("/api/sources").status_code == 422


def test_index_page_is_served(client: TestClient) -> None:
    res = client.get("/")

    assert res.status_code == 200
    assert "LLM Gateway" in res.text


def test_static_assets_are_served(client: TestClient) -> None:
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_openapi_is_available(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_no_traceback_leaks_to_the_client(client: TestClient) -> None:
    """A stack trace in a response is an information disclosure, not a crash report."""
    res = client.post("/api/ask", json={"question": "test"})

    assert "Traceback" not in res.text
    assert "File \"" not in res.text


def test_with_corpus_the_health_reports_chunks(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# Caching\n\n" + ("Cache invalidation is hard. " * 80), encoding="utf-8"
    )

    body = TestClient(create_app(corpus_dir=corpus)).get("/api/health").json()

    assert body["chunks"] >= 1
    assert body["documents"] == 1
    assert body["retriever"] == "bm25"


def test_with_corpus_retrieval_finds_the_document(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cache.md").write_text(
        "# Caching\n\n" + "Cache invalidation is a famously difficult problem. " * 40,
        encoding="utf-8",
    )
    (corpus / "other.md").write_text(
        "# Routing\n\n" + "Routing maps request paths to handler functions. " * 40,
        encoding="utf-8",
    )

    body = TestClient(create_app(corpus_dir=corpus)).get(
        "/api/sources", params={"q": "cache invalidation", "k": 2}
    ).json()

    assert body["results"]
    assert body["results"][0]["doc_id"] == "cache.md"
