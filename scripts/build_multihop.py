"""Build the multi-hop and unanswerable question sets.

Answers are composed of sentences pulled from the corpus by exact substring match, and
the build fails loudly when a match is missing. That is the whole point: a
hand-written multi-hop set is where fabricated answers get in, and an unverifiable
answer is worse than no question at all.

A multi-hop question here means both cited documents are needed to answer it — either
because the answer draws on one sentence from each, or because one document only
defines the term the other then uses. Questions whose two documents happen to be
topically adjacent, but where one alone would answer it, are not multi-hop and do not
belong in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from gateway.rag.ingest import load_documents
from gateway.text import clean_markup

# (question, [(doc_id, substring that must exist), ...])
PAIRS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "In FastAPI, what is a dependency, and how can one be applied to the whole "
        "application instead of a single route?",
        [
            (
                "tutorial/dependencies/index.md",
                "will take care of doing whatever is needed to provide your code with",
            ),
            (
                "tutorial/dependencies/global-dependencies.md",
                "you might want to add dependencies to the whole application",
            ),
        ],
    ),
    (
        "When one dependency depends on another, what happens to the shared result "
        "inside a single request?",
        [
            (
                "tutorial/dependencies/sub-dependencies.md",
                'it will save the returned value in a "cache" and pass it to all the '
                '"dependants"',
            ),
            (
                "tutorial/dependencies/index.md",
                "will take care of doing whatever is needed to provide your code with",
            ),
        ],
    ),
    (
        "What makes something usable as a FastAPI dependency, and how does the framework "
        "supply it to your code?",
        [
            (
                "tutorial/dependencies/classes-as-dependencies.md",
                'a dependency should be a "callable"',
            ),
            (
                "tutorial/dependencies/index.md",
                "will take care of doing whatever is needed to provide your code with",
            ),
        ],
    ),
    (
        "How can a dependency perform work after a request completes, and how does that "
        "differ from a background task?",
        [
            (
                "tutorial/dependencies/dependencies-with-yield.md",
                "dependencies that do some extra steps after finishing",
            ),
            (
                "tutorial/background-tasks.md",
                "background tasks to be run after returning a response",
            ),
        ],
    ),
    (
        "FastAPI communicates over HTTP and can also hold a persistent connection. What "
        "does the documentation say about each?",
        [
            (
                "tutorial/background-tasks.md",
                "background tasks to be run after returning a response",
            ),
            (
                "advanced/websockets.md",
                "You can use WebSockets with FastAPI",
            ),
        ],
    ),
    (
        "How does a FastAPI application report an error to a client, and where does "
        "middleware fit relative to that?",
        [
            (
                "tutorial/handling-errors.md",
                "To return HTTP responses with errors to the client you use HTTPException",
            ),
            (
                "advanced/middleware.md",
                "implements the ASGI specification, you can use any ASGI middleware",
            ),
        ],
    ),
    (
        "What problem does CORS address, and how is the middleware added to an "
        "application?",
        [
            (
                "tutorial/cors.md",
                "the combination of protocol (http, https), domain (myapp.com, localhost",
            ),
            (
                "advanced/middleware.md",
                "you also read how to handle CORS with the CORSMiddleware",
            ),
        ],
    ),
    (
        "How does FastAPI secure an application with tokens, and what are OAuth2 scopes "
        "used for on top of that?",
        [
            (
                "tutorial/security/oauth2-jwt.md",
                "JWT means",
            ),
            (
                "advanced/security/oauth2-scopes.md",
                "manage authentication and authorization with the same OAuth2 with scopes",
            ),
        ],
    ),
    (
        "A dependency is described as a callable, but there is a restriction on how many "
        "times a dependency may yield. What is it?",
        [
            (
                "tutorial/dependencies/classes-as-dependencies.md",
                'a dependency should be a "callable"',
            ),
            (
                "tutorial/dependencies/dependencies-with-yield.md",
                "use yield one single time per dependency",
            ),
        ],
    ),
    (
        "Dependencies can share a cached value within a request, and a dependency can "
        "open a database session. How do these two features combine?",
        [
            (
                "tutorial/dependencies/sub-dependencies.md",
                'it will save the returned value in a "cache" and pass it to all the '
                '"dependants"',
            ),
            (
                "tutorial/dependencies/dependencies-with-yield.md",
                "create a database session and close it after finishing",
            ),
        ],
    ),
    (
        "How can an exception raised deep inside a helper be turned into an HTTP error "
        "response, and what does the ASGI specification allow in between?",
        [
            (
                "tutorial/handling-errors.md",
                "Because it's a Python exception, you don't return it, you raise it",
            ),
            (
                "advanced/middleware.md",
                "implements the ASGI specification, you can use any ASGI middleware",
            ),
        ],
    ),
    (
        "What is shared logic in FastAPI good for, and how does an application-wide "
        "dependency relate to a decorator-scoped one?",
        [
            (
                "tutorial/dependencies/index.md",
                "Have shared logic",
            ),
            (
                "tutorial/dependencies/global-dependencies.md",
                "Similar to the way you can add dependencies to the path operation "
                "decorators",
            ),
        ],
    ),
]

# Questions the corpus cannot answer at all. The retriever is not measured on these and
# they are not part of the retrieval gate: an unanswerable question has no relevant
# document, so nDCG is undefined. They exist to measure whether the gateway withholds
# an answer instead of returning a confidently grounded one that does not respond to
# the question — which is the failure this project exists to avoid.
UNANSWERABLE: list[str] = [
    "How do I configure a Kubernetes ingress controller for blue-green deployments?",
    "What is the capital of Australia?",
    "How do I bake sourdough bread at home?",
    "Explain quantum entanglement to a five year old.",
    "What is the difference between TCP and UDP?",
    "Who won the 1998 FIFA World Cup?",
    "How do I calculate the standard deviation of a list in Rust?",
    "What is the current price of a Raspberry Pi?",
]


def main() -> None:
    docs = {d.doc_id: d for d in load_documents(Path("corpus"))}

    rows: list[dict] = []
    for index, (question, wanted) in enumerate(PAIRS, start=1):
        parts: list[str] = []
        for doc_id, needle in wanted:
            if doc_id not in docs:
                raise SystemExit(f"FATAL: no such document {doc_id!r}")
            # Cleaned first, because that is the text the retriever actually scores
            # and the text the answer will be compared against. Searching the raw
            # markdown misses any sentence containing a <dfn> or an emphasis marker.
            text = clean_markup(docs[doc_id].text).replace("\n", " ")
            position = text.find(needle)
            if position < 0:
                # Loud, because a silent miss would let an invented answer through.
                raise SystemExit(f"FATAL: {needle!r} not found in {doc_id}")
            # rfind returns -1 when the needle is in the document's first sentence;
            # +2 on that would start the excerpt at index 1 and produce "astAPI".
            boundary = text.rfind(". ", 0, position)
            start = 0 if boundary < 0 else boundary + 2
            end = text.find(". ", position)
            sentence = text[start : end + 1] if end > 0 else text[start : start + 200]
            parts.append(sentence.strip())
        rows.append(
            {
                "id": f"m{index:03d}",
                "question": question,
                "answer": " ".join(parts),
                "relevant_doc_ids": [d for d, _ in wanted],
                # "multi-doc", not "multi-hop". These questions need two documents
                # retrieved, which is a real capability, but they do not require
                # multi-hop *reasoning*: measured against the corpus, 11 of the 12 have
                # one document that already covers 75% or more of the answer. Calling
                # them multi-hop would claim more than the set tests. See
                # scripts/report_multidoc.py, which recomputes that number.
                "difficulty": "multi-doc",
                "notes": (
                    f"authored; answer composed of real sentences from "
                    f"{len(wanted)} documents"
                ),
                "review": "PENDING",
            }
        )

    Path("evals/proposed_multihop.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    Path("evals/unanswerable.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"u{i:03d}",
                    "question": q,
                    "relevant_doc_ids": [],
                    "difficulty": "no-answer",
                    "notes": "deliberately outside the corpus",
                    "review": "PENDING",
                }
            )
            for i, q in enumerate(UNANSWERABLE, start=1)
        )
        + "\n"
    )
    print(f"wrote {len(rows)} multi-hop rows and {len(UNANSWERABLE)} unanswerable rows")


if __name__ == "__main__":
    main()
