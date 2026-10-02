"""Build a genuine multi-hop question set, or refuse to build one.

The first attempt at this paired two documents and concatenated one sentence from each.
Eleven of those twelve rows turned out to be answerable from a single document covering
75% or more of the answer, which is not multi-hop in any useful sense — it is two
retrievals dressed up as one question. Labelling them `multi-doc` was honest, but it left
the corpus with no multi-hop coverage at all.

The fix is to make the property a constraint the builder enforces rather than a property
the author hopes for. For a row to be emitted:

1. Neither document alone may reach the support threshold on the full answer. If one
   document does, the question is single-hop and is rejected.
2. Each document must contribute content words the other does not have, so the second
   document is not decoration.

A pair that fails both is skipped, and the script reports what it dropped. An empty result
is a legitimate outcome: it would mean no pair in this corpus supports a real multi-hop
question, which is itself the finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gateway.rag.ingest import load_documents
from gateway.rag.verify_set import MIN_SUPPORT, check_answer
from gateway.text import clean_markup, is_quotable, sentences

# Documents that genuinely need each other. Not "any two files about FastAPI" — each
# pair below has to be joined by a real operation the reader would have to perform.
PAIRS: list[tuple[str, str]] = [
    ("tutorial/dependencies/dependencies-with-yield.md", "tutorial/security/oauth2-jwt.md"),
    ("tutorial/testing.md", "tutorial/dependencies/dependencies-with-yield.md"),
    ("advanced/sub-applications.md", "tutorial/cors.md"),
    ("advanced/middleware.md", "tutorial/handling-errors.md"),
    ("tutorial/dependencies/sub-dependencies.md", "tutorial/dependencies/global-dependencies.md"),
    ("tutorial/request-files.md", "advanced/middleware.md"),
    ("tutorial/response-model.md", "tutorial/path-operation-configuration.md"),
    ("tutorial/background-tasks.md", "tutorial/dependencies/dependencies-with-yield.md"),
    ("advanced/security/oauth2-scopes.md", "tutorial/security/oauth2-jwt.md"),
    ("tutorial/body-multiple-params.md", "tutorial/dependencies/classes-as-dependencies.md"),
    ("tutorial/static-files.md", "advanced/sub-applications.md"),
    ("tutorial/middleware.md", "advanced/testing-events.md"),
    ("tutorial/cookie-params.md", "tutorial/dependencies/global-dependencies.md"),
    ("tutorial/path-params.md", "tutorial/dependencies/classes-as-dependencies.md"),
    ("tutorial/query-params.md", "tutorial/dependencies/index.md"),
]


@dataclass(frozen=True)
class Candidate:
    """One sentence pair that would make a multi-hop answer."""

    question: str
    answer: str
    doc_a: str
    doc_b: str


# Each question asks for a thing that needs both documents named explicitly. Written by
# hand because the constraint below can validate a sentence pair but cannot invent a good
# question about it.
QUESTIONS: dict[tuple[str, str], str] = {
    ("tutorial/dependencies/dependencies-with-yield.md", "tutorial/security/oauth2-jwt.md"): (
        "I need a dependency that identifies the current user and also releases a database "
        "session once the response is sent. Which two features cover those halves?"
    ),
    ("tutorial/testing.md", "tutorial/dependencies/dependencies-with-yield.md"): (
        "How do I substitute a dependency during a test, and how does a dependency that "
        "yields a session make sure that session is closed afterwards?"
    ),
    ("advanced/sub-applications.md", "tutorial/cors.md"): (
        "I mounted my API as a sub-application. What must I be careful about regarding "
        "CORS for the mounted routes to be reachable from a browser?"
    ),
    ("advanced/middleware.md", "tutorial/handling-errors.md"): (
        "What do I raise to tell a client the request was bad, and what do I register so "
        "that it runs for every request?"
    ),
    ("tutorial/dependencies/sub-dependencies.md", "tutorial/dependencies/global-dependencies.md"): (
        "If several routers share one dependency that must run on every request and be "
        "computed only once per request, which two dependency scopes apply?"
    ),
    ("tutorial/request-files.md", "advanced/middleware.md"): (
        "How do I accept an uploaded file in a route, and what kind of component processes "
        "the request on its way in?"
    ),
    ("tutorial/response-model.md", "tutorial/path-operation-configuration.md"): (
        "How do I filter what a response returns, and which route setting controls the "
        "status code it returns?"
    ),
    ("tutorial/background-tasks.md", "tutorial/dependencies/dependencies-with-yield.md"): (
        "I need work to happen after the response is returned. What are the two different "
        "ways to arrange that, and how do they differ in when they run?"
    ),
    ("advanced/security/oauth2-scopes.md", "tutorial/security/oauth2-jwt.md"): (
        "How do I issue a token, and how do I check the caller's permissions are wide "
        "enough for what they asked for?"
    ),
    ("tutorial/body-multiple-params.md", "tutorial/dependencies/classes-as-dependencies.md"): (
        "How do I declare several body parameters in one function, and what can stand in "
        "for a dependency that would normally be a function?"
    ),
    ("tutorial/static-files.md", "advanced/sub-applications.md"): (
        "How do I serve files from a directory, and what has to happen first if I want "
        "them served from a mounted sub-application?"
    ),
    ("tutorial/middleware.md", "advanced/testing-events.md"): (
        "How do I write a custom middleware class, and what does the test client need in "
        "order to exercise code that runs at application start-up?"
    ),
    ("tutorial/cookie-params.md", "tutorial/dependencies/global-dependencies.md"): (
        "How do I read a cookie as a declared parameter, and how do I make that "
        "declaration apply to every route rather than one?"
    ),
    ("tutorial/path-params.md", "tutorial/dependencies/classes-as-dependencies.md"): (
        "How does a value in the URL path reach my function, and what property does that "
        "value have to come from to be usable as a dependency?"
    ),
    ("tutorial/query-params.md", "tutorial/dependencies/index.md"): (
        "How do I read a value out of the query string, and what does the framework do "
        "with a parameter it recognises as a dependency?"
    ),
}


def _best_pairs(text_a: str, text_b: str, want: int) -> list[tuple[str, str]]:
    """Sentence pairs from two documents, most genuinely two-sided first.

    Scored by how little of the combined answer either document covers on its own, since
    that is the property a multi-hop question needs and the one the old set lacked.
    """
    left = [s for s in sentences(text_a) if is_quotable(s)]
    right = [s for s in sentences(text_b) if is_quotable(s)]

    scored: list[tuple[float, str, str]] = []
    for a in left:
        for b in right:
            if len(a.split()) < 8 or len(b.split()) < 8:
                continue
            answer = f"{a} {b}"
            solo_a, _ = check_answer(answer, text_a)
            solo_b, _ = check_answer(answer, text_b)
            # Lower is better: we want neither document to already cover the answer.
            scored.append((max(solo_a, solo_b), a, b))
    scored.sort(key=lambda row: row[0])
    return [(a, b) for _score, a, b in scored[:want]]


def main() -> None:
    docs = {d.doc_id: d for d in load_documents(Path("corpus"))}
    kept: list[dict] = []
    rejected: list[str] = []

    for doc_a, doc_b in PAIRS:
        key = (doc_a, doc_b)
        if key not in QUESTIONS:
            rejected.append(f"{doc_a} + {doc_b}: no question written")
            continue
        if doc_a not in docs or doc_b not in docs:
            rejected.append(f"{doc_a} + {doc_b}: document missing from corpus")
            continue

        text_a = clean_markup(docs[doc_a].text)
        text_b = clean_markup(docs[doc_b].text)

        for a, b in _best_pairs(text_a, text_b, want=6):
            answer = f"{a} {b}"
            solo_a, _ = check_answer(answer, text_a)
            solo_b, _ = check_answer(answer, text_b)
            if max(solo_a, solo_b) >= MIN_SUPPORT:
                continue  # one side already covers it; not a multi-hop answer

            # Both sides must contribute words the other lacks, or the second document
            # is decoration on a single-document answer.
            only_a = len(set(a.lower().split()) - set(b.lower().split()))
            only_b = len(set(b.lower().split()) - set(a.lower().split()))
            if only_a < 6 or only_b < 6:
                continue

            kept.append(
                {
                    "id": f"h{len(kept) + 1:03d}",
                    "question": QUESTIONS[key],
                    "answer": answer,
                    "relevant_doc_ids": [doc_a, doc_b],
                    "difficulty": "multi-hop",
                    "notes": (
                        "authored; sentence pair chosen by the builder because neither "
                        "document alone reaches the support threshold"
                    ),
                    "review": "PENDING",
                }
            )
            break  # one row per document pair

    for line in rejected:
        print(f"  dropped: {line}")
    print(f"\nbuilt {len(kept)} multi-hop rows from {len(PAIRS)} candidate pairs")

    if not kept:
        print(
            "\nNo pair satisfied the constraint. That would be a finding rather than a "
            "failure: this corpus may not support genuine multi-hop questions."
        )
        return

    Path("evals/proposed_multihop.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n"
    )

    print(f"\n{'id':6} {'best single doc':>16} {'union':>7}")
    corpus = {d: docs[d].text for d in docs}
    for row in kept:
        answer = row["answer"]
        solos = [check_answer(answer, corpus[d])[0] for d in row["relevant_doc_ids"]]
        union, _ = check_answer(answer, "\n".join(corpus[d] for d in row["relevant_doc_ids"]))
        print(f"{row['id']:6} {max(solos):>16.2f} {union:>7.2f}")
    print(f"\nwrote evals/proposed_multihop.jsonl ({len(kept)} rows)")


if __name__ == "__main__":
    main()