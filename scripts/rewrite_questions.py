"""Rewrite the single-hop questions as natural language, then re-measure.

The evaluation set had a problem nobody looked at closely enough: all 51 single-hop
questions were one template with the tail swapped.

    In the FastAPI documentation, what is described regarding <topic>?

Fifty-one copies of one stem. "What is described regarding burger conclusion?" is not a
sentence anyone types. Every query therefore carried the same eight content words, which
BM25 scored identically across the whole set — so the number reported as the difficulty
baseline was measuring how well the retriever handles a template, not how well it handles
questions. The 0.891 single-hop nDCG in `docs/RESULTS.md` was a measurement of the
evaluation set as much as of the retriever, and comparing it against the multi-hop rows
that *were* hand-written was not a comparison of two difficulties.

The fix is not to keep the template and add variety around it. It is to write questions a
person would actually type, which means mixed register and mixed intent:

- why questions        ("If I use async def but my database call blocks, what happens?")
- how-to questions     ("How do I set up server-sent events?")
- difference questions ("What's the difference between FastAPI's HTTPException and Starlette's?")
- counterintuitive ones ("If I add several middleware classes, which runs first?")

Answers and their documents are untouched. That matters: the mechanical verifier checks
each answer against the document it cites, so rewording a question cannot quietly weaken
the set. What *does* change is retrieval, and that is the point — the new numbers are the
honest ones, and the old ones are kept in the commit message so the difference is visible
rather than quietly overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

STEM = "In the FastAPI documentation, what is described regarding "

# id -> natural question. Each targets the same document and the same answer as the
# template it replaces.
QUESTIONS: dict[str, str] = {
    # Counterintuitive or easy-to-get-wrong: phrased so a lazy answer is visibly wrong.
    "q022": "If I add several middleware classes, which one runs first?",
    "q048": "If I declare the same path twice, which route wins?",
    "q043": "Can I send back data that matches the default values exactly?",
    "q018": "How should I choose the value of the sub claim in a JWT?",
    "q006": "Why do TLS certificates have to be issued for a domain name rather than an IP address?",
    # Why questions: ask for the reason, which is what the answer sentence gives.
    "q002": "Why build FastAPI at all instead of using NestJS or Angular?",
    "q004": "If I use async def but my database call is blocking, what actually happens to my server?",
    "q007": "I remember Starlette shipping a GraphQLApp class. What happened to it?",
    "q013": "Why is HeroUpdate a separate model instead of reusing Hero?",
    "q015": "If my application crashes, who should be responsible for restarting it?",
    "q047": "Why do I need a separate HeroPublic model instead of returning Hero directly?",
    "q050": "What is the point of the FastAPI people page?",
    "q005": "In the burger example, what does the cashier telling the cook actually illustrate?",
    # Difference questions: force a contrast the answer sentence actually draws.
    "q023": "Does FastAPI have its own Jinja2 templating, or is it Starlette's?",
    "q049": "What's the difference between FastAPI's HTTPException and Starlette's?",
    "q011": "When testing async code, what is different about AsyncClient?",
    "q044": "Does Header do anything beyond what Path, Query and Cookie do?",
    "q042": "I need to change the Swagger UI JavaScript. How do I override just that part?",
    # Plain how-to questions.
    "q017": "How do I stream server-sent events from FastAPI?",
    "q045": "How do I get FastAPI to read settings from a .env file?",
    "q033": "How do I get a new language added to the FastAPI documentation?",
    "q031": "How long does a JWT last before the user has to sign in again?",
    "q020": "How are datetime values represented in request and response bodies?",
    "q026": "How does the browser know an HTTPS response really came from my server?",
    "q021": "Can a path parameter itself contain a slash, and how do I declare that?",
    "q003": "How do I describe the schema for apispec — do I write it in the docstring?",
    "q025": "What do I need to install and import to mount an existing WSGI app?",
    "q051": "How do I document the callback that my API will call?",
    "q016": "In the Docker deployment example, how does a request get from the load balancer to the container?",
    "q035": "Which cloud services handle replication automatically?",
    "q036": "Am I at risk of CSRF if I am not sending a strict Content-Type?",
    "q030": "How do I see which scopes a path operation actually required?",
    "q040": "The author based design decisions partly on a Python developer survey. What did it show?",
    "q041": "How did the author work out what other frameworks did well?",
    "q032": "My Pydantic v1 models will not work with bump-pydantic. What is the alternative?",
    # Remaining specific questions.
    "q001": "GZipMiddleware has a minimum_size option. What changes if I set it low versus high?",
    "q008": "What does the Path Operation Explorer show me in the editor support docs?",
    "q009": "How do type annotations on a response model help my editor and mypy?",
    "q010": "How do I declare a response that is one of two types without using the new | syntax?",
    "q012": "In the streaming docs, how is a fake file object made so it still gets closed?",
    "q014": "If I use a class as a dependency, do I get the same parameter conversion as a path operation?",
    "q019": "In the Docker docs they mention distributed container managers for replication. How many processes should I run?",
    "q024": "What actually counts as a context manager in Python?",
    "q027": "How do I write my own context manager class?",
    "q028": "What kinds of object can I hand to StreamingResponse?",
    "q029": "What are EventSourceResponse and ServerSentEvent for?",
    "q034": "If I nest dataclasses inside other dataclasses, what does FastAPI use to build the response?",
    "q037": "If I do not set a status code on a RedirectResponse, what status does the client get?",
    "q038": "My proxy strips the path prefix before forwarding. Anything I need to remember?",
    "q039": "Should I use strawberry-graphql with FastAPI, or something else?",
    "q046": "What type is app.webhooks — can I use my usual router patterns there?",
}


def main() -> None:
    path = Path("evals/goldens.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    single = [r for r in rows if r["difficulty"] == "single-hop"]
    unknown = sorted({r["id"] for r in single} - set(QUESTIONS))
    if unknown:
        raise SystemExit(f"no rewrite for {unknown}")
    if len(QUESTIONS) != len(single):
        raise SystemExit(f"{len(QUESTIONS)} rewrites for {len(single)} single-hop rows")

    for row in rows:
        if row["difficulty"] != "single-hop":
            continue
        before = row["question"]
        if not before.startswith(STEM):
            raise SystemExit(f"{row['id']} no longer matches the template: {before!r}")
        row["question"] = QUESTIONS[row["id"]]
        row["notes"] = (
            f"authored; question reworded from a single templated stem, answer and document "
            f"unchanged (was: {before!r})"
        )

    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    stems = {r["question"][:38] for r in rows if r["difficulty"] == "single-hop"}
    print(f"rewrote {len(single)} single-hop questions")
    print(f"distinct 38-char prefixes now: {len(stems)} (was 1)")
    print("\nsample:")
    for row in [r for r in rows if r["difficulty"] == "single-hop"][:6]:
        print(f"  {row['question']}")


if __name__ == "__main__":
    main()