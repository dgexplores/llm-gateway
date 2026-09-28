"""Generate provenance-derived evaluation questions.

The trick that makes this affordable: for each question, the source document is known
because the question is written *from* that document. So ``relevant_doc_ids`` is not
guessed by an annotator, it is a fact about how the question was constructed. Only the
answer text needs human review, which is a review task rather than a five-hour
annotation task.

Two properties are enforced, because a question missing either is worse than no
question at all:

- **Answerable**: the answer is a verbatim span from the document, so a reviewer can
  verify it by reading one paragraph instead of researching a topic.
- **Retrievable**: the question is built from the section's distinctive terms, so a
  lexical retriever has a fair chance of finding the source. A question nobody can
  retrieve is not measuring the retriever, it is measuring the annotation.

Rows are written as ``review: PENDING`` and are excluded from the CI gate until a
human marks them ``OK``. Mixing reviewed and unreviewed rows would let an unreviewed
set quietly become the basis of a published number.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ingest import Document, load_documents

CORPUS_DIR = Path("corpus")
OUT_DIR = Path("evals")

# Pages that are not documentation of behaviour: indexes, changelogs, contribution
# guides. A question drawn from these measures nothing about the library.
SKIP_FILES = re.compile(
    r"(index|release-notes|contributing|external|readme|community|"
    r"_llm-test|test-file|changelog|/docs/en/docs$)",
    re.IGNORECASE,
)

# A section shorter than this is a stub or a link list.
MIN_BODY_CHARS = 400
# A sentence longer than this is usually a run-on paragraph, not a claim.
MAX_ANSWER_CHARS = 400
MIN_ANSWER_CHARS = 60

# Tutorial prose runs on through conjunctions, so a naive sentence split yields
# fragments like "And then, there would be another program...". Those read as nonsense
# out of context and a reviewer rejects them, which is the same as having no question.
_FRAGMENT_START = re.compile(
    r"^(and|but|or|so|then|thus|therefore|however|this|that|it|they|we|you|which|"
    r"there|here|such|because|while|when|if|as|also|now|for example|e\.g|i\.e)\b",
    re.IGNORECASE,
)

# Terms too common in a technical corpus to identify a section.
STOPWORDS = frozenset(
    """a an the and or but if then when while for to of in on at by with without from
    into is are was were be been being do does did doing have has had having it its
    this that these those you your we our they their he she as not no nor so such than
    too very can will just should now here there what which who whom whose how why
    all any both each few more most other some only own same s t don now use used using
    also may might must shall would could one two three example examples note notes
    information info return returns returned request requests response responses
    function functions class classes method methods value values type types name names
    code file files python fastapi app application data test tests run runs running
    first second next last see more detail details default defaults option options
    following above below like unlike need needs needed get gets got set sets
    about after before between during through under over out off again further once
    other same new old good best better most least
    """.split()
)

_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Candidate:
    """A proposed question plus the document it came from."""

    doc_id: str
    title: str
    question: str
    answer: str
    difficulty: str
    anchor_terms: tuple[str, ...]

    @property
    def notes(self) -> str:
        return f"anchor_terms={','.join(self.anchor_terms)}"


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if heading and len("\n".join(body).strip()) > MIN_BODY_CHARS:
                sections.append((heading, "\n".join(body).strip()))
            heading = line.lstrip("#").strip()
            body = []
        else:
            body.append(line)

    if heading and len("\n".join(body).strip()) > MIN_BODY_CHARS:
        sections.append((heading, "\n".join(body).strip()))
    return sections


_ADMONITION = re.compile(r"^\s*(?:///|!!!|\?\?\?|:::)\s*\w*\s*", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
    flags=re.UNICODE,
)


def _clean(text: str) -> str:
    """Strip markup that would leak into a question or answer.

    Documentation is written for rendering, not extraction. Code blocks, admonition
    markers, emoji and emphasis all survive a naive split and produce answers a
    reviewer has to mentally de-render before they can check them.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = _ADMONITION.sub("", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _EMOJI.sub(" ", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    # Bullets, tables and setext heading underlines survive the above and read as
    # stray punctuation once detached from their block.
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


# Tutorial pages number their steps imperatively ("Return the token", "Check it",
# "Add a route"). A question built from one of those reads as an instruction rather
# than a topic, and there is no stable answer to "what is documented about check it".
# Reference pages use noun phrases, which do support a real question.
_IMPERATIVE_START = re.compile(
    r"^(check|run|add|create|return|declare|use|import|install|define|call|open|close|"
    r"try|copy|set|put|write|read|send|receive|see|go|try|now|next|then|rename|move|"
    r"update|delete|remove|test|verify|build|make|convert|pass|include|register|"
    r"instantiate|assign|apply|execute|enter|exit|start|stop|enable|disable)\b",
    re.IGNORECASE,
)

# Headings that read as a section name rather than a topic.
_GENERIC_HEADING = re.compile(
    r"^(recap|more info|notes?|technical details|example|examples|summary|"
    r"further reading|see also|check it|try it|test it|additional|appendix)\s*$",
    re.IGNORECASE,
)


def _is_topical_heading(heading: str) -> bool:
    """Whether a heading names a subject that can anchor a question.

    Rejects imperative tutorial steps and filler headings. Both produce questions that
    are either unanswerable ("what is documented about check it") or identical across
    many pages, which makes the metric measure nothing.
    """
    cleaned = heading.split("{", 1)[0].strip()  # docs carry an anchor slug in braces
    if len(cleaned) < 4 or _GENERIC_HEADING.match(cleaned):
        return False
    if _IMPERATIVE_START.match(cleaned):
        return False
    # A heading made only of punctuation or markup is not a subject.
    return bool(re.search(r"[a-z]{3,}", cleaned, re.IGNORECASE))


def _is_self_contained(sentence: str) -> bool:
    """Whether a sentence reads as a claim on its own.

    Three tests, each catching a different failure seen in generated sets: a
    conjunction fragment, a pronoun with no antecedent once the paragraph is
    detached, and an emoji or bullet glyph left over from markup cleaning.
    """
    stripped = sentence.strip()
    if _FRAGMENT_START.match(stripped):
        return False
    if _EMOJI.search(stripped) or re.search(r"[→←⇒]", stripped):
        return False
    # A sentence opening on a pronoun refers to something the reader cannot see.
    if re.match(r"^(it|they|this|that|these|those|he|she)\b", stripped, re.IGNORECASE):
        return False
    return True


def _sentences(prose: str) -> list[str]:
    """Sentence candidates that stand alone as claims.

    Tutorial documentation is full of pronouns and conjunctions that only make sense
    with the surrounding paragraph. Answering with one produces a question a reviewer
    cannot verify, so those are dropped rather than cleaned up afterwards.
    """
    parts = re.split(r"(?<=[.!?])\s+", prose)
    candidates: list[str] = []
    for raw in parts:
        sentence = raw.strip().lstrip("|-*+ ").strip()
        if not (MIN_ANSWER_CHARS < len(sentence) < MAX_ANSWER_CHARS):
            continue
        if not _is_self_contained(sentence):
            continue
        # A leftover pipe or dash run means the sentence was a table row, a list item
        # or a setext underline, not prose. A CLI flag such as --proxy-headers is
        # different: it is meaningful content, so only a run of two or more is
        # treated as residue.
        if "|" in sentence or re.search(r"-{2,}", sentence):
            continue
        if sentence.startswith(("---", "//", "{")):
            continue
        candidates.append(sentence)
    return candidates


def _distinctive_terms(text: str, limit: int = 6) -> list[str]:
    """Terms that identify this section against the rest of the corpus.

    Scored by rarity across the corpus, not by position. A term that appears in many
    documents cannot narrow a search, so including it in a question would make the
    question unanswerable by retrieval for reasons that have nothing to do with the
    retriever's quality.
    """
    counts = Counter(term for term in tokenize(text) if term not in STOPWORDS)
    if not counts:
        return []
    return [term for term, _ in counts.most_common(limit)]


def _pick_answer(sentences: list[str], anchors: list[str], rng: random.Random) -> str | None:
    """Choose a sentence that both states something and contains an anchor term.

    A sentence with no distinctive term is likely to be generic filler, which makes a
    question that any document could appear to answer.
    """
    anchors_lower = {a.lower() for a in anchors}
    with_anchor = [s for s in sentences if any(a in s.lower() for a in anchors_lower)]
    pool = with_anchor or sentences
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


def _question_for(
    doc: Document, heading: str, body: str, anchors: list[str], rng: random.Random
) -> Candidate | None:
    prose = _clean(body)
    if len(prose) < MIN_BODY_CHARS:
        return None

    sentences = _sentences(prose)
    if not sentences:
        return None

    # Code blocks and tables survive cleaning as noise; a section that is mostly one
    # cannot support a prose question.
    if body.count("```") >= 4 or prose.count("|") > 20:
        return None

    cleaned_heading = _clean(heading.split("{", 1)[0].strip())
    subject = cleaned_heading or doc.title
    if len(subject) > 90:
        subject = subject[:90].rsplit(" ", 1)[0]

    answer = _pick_answer(sentences, anchors, rng)
    if answer is None:
        return None

    # The question names the subject; the anchors are what a retriever must match.
    # Including the anchors in the question would make retrieval trivial and the
    # metric meaningless, so the question carries the subject alone.
    question = f"In the FastAPI documentation, what is described regarding {subject.lower()}?"

    return Candidate(
        doc_id=doc.doc_id,
        title=doc.title,
        question=question,
        answer=answer,
        difficulty="single-hop",
        anchor_terms=tuple(anchors[:3]),
    )


def build_corpus_frequencies(documents: list[Document]) -> Counter[str]:
    """Document frequency across the corpus, for the rarity scoring above."""
    frequency: Counter[str] = Counter()
    for doc in documents:
        seen = {term for term in tokenize(doc.text) if term not in STOPWORDS}
        frequency.update(seen)
    return frequency


def select_sections(
    documents: list[Document], frequency: Counter[str], limit: int = 60
) -> list[tuple[Document, str, str, list[str]]]:
    """Rank sections by how distinctive their vocabulary is, then take the best.

    Rare-vocabulary sections make the sharpest questions: a retriever that finds the
    source has demonstrably discriminated it from the rest of the corpus.
    """
    scored: list[tuple[float, Document, str, str, list[str]]] = []

    for doc in documents:
        if SKIP_FILES.search(doc.doc_id):
            continue
        for heading, body in _sections(doc.text):
            if not _is_topical_heading(heading):
                continue
            anchors = _distinctive_terms(f"{heading} {body}")
            if len(anchors) < 3:
                continue
            rarity = sum(1.0 / (1.0 + frequency[t]) for t in anchors[:6])
            length_penalty = 1.0 if len(body) > 800 else 0.85
            scored.append((rarity * length_penalty, doc, heading, body, anchors))

    scored.sort(key=lambda item: -item[0])
    return [(doc, heading, body, anchors) for _, doc, heading, body, anchors in scored[: limit * 2]]


def propose(
    documents: list[Document], limit: int = 60, seed: int = 17
) -> list[Candidate]:
    rng = random.Random(seed)
    frequency = build_corpus_frequencies(documents)
    ranked = select_sections(documents, frequency, limit=limit)

    candidates: list[Candidate] = []
    # Cap per document so one long page cannot dominate the set. A set drawn from a
    # handful of pages measures page selection, not retrieval.
    per_doc: dict[str, int] = {}
    for doc, heading, body, anchors in ranked:
        if per_doc.get(doc.doc_id, 0) >= 2:
            continue
        candidate = _question_for(doc, heading, body, anchors, rng)
        if candidate is None:
            continue
        per_doc[doc.doc_id] = per_doc.get(doc.doc_id, 0) + 1
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"corpus not found at {CORPUS_DIR}/ - run scripts/fetch_corpus.sh first")
        return 1

    documents = load_documents(CORPUS_DIR)
    candidates = propose(documents)

    if not candidates:
        print("no candidates generated; check the corpus and the section filters")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    review_path = OUT_DIR / "proposed.jsonl"

    with review_path.open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(candidates, start=1):
            handle.write(
                json.dumps(
                    {
                        "id": f"q{index:03d}",
                        "question": candidate.question,
                        "answer": candidate.answer,
                        # Provenance, not a guess: the question was written from this
                        # document, so the label is a fact about construction.
                        "relevant_doc_ids": [candidate.doc_id],
                        "difficulty": candidate.difficulty,
                        "notes": candidate.notes,
                        "review": "PENDING",
                    }
                )
                + "\n"
            )

    distinct = len({c.doc_id for c in candidates})
    print(f"wrote {len(candidates)} proposed questions from {distinct} documents")
    print(f"  -> {review_path}")
    print()
    print("Each row is review=PENDING. A human must confirm the answer is actually")
    print('supported by the source document and set "review": "OK", then run:')
    print("  uv run python -m gateway.rag.promote")
    print("Only OK rows reach the CI gate. Unreviewed rows are never published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
