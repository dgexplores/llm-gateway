# llm-gateway

A resilient gateway between an application and an LLM provider. It owns the four
things that break in production and get ignored in demos: provider failure, repeated
cost, unsupported answers, and missing traces.

**Status:** Module 1 complete (resilient client, cache, retrieval, evaluation
harness). Module 2 planned in [PLAN.md](PLAN.md). Design rationale in
[ARCHITECTURE.md](ARCHITECTURE.md).

> **No results published yet, on purpose.** The evaluation harness runs, but its CI
> gate refuses to execute over unreviewed questions. See
> [Why there are no numbers](#why-there-are-no-numbers-yet).

## What it does today

```python
from pathlib import Path

from gateway import LLMClient, Message, OllamaProvider, OpenAIProvider, UsageRecorder

# Ordered by preference. A local model first means the gateway is demonstrable
# with no credentials at all.
with UsageRecorder(path=Path(".usage.jsonl")) as recorder:
    client = LLMClient([OllamaProvider.from_env(), OpenAIProvider.from_env()], recorder)
    answer = client.complete([Message.user("hi")], request_id="r1")
```

- Retries a failing provider with jittered backoff, then fails over to the next one.
- Raises `ProviderExhausted` rather than returning a fabricated answer.
- Opens a circuit breaker after repeated failures so a dead provider stops adding
  latency to every request.
- Retries only what can succeed on a second attempt: 429 and 5xx, never a 401.
- Writes exactly one usage record per provider call. `docs/RESULTS.md` is generated
  from these records, not written by hand.

### Latency is recorded twice, on purpose

| Field | Measures | Use it for |
|---|---|---|
| `latency_ms` | one provider round trip | comparing providers |
| `total_latency_ms` | the whole request, including retries and backoff | the p95 a user actually waits |

Quoting a p95 without saying which one it is makes the number unanswerable in an
interview. `Usage.overhead_ms` exposes the difference directly.

## Providers

| Provider | Credentials | Notes |
|---|---|---|
| `OllamaProvider` | none | Local. The default, so the project runs with no setup |
| `OpenAIProvider` | optional | Speaks the OpenAI wire format, so vLLM, Groq, Together and LM Studio work unchanged |
| `Provider` protocol | — | Your own: implement `complete` and nothing else changes |

`test_providers.py` covers the status-to-retryability table, which is where a wrong
decision costs real latency.

## Retrieval

The boundary between "find relevant text" and "answer the question" is explicit, so a
retriever can be swapped and compared without touching generation.

```python
from gateway.rag import BM25Retriever, chunk_corpus, load_documents

chunks = chunk_corpus(load_documents(Path("corpus")))
retriever = BM25Retriever.build(chunks)

for hit in retriever.retrieve("how do i add a response header", k=5):
    print(f"{hit.score:6.3f}  {hit.doc_id}  {hit.text[:60]}...")
```

`Retriever` is a `Protocol` with four implementations so far: `NullRetriever` (RAG
off), `BM25Retriever`, and later the hybrid and graph retrievers. A retriever that
finds nothing returns an empty list — never low-scoring filler, because the generator
treats the chunk list as ground truth.

## Evaluating a retriever

```bash
./scripts/fetch_corpus.sh              # pinned FastAPI docs, MIT, 155 files
uv run python -m gateway.rag.label     # propose questions from the corpus
#   a human reviews each row and sets "review": "OK"
uv run python -m gateway.rag.promote    # only OK rows reach the gate
uv run python scripts/run_eval.py      # nDCG@10, recall@10, MRR, hit@10, p95
```

Metrics are split by whether they are deterministic:

| Metric | Deterministic | Gates CI |
|---|---|---|
| nDCG@10, recall@10, MRR, hit@10 | yes | **yes** |
| Faithfulness, answer quality | no (LLM-judged) | no |

A non-deterministic metric used as a gate produces flaky failures that teams learn to
retry past, which is worse than no gate: it costs time while catching nothing.

### Provenance-derived labels

Each question is written *from* a specific section, so `relevant_doc_ids` is a fact
about how the question was constructed rather than an annotator's guess. That turns a
five-hour annotation task into a review task, and makes the labels checkable. Only the
answer text needs human eyes.

## Development

```bash
uv sync --group dev          # core only, no heavy ML deps
uv run pytest                # 112 tests, no network required
uv run ruff check .
```

Optional extras: `--extra rag` (embeddings, reranking, NLI), `--extra serve`
(FastAPI), `--extra cache` (Redis).

## Why there are no numbers yet

A provisional run over 51 unreviewed questions produced BM25 nDCG@10 ≈ 0.89 at 2.7 ms
p95 across 1,448 chunks. It did two useful things: it proved the harness works, and it
exposed that nDCG could exceed 1.0 when a retriever returned many chunks from one
document (now bounded, with a regression test). The result file was then **deleted**,
because the questions had not been verified by a human.

That is the operating rule: `docs/RESULTS.md` is generated from usage records and a
human-reviewed golden set, never written by hand. An unmeasured number is a liability
in an interview, and an unverified one is worse.
