# llm-gateway

A resilient gateway between an application and an LLM provider. It owns the four
things that break in production and get ignored in demos: provider failure, repeated
cost, unsupported answers, and missing traces.

**Status:** Modules 1–2 complete (client with retry and failover, cache). Modules 3–6
planned in [PLAN.md](PLAN.md). Design rationale in
[ARCHITECTURE.md](ARCHITECTURE.md).

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

## Development

```bash
uv sync --group dev          # core only, no heavy ML deps
uv run pytest                # gateway suite, no network required
uv run ruff check .
```

Optional extras: `uv sync --extra rag` (embeddings, reranking, NLI), `--extra serve`
(FastAPI), `--extra cache` (Redis).

## Why the numbers are not here yet

No retrieval, cost, or latency figure appears in this README until a run has produced
it. The eval harness (Module 5) writes `docs/RESULTS.md` from usage records; every
claim on the resume comes from that file. An unmeasured number is a liability in an
interview.
