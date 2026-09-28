# Architecture — LLM Reliability Gateway

## Problem

LLM applications fail in ways that are invisible in a demo:

- The provider 429s or times out, and the app returns a raw error to the user.
- The same question is asked repeatedly and paid for repeatedly.
- The model answers confidently with facts that exist in no source document.
- Nobody can reconstruct what happened, what it cost, or why it was wrong.

This gateway sits between an application and the model provider and owns all four.

## Non-goals

- Not a framework. No plugin DSL, no abstract base classes per provider.
- Not a fine-tuning or model-training tool.
- Not an orchestrator of agents. It is the layer *under* them.

## Design principles

1. **Every model call goes through one client.** Provider selection, retries, caching,
   and accounting are not optional per call site — they are the default.
2. **No silent degradation.** If the answer cannot be supported, the gateway abstains
   and says so. A wrong answer that reads as confident is worse than a refusal.
3. **Measurement before optimization.** The eval harness exists before the caching and
   routing layers, so their effect can be attributed rather than assumed.
4. **Cheap dependencies.** The core (`src/gateway`) uses the standard library plus
   `httpx`. Heavyweight dependencies (rerankers, embeddings, NLI models) live in the
   optional `rag` extra and are imported lazily, so the package installs and tests run
   with `uv sync --no-extra rag`.

## Layers

```
                     application
                          |
                   [ HTTP / SSE ]   FastAPI, streaming tokens
                          |
  +--------------------- gateway -------------------------+
  |  middleware   request-id, PII redaction, rate limit   |
  |  cache        Redis semantic cache -> miss          |
  |  rag          retrieve -> rerank -> verify (optional)|
  |  llm          provider router, retries, circuit break|
  |  verify       per-claim NLI entailment + abstention  |
  |  account      tokens, latency, cost, cache hit rate  |
  |  trace        structured span per request            |
  +------------------------------------------------------+
       |              |               |            |
    OpenAI       Anthropic       Ollama (local)   Redis
```

### Request lifecycle

1. `middleware` assigns `request_id`, redacts PII from the query.
2. `cache` computes an embedding of the query (or a hash, in `hash` mode) and looks up
   a semantic cache. A hit returns the stored response and is accounted as
   `cache_hit=true`; no provider call occurs.
3. `rag` (only when enabled) retrieves chunks, reranks them, and attaches them as
   context. The retrieval trace is recorded with `source` and `rank`.
4. `llm` selects a provider by policy, executes with retries and a circuit breaker,
   and records tokens, latency, and cost.
5. `verify` (only when enabled) splits the answer into claims, checks each against the
   retrieved context, and either returns the answer, returns a revised answer, or
   returns an abstention.
6. `account` writes one usage record. `trace` writes one span.

Every layer is optional except `account`, which is unconditional.

## Failure handling

| Failure | Behaviour |
|---|---|
| Primary provider 429/5xx | Retry with jittered backoff, then fail over per policy |
| All providers exhausted | Return `ProviderExhausted`; never a fabricated answer |
| Cache backend down | Log and treat as a miss; do not fail the request |
| Reranker unavailable | Degrade to unreranked retrieval, flag `reranked=false` |
| Verification model unavailable | Return the answer with `verified=false` rather than blocking |
| Context too large | Trim by reranker score, never truncate silently — record what was dropped |

## Ports and adapters

Concrete implementations only. Interfaces exist where a seam is genuinely needed for
testing or for swapping a dependency the user asked to swap.

- `gateway.llm.providers.base.Provider` — the one protocol. Implementations:
  `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`.
- `gateway.cache.Cache` — `InMemoryCache` (default, zero deps), `RedisCache`.
- `gateway.rag.retriever.Retriever` — `NullRetriever` (default),
  `LightRagRetriever` (the fork), `BM25Retriever`.
- `gateway.verify.Verifier` — `NullVerifier` (default), `NliVerifier`.

`Null*` defaults are what let the package run with zero optional dependencies. They
are not stubs to be filled in later; they are the correct behaviour when the feature
is off.

## Data model

```python
@dataclass(frozen=True)
class Usage:
    request_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    verified: bool
    abstained: bool
```

Usage records are emitted as JSON lines. One writer, append-only. `docs/RESULTS.md`
is generated from these records, so the numbers on the resume are the numbers the
code produced.

## Evaluation

Two suites, both runnable locally:

- **Gateway suite** — provider failover, cache correctness, PII redaction, streaming.
  Fast, no network required (uses fakes).
- **RAG suite** — 50 hand-labeled multi-hop questions. Reports nDCG@10, faithfulness,
  answer relevance, unsupported-claim rate, abstention precision, cost per query, and
  p95 latency.

The RAG suite runs in CI as a gate: a variant that regresses nDCG@10 by more than 2%
against the base branch fails the pull request.

## Deployment

- Local development: `uv run` with in-memory adapters; no Docker required.
- Full stack: `docker compose up` for Redis, Postgres, Phoenix, and Ollama.
- The application is stateless. Cache and usage records live outside the process.

## Security

- Provider keys are read from the environment only, never logged.
- Usage records store redacted text; the redaction step runs before anything is
  persisted.
- The gateway does not accept arbitrary provider URLs from request bodies, which
  would otherwise be a server-side request forgery vector.

## Trade-offs accepted

- **Semantic cache** costs an embedding call per query. A hash cache is free and is the
  default; semantic mode is opt-in and its hit rate is measured, not assumed.
- **Verification** adds a model call. It is off by default and its cost/benefit is
  reported in `docs/RESULTS.md` rather than asserted in a README.
- **Python 3.14** is ahead of most ML package support. Core code is stdlib-only for
  that reason; the `rag` extra pins nothing and may need lower pins on this
  interpreter.
