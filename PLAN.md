# Plan

Six modules. Each ends in a commit and leaves a runnable system. A half-finished
project ships nothing, so nothing starts before the previous module runs.

## Module 1 — Core client and accounting

- `Provider` protocol; `OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`.
- Retry with jittered backoff, then fail over per policy.
- `Usage` records emitted as JSON lines.

Verify: `pytest` proves a 429 on the primary provider falls over and that exactly one
usage record is written per provider call.

## Module 2 — Cache

- `InMemoryCache` (default) and `RedisCache`.
- Hash mode (free) and semantic mode (embedding-based, opt-in).
- Cache failures degrade to misses and are logged.

Verify: `pytest` proves a repeated query makes zero provider calls, and that a broken
cache backend still returns answers.

## Module 3 — RAG

- `Retriever` protocol; `NullRetriever`, `BM25Retriever`, `LightRagRetriever`.
- Cross-encoder reranking with graceful degradation when unavailable.
- HyDE query rewriting.

Depends on: Module 1.
Verify: a golden-set run produces nDCG@10 and the number lands in `docs/RESULTS.md`.

## Module 4 — Verification and abstention

- Per-claim splitting, NLI entailment, calibrated abstention.
- Output carries `verified` and `abstained`.

Verify: `pytest` proves a fabricated claim triggers abstention, and a supported answer
passes through unchanged.

## Module 5 — Eval harness and CI gate

- 50-question golden set, written by hand.
- Ablation runner across retrieval modes and frameworks.
- GitHub Actions job that fails a pull request on an nDCG@10 regression above 2%.

Depends on: Module 3.
Verify: the gate demonstrably fails on a deliberately degraded variant.

## Module 6 — Serving

- FastAPI with SSE streaming.
- React + Vite front end: streamed tokens, clickable source spans, usage panel.
- Docker Compose for Redis, Postgres, Phoenix, Ollama.

Depends on: Modules 1, 2, 3, 4.
Verify: `docker compose up` reaches a working UI from a clean machine.

## Explicitly out of scope

Agent orchestration, fine-tuning, a plugin system, and a web UI theme. Each adds
surface area without adding signal.

## Milestone mapping

| Week | Modules | Ship state |
|---|---|---|
| 1 | 1, 2 | Resilient client with caching, measured |
| 2 | 3 | RAG with a real retrieval number |
| 3 | 4, 5 | Verification and a CI gate that catches regressions |
| 4 | 6 | Full stack, demo video, README |

## Resume bullets

Numbers are placeholders. Fill them from `docs/RESULTS.md` only after the run
produces them. A number you did not measure is a liability in an interview.

- Built an LLM reliability gateway: multi-provider failover, semantic caching, and
  token/cost accounting — p95 latency down __%, cache hit rate __%.
- Benchmarked graph vs vector retrieval across __ modes × 2 frameworks on 50
  hand-labeled multi-hop questions — __ nDCG@10.
- Added per-claim NLI citation verification with calibrated abstention —
  unsupported-claim rate __% → __%.
- CI eval gate fails pull requests on an nDCG@10 regression above 2%.
