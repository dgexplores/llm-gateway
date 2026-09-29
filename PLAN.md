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

| Module | Ship state | Done |
|---|---|---|
| 1 | Resilient client, cache, usage accounting | yes |
| 2 | Retrieval, evaluation harness, API, UI | yes |
| 3 | Verification, abstention, hybrid fusion, ablation | yes |
| 4 | Container image, published results, documentation | yes |

## Status

Complete, with two documented limitations rather than two open items.

Shipped: provider failover with circuit breaking, usage accounting, response cache, the
`Retriever` contract, chunking, `BM25Retriever`, a TF-IDF cosine retriever, reciprocal-rank
fusion, an evaluation harness whose CI gate refuses to run over unverified questions, a
FastAPI service, a zero-build UI, claim-level verification with abstention, LangChain and
LightRAG adapters, a multi-stage container image, and a measured result set.

Measured, on 63 mechanically verified questions over 155 documents:

| retriever | nDCG@10 | recall@10 | p95 ms |
|---|---|---|---|
| `none` (control) | 0.000 | 0.000 | 0.0 |
| `bm25` (baseline) | **0.869** | 0.905 | 2.7 |
| `hybrid` (BM25 + TF-IDF) | 0.852 | 0.905 | 5.4 |

### Two limitations, both measured

**1. The gateway cannot detect a question the corpus cannot answer.** On eight such
questions it answered all eight and reported all eight as grounded: 0% abstention. Two
candidate gates were measured and both rejected, because the answerable and unanswerable
score distributions overlap completely. No grounding check can catch this, and the UI
footer no longer claims it can. See `docs/RESULTS.md`.

**2. The NLI verifier is not usable as a drop-in.** It rejects every true claim (0/60)
because it collapses on the ~1,100-character multi-source context the pipeline supplies:
the same claim scores 0.991 against a short premise and 0.004 against the real one.
Lexical overlap stays the default, and the comparison is in `docs/RESULTS.md`.

Neither is a task left undone. Both are results, and the honest thing is to publish them
rather than build a gate that does not work.

### Not built, on purpose

- **Nothing, here — this was wrong when written.** The semantic path was excluded on the
  assumption that dense retrieval needs an API key. It does not: `all-MiniLM-L6-v2`
  downloads and runs locally. The LangChain adapter is now measured in the ablation
  (0.655 nDCG@10 against BM25's 0.869) and the LightRAG adapter remains unmeasured
  because it needs an LLM for entity extraction at index time.
- **A Docker build.** The image and compose file are written but Docker is not installed
  here, so the image has never been built. The docs say so.

## Resume bullets

Every number below is from a committed run in `docs/results/`. A number that was not
measured is a liability, so the two things this project does *not* do are stated in the
same list as the things it does.

- Built an LLM reliability gateway: multi-provider failover with circuit breaking,
  token/cost accounting, and claim-level citation verification. BM25 retrieval over
  1,448 chunks from 155 documents scores **nDCG@10 0.869, recall@10 0.905, p95 2.7 ms**
  against a no-retrieval control at 0.000, on 63 questions.
- Measured what the response cache is worth instead of asserting it: **55.6% hit rate**
  on a Zipf-shaped workload, 79 of 142 provider calls avoided. Found the cache was
  implemented, unit-tested, and **never wired into the client**; a unit test of a class
  cannot tell you nothing calls it.
- Benchmarked **9 retrieval configurations** including dense retrieval with a local
  embedding model. Dense scores 0.655 nDCG@10 against BM25's 0.869, and **both rank-fusion
  variants lose** — BM25+TF-IDF 0.852, BM25+dense 0.795 — because equal-weight fusion
  averages in a weaker member rather than using it to break ties. Chunk size turned out to
  be worth a third of dense's accuracy (0.465 at 1,200c vs 0.655 at 250c), so the tuned
  number is the one reported.
- Evaluated the obvious verification upgrade and rejected it on evidence: an NLI
  cross-encoder scores **0/60** on claims the pipeline itself retrieved, because it
  collapses from 0.991 to 0.004 as the premise grows to the real multi-source context.
  Found and fixed an inverted logit index in the process, which no test had caught
  because the class was never executed.
- **Measured the product's main failure mode rather than hiding it.** On 8 questions the
  corpus cannot answer, the gateway answered all 8 and reported all 8 as grounded —
  0% abstention. Two candidate relevance gates were measured and both rejected for
  overlapping score distributions. The UI footer previously claimed otherwise and was
  corrected.
- CI runs lint, 380 tests, a retrieval gate that refuses to execute without a verified
  golden set, the ablation, and the abstention measurement.

### Not claimed

- **No multi-hop reasoning results.** The 12 multi-document questions need two documents
  retrieved, but 11 of 12 are answerable from one, so they are labelled `multi-doc` and
  the difficulty gap (0.891 → 0.777) is reported as a retrieval finding, not a reasoning
  one.
- **Dense retrieval was measured, and lost.** With a local `all-MiniLM-L6-v2` it scores
  0.655 nDCG@10 against BM25's 0.869, and fusing it in scores 0.795. Chunk size was
  worth a third of dense's accuracy (0.465 at 1,200c vs 0.655 at 250c), so the tuned
  number is the one reported. The LangChain adapter is measured; the LightRAG adapter is
  not, because it needs an LLM for entity extraction at index time.
- **No verified Docker build.** The image and compose file exist; Docker is not installed
  in the environment they were written in, so the image has never been built.
