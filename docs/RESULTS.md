# Results

Every number in this file comes from a committed run of `scripts/run_eval.py` or
`scripts/ablation.py`. The machine-readable form is in `docs/results/`.

## Golden set

**51 questions, mechanically verified — not hand-verified.**

`gateway.rag.verify_set` checks each proposed answer against the document its label
cites: the answer's content words must be at least 75% present in that document, and no
two answers may be near-duplicates. 51 of 51 passed.

That number is only meaningful because the verifier was shown to reject. The control
cases in `tests/test_verify_set.py` assert that a fabricated answer, an answer citing the
wrong document, a label pointing at a missing document, an empty answer, and a
near-duplicate pair are all rejected. Without those controls, "51 of 51" would be
consistent with a verifier that never rejects anything.

What this establishes: the answers are supported by the documents they cite, and the
labels are not broken. What it does not establish: that these are good questions, that
the difficulty labels are meaningful, or that a person would find them natural. The set
is therefore described as *mechanically verified* throughout, never as *hand-verified*.

### Known weaknesses in this set

- **All 51 questions are single-hop.** The difficulty breakdown collapses to one row, so
  it measures nothing yet. Multi-hop and no-answer questions are the next thing to add.
- **Questions are generated from the corpus**, so the set inherits the corpus's
  vocabulary. This favours lexical retrieval and understates the gap a dense retriever
  would close.
- **51 questions is small.** Wide error bars. Treat differences below roughly 0.02 nDCG
  as noise.

## Retrieval

Corpus: 155 FastAPI documents, 1,448 chunks.

| retriever | nDCG@10 | recall@10 | MRR | hit@10 | p95 ms | n |
|---|---|---|---|---|---|---|
| `none` (control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 51 |
| `bm25` (baseline) | **0.891** | 0.922 | 0.846 | 0.922 | 3.5 | 51 |

The `none` row is the point of the table. A retriever that returns nothing scores zero,
so the baseline's 0.891 is measuring retrieval rather than the shape of the metric.

## Ablation

| retriever | nDCG@10 | recall@10 | MRR | hit@10 | p95 ms |
|---|---|---|---|---|---|
| `bm25_no_overlap` | 0.883 | 0.941 | 0.845 | 0.941 | 1.9 |
| `bm25_long` (2400 chars) | 0.877 | **0.961** | 0.829 | **0.961** | 1.4 |
| `bm25_short` (400 chars) | 0.871 | 0.902 | 0.819 | 0.902 | 4.8 |

**The baseline does not win.** It is a middle point on a precision/recall trade-off, not
an optimum:

- Larger chunks retrieve the right document more often (recall 0.961 vs 0.922) but
  dilute it across more text, so the ideal document lands lower (nDCG 0.877 vs 0.891).
- Smaller chunks do the opposite and cost 2.2x the latency.
- Disabling overlap *lowers* nDCG while *raising* recall. The overlap window is
  apparently carrying sentence context that helps ranking, without preventing a document
  from being retrieved by either of its halves.

The honest reading is that chunk size is the dominant lever here, ahead of the ranking
model — which is the usual finding, and the reason the ablation varies chunking at all.
Choosing the baseline as "best" because it has the top nDCG would be selecting on the
metric rather than on the use case. For a gateway that shows the reader the retrieved
passages, recall is the metric that matters: a missing document cannot be shown to
anyone.

## What is measured, and what is not

| Metric | Deterministic | Gates CI |
|---|---|---|
| nDCG@10 | yes | yes |
| recall@10 | yes | yes |
| MRR | yes | yes |
| hit@10 | yes | yes |
| p95 retrieval latency | reported | no |
| claim grounding rate | reported | no |
| faithfulness (LLM-judged) | no | never |

The last row is excluded on purpose. A non-deterministic gate produces flaky failures
that get retried past, which costs time while catching nothing.

Latency is reported but not gated: on a shared CI runner the p95 is dominated by
scheduling noise, and a gate on it would fail for reasons unrelated to the code.

## A provisional run, and why it was deleted

An earlier run over the same 51 questions scored nDCG@10 **0.89** at **2.7 ms** p95.
That run was deleted because the questions had not been verified. A plausible number
from an unverified set is worse than no number, because it looks like evidence.

It did earn its keep by exposing a bug: nDCG read **1.95** when a retriever returned
several chunks from one document, because every chunk earned a full position discount.
The metric is now bounded to 1.0, with a regression test pinning the case.

## Retrievers that are not in the table above

`gateway.rag.langchain.LangChainRetriever` and
`gateway.rag.lightrag_retriever.LightRAGRetriever` both satisfy the `Retriever` contract
and are therefore runnable through `scripts/run_eval.py`. **Neither appears in the
ablation, and no number here comes from either.**

The LangChain retriever is tested against real `langchain-core` 1.6.5 for mechanics:
ordering, the distance-to-similarity direction, metadata round-trip, k-clamping, and the
empty result. It is not measured for retrieval quality, because the only embedding
available without a network or an API key is `DeterministicFakeEmbedding`, whose vectors
are random. Measured against it, a query about status codes ranks a background-tasks
passage first. A semantic-quality number from it would be noise with a decimal point.

LightRAG returns a synthesized context string rather than scored chunks, so it cannot be
scored per document at all, and it is excluded from the harness for that reason rather
than included with a caveat.

Producing a comparable semantic number needs a real embedding model, which needs a key.
That is a deliberate omission, not an oversight.

## Reproducing

```bash
./scripts/fetch_corpus.sh
uv run python -m gateway.rag.verify_set   # re-verify the proposed set
uv run python -m gateway.rag.promote      # promote mechanically OK rows
uv run python scripts/run_eval.py         # writes docs/results/bm25.{json,md}
uv run python scripts/ablation.py         # writes docs/results/ablation.{json,md}
```

The corpus is pinned by commit `a3d205b` in `scripts/fetch_corpus.sh`, and the script
refuses to continue if the checkout resolves to a different SHA. Retrieval scores are
stable across re-fetches; the p95 latency column is not, and moves with the machine.
