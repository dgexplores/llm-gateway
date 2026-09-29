# Results

No numbers are published here yet.

## Why this file is empty

The evaluation harness works and produces a table. It refuses to publish one, because
the question set it would measure has never been checked by a person.

`scripts/ablation.py` and `scripts/run_eval.py` both exit non-zero when
`evals/goldens.jsonl` is absent. CI does the same. This is deliberate: a retrieval
score computed over machine-generated, unreviewed questions is a statement about the
label generator, not about retrieval.

## What happens once the set is reviewed

```bash
uv run python -m gateway.rag.label     # propose questions from the corpus
#   a human checks each answer against its source document and sets
#   "review": "OK" or "REJECT"
uv run python -m gateway.rag.promote    # only OK rows reach the gate
uv run python scripts/ablation.py       # writes docs/results/ablation.{json,md}
```

The table this produces is committed, and `README.md` links to it. Every number in this
repository comes from that run.

## What will be measured

| Metric | Deterministic | Gates CI |
|---|---|---|
| nDCG@10 | yes | yes |
| recall@10 | yes | yes |
| MRR | yes | yes |
| hit@10 | yes | yes |
| p95 retrieval latency | yes | no |
| claim grounding rate | yes | no |
| faithfulness (LLM-judged) | no | never |

The last row is excluded from the gate on purpose. A non-deterministic gate produces
flaky failures that get retried past, which costs time while catching nothing.

## Variants under comparison

Each isolates one decision, and the set includes a control:

| Variant | What it changes |
|---|---|
| `none` | no retrieval — the control |
| `bm25` | the baseline: 1200-char chunks, 200-char overlap |
| `bm25_no_overlap` | chunk overlap disabled |
| `bm25_long` | 2400-char chunks |
| `bm25_short` | 400-char chunks |

Chunking is included because it is usually where a retrieval system quietly loses more
accuracy than the ranking model does, and a benchmark that varies only the ranker would
miss that entirely.

## A provisional run, and why it was deleted

Over 51 unreviewed questions, BM25 scored nDCG@10 **0.89** at **2.7 ms** p95 over
**1,448 chunks** from **155 documents**.

That run did two useful things and was then deleted. It proved the harness works, and
it exposed that nDCG could exceed 1.0 when a retriever returned several chunks from one
document — every chunk earned a full position discount, so the score read **1.95**. The
metric is now bounded to 1.0 with a regression test pinning the case.

The result file was removed because the questions had not been verified. A plausible
number from an unverified set is worse than no number, because it looks like evidence.
