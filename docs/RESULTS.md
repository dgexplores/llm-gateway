# Results

Every number in this file comes from a committed run of `scripts/run_eval.py` or
`scripts/ablation.py`. The machine-readable form is in `docs/results/`.

## Golden set

**65 questions, mechanically verified — not hand-verified.** 51 single-hop plus 14
multi-hop.

The single-hop questions were rewritten after the first run of this table, and the rewrite
changed the numbers. Both versions are reported below, because the honest reading is that
**they are not comparable** — see [The templated question set](#the-templated-question-set).

`gateway.rag.verify_set` checks each proposed answer against the document its label
cites: the answer's content words must be at least 75% present in that document, and no
two answers may be near-duplicates. 63 of 63 passed. The multi-document answers are
composed of sentences pulled from the corpus by exact substring match, and
`scripts/build_multihop.py` exits non-zero if a match is missing, so no answer in the
set was written by hand.

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

- **The multi-hop questions are retrieval-hard, not reasoning-hard.** Each needs two
  documents retrieved; none asks the reader to chain an inference across them. A model
  could still answer all 14 from one document given enough context.
 They have no relevant document,
  so nDCG is undefined for them. They are measured separately, by abstention rate, below.
- **Questions are generated from the corpus**, so the set inherits the corpus's
  vocabulary. This favours lexical retrieval and understates the gap a dense retriever
  would close.
- **51 questions is small.** Wide error bars. Treat differences below roughly 0.02 nDCG
  as noise.

## Retrieval

Corpus: 155 FastAPI documents, 1,448 chunks.

| retriever | nDCG@10 | recall@10 | MRR | hit@10 | p95 ms | n |
|---|---|---|---|---|---|---|
| `none` (control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 63 |
| `bm25` (baseline) | **0.869** | 0.905 | 0.828 | 0.921 | 2.7 | 63 |

The `none` row is the point of the table. A retriever that returns nothing scores zero,
so the baseline's 0.891 is measuring retrieval rather than the shape of the metric.

### By difficulty

| difficulty | n | nDCG@10 |
|---|---|---|
| single-hop | 51 | 0.891 |
| multi-hop | 14 | 0.570 |

The multi-hop questions score **0.32 lower**, which is the gap a difficulty label is
supposed to reveal. It took two attempts to get a set that measures it.

The first attempt paired two documents and concatenated a sentence from each. Measured
against the corpus, 11 of those 12 rows were answerable from a single document covering
75% or more of the answer — two retrievals dressed up as one question. They were relabelled
`multi-doc`, which was honest but left the corpus with no multi-hop coverage at all, and
the difficulty gap was a meaningless 0.11.

`scripts/build_multihop.py` now enforces the property instead of hoping for it. A sentence
pair is emitted only when **neither** document alone reaches the support threshold and each
contributes content words the other lacks. Candidate pairs are scored by how little of the
combined answer either side covers, and the best-balanced pair per document pair is taken.
The result: all 14 rows sit at 0.55–0.74 best-single-document, against 0.75–0.94 before.

A set that cannot be built under the constraint is a legitimate outcome — it would mean
this corpus does not support genuine multi-hop questions — so the script reports what it
dropped rather than quietly emitting weaker rows.

What the 0.32 gap means: BM25 has to get **both** documents into the top 10 for a
multi-hop question, which is a different and harder task than getting one. A retriever can
score well on single-hop and still be poor at this, so the headline number alone hides the
difference between the two.

## Ablation

| retriever | nDCG@10 | recall@10 | MRR | hit@10 | p95 ms |
|---|---|---|---|---|---|
| `bm25_no_overlap` | 0.815 | 0.854 | 0.779 | 0.908 | 18.8 |
| `bm25_long` (2400 chars) | 0.845 | 0.892 | 0.774 | **0.954** | 9.2 |
| `bm25_short` (400 chars) | 0.838 | 0.877 | 0.762 | 0.938 | 40.9 |
| `hybrid` (BM25 + TF-IDF RRF) | 0.796 | 0.831 | 0.718 | 0.892 | 20.1 |

| `semantic` (dense, 1200c chunks) | 0.745 | 0.854 | 0.364 | 0.938 | 123.5 |
| `semantic_250c` (dense, 250c chunks) | 0.784 | 0.846 | 0.519 | 0.908 | 227.4 |
| `semantic+bm25` (RRF of dense and BM25) | 0.814 | 0.892 | 0.585 | 0.954 | 72.2 |

**Dense retrieval loses to BM25 on this corpus, and the reason is specific.** The dense
rows use `sentence-transformers/all-MiniLM-L6-v2`, loaded locally — no API key, and the
first version of this analysis assumed otherwise, which is why the semantic path was
originally excluded from the harness. It was measurable all along.

Two separate effects, and the first one was a measurement artefact:

**Chunk size dominated the dense result.** A sentence encoder embeds sentences, not
1,200-character technical prose. Over 1200 / 500 / 250 / 100-character chunks, dense
scored:

| chunk size | nDCG@10 | recall@10 |
|---|---|---|
| 1200c (the baseline) | 0.434 | 0.623 |
| 500c | 0.586 | 0.754 |
| **250c** | **0.665** | **0.739** |
| 100c | 0.627 | 0.738 |

(measured on the earlier 63-question set; the ordering is what matters here)

So the baseline chunking was costing dense retrieval about a third of its accuracy, and
100c is too small — passages lose the context that makes them matchable. `semantic_250c`
is in the table so the comparison is tuned against tuned; reporting only the 0.465 would
have blamed the retriever for a chunking mismatch.

**Even tuned, BM25 wins, but by 0.068 rather than the 0.157 the templated set suggested.**
The reason is in the next section, and it is a correction to a claim this file used to
make. Documentation questions are made of exact
identifiers — `status_code`, `BackgroundTasks`, `OAuth2PasswordRequestForm` — and lexical
matching is precisely right for those while a 384-dimensional embedding blurs them into
the surrounding prose. This is the expected result for technical documentation with rare
terms, and it is why a keyword index is the sensible default here rather than a
conservative one.

**Rank fusion hurts twice, and the second time is the informative one.** Fusing BM25 with
TF-IDF scores 0.852 and BM25 with dense scores 0.795, both below BM25's 0.869. With
equal weights, reciprocal-rank fusion averages in a weaker member's opinions instead of
using it to break ties, so adding a retriever that is worse than the incumbent makes the
result worse. Rank fusion is a way to combine *comparable* rankers. It paid off nowhere
here, and the honest conclusion is that it is not worth its latency on this corpus.

The latency column is the other cost: dense is 15x BM25's p95 and fusion of the two is
7x, for worse accuracy.

****The baseline does not win.** It is a middle point on a precision/recall trade-off, not
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

## Abstention on questions the corpus cannot answer

**0 of 8 without a relevance gate; 6 of 8 with one.** The gateway answered every one, and
reported every one as grounded.

Run it yourself:

```bash
uv run python scripts/abstention.py --json docs/results/abstention.json
```

| question | abstained | grounded |
|---|---|---|
| How do I configure a Kubernetes ingress controller for blue-green deployments? | no | yes |
| What is the capital of Australia? | no | yes |
| How do I bake sourdough bread at home? | no | yes |
| Explain quantum entanglement to a five year old. | no | yes |
| What is the difference between TCP and UDP? | no | yes |
| Who won the 1998 FIFA World Cup? | no | yes |
| How do I calculate the standard deviation of a list in Rust? | no | yes |
| What is the current price of a Raspberry Pi? | no | yes |

This is not hallucination. Every sentence returned really does appear in a retrieved
passage, which is exactly what the verifier checks and exactly what the UI now labels
**Grounded** rather than **Verified**. What is missing is any check that the passages
address the question.

Two candidate gates were measured and both are rejected. Both distributions overlap, so
a threshold would abstain on good questions while still answering bad ones:

| signal | answerable (n=63) | unanswerable (n=8) | separates? |
|---|---|---|---|
| question ↔ retrieved passage overlap | 0.167 – 0.667 | 0.000 – 0.500 | no |
| question ↔ generated answer overlap | 0.000 – 0.667 (median 0.200) | 0.000 – 0.200 | no |

The second is the more tempting of the two and still fails: the unanswerable maximum
equals the answerable median.

### A gate that works: cross-encoder relevance

The two lexical gates above fail because word overlap is the wrong question. A
cross-encoder trained for query-passage relevance is the right tool, and one runs locally
without an API key — `cross-encoder/ms-marco-MiniLM-L-6-v2`.

`RelevanceGate` scores the question against each retrieved passage and judges the best
one. End to end, through the real pipeline:

| threshold | answerable answered | unanswerable caught |
|---|---|---|
| **-9.0 (default)** | **65/65 (100%)** | **6/8 (75%)** |
| -6.0 | 64/65 (98%) | 6/8 (75%) |
| -3.0 | 56/65 (86%) | 8/8 (100%) |

**0% → 75% abstention, at no measured cost to answerable questions.** The two it still
answers are the ones the corpus does partially discuss; tightening to -3.0 catches them
too, at the price of withholding one legitimate answer in nine.

```bash
uv run python scripts/abstention.py --relevance --json docs/results/abstention_gated.json
```

**It is opt-in and off by default.** It costs a 90 MB model download and roughly 10 ms
per query, and -3.0 imposes an 11% false-abstention rate. Those are real costs and the
caller should choose, not inherit. The threshold and both numbers are in
`src/gateway/relevance.py`.

One correction worth recording: an earlier estimate of this curve, taken by scoring only
the top-ranked passage, predicted -9.0 would withhold 7 of 63 answerable questions. It
withholds none. The gate takes the best of *k* passages — if any retrieved passage
addresses the question, the gateway can answer from it — so scoring top-1 alone measured
something the gate does not do.

This closes the largest limitation in the project, and it is the third attempt: the two
lexical gates are kept above, rejected on evidence, so the obvious next person does not
retry them.

## Verifier comparison: is the NLI upgrade worth it?

The obvious upgrade from lexical overlap is a cross-encoder that judges entailment rather
than word presence. It is implemented, it loads, and it is not usable. Both verifiers were
run over the same 120 claims, all drawn from the context the pipeline actually retrieves:

- 60 ``supported`` — a sentence present in the retrieved context, so the right answer is yes.
- 60 ``swapped`` — a real corpus sentence from a *different* question's context, judged
  against this one. Its vocabulary is equally ordinary and it is still false.

| verifier | accepts true | accepts swapped |
|---|---|---|
| `overlap` (shipped) | **60/60** | 1/60 |
| `nli` (cross-encoder) | **0/60** | 0/60 |

The NLI verifier rejects every true claim. It is not slightly worse; it is unusable, and
the reason is specific:

| premise | length | entailment |
|---|---|---|
| the claim itself | ~110 chars | 0.991 |
| three sentences, containing the claim | ~230 chars | 0.984 |
| the full rendered multi-source context | ~1,100 chars | 0.004 – 0.018 |

The model handles a claim quoted against a short premise and collapses once the premise is
the full multi-source context the pipeline hands it. Held constant while the premise grows,
the same verbatim claim falls from 0.99 to 0.004. Swapping it in would make the gateway
abstain on essentially everything.

```bash
uv run python scripts/verifier_comparison.py --json docs/results/verifiers.json
```

A per-source variant might work, since ~230 chars is fine, but that is not what the
pipeline does and it is not what was measured.

### The bug this work uncovered

`NliVerifier` read `probabilities[-1]` as the entailment probability. For the
cross-encoders this is used with, that index is **neutral**; entailment is index 1,
resolved from the model's `id2label`. Every verdict was inverted — an entailed claim
scored 0.992 at the right index and 0.007 at the one being read.

Nothing caught it, because the class was defined, imported, and never executed. The only
test was `hasattr(module, "NliVerifier")`, which passes just as happily against a class
that is wrong in every case. The index is now resolved by label name, and a test pins the
regression by asserting the entailment index is *not* the last one. The end-to-end tests
need a checkpoint download, so they are opt-in via `GATEWAY_NLI_TESTS=1`; the index
resolution is tested unconditionally.

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

An earlier run over 51 single-hop questions scored nDCG@10 **0.89** at **2.7 ms** p95.
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
