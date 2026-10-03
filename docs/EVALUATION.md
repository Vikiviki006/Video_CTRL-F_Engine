# Evaluation reference

Every metric used in this project, with its exact definition, the matching rule
it depends on, and the edge cases where it is deliberately undefined.

Implementations: `experiments/evaluation/metrics.py`.
Verification: `backend/tests/test_metrics.py` (47 tests, every expected value
computed by hand in a comment next to the assertion).

---

## 0. Conventions

A **run** for one query is a list of retrieved items in best-first order. Each
item may carry `id`/`frame_id` and/or `timestamp_seconds`.

### Relevance matching rule (binary, used everywhere)

An item is **relevant** if:

1. its id matches a ground-truth id, **or**
2. `|item.timestamp − gt| ≤ tolerance_seconds` for some ground-truth timestamp.

`tolerance_seconds` defaults to **2.0 s** and is overridable per query. The
comparison is inclusive (`≤`).

The default of 2.0 s is chosen to be *narrower than the spacing between
genuinely different shots*. A tolerance wider than the inter-shot interval would
let unrelated frames count as hits and inflate every metric.

**Binary relevance is used throughout.** No dataset in this project carries
graded relevance, and inventing grades would fabricate information. `dcg_at_k`
and `ndcg_at_k` accept a `grades` mapping for a future graded dataset, and the
binary metrics ignore grades (correctly, since binary metrics have no notion of
grade).

### Undefined cases, and what is returned instead

| Situation | Returned | Why |
|---|---|---|
| Query has no ground-truth relevant item | `0.0` | The ratio has no denominator. Returning `1.0` would reward the system for an unanswerable question. |
| Empty run passed to `evaluate_run` | raises `ValueError` | An empty run has *no defined metrics*. Returning zeros would be indistinguishable from "scored and got 0" - the exact mechanism by which a fabricated number enters a report. Callers must handle "nothing to score" explicitly. |
| Latency sample is empty | `None` → renders `N/A` | `0 ms` reads as a real measurement. |
| `original_frames ≤ 0` in frame reduction | `None` → `N/A` | The ratio is undefined. |
| `raw_bytes ≤ 0` in storage overhead | `None` → `N/A` | Same. |

---

## 1. Precision@K

```
Precision@K(q) = |{ i ∈ top-K : relevant(i) }| / K
```

**The denominator is `K`, not `min(K, len(ranked))`.** A system that returns
fewer than K results is penalised for the missing ones, which is the honest
behaviour: an empty result list is not a perfect precision of 1.

`K ∈ {1, 5, 10, 20}`

*Worked example.* 4 results returned, K = 10, 1 relevant → `1/10 = 0.1`.

## 2. Recall@K

```
Recall@K(q) = |{ i ∈ top-K : relevant(i) }| / |relevant(q)|
```

`K ∈ {1, 5, 10, 20}`

*Worked example.* Ground truth = `{f1}`; retrieved order is `[f2, f3, f1, f4]`.
* `Recall@2 = 0/1 = 0.0` (f1 is at rank 3)
* `Recall@5 = 1/1 = 1.0`

**Single-ground-truth-timestamp datasets.** With one annotated timestamp `τ` and
tolerance `ε`, exactly one retrieved frame can be relevant (barring
near-duplicates), so Recall@K ∈ {0, 1} and Precision@K = 1/K at best. Recall@K
is therefore the metric to read on such a dataset, and Precision@K is mostly a
measure of how many distractors the system returns. This is stated explicitly
because reading a low Precision@10 as "the system is bad" would be a
misreading.

## 3. HitRate@K

```
HitRate@K(q) = 1.0 if any i ∈ top-K is relevant, else 0.0
```

`K ∈ {1, 5, 10}`

## 4. MRR (Mean Reciprocal Rank)

```
RR(q)   = 1 / rank_of_first_relevant_result      (rank is 1-based)
MRR     = mean over queries of RR(q)
```

`0.0` when no result is relevant.

*Worked example.* First relevant at rank 3 → `1/3 ≈ 0.3333`.

## 5. MAP (Mean Average Precision)

Supports **multiple** relevant items per query.

```
AP(q) = (1 / R) · Σ_{i=1..ranked} P@i · relevant(i)
```

`R` = total ground-truth relevant items. The sum runs over the **full ranked
list**, not just top-K, because truncating it would make MAP depend on K.

*Worked example.* ranked `[f1, f2, f3]`, relevant `{f1, f3}`, R = 2.
* i=1 hit → P@1 = 1 → contributes 1
* i=2 miss → contributes 0
* i=3 hit → P@3 = 2/3 → contributes 2/3

`AP = (1 + 2/3) / 2 = 5/6 ≈ 0.8333`

## 6. NDCG@K

```
DCG@K  = Σ_{i=1..K} gain_i / log2(i + 1)
IDCG@K = Σ_{i=1..min(K, R)} 1 / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

`K ∈ {1, 5, 10, 20}`. `gain_i` is **binary** (1 relevant, 0 otherwise) unless a
`grades` mapping is supplied.

`log2(i + 1)` rather than `log2(i)` avoids division by zero at the first
position. The ideal ranking is truncated at `min(K, R)` because only `R`
relevant items exist.

`0.0` when `IDCG@K == 0` (nothing relevant exists).

*Worked examples.*
* relevant first, K = 3, R = 1: `DCG = 1/log2(2) = 1`, `IDCG = 1` → **1.0**
* relevant at rank 3, K = 3, R = 1: `DCG = 1/log2(4) = 0.5`, `IDCG = 1` → **0.5**
* relevant at rank 2, K = 4, R = 1: `DCG = 1/log2(3) ≈ 0.6309` → **0.6309**

## 7. Percentiles and latency

```
avg    = mean(samples)
median = middle value (mean of the two middles for an even count)
P95    = percentile(samples, 95)
P99    = percentile(samples, 99)
```

Percentiles use **linear interpolation between closest ranks** (NumPy's default
"inclusive" method):

```
pos       = (n − 1) · p / 100
value     = data[floor(pos)] + (data[ceil(pos)] − data[floor(pos)]) · (pos − floor(pos))
```

This convention is stated because percentile definitions differ between tools,
and silently mixing nearest-rank with interpolated P95 makes two latency tables
incomparable.

*Worked example.* `1..20`, P95: `pos = 19 × 0.95 = 18.05` → `19 + (20−19)×0.05 = 19.05`.

**Stages measured** (from `graph_search_service.graph_search`, timed with
`time.perf_counter`):

| Stage | Covers |
|---|---|
| `embedding` | SigLIP-2 text encoding of the query |
| `vector_retrieval` | the pgvector query |
| `graph_expansion` | the bounded graph traversal |
| `graph_reranking` | scoring + weighted re-ranking |
| `total` | sum of the above, and the response's `latency_ms` |

## 8. Indexing metrics

```
Frame Reduction % = (1 − retained / original) × 100
```

* `original` = `videos.frame_count`, the source video's true decoded frame count
* `retained` = rows actually written to `video_frames`

`N/A` when `original ≤ 0`. This is the honest denominator for the
adaptive-extraction claim: it measures against every decoded frame, not against
some pre-filtered subset.

Also collected per video: duration, fps, original/candidate/retained frame
counts, graph node and edge counts.

## 9. Storage metrics

| Quantity | Source |
|---|---|
| raw video bytes | `os.path.getsize(video.path)`, summed over ready videos |
| thumbnail bytes | recursive `os.path.getsize` over `FRAME_ROOT` |
| clip bytes | recursive `os.path.getsize` over `CLIP_ROOT` |
| embedding bytes | `row_count × 1152 × 4` — an **arithmetic estimate** of the column payload, labelled as such, not a database-reported size |
| overhead ratio | `derived / raw`, `N/A` when `raw ≤ 0` |

## 10. Graph metrics

| Metric | Meaning |
|---|---|
| `nodes_reached_avg` / `_max` | distinct nodes reached per query, after the node cap |
| `edges_traversed_avg` / `_max` | edges actually traversed |
| `edge_type_totals` | counts per relationship type, from real traversal |
| `depth_limit_hit_queries` | queries where expansion stopped at `GRAPH_MAX_DEPTH` |
| `node_limit_hit_queries` | queries where it stopped at `GRAPH_MAX_NODES` |
| `expansion_latency_ms_avg` / `_p95` | traversal cost |

`depth_effect_report` additionally reports how each quality metric moves with
depth and names the best-scoring depth — but states the direction as an
**observation**. Whether deeper expansion helps is an empirical question this
project has deliberately not pre-decided.

## 11. Model comparison

The SigLIP vs SigLIP-2 table requires **both** checkpoints indexed over
identical frames, scored on the same queries, ground truth, top-K and metric
set. When only one checkpoint has been evaluated, every comparison cell is
`N/A` with the reason recorded. A single measured model plus an invented second
series would be the worst possible outcome, so the table is left empty instead.

## 12. What is measured, today

| Metric family | Status |
|---|---|
| Recall@K, Precision@K, MRR, MAP, NDCG@K, HitRate@K | **Implemented and unit-tested.** Values: **N/A** — no ground truth annotated. |
| Latency (avg/median/P95/P99 per stage) | **Implemented.** Values: **N/A** — no indexed video. |
| Frame reduction | **Implemented.** Values: **N/A** — no indexed video. |
| Storage | **Implemented.** Values: **N/A** — no indexed video. |
| Graph size / expansion counts | **Implemented.** Values: **N/A** — no indexed video. |
| SigLIP vs SigLIP-2 | **Implemented.** Values: **N/A** — one checkpoint. |

The distinction between "implemented" and "measured" is the whole point of this
table. The formulas above are real, tested code; the numbers are not yet real,
and no number in this repository pretends otherwise.
