# Evaluation datasets

## What is here

| File | Purpose |
|---|---|
| `evaluation_dataset.template.csv` | The annotation template. Column definitions below. |
| `evaluation.csv` | The working dataset. **Currently has no ground truth** - see below. |

The pre-existing `experiments/evaluation_dataset.csv` is left byte-identical; it
is the original query list, and it has the same problem (empty `video_id` and
empty `ground_truth_timestamp`).

## Current state: no ground truth, so metrics are N/A

`evaluation.csv` contains 5 real queries but **no annotations**. Every retrieval
metric is therefore undefined and the evaluation harness reports `N/A` with the
reason recorded in the CSV. This is deliberate: Recall@K, Precision@K, MAP and
NDCG@K are ratios against ground truth, and with no ground truth a `0.0` would
mean "the system retrieved nothing relevant" when the truth is "nobody said
what relevant was".

`python -m experiments.evaluation.evaluate_retrieval --dry-run` reports exactly
how many queries are scorable. Run it first, before spending GPU time.

## Columns

| Column | Required | Meaning |
|---|---|---|
| `query` | **yes** | The natural-language query. |
| `id` | no | Stable query identifier. Defaults to `q1`, `q2`, ... |
| `video_id` | one of these three | UUID of the video the query refers to. |
| `ground_truth_timestamp` | | A single relevant moment, in **seconds from the start of the video**. |
| `ground_truth_timestamps` | | Several relevant moments. JSON array `["10.5","42.0"]` or `10.5|42.0`. |
| `relevant_ids` | | Explicit frame UUIDs instead of timestamps. JSON array or `10.5|42.0` style. |
| `tolerance_seconds` | no | How far from the annotated timestamp still counts as a hit. Default `2.0`. |
| `relevant_relationships` | no | Ground-truth relationship names. Optional. |
| `notes` | no | Free text. Never used in scoring. |

A query is **scorable** when it has `video_id` + a timestamp, *or*
`ground_truth_timestamps`, *or* `relevant_ids`.

## Choosing `tolerance_seconds`

The default of `2.0` s is deliberately conservative: it must be narrower than
the spacing between genuinely different shots, or unrelated frames count as
hits. Use it unless your video has very fast cuts.

## How to build a real annotated set

1. Index your video through the normal API (`POST /api/v1/videos`) and wait for
   `status = "ready"`.
2. Get the `video_id` from `GET /api/v1/videos`.
3. For each query, use `build_labels.py` to print the candidate frames the
   retriever actually surfaced:

   ```bash
   python -m experiments.evaluation.build_labels \
       --video-id <VIDEO_UUID> \
       --queries experiments/datasets/evaluation.csv \
       --candidates 20
   ```

4. **Watch the candidate frames and record the second at which the query is
   genuinely answered.** This is the human step, and it cannot be skipped: the
   retriever's own suggestions are biased toward what the retriever already
   finds, so using them uncritically would grade the system on its own
   preferences.
5. Fill `ground_truth_timestamp` (and `video_id`) in `evaluation.csv`.
6. Re-run `evaluate_retrieval --dry-run` and confirm `num_scorable` matches the
   number of queries.

## Honest note on scale

Five annotated queries is enough to demonstrate the pipeline end to end and to
produce a real measurement, but it is far too few for the differences between
conditions A/B/C/D to be statistically meaningful. Any conclusion drawn from
five queries should be reported as a single observation, not a benchmark.
