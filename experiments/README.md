# VideoCtrl-F experiments

Research code for evaluating the graph-enhanced retrieval architecture against a
vector-only baseline. Everything here runs against **real data only**: a real
indexed video, a real embedding model, a real database. Nothing in this
directory invents a metric value.

---

## Layout

```
experiments/
├── __init__.py
├── datasets/                      # Annotation CSVs
│   ├── README.md                  # Column definitions + how to build a labelled set
│   ├── evaluation.csv             # The working dataset
│   └── evaluation_dataset.template.csv
├── evaluation/
│   ├── metrics.py                 # Recall@K, Precision@K, MRR, MAP, NDCG@K, HitRate@K
│   ├── graph_metrics.py           # Expanded-node counts, depth effects, graph sizes
│   ├── ranking.py                 # Ablation configurations A/B/C/D, depth sweep
│   ├── dataset.py                 # Dataset loading + strict ground-truth validation
│   ├── reporting.py               # CSV / JSON / generated Markdown summary
│   ├── evaluate_retrieval.py      # CLI entrypoint
│   ├── build_labels.py            # Annotation helper (candidate-frame lister)
│   └── __init__.py
├── visualization/
│   ├── _common.py                 # Shared CSV reading, N/A handling, plot helpers
│   ├── plot_recall.py             # Figure 1
│   ├── plot_precision.py          # Figure 2
│   ├── plot_mrr.py                # Figure 3
│   ├── plot_ndcg.py               # Figure 4
│   ├── plot_latency.py            # Figure 5
│   ├── plot_frame_reduction.py    # Figure 6
│   ├── plot_comparison.py         # Figure 7
│   ├── plot_ablation.py           # Figure 8
│   └── __init__.py
└── results/                       # Generated. Git-ignored except .gitkeep + README.md
```

`evaluate.py`, `compare_siglip.py` and `plot_results.py` at this level are
**pre-existing scaffolding** and are left byte-identical:

* `evaluate.py` scores *supplied prediction files* for a baseline-vs-proposed
  comparison. It never runs the retrieval system, and `backend/tests/
  test_evaluate.py` depends on its exact contract, so it was not repurposed.
* `compare_siglip.py` and `plot_results.py` only print instructions. The real
  comparison and plotting now live in `evaluation/` and `visualization/`.

---

## Quick start

```bash
# 1. Always start here. Validates annotations, reports what is scorable,
#    touches neither the database nor the model.
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --dry-run

# 2. The full study.
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --model siglip2 \
    --top-k 10 \
    --graph --graph-depth 2 \
    --ablation --depth-sweep --index-metrics \
    --output experiments/results/

# 3. Figures, from the CSVs the previous step produced.
make plots
```

`make plots` runs all eight plot scripts. Each accepts `--results-dir` so a
figure can be rendered from an alternative results directory.

---

## Outputs

| File | Contents |
|---|---|
| `results/retrieval_metrics.csv` / `.json` | Vector vs. vector+graph, all metrics at K = 1, 5, 10, 20 |
| `results/ablation_results.csv` / `.json` | Conditions A/B/C/D |
| `results/latency_results.csv` | avg / median / P95 / P99 per pipeline stage |
| `results/indexing_results.csv` | Frame counts, frame reduction %, graph sizes, storage |
| `results/graph_results.csv` | Expanded nodes, traversed edges, depth sweep |
| `results/model_comparison.csv` / `.json` | SigLIP vs SigLIP-2 |
| `results/SUMMARY.md` | **Generated** from the same rows as the CSVs |
| `results/figures/*.png` | The eight figures |

A CSV whose experiment produced nothing is not left empty: it contains a single
`# N/A - no measured records were produced for this experiment.` line, which
`reporting.read_csv` recognises. A real zero and a missing measurement are
therefore always distinguishable.

---

## Ablation conditions

| Key | Description | vector | graph | temporal | confidence |
|---|---|---|---|---|---|
| `A` | vector only (baseline) | 1.00 | 0.00 | 0.00 | 0.00 |
| `B` | vector + temporal | 0.65 | 0.00 | 0.35 | 0.00 |
| `C` | vector + graph | 0.80 | 0.20 | 0.00 | 0.00 |
| `D` | vector + graph + temporal | 0.65 | 0.20 | 0.15 | 0.00 |

These are `RankingConfig` values defined once, in
`backend/app/services/graph_ranking_service.py`, and re-exported by
`evaluation/ranking.py`. There is no second implementation of the scoring
formula, so the published ablation necessarily describes the code that runs in
production.

`confidence_weight` is `0` in every condition on purpose. With no extractor
registered, every node confidence is exactly `1.0`, so the term is
ranking-neutral. Including it would add a constant to every candidate and make
the numbers *look* like the confidence signal is contributing when it
demonstrably is not. `ranking.confidence_term_diagnostic` reports this as a
measured fact instead.

---

## Current status: no measured values yet

| What | Status |
|---|---|
| `graph_nodes` / `graph_edges` schema | **Done** — Alembic revision `0001_graph_layer`, verified via `alembic upgrade head --sql` |
| Traversal boundedness and batching | **Done** — 43 unit tests, including a query-count assertion |
| Scoring, ranking, tie-breaking | **Done** — 42 unit tests |
| Metrics implementation | **Done** — 47 unit tests against hand-computed values |
| Figures render from CSV | **Done** — 8 scripts exercised end-to-end in `test_reporting_and_plots.py` |
| **Recall / Precision / MRR / NDCG values** | **N/A — no ground truth** |
| **Latency per stage** | **N/A — no indexed video** |
| **Frame reduction %** | **N/A — no indexed video** |
| **SigLIP vs SigLIP-2** | **N/A — only one checkpoint has ever been indexed** |

The two blocking environment problems are documented in `../scripts/README.md`:

1. **pgvector is not installed** on the local PostgreSQL 18, so
   `video_frames.embedding vector(1152)` cannot be created and no video can be
   indexed.
2. **No ground truth has been annotated**, so retrieval metrics are undefined.

Both are fixable without any code change - see the two linked documents.

---

## Honest limitations

* **Five queries is not a benchmark.** Once annotations exist, the measured
  numbers will come from a very small sample. Any difference between
  conditions should be reported as a single observation, with the caveat stated.
* **One video is one video.** Frame-reduction and storage figures from a single
  clip do not generalise to a corpus.
* **The graph currently has no concept layer.** With no extractor registered,
  the graph contains only `VIDEO`, `SCENE` and `FRAME` nodes, and the
  "Concept Relations" branch of the architecture diagram is empty. This limits
  how much the graph term can possibly contribute, and the ablation is
  precisely the measurement that would reveal it.
