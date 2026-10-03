# Generated results

Everything in this directory except `.gitkeep` and this README is **generated**
and git-ignored. Nothing here is hand-written.

Regenerate with:

```bash
python -m experiments.evaluation.evaluate_retrieval \
    --dataset experiments/datasets/evaluation.csv \
    --model siglip2 --top-k 10 \
    --graph --graph-depth 2 \
    --ablation --depth-sweep --index-metrics \
    --output experiments/results/

make plots
```

## What is currently here

| File | Present | State |
|---|---|---|
| `retrieval_metrics.csv` / `.json` | no | Not produced yet - no ground truth, no indexed video |
| `ablation_results.csv` / `.json` | no | Not produced yet - same reason |
| `latency_results.csv` | no | Not produced yet - no indexed video |
| `indexing_results.csv` | no | Not produced yet - no indexed video |
| `graph_results.csv` | no | Not produced yet - no indexed video |
| `model_comparison.csv` / `.json` | no | Not produced yet - only one checkpoint indexed |
| `SUMMARY.md` | no | Not produced yet |
| `figures/*.png` | yes | Generated. All eight currently carry an explicit **N/A** caption, because no measurement exists to plot |

The eight figures exist and render, but each one states in its own body that it
has nothing to draw and why. That is deliberate: an empty-looking chart is
indistinguishable from a real measurement of nothing, so the figure says so in
words instead.

## Reading the files when they do appear

* `N/A` means **not measured**. `0.0` means **measured, and it was zero**.
  They are never used interchangeably - see `reporting.normalise`.
* `metrics_status` and `metrics_reason` on every row explain, in words, why a
  number is present or absent.
* `SUMMARY.md` is rendered from the same row objects as the CSVs, so it cannot
  disagree with them.
