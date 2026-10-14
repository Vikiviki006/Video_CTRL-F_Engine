# VideoCtrl-F Evaluation Summary

_Generated 2026-10-14 06:00:38 UTC by `experiments.evaluation.reporting.build_summary`._

Every table on this page is rendered from the same records that were written to the CSV files in this directory. No value here is typed by hand. A cell reading **N/A** means the measurement was not taken - it is not a zero.

## Run conditions

### Run metadata

| dataset | model | model_checkpoint | top_k | graph_enabled | graph_depth | depth_sweep | num_queries | num_scorable | metrics_status | generated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments\datasets\evaluation.csv | siglip2 | google/siglip2-so400m-patch14-384 | `10` | true | `2` | [0, 1, 2, 3] | `12` | `12` | measured | 2026-10-14 05:59:55 UTC |


## Retrieval metrics

### Vector vs. vector + graph

| condition | description | metrics_status | metrics_reason | precision@1 | recall@1 | ndcg@1 | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | hitrate@1 | hitrate@5 | hitrate@10 | mrr | map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | vector + graph + temporal (production) | measured | *N/A* | `0.083333` | `0.006944` | `0.083333` | `0.1` | `0.055556` | `0.095918` | `0.091667` | `0.104167` | `0.102205` | `0.045833` | `0.104167` | `0.09379` | `0.083333` | `0.166667` | `0.166667` | `0.1` | `0.073611` |


## Ablation (A/B/C/D)

### Ablation conditions

| condition | description | vector_weight | graph_weight | temporal_weight | confidence_weight | mode | metrics_status | num_queries | precision@1 | recall@1 | ndcg@1 | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | hitrate@1 | hitrate@5 | hitrate@10 | mrr | map | num_relevant_items | latency_count | latency_avg_ms | latency_median_ms | latency_p95_ms | latency_p99_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A: vector only | `1.0` | `0.0` | `0.0` | `0.0` | live | measured | `12.0` | `0.166667` | `0.027778` | `0.166667` | `0.116667` | `0.076389` | `0.13639` | `0.091667` | `0.104167` | `0.131088` | `0.045833` | `0.104167` | `0.122674` | `0.166667` | `0.166667` | `0.166667` | `0.166667` | `0.104167` | `54.0` | `12` | `2479.2528` | `72.37855` | `13083.16738` | `25727.133396` |
| B | B: vector + temporal | `0.65` | `0.0` | `0.35` | `0.0` | live | measured | `12.0` | `0.083333` | `0.006944` | `0.083333` | `0.1` | `0.055556` | `0.095918` | `0.091667` | `0.104167` | `0.102205` | `0.045833` | `0.104167` | `0.09379` | `0.083333` | `0.166667` | `0.166667` | `0.1` | `0.073611` | `54.0` | `12` | `57.543258` | `51.25685` | `82.811975` | `94.543915` |
| C | C: vector + graph | `0.8` | `0.2` | `0.0` | `0.0` | live | measured | `12.0` | `0.083333` | `0.006944` | `0.083333` | `0.116667` | `0.076389` | `0.120124` | `0.091667` | `0.104167` | `0.114823` | `0.045833` | `0.104167` | `0.106408` | `0.083333` | `0.166667` | `0.166667` | `0.125` | `0.086806` | `54.0` | `12` | `97.910808` | `79.14425` | `194.40631` | `287.811182` |
| D | D: vector + graph + temporal | `0.65` | `0.2` | `0.15` | `0.0` | live | measured | `12.0` | `0.083333` | `0.006944` | `0.083333` | `0.1` | `0.055556` | `0.095918` | `0.091667` | `0.104167` | `0.102205` | `0.045833` | `0.104167` | `0.09379` | `0.083333` | `0.166667` | `0.166667` | `0.1` | `0.073611` | `54.0` | `12` | `82.9208` | `72.66975` | `130.858595` | `142.875479` |


## Graph depth sweep

### Graph depth

| depth | queries | nodes_reached_avg | edges_traversed_avg | expansion_latency_ms_avg | expansion_latency_ms_p95 | depth_limit_hit_queries | node_limit_hit_queries | metrics_status | metrics_reason | recall@10 | precision@10 | ndcg@10 | mrr | map | latency_avg_ms | latency_p95_ms | nodes_reached_max | edges_traversed_max | edge_type_totals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2` | `11` | `40.0` | `257.818182` | `15.114545` | `24.335` | `11` | `0` | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| `0` | `12` | `0.0` | `0.0` | `0.0` | *N/A* | *N/A* | *N/A* | measured | *N/A* | `0.104167` | `0.091667` | `0.131088` | `0.166667` | `0.104167` | `50.899367` | `61.26376` | *N/A* | *N/A* | *N/A* |
| `1` | `11` | `25.545455` | `114.818182` | `7.126364` | `11.39` | `11` | `0` | measured | *N/A* | `0.104167` | `0.091667` | `0.102205` | `0.1` | `0.073611` | `61.957075` | `76.00563` | `39` | `171` | {} |
| `2` | `11` | `40.0` | `257.818182` | `18.907273` | `27.84` | `11` | `0` | measured | *N/A* | `0.104167` | `0.091667` | `0.102205` | `0.1` | `0.073611` | `79.849508` | `98.779175` | `40` | `317` | {} |
| `3` | `11` | `40.0` | `257.818182` | `13.776364` | `18.665` | `11` | `0` | measured | *N/A* | `0.104167` | `0.091667` | `0.102205` | `0.1` | `0.073611` | `71.020217` | `77.92045` | `40` | `317` | {} |


## Latency by pipeline stage

### Latency

| condition | stage | samples | avg_ms | median_ms | p95_ms | p99_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | embedding | `12` | `2426.550833` | `51.475` | `12854.6485` | `25307.3217` | `44.51` | `28420.49` |
| A | vector_retrieval | `12` | `52.013333` | `14.49` | `228.0215` | `419.2763` | `3.5` | `467.09` |
| A | graph_expansion | `11` | `0.001818` | `0.0` | `0.01` | `0.01` | `0.0` | `0.01` |
| A | graph_reranking | `11` | `0.209091` | `0.18` | `0.43` | `0.55` | `0.08` | `0.58` |
| A | total | `12` | `2479.2528` | `72.37855` | `13083.16738` | `25727.133396` | `48.329` | `28888.1249` |
| B | embedding | `12` | `46.810833` | `44.74` | `57.7695` | `61.1619` | `41.67` | `62.01` |
| B | vector_retrieval | `12` | `10.191667` | `4.145` | `39.194` | `51.3028` | `3.02` | `54.33` |
| B | graph_expansion | `11` | `0.002727` | `0.0` | `0.01` | `0.01` | `0.0` | `0.01` |
| B | graph_reranking | `11` | `0.216364` | `0.09` | `0.68` | `0.96` | `0.05` | `1.03` |
| B | total | `12` | `57.543258` | `51.25685` | `82.811975` | `94.543915` | `45.303` | `97.4769` |
| C | embedding | `12` | `43.219167` | `42.725` | `46.29` | `47.346` | `41.45` | `47.61` |
| C | vector_retrieval | `12` | `7.185833` | `3.87` | `23.6185` | `25.7877` | `3.08` | `26.33` |
| C | graph_expansion | `11` | `38.993636` | `14.93` | `145.695` | `232.427` | `11.96` | `254.11` |
| C | graph_reranking | `11` | `12.125455` | `11.18` | `19.75` | `24.502` | `6.33` | `25.69` |
| C | total | `12` | `97.910808` | `79.14425` | `194.40631` | `287.811182` | `45.9115` | `311.1624` |
| D | embedding | `12` | `44.311667` | `43.3` | `49.2815` | `50.1043` | `41.57` | `50.31` |
| D | vector_retrieval | `12` | `14.446667` | `4.485` | `48.178` | `59.1956` | `3.31` | `61.95` |
| D | graph_expansion | `11` | `15.114545` | `12.72` | `24.335` | `26.299` | `10.51` | `26.79` |
| D | graph_reranking | `11` | `10.586364` | `10.37` | `14.5` | `14.74` | `6.83` | `14.8` |
| D | total | `12` | `82.9208` | `72.66975` | `130.858595` | `142.875479` | `48.2521` | `145.8797` |


## Indexing metrics

### Indexing

| video_id | filename | duration_seconds | fps | original_frames | retained_frames | frame_reduction_percent | graph_nodes | graph_edges | raw_video_bytes | storage_overhead_ratio | derived_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ff39e300-0788-450d-8100-1ab38c060409 | e11a89c8-37bf-41b6-be8b-bceb9b0096da.mp4 | `103.6` | `30.0` | `3108` | `38` | `98.777349` | `40` | `270` | *N/A* | *N/A* | `175104` |


## Model comparison

### SigLIP vs SigLIP-2

| model | checkpoint | metrics_status | metrics_reason |
| --- | --- | --- | --- |
| siglip | google/siglip-so400m-patch14-384 | *N/A* | not evaluated in this run. The comparison requires both checkpoints to be indexed over identical frames and scored on the same queries; run the pipeline once per MODEL_NAME and merge the results. |
| siglip2 | google/siglip2-so400m-patch14-384 | *N/A* | not evaluated in this run. The comparison requires both checkpoints to be indexed over identical frames and scored on the same queries; run the pipeline once per MODEL_NAME and merge the results. |


## Files

- `retrieval_metrics.csv` - present
- `retrieval_metrics.json` - present
- `model_comparison.csv` - present
- `model_comparison.json` - present
- `ablation_results.csv` - present
- `ablation_results.json` - present
- `latency_results.csv` - present
- `indexing_results.csv` - present
- `graph_results.csv` - present
