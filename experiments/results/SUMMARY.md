# VideoCtrl-F Evaluation Summary

_Generated 2026-10-06 07:52:23 UTC by `experiments.evaluation.reporting.build_summary`._

Every table on this page is rendered from the same records that were written to the CSV files in this directory. No value here is typed by hand. A cell reading **N/A** means the measurement was not taken - it is not a zero.

## Run conditions

### Run metadata

| dataset | model | model_checkpoint | top_k | min_score | graph_enabled | graph_depth | depth_sweep | num_queries | num_scorable | metrics_status | generated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments\datasets\evaluation.csv | siglip2 | google/siglip2-so400m-patch14-384 | `10` | `-1.0` | true | `2` | [0, 1, 2, 3] | `25` | `25` | measured | 2026-10-06 07:51:12 UTC |


## Retrieval metrics

### Vector vs. vector + graph

| condition | description | metrics_status | metrics_reason | precision@1 | recall@1 | ndcg@1 | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | hitrate@1 | hitrate@5 | hitrate@10 | mrr | map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | vector + graph + temporal (production) | measured | *N/A* | `0.24` | `0.116111` | `0.24` | `0.216` | `0.393889` | `0.355544` | `0.196` | `0.679444` | `0.446301` | `0.098` | `0.679444` | `0.446301` | `0.24` | `0.52` | `0.72` | `0.37` | `0.355069` |


## Ablation (A/B/C/D)

### Ablation conditions

| condition | description | vector_weight | graph_weight | temporal_weight | confidence_weight | mode | metrics_status | num_queries | precision@1 | recall@1 | ndcg@1 | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 | hitrate@1 | hitrate@5 | hitrate@10 | mrr | map | num_relevant_items | latency_count | latency_avg_ms | latency_median_ms | latency_p95_ms | latency_p99_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A: vector only | `1.0` | `0.0` | `0.0` | `0.0` | live | measured | `25.0` | `0.2` | `0.109444` | `0.2` | `0.208` | `0.407222` | `0.352718` | `0.192` | `0.666111` | `0.437879` | `0.096` | `0.666111` | `0.437879` | `0.2` | `0.52` | `0.72` | `0.351667` | `0.34999` | `63.0` | `25` | `202.338644` | `184.4395` | `220.39598` | `432.502404` |
| B | B: vector + temporal | `0.65` | `0.0` | `0.35` | `0.0` | live | measured | `25.0` | `0.24` | `0.116111` | `0.24` | `0.216` | `0.393889` | `0.355544` | `0.196` | `0.679444` | `0.446301` | `0.098` | `0.679444` | `0.446301` | `0.24` | `0.52` | `0.72` | `0.37` | `0.355069` | `63.0` | `25` | `181.127236` | `170.6416` | `223.51726` | `264.567148` |
| C | C: vector + graph | `0.8` | `0.2` | `0.0` | `0.0` | live | measured | `25.0` | `0.28` | `0.136111` | `0.28` | `0.208` | `0.380556` | `0.360164` | `0.196` | `0.679444` | `0.459237` | `0.098` | `0.679444` | `0.459237` | `0.28` | `0.52` | `0.72` | `0.397` | `0.369847` | `63.0` | `25` | `214.440652` | `202.4336` | `299.72728` | `313.773648` |
| D | D: vector + graph + temporal | `0.65` | `0.2` | `0.15` | `0.0` | live | measured | `25.0` | `0.24` | `0.116111` | `0.24` | `0.216` | `0.393889` | `0.355544` | `0.196` | `0.679444` | `0.446301` | `0.098` | `0.679444` | `0.446301` | `0.24` | `0.52` | `0.72` | `0.37` | `0.355069` | `63.0` | `25` | `202.741212` | `202.7963` | `229.88084` | `231.989532` |


## Graph depth sweep

### Graph depth

| depth | queries | metrics_status | metrics_reason | recall@10 | precision@10 | ndcg@10 | mrr | map | latency_avg_ms | latency_p95_ms | nodes_reached_avg | edges_traversed_avg | expansion_latency_ms_avg | nodes_reached_max | edges_traversed_max | expansion_latency_ms_p95 | depth_limit_hit_queries | node_limit_hit_queries | edge_type_totals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0` | `25` | measured | *N/A* | `0.666111` | `0.192` | `0.437879` | `0.351667` | `0.34999` | `183.343336` | `227.44094` | `0.0` | `0.0` | `0.0` | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| `1` | `25` | measured | *N/A* | `0.679444` | `0.196` | `0.446301` | `0.37` | `0.355069` | `192.942592` | `228.9785` | `23.72` | `130.84` | `6.4248` | `39` | `171` | `10.124` | `25` | `0` | {} |
| `2` | `25` | measured | *N/A* | `0.679444` | `0.196` | `0.446301` | `0.37` | `0.355069` | `198.382516` | `236.07408` | `24.76` | `191.48` | `8.2856` | `40` | `311` | `12.832` | `25` | `0` | {} |
| `3` | `25` | measured | *N/A* | `0.679444` | `0.196` | `0.446301` | `0.37` | `0.355069` | `193.55598` | `224.20554` | `24.76` | `191.48` | `9.7536` | `40` | `311` | `18.276` | `25` | `0` | {} |


## Latency by pipeline stage

### Latency

| condition | stage | samples | avg_ms | median_ms | p95_ms | p99_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | embedding | `25` | `195.1952` | `179.91` | `216.574` | `384.8212` | `171.7` | `437.95` |
| A | vector_retrieval | `25` | `6.5776` | `4.18` | `6.868` | `47.142` | `3.21` | `59.85` |
| A | graph_expansion | `25` | `0.0012` | `0.0` | `0.008` | `0.0176` | `0.0` | `0.02` |
| A | graph_reranking | `25` | `0.2272` | `0.18` | `0.44` | `0.8956` | `0.05` | `1.03` |
| A | total | `25` | `202.338644` | `184.4395` | `220.39598` | `432.502404` | `175.7091` | `499.4667` |
| B | embedding | `25` | `176.8084` | `166.12` | `219.046` | `259.754` | `158.26` | `272.39` |
| B | vector_retrieval | `25` | `3.8192` | `3.79` | `4.238` | `4.4224` | `3.36` | `4.48` |
| B | graph_expansion | `25` | `0.0004` | `0.0` | `0.0` | `0.0076` | `0.0` | `0.01` |
| B | graph_reranking | `25` | `0.1892` | `0.17` | `0.406` | `0.4908` | `0.03` | `0.51` |
| B | total | `25` | `181.127236` | `170.6416` | `223.51726` | `264.567148` | `162.2995` | `277.2697` |
| C | embedding | `25` | `186.7436` | `178.49` | `210.624` | `273.8044` | `162.66` | `293.65` |
| C | vector_retrieval | `25` | `3.8264` | `3.78` | `4.336` | `4.6668` | `2.82` | `4.77` |
| C | graph_expansion | `25` | `14.7112` | `10.23` | `17.884` | `86.2896` | `4.84` | `107.88` |
| C | graph_reranking | `25` | `8.3824` | `7.55` | `19.868` | `19.9276` | `0.24` | `19.93` |
| C | total | `25` | `214.440652` | `202.4336` | `299.72728` | `313.773648` | `179.4893` | `313.83` |
| D | embedding | `25` | `179.7184` | `176.22` | `208.536` | `212.5076` | `161.64` | `212.99` |
| D | vector_retrieval | `25` | `4.1908` | `4.12` | `4.926` | `5.0844` | `3.35` | `5.13` |
| D | graph_expansion | `25` | `9.6696` | `9.02` | `14.88` | `16.4104` | `3.95` | `16.78` |
| D | graph_reranking | `25` | `8.3956` | `6.86` | `18.048` | `21.9916` | `0.25` | `23.23` |
| D | total | `25` | `202.741212` | `202.7963` | `229.88084` | `231.989532` | `175.5334` | `232.3005` |


## Indexing metrics

### Indexing

| video_id | filename | duration_seconds | fps | original_frames | retained_frames | frame_reduction_percent | graph_nodes | graph_edges | raw_video_bytes | storage_overhead_ratio | derived_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a19c1351-3459-4d1c-ad9b-d14e4808b368 | Screen Recording 2026-07-21 194357.mp4 | `82.1` | `30.0` | `2463` | `29` | `98.822574` | `31` | `204` | `72783848` | `0.009753` | `1993662` |
| 7778dd4b-c53c-4bc9-b12e-87be2a165444 | e11a89c8-37bf-41b6-be8b-bceb9b0096da.mp4 | `103.6` | `30.0` | `3108` | `38` | `98.777349` | `40` | `270` | `92123729` | `0.009753` | `1993662` |
| a518e791-3974-48cb-802d-2758c4c26e78 | flower.mp4 | `5.005` | `29.97003` | `150` | `2` | `98.666667` | `4` | `8` | *N/A* | `0.009753` | `1993662` |
| d3ed05a9-36db-472b-be10-79dd0d511656 | 5676150-hd_1920_1080_25fps.mp4 | `15.64` | `25.0` | `391` | `6` | `98.465473` | `9` | `39` | `7407222` | `0.009753` | `1993662` |
| bc665aa7-55a3-4022-97a9-de110830e44a | 10520391-uhd_2160_3840_30fps.mp4 | `8.066667` | `30.0` | `242` | `17` | `92.975207` | `19` | `118` | `11901180` | `0.009753` | `1993662` |
| 07795017-9760-4f12-8186-6a840558eeb1 | Screen Recording 2024-10-15 214753.mp4 | `48.5` | `30.0` | `1455` | `17` | `98.831615` | `19` | `129` | `13183435` | `0.009753` | `1993662` |
| be6c6a79-9e93-44d7-8cc5-2874ce9cd30b | 3002456-hd_1920_1080_25fps.mp4 | `10.64` | `25.0` | `266` | `21` | `92.105263` | `23` | `154` | `7007287` | `0.009753` | `1993662` |


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
