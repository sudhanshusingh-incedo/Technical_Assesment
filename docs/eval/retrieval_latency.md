# Retrieval evaluation

13 answerable questions with hand-labelled gold (document, page) sources. A retrieved chunk is a hit if it comes from the gold document and its page range covers the gold page. Multi-step questions are retrieved with the raw question here (the agent decomposes them at runtime), so these numbers are a lower bound for that category.

| Variant | Hit@1 | Hit@3 | Hit@6 | Recall@6 | MRR | Mean latency | p95 latency | Gate: answerable pass | Gate: unanswerable blocked |
|---|---|---|---|---|---|---|---|---|---|
| hybrid (RRF) | 0.69 | 0.92 | 1.00 | 0.95 | 0.82 | 38 ms | 48 ms | n/a | n/a |
| hybrid + rerank (20 cand.) | 0.77 | 0.92 | 1.00 | 0.95 | 0.85 | 11219 ms | 15031 ms | 100% | 100% |
| hybrid + rerank (20 cand., 700 chars) | 0.77 | 0.92 | 1.00 | 0.87 | 0.85 | 1882 ms | 2236 ms | 100% | 100% |

## Relevance-gate calibration (min_relevance = 0.1)

Best cross-encoder relevance (sigmoid) per question. Questions that must be declined should fall below the threshold; answerable ones above it.

| Id | Category | Should decline | Best relevance |
|---|---|---|---|
| O1 | out_of_scope | yes | 0.0000 |
| U2 | unanswerable | yes | 0.0006 |
| U1 | unanswerable | yes | 0.0029 |
| S4 | simple | no | 0.6825 |
| M3 | multi_step | no | 0.8584 |
| P1 | partial | no | 0.9483 |
| T1 | tool_use | no | 0.9717 |
| M2 | multi_step | no | 0.9879 |
| S1 | simple | no | 0.9894 |
| S2 | simple | no | 0.9913 |
| M4 | multi_step | no | 0.9940 |
| S5 | simple | no | 0.9973 |
| S3 | simple | no | 0.9979 |
| M1 | multi_step | no | 0.9993 |
| S6 | simple | no | 0.9997 |
