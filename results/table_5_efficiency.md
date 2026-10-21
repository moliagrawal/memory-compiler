# Table 5: Systems & Efficiency Profiling

### A. Indexing Structure Comparison (10,000 Frames, D=512)
| Index Type | Indexing Time (ms) | Storage Size (KB) | Memory Compression | p50 Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|---|
| Exact Flat (IP) | 1.8 ms | 20,000 KB | 1.0x (Reference) | 0.42 ms | 0.78 ms |
| IVF-PQ (nlist=16, m=16) | 14.5 ms | 185 KB | **108.1x reduction** | 0.18 ms | 0.35 ms |
| HNSW (M=32, ef=64) | 48.2 ms | 22,560 KB | 0.88x | **0.08 ms** | **0.15 ms** |

### B. Second-Stage Reranker Budget (K) vs Retrieval Recall
| Reranker Top-K Budget | R@1 @ 0.5 | R@5 @ 0.5 | Rerank Stage Latency (ms) | Total Pipeline p95 Latency (ms) |
|---|---|---|---|---|
| K = 5 | 0.412 | 0.612 | 1.8 ms | 10.3 ms |
| K = 10 (Default) | 0.435 | 0.648 | 3.2 ms | 14.2 ms |
| K = 20 | 0.439 | 0.655 | 6.5 ms | 18.5 ms |
| K = 50 | 0.441 | 0.658 | 15.8 ms | 28.4 ms |

### C. Hierarchical Coarse-to-Fine Speedup
| Retrieval Mode | Evaluated Frames / 30m Video | Candidate Reduction | Search Latency (ms) | Speedup Factor |
|---|---|---|---|---|
| Exhaustive Dense Flat Search | 1,800 frames | 0.0% | 34.2 ms | 1.0x |
| Hierarchical Search (16s chunks) | 320 frames | **82.2%** | **6.8 ms** | **5.0x speedup** |
| Adaptive Sampling (0.5 fps + zoom) | 450 frames | **75.0%** | **8.1 ms** | **4.2x speedup** |
