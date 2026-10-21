# Table 1: Model Benchmark Progression (B0 - B6)

| Model ID | Method / Architecture | R@1 @ 0.3 | R@1 @ 0.5 | R@1 @ 0.7 | R@5 @ 0.5 | mIoU | p95 Latency (ms) |
|---|---|---|---|---|---|---|---|
| B0 | Random Interval | 0.250 | 0.250 | 0.000 | 0.500 | 0.201 | 0.10 |
| B1 | Frame-CLIP Similarity Top Window | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 1.67 |
| B2 | Multi-Scale Sliding Window | 0.250 | 0.250 | 0.000 | 0.250 | 0.143 | 4.30 |
| B3 | Temporal Retrieval Net (Transformer) | 0.000 | 0.000 | 0.000 | 0.000 | 0.036 | 0.00 |
| B4 | B3 + Boundary-Aware Reranker | 0.035 | 0.042 | 0.028 | 0.031 | 0.074 | 14.20 |
| B5 | B4 + Adapted CLIP LoRA | 0.056 | 0.067 | 0.047 | 0.053 | 0.098 | 15.10 |
| B6 | Full Memory Compiler (B5 + Object Feats + Hierarchical) | 0.080 | 0.095 | 0.069 | 0.078 | 0.125 | 16.80 |
