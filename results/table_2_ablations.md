# Table 2: Ablation Analysis across 3 Random Seeds (A1 - A8)

| ID | Ablation Description | Val R@1 @ 0.5 (Mean ± Std) | Delta vs Full | 95% Bootstrap CI | Sig. (p < 0.05) |
|---|---|---|---|---|---|
| Full | Full Memory Compiler System | 0.435 ± 0.006 | Reference | [-] | - |
| A1 | No temporal Transformer (MLP only) | 0.312 ± 0.009 | -0.123 | [-0.145, -0.101] | Yes (p < 0.001) |
| A2 | No query-conditioned fusion (dot-product only) | 0.354 ± 0.008 | -0.081 | [-0.098, -0.063] | Yes (p < 0.001) |
| A3 | No boundary head (relevance threshold only) | 0.348 ± 0.007 | -0.087 | [-0.106, -0.069] | Yes (p < 0.001) |
| A4 | No hard-negative ranking loss | 0.382 ± 0.008 | -0.053 | [-0.071, -0.035] | Yes (p < 0.001) |
| A5 | No LoRA encoder adaptation | 0.395 ± 0.007 | -0.040 | [-0.058, -0.022] | Yes (p < 0.001) |
| A6 | No object features (CLIP visual only) | 0.407 ± 0.006 | -0.028 | [-0.044, -0.012] | Yes (p = 0.003) |
| A7 | No hierarchical retrieval (Flat frame scan) | 0.431 ± 0.007 | -0.004 | [-0.018, +0.010] | No (p = 0.58) |
| A8a | Temporal sampling at 0.5 fps (sparse) | 0.378 ± 0.008 | -0.057 | [-0.076, -0.039] | Yes (p < 0.001) |
| A8b | Temporal sampling at 2.0 fps (dense) | 0.442 ± 0.005 | +0.007 | [-0.006, +0.021] | No (p = 0.32) |
