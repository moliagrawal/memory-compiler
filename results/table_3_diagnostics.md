# Table 3: Fine-Grained Diagnostics & Robustness Evaluation

### A. Grounding Accuracy by Query Taxonomy
| Query Category | Query Count (%) | B2 Baseline R@1 @ 0.5 | Full System R@1 @ 0.5 | Relative Gain |
|---|---|---|---|---|
| Object / Attribute | 24.2% | 0.245 | 0.468 | +91.0% |
| Single Action | 29.5% | 0.278 | 0.482 | +73.4% |
| Subject-Object Interaction | 31.8% | 0.214 | 0.426 | +99.1% |
| Multi-Event / Temporal Order | 14.5% | 0.142 | 0.334 | +135.2% |

### B. Linguistic Paraphrase Invariance
| Query Version | R@1 @ 0.3 | R@1 @ 0.5 | mIoU | Agreement Rate with Original (tIoU >= 0.5) |
|---|---|---|---|---|
| Original Annotations | 0.554 | 0.435 | 0.447 | 100.0% |
| Paraphrased Queries | 0.538 | 0.419 | 0.431 | 93.8% |

### C. Hard-Negative Discrimination Accuracy
| Negative Type | Negative Source | Identification Accuracy (Score_pos > Score_neg) |
|---|---|---|
| Action Reversal | e.g. open vs close, pick up vs put down | 88.4% |
| Object Swap | e.g. cup vs glass, door vs window | 84.6% |
| Chronological Order Inversion | e.g. first X then Y vs first Y then X | 81.2% |
| Same-Video Disjoint Segment | Temporal offset with tIoU < 0.1 | 94.7% |
