# Memory Compiler: Full Implementation Plan & Execution Architecture

## Executive Overview
Memory Compiler is a low-latency, resource-constrained temporal video grounding and retrieval system designed to run on WSL2 with a 4 GB VRAM GPU (RTX 3050 Laptop).
It extracts and caches multi-rate multimodal representations, builds hierarchical indexing and temporal segment proposals, refines predictions using a query-conditioned temporal transformer and cross-attention reranker, and serves queries via a responsive FastAPI backend and React+Vite user interface.

## System Architecture & Flow
```
Video Input (mp4/mkv/webm)
    │
    ▼
[ Decord Fixed-FPS Sampler (0.5 / 1.0 / 2.0 fps) ]
    │
    ▼
[ Feature Extraction: CLIP ViT-B/32 fp16 + YOLO Object Embeddings ]
    │
    ├──> Feature Cache: `data/features/{encoder_id}/{video_id}.npy` [T, D]
    │
    ▼
[ Hierarchical & Adaptive Search / Baselines / Temporal Retrieval Network ]
    │
    ├── B0: Random Interval
    ├── B1: Frame-CLIP Similarity (expanded top frame)
    ├── B2: Sliding Window Mean Similarity
    ├── B3: Temporal Retrieval Network (Transformer + Multi-head proposals)
    └── B4: Temporal Retrieval Network + Boundary-Aware Reranker (B3 + Rerank)
    │
    ▼
[ Candidate Generation & NMS Merge ] -> `preds.jsonl`
    │
    ▼
[ Evaluation & Statistical Significance: tIoU @ {0.3, 0.5, 0.7}, mIoU, Bootstrap CI ]
    │
    ▼
[ Interactive Interface: FastAPI + React/Vite with Live Model Comparison ]
```

## Implementation Phases
- **Phase 1: Data & Evaluation Core**
  - Dataset parser (`charades_sta.py`), 10% seeded video-level validation split.
  - Deterministic video sampler (`sampler.py`).
  - Metrics engine (`metrics.py`): tIoU, R@K, mIoU, mAP, query type breakdowns.
  - Paired bootstrap statistical significance testing (`bootstrap.py`).
  - Query classification (`query_types.py`).
- **Phase 2: Frozen Baselines & Demo**
  - Feature extraction pipeline with resume support (`extract_features.py`).
  - Baseline algorithms B0, B1, B2 in `baselines.py`.
  - Baseline runner generating first quantitative tables (`run_baselines.py`).
- **Phase 3: Temporal Retrieval Network (B3)**
  - PyTorch Temporal Retrieval Transformer (`temporal_net.py`).
  - Multi-task heads: relevance, start, end boundary logits (`heads.py`).
  - Balanced multi-objective loss with label smoothing & ranking margin (`losses.py`).
  - Peak candidate merging and 1D temporal NMS (`merge.py`).
  - Training loop with AMP, gradient clipping, early stopping (`trainer.py`).
- **Phase 4: Hard Negatives & Query-Aware Reranker (B4)**
  - Synthesizer for action reversal, recipient swap, and temporal inversion (`hard_negatives.py`).
  - Second-stage cross-encoder reranker over top candidates (`rerank.py`).
- **Phase 5: Encoder Adaptation with LoRA**
  - Parameter-efficient LoRA targeting CLIP text tower and final 4 vision blocks (`lora.py`).
  - Contrastive InfoNCE tuning with micro-batch 8 and gradient accumulation.
- **Phase 6: Object-Aware Representations**
  - YOLOv8 frame object detection pooled into class embeddings (`object_feats.py`).
- **Phase 7: Challenge Benchmark Set**
  - Challenge set with 4 query categories and Cohen's Kappa agreement (`challenge_set.py`).
  - Linguistic paraphrase robustness evaluation (`paraphrase.py`).
- **Phase 8: Efficiency, FAISS & System Optimizations**
  - FAISS indexer with Flat, IVF-PQ, and HNSW benchmarks (`faiss_index.py`).
  - Coarse-to-fine chunk retrieval (`hierarchical.py`).
  - Sparse 0.5 fps adaptive sampling with high-res zoom (`adaptive_sampling.py`).
  - Latency, memory, and throughput profiler (`efficiency.py`).
- **Phase 9: Comprehensive Ablations (A1–A8) & Result Generation**
  - Automated evaluation across 8 ablations and 3 random seeds.
  - Table auto-generator for all 5 required tables (`make_tables.py`).
- **Phase 10: App Polish, Verification & Reproduction**
  - FastAPI server with search, preview, and comparison endpoints (`api/main.py`).
  - React + Vite web dashboard (`app/`).
  - Reproduction harness (`Makefile`, `reproduce.sh`).
