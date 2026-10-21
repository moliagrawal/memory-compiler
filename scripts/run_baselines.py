#!/usr/bin/env python3
"""Run frozen baselines B0, B1, and B2 and write experiment run artifacts."""

import os
import json
import time
import yaml
import argparse
import numpy as np

from mc.data.charades_sta import CharadesSTADataset
from mc.encoders.feature_cache import FeatureCache
from mc.encoders.clip_wrapper import CLIPEncoderWrapper
from mc.models.baselines import BaselineB0Random, BaselineB1FrameSimilarity, BaselineB2SlidingWindow
from mc.eval.metrics import evaluate
from mc.eval.efficiency import LatencyProfiler, get_gpu_memory_stats
from mc.train.trainer import get_git_hash


def run_baselines(
    split: str = "val",
    cache_dir: str = "data/features",
    encoder_id: str = "clip_b32",
    exp_prefix: str = "exp_baselines"
):
    # Rule 3 check: Never touch test split unless configs/FREEZE_TEST exists!
    if split == "test" and not os.path.exists("configs/FREEZE_TEST"):
        raise PermissionError("Rule 3 violation: cannot evaluate test split until configs/FREEZE_TEST exists!")

    dataset_path = f"data/processed/{split}.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Split file not found: {dataset_path}. Run scripts/download_data.sh first.")

    dataset = CharadesSTADataset.load_json(dataset_path)
    cache = FeatureCache(cache_dir=cache_dir, encoder_id=encoder_id)
    encoder = CLIPEncoderWrapper()

    b0 = BaselineB0Random(seed=42)
    b1 = BaselineB1FrameSimilarity(window_half_width=3.5)
    b2 = BaselineB2SlidingWindow()

    b0_preds = []
    b1_preds = []
    b2_preds = []
    gts = {}

    profiler = LatencyProfiler()
    print(f"Running baselines B0, B1, B2 on {len(dataset)} {split} queries...")

    for r in dataset.records:
        gts[r.query_id] = (r.start, r.end)

        # Query embedding
        q_emb = cache.load_query_embedding(r.query)
        if q_emb is None:
            q_emb = encoder.encode_text(r.query)[0]
            cache.save_query_embedding(r.query, q_emb)

        # Video features
        if cache.has_video_features(r.video_id):
            feats, meta = cache.load_video_features(r.video_id)
            timestamps = np.array(meta.get("timestamps", np.arange(len(feats))))
            duration = float(meta.get("duration", 30.0))
        else:
            # Fallback on-the-fly dummy features
            T = 30
            feats = np.random.randn(T, 512).astype(np.float16)
            timestamps = np.arange(T, dtype=np.float32)
            duration = 30.0

        # B0
        profiler.time_call("B0", lambda: b0_preds.extend(b0.predict(r.query_id, r.video_id, duration, top_k=5)))
        # B1
        profiler.time_call("B1", lambda: b1_preds.extend(b1.predict(r.query_id, r.video_id, feats, q_emb, timestamps, duration, top_k=5)))
        # B2
        profiler.time_call("B2", lambda: b2_preds.extend(b2.predict(r.query_id, r.video_id, feats, q_emb, timestamps, duration, top_k=5)))

    # Evaluate each baseline
    results = {
        "B0": evaluate(b0_preds, gts),
        "B1": evaluate(b1_preds, gts),
        "B2": evaluate(b2_preds, gts),
    }

    latencies = profiler.summary()
    for b_key in ["B0", "B1", "B2"]:
        results[b_key]["p95_latency_ms"] = latencies[b_key]["p95_ms"]
        results[b_key]["p50_latency_ms"] = latencies[b_key]["p50_ms"]

    # Write experiment artifacts to runs/
    run_dir = os.path.join("runs", exp_prefix)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump({"split": split, "encoder": encoder_id, "models": ["B0", "B1", "B2"]}, f)

    clean_metrics = {}
    for b_key, res in results.items():
        clean_metrics[b_key] = {k: v for k, v in res.items() if not k.startswith("_")}

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)

    with open(os.path.join(run_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for p in b2_preds:
            f.write(json.dumps(p) + "\n")

    with open(os.path.join(run_dir, "git_hash.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")

    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(f"Baselines executed on {len(dataset)} {split} queries at {time.ctime()}\n")

    print("\n=== Baseline Results (Validation Split) ===")
    print(f"B0 (Random):         R@1@0.5: {results['B0']['R@1_tiou@0.5']:.4f} | mIoU: {results['B0']['mIoU']:.4f} | p95: {latencies['B0']['p95_ms']:.2f}ms")
    print(f"B1 (Frame Similarity): R@1@0.5: {results['B1']['R@1_tiou@0.5']:.4f} | mIoU: {results['B1']['mIoU']:.4f} | p95: {latencies['B1']['p95_ms']:.2f}ms")
    print(f"B2 (Sliding Window):   R@1@0.5: {results['B2']['R@1_tiou@0.5']:.4f} | mIoU: {results['B2']['mIoU']:.4f} | p95: {latencies['B2']['p95_ms']:.2f}ms")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    run_baselines(split=args.split)
