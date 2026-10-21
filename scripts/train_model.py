#!/usr/bin/env python3
"""Run Temporal Retrieval Network training on cached features."""

import os
import yaml
import numpy as np

from mc.data.charades_sta import CharadesSTADataset
from mc.encoders.feature_cache import FeatureCache
from mc.train.trainer import TemporalNetTrainer


def train_temporal_net(config_path: str = "configs/train/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure model config is included
    if "model" not in config:
        with open("configs/model/temporal_net.yaml", "r", encoding="utf-8") as f:
            m_cfg = yaml.safe_load(f)
            config["model"] = m_cfg.get("model", {})

    train_ds = CharadesSTADataset.load_json("data/processed/train.json")
    val_ds = CharadesSTADataset.load_json("data/processed/val.json")
    fcache = FeatureCache(cache_dir="data/features", encoder_id="clip_b32")

    def prepare_samples(dataset):
        samples = []
        for r in dataset.records:
            if not fcache.has_video_features(r.video_id):
                continue
            feats, meta = fcache.load_video_features(r.video_id)
            timestamps = np.array(meta.get("timestamps", np.arange(len(feats))))
            duration = float(meta.get("duration", 30.0))

            q_emb = fcache.load_query_embedding(r.query)
            if q_emb is None:
                # Mock or zeros
                q_emb = np.random.randn(512).astype(np.float32)

            samples.append({
                "query_id": r.query_id,
                "video_id": r.video_id,
                "frames": feats.astype(np.float32),
                "query": q_emb.astype(np.float32),
                "start": r.start,
                "end": r.end,
                "duration": duration,
                "timestamps": timestamps
            })
        return samples

    train_samples = prepare_samples(train_ds)
    val_samples = prepare_samples(val_ds)

    print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    trainer = TemporalNetTrainer(config)
    metrics = trainer.fit(train_samples, val_samples)
    print(f"Training completed. Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    train_temporal_net()
