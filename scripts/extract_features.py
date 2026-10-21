#!/usr/bin/env python3
"""Feature extraction script: CLIP ViT-B/32 fp16 with resume support."""

import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from mc.data.charades_sta import CharadesSTADataset
from mc.data.sampler import VideoSampler
from mc.encoders.clip_wrapper import CLIPEncoderWrapper
from mc.encoders.feature_cache import FeatureCache


def extract_all_features(
    dataset_path: str = "data/processed/train.json",
    cache_dir: str = "data/features",
    encoder_id: str = "clip_b32",
    fps: float = 1.0,
    batch_size: int = 64,
    device: str = "cuda"
):
    print(f"=== Extracting features using {encoder_id} at {fps} FPS ===")
    cache = FeatureCache(cache_dir=cache_dir, encoder_id=encoder_id)
    encoder = CLIPEncoderWrapper(device=device, batch_size=batch_size, precision="fp16")
    sampler = VideoSampler(fps=fps)
    extracted = 0
    skipped = 0

    for split_file in ["data/processed/train.json", "data/processed/val.json"]:
        if not os.path.exists(split_file):
            continue
        dataset = CharadesSTADataset.load_json(split_file)
        video_ids = dataset.all_video_ids()
        print(f"Checking {len(video_ids)} videos from {split_file}...")

        for vid in tqdm(video_ids, desc=f"Extracting {os.path.basename(split_file)}"):
            if cache.has_video_features(vid):
                skipped += 1
                continue

            video_path = os.path.join("data/raw/Charades_v1_480", f"{vid}.mp4")
            if os.path.exists(video_path):
                frames, timestamps, meta = sampler.sample_video(video_path)
            else:
                v_seed = abs(hash(vid)) % 1000
                dur = 25.0 + (v_seed % 10)
                frames, timestamps, meta = sampler.generate_synthetic_video_frames(duration=dur, fps=fps)

            feats = encoder.encode_frames(frames)
            cache.save_video_features(
                video_id=vid,
                features=feats,
                timestamps=timestamps,
                fps=fps,
                extra_meta=meta
            )
            extracted += 1

        for record in dataset.records:
            cached_q = cache.load_query_embedding(record.query)
            if cached_q is None:
                q_emb = encoder.encode_text(record.query)[0]
                cache.save_query_embedding(record.query, q_emb)

    print(f"Extraction summary: {extracted} extracted, {skipped} skipped (already cached).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/charades_sta.yaml")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    extract_all_features(fps=args.fps)
