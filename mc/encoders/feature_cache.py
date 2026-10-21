"""Feature cache management for frame embeddings and query vectors (fp16)."""

import os
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple


class FeatureCache:
    """
    Manages cached embeddings:
      - Video features: `data/features/{encoder_id}/{video_id}.npy` [T, D] fp16
      - Video metadata: `data/features/{encoder_id}/{video_id}_meta.json`
      - Query embeddings: `data/features/{encoder_id}/queries/{query_hash}.npy` [D] fp16
    """

    def __init__(self, cache_dir: str = "data/features", encoder_id: str = "clip_b32"):
        self.cache_dir = cache_dir
        self.encoder_id = encoder_id
        self.encoder_dir = os.path.join(cache_dir, encoder_id)
        self.query_dir = os.path.join(self.encoder_dir, "queries")
        os.makedirs(self.encoder_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Compute SHA256 content hash of video file for cache validation."""
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in 64KB chunks
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def has_video_features(self, video_id: str, video_path: Optional[str] = None) -> bool:
        """Check if cached features exist and match video content hash if path is given."""
        feat_path = os.path.join(self.encoder_dir, f"{video_id}.npy")
        meta_path = os.path.join(self.encoder_dir, f"{video_id}_meta.json")

        if not (os.path.exists(feat_path) and os.path.exists(meta_path)):
            return False

        if video_path and os.path.exists(video_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cached_hash = meta.get("file_hash")
            if cached_hash:
                curr_hash = self.compute_file_hash(video_path)
                return cached_hash == curr_hash

        return True

    def save_video_features(
        self,
        video_id: str,
        features: np.ndarray,
        timestamps: np.ndarray,
        fps: float,
        video_path: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save video features as float16 numpy array along with meta.json."""
        feat_path = os.path.join(self.encoder_dir, f"{video_id}.npy")
        meta_path = os.path.join(self.encoder_dir, f"{video_id}_meta.json")

        # Force fp16
        features_fp16 = features.astype(np.float16)
        np.save(feat_path, features_fp16)

        file_hash = self.compute_file_hash(video_path) if video_path else ""

        meta = {
            "video_id": video_id,
            "encoder_id": self.encoder_id,
            "num_frames": int(features_fp16.shape[0]),
            "feature_dim": int(features_fp16.shape[1]),
            "fps": float(fps),
            "timestamps": timestamps.tolist() if isinstance(timestamps, np.ndarray) else list(timestamps),
            "file_hash": file_hash
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_video_features(self, video_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load features and metadata. Returns: (features [T, D] fp16, meta dict)."""
        feat_path = os.path.join(self.encoder_dir, f"{video_id}.npy")
        meta_path = os.path.join(self.encoder_dir, f"{video_id}_meta.json")

        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        features = np.load(feat_path)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        return features, meta

    def save_query_embedding(self, query_id_or_text: str, embedding: np.ndarray) -> None:
        """Save query embedding [D] fp16."""
        q_hash = hashlib.md5(query_id_or_text.encode("utf-8")).hexdigest()
        path = os.path.join(self.query_dir, f"{q_hash}.npy")
        np.save(path, embedding.astype(np.float16))

    def load_query_embedding(self, query_id_or_text: str) -> Optional[np.ndarray]:
        """Load query embedding [D] fp16 if cached."""
        q_hash = hashlib.md5(query_id_or_text.encode("utf-8")).hexdigest()
        path = os.path.join(self.query_dir, f"{q_hash}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None
