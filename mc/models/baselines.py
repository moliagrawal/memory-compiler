"""Baseline grounding algorithms: B0 (Random), B1 (Top Frame Window), B2 (Sliding Window)."""

import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class BaselineB0Random:
    """
    B0 Baseline: Random Interval
    Samples segment duration from train distribution (or defaults), placed uniformly in the video.
    """

    def __init__(self, mean_duration: float = 7.5, std_duration: float = 3.0, seed: int = 42):
        self.mean_duration = mean_duration
        self.std_duration = std_duration
        self.rng = random.Random(seed)

    def predict(
        self,
        query_id: str,
        video_id: str,
        video_duration: float,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        preds = []
        for rank in range(1, top_k + 1):
            dur = max(1.0, min(video_duration, self.rng.gauss(self.mean_duration, self.std_duration)))
            max_start = max(0.0, video_duration - dur)
            start = self.rng.uniform(0.0, max_start)
            end = min(video_duration, start + dur)
            score = 1.0 / rank

            preds.append({
                "query_id": query_id,
                "video_id": video_id,
                "rank": rank,
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(score, 4)
            })
        return preds


class BaselineB1FrameSimilarity:
    """
    B1 Baseline: Frame-CLIP Similarity
    Finds top-K scoring frames by cosine similarity with the query,
    then expands each frame to a fixed window (e.g. +/- 4 seconds).
    """

    def __init__(self, window_half_width: float = 3.5):
        self.window_half_width = window_half_width

    def predict(
        self,
        query_id: str,
        video_id: str,
        frame_features: np.ndarray,
        query_vector: np.ndarray,
        timestamps: np.ndarray,
        video_duration: float,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        # Compute cosine similarity per frame: [T]
        frames_norm = frame_features / (np.linalg.norm(frame_features, axis=-1, keepdims=True) + 1e-8)
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        sims = np.dot(frames_norm, q_norm).flatten()

        # Find top frames with non-maximum suppression / distance separation
        ranked_indices = np.argsort(-sims)
        selected_indices = []
        min_dist_frames = 3

        for idx in ranked_indices:
            if all(abs(idx - prev) >= min_dist_frames for prev in selected_indices):
                selected_indices.append(idx)
                if len(selected_indices) >= top_k:
                    break

        # Fallback if not enough non-overlapping frames
        if len(selected_indices) < top_k:
            for idx in ranked_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= top_k:
                        break

        preds = []
        for rank, frame_idx in enumerate(selected_indices, 1):
            center_t = float(timestamps[frame_idx]) if frame_idx < len(timestamps) else 0.0
            start = max(0.0, center_t - self.window_half_width)
            end = min(video_duration, center_t + self.window_half_width)
            score = float(sims[frame_idx])

            preds.append({
                "query_id": query_id,
                "video_id": video_id,
                "rank": rank,
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(score, 4)
            })

        return preds


class BaselineB2SlidingWindow:
    """
    B2 Baseline: Multi-Scale Sliding Window
    Slides windows of multiple lengths (e.g. 4s, 8s, 12s, 16s) over frame similarities,
    computes mean similarity in each window, and returns top non-overlapping windows.
    """

    def __init__(self, window_durations: Tuple[float, ...] = (4.0, 8.0, 12.0, 16.0), stride_s: float = 1.0):
        self.window_durations = window_durations
        self.stride_s = stride_s

    def predict(
        self,
        query_id: str,
        video_id: str,
        frame_features: np.ndarray,
        query_vector: np.ndarray,
        timestamps: np.ndarray,
        video_duration: float,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        frames_norm = frame_features / (np.linalg.norm(frame_features, axis=-1, keepdims=True) + 1e-8)
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        sims = np.dot(frames_norm, q_norm).flatten()

        candidates = []
        # Multi-scale sliding window
        for win_dur in self.window_durations:
            start_t = 0.0
            while start_t + 1.0 <= video_duration:
                end_t = min(video_duration, start_t + win_dur)
                mask = (timestamps >= start_t) & (timestamps <= end_t)
                if np.any(mask):
                    mean_sim = float(np.mean(sims[mask]))
                else:
                    mean_sim = 0.0

                candidates.append((start_t, end_t, mean_sim))
                start_t += self.stride_s

        # Sort candidates descending by score
        candidates.sort(key=lambda x: -x[2])

        # Simple 1D NMS (temporal IoU < 0.4) to pick diverse top-K
        from mc.eval.metrics import compute_tiou

        selected = []
        for c in candidates:
            c_span = (c[0], c[1])
            if all(compute_tiou(c_span, (s[0], s[1])) < 0.4 for s in selected):
                selected.append(c)
                if len(selected) >= top_k:
                    break

        preds = []
        for rank, c in enumerate(selected, 1):
            preds.append({
                "query_id": query_id,
                "video_id": video_id,
                "rank": rank,
                "start": round(c[0], 2),
                "end": round(c[1], 2),
                "score": round(c[2], 4)
            })

        return preds


def run_all_baselines(
    query_id: str,
    video_id: str,
    frame_features: np.ndarray,
    query_vector: np.ndarray,
    timestamps: np.ndarray,
    video_duration: float,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Execute B0, B1, and B2 concurrently on a single query."""
    b0 = BaselineB0Random()
    b1 = BaselineB1FrameSimilarity()
    b2 = BaselineB2SlidingWindow()

    return {
        "B0": b0.predict(query_id, video_id, video_duration, top_k=top_k),
        "B1": b1.predict(query_id, video_id, frame_features, query_vector, timestamps, video_duration, top_k=top_k),
        "B2": b2.predict(query_id, video_id, frame_features, query_vector, timestamps, video_duration, top_k=top_k),
    }
