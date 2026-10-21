"""System efficiency benchmarking, latency profiling (p50/p95), and hardware tracking."""

import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple


class LatencyProfiler:
    """
    Measures end-to-end and stage latencies with empirical p50, p90, and p95 percentiles.
    """

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def record(self, stage_name: str, duration_ms: float) -> None:
        self.measurements.setdefault(stage_name, []).append(duration_ms)

    def time_call(self, stage_name: str, func: Callable, *args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000.0
        self.record(stage_name, duration_ms)
        return result

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary_dict = {}
        for stage, times in self.measurements.items():
            arr = np.array(times, dtype=np.float64)
            summary_dict[stage] = {
                "count": len(arr),
                "mean_ms": round(float(np.mean(arr)), 2),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p90_ms": round(float(np.percentile(arr, 90)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "p99_ms": round(float(np.percentile(arr, 99)), 2),
                "min_ms": round(float(np.min(arr)), 2),
                "max_ms": round(float(np.max(arr)), 2),
            }
        return summary_dict


def get_gpu_memory_stats() -> Dict[str, Any]:
    """Retrieve current and peak GPU VRAM allocations."""
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        device_name = torch.cuda.get_device_name(0)
    else:
        allocated_mb = 0.0
        reserved_mb = 0.0
        max_allocated_mb = 0.0
        device_name = "CPU"

    return {
        "device": device_name,
        "allocated_mb": round(allocated_mb, 2),
        "reserved_mb": round(reserved_mb, 2),
        "peak_vram_mb": round(max_allocated_mb, 2),
        "vram_limit_safe": max_allocated_mb < 3500.0  # Constraint: stay under 3.5 GB
    }


def profile_index_configurations(
    features: np.ndarray,
    queries: np.ndarray,
    top_k: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Profile Flat vs IVF-PQ vs HNSW across:
      - Indexing time
      - Storage size
      - p50 and p95 retrieval latency
    """
    from mc.index.faiss_index import VideoFAISSIndex

    results = {}
    index_types = ["Flat", "IVF-PQ", "HNSW"]
    d_model = features.shape[1]

    for itype in index_types:
        idx = VideoFAISSIndex(d_model=d_model, index_type=itype)
        build_stats = idx.build(features)

        profiler = LatencyProfiler()
        for q in queries:
            profiler.time_call("search", idx.search, q, top_k=top_k)

        search_stats = profiler.summary()["search"]
        results[itype] = {
            "build_time_ms": build_stats["build_time_ms"],
            "storage_kb": build_stats["estimated_size_kb"],
            "p50_latency_ms": search_stats["p50_ms"],
            "p95_latency_ms": search_stats["p95_ms"],
            "mean_latency_ms": search_stats["mean_ms"]
        }

    return results


def build_long_video_composite(
    clip_features_list: List[np.ndarray],
    clip_durations: List[float]
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Concatenate 20 to 60 Charades clips into a 10 to 30 minute long video stream
    with ground truth offset bookkeeping for scalability testing (Phase 8).
    """
    concatenated_feats = []
    concatenated_timestamps = []
    offsets = []
    current_offset = 0.0

    for i, (feats, dur) in enumerate(zip(clip_features_list, clip_durations)):
        T = feats.shape[0]
        ts = np.linspace(0.0, dur, T, endpoint=False) + current_offset
        concatenated_feats.append(feats)
        concatenated_timestamps.append(ts)
        offsets.append({
            "clip_idx": i,
            "global_start": round(current_offset, 2),
            "global_end": round(current_offset + dur, 2),
            "duration": round(dur, 2)
        })
        current_offset += dur

    all_feats = np.concatenate(concatenated_feats, axis=0)
    all_ts = np.concatenate(concatenated_timestamps, axis=0)

    return all_feats, all_ts, offsets
