"""Candidate interval generation, peak merging, and 1D temporal NMS."""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from mc.eval.metrics import compute_tiou


@dataclass
class CandidateSegment:
    start_time: float
    end_time: float
    start_idx: int
    end_idx: int
    score: float
    rel_score: float
    boundary_score: float

    @property
    def span(self) -> Tuple[float, float]:
        return (self.start_time, self.end_time)


def temporal_nms_1d(
    candidates: List[CandidateSegment],
    nms_threshold: float = 0.4,
    top_k: int = 5
) -> List[CandidateSegment]:
    """
    1D Non-Maximum Suppression (NMS) over candidate intervals based on tIoU.
    """
    if not candidates:
        return []

    # Sort descending by score
    sorted_candidates = sorted(candidates, key=lambda c: -c.score)
    selected: List[CandidateSegment] = []

    for cand in sorted_candidates:
        # Check overlap with already selected segments
        keep = True
        for sel in selected:
            iou = compute_tiou(cand.span, sel.span)
            if iou >= nms_threshold:
                keep = False
                break
        if keep:
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected


def generate_candidates_and_merge(
    rel_logits: np.ndarray,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    timestamps: np.ndarray,
    video_duration: float,
    query_id: str = "query",
    video_id: str = "video",
    rel_threshold: float = 0.25,
    top_k_peaks: int = 8,
    nms_threshold: float = 0.4,
    final_top_k: int = 5,
    w_rel: float = 0.6,
    w_bound: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Generate candidate segments by:
      1. Thresholding relevance logits -> finding contiguous connected components
      2. Finding top peaks in start and end probability distributions
      3. Forming interval combinations (s, e) where s < e
      4. Scoring intervals = w_rel * mean_relevance + w_bound * (start_p * end_p)^0.5
      5. Applying 1D temporal NMS
    """
    T = len(timestamps)
    if T == 0:
        return []

    # Softmax / Sigmoid activations
    rel_probs = 1.0 / (1.0 + np.exp(-np.clip(rel_logits, -15, 15)))

    # Stable softmax for start and end
    exp_s = np.exp(start_logits - np.max(start_logits))
    start_probs = exp_s / (np.sum(exp_s) + 1e-8)

    exp_e = np.exp(end_logits - np.max(end_logits))
    end_probs = exp_e / (np.sum(exp_e) + 1e-8)

    candidates: List[CandidateSegment] = []

    # 1. Contiguous relevance components
    is_rel = rel_probs >= rel_threshold
    diff = np.diff(np.pad(is_rel.astype(int), (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    for s_idx, e_idx in zip(starts, ends):
        if e_idx <= s_idx:
            e_idx = min(T - 1, s_idx + 1)
        s_time = float(timestamps[s_idx])
        e_time = float(timestamps[e_idx])
        if e_time <= s_time:
            e_time = min(video_duration, s_time + 1.0)

        mean_rel = float(np.mean(rel_probs[s_idx : e_idx + 1]))
        bound_s = float(np.sqrt(start_probs[s_idx] * end_probs[e_idx]))
        score = w_rel * mean_rel + w_bound * bound_s

        candidates.append(CandidateSegment(
            start_time=round(s_time, 2),
            end_time=round(e_time, 2),
            start_idx=int(s_idx),
            end_idx=int(e_idx),
            score=round(score, 4),
            rel_score=round(mean_rel, 4),
            boundary_score=round(bound_s, 4)
        ))

    # 2. Top-k boundary peaks pairs
    top_start_indices = np.argsort(-start_probs)[:top_k_peaks]
    top_end_indices = np.argsort(-end_probs)[:top_k_peaks]

    for s_idx in top_start_indices:
        for e_idx in top_end_indices:
            if s_idx >= e_idx:
                continue

            s_time = float(timestamps[s_idx])
            e_time = float(timestamps[e_idx])
            dur = e_time - s_time
            # Ignore unnaturally short (<0.5s) or long (>60s) windows
            if dur < 0.5 or dur > 60.0:
                continue

            mean_rel = float(np.mean(rel_probs[s_idx : e_idx + 1]))
            bound_s = float(np.sqrt(start_probs[s_idx] * end_probs[e_idx]))
            score = w_rel * mean_rel + w_bound * bound_s

            candidates.append(CandidateSegment(
                start_time=round(s_time, 2),
                end_time=round(e_time, 2),
                start_idx=int(s_idx),
                end_idx=int(e_idx),
                score=round(score, 4),
                rel_score=round(mean_rel, 4),
                boundary_score=round(bound_s, 4)
            ))

    # Fallback if no candidate meets criteria
    if not candidates:
        center = T // 2
        s_time = float(timestamps[max(0, center - 2)])
        e_time = float(timestamps[min(T - 1, center + 2)])
        candidates.append(CandidateSegment(
            start_time=round(s_time, 2),
            end_time=round(e_time, 2),
            start_idx=max(0, center - 2),
            end_idx=min(T - 1, center + 2),
            score=0.1,
            rel_score=0.1,
            boundary_score=0.1
        ))

    # Apply 1D Temporal NMS
    selected = temporal_nms_1d(candidates, nms_threshold=nms_threshold, top_k=final_top_k)

    # Format predictions dictionary
    preds = []
    for rank, cand in enumerate(selected, 1):
        preds.append({
            "query_id": query_id,
            "video_id": video_id,
            "rank": rank,
            "start": cand.start_time,
            "end": cand.end_time,
            "score": cand.score,
            "rel_score": cand.rel_score,
            "boundary_score": cand.boundary_score
        })

    return preds
