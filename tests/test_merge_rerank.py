"""Unit tests for merge, 1D NMS, and interval reranking."""

import numpy as np
import pytest
from mc.index.merge import generate_candidates_and_merge, temporal_nms_1d, CandidateSegment
from mc.index.rerank import IntervalReranker


def test_1d_temporal_nms():
    # 3 overlapping candidates
    c1 = CandidateSegment(10.0, 20.0, 10, 20, 0.9, 0.9, 0.9)
    # c2 overlaps heavily with c1 (11 to 21 -> IoU ~ 0.82)
    c2 = CandidateSegment(11.0, 21.0, 11, 21, 0.85, 0.85, 0.85)
    # c3 is disjoint (30 to 40)
    c3 = CandidateSegment(30.0, 40.0, 30, 40, 0.80, 0.80, 0.80)

    selected = temporal_nms_1d([c1, c2, c3], nms_threshold=0.4, top_k=5)

    # c2 should be suppressed, so only c1 and c3 remain
    assert len(selected) == 2
    assert selected[0].span == (10.0, 20.0)
    assert selected[1].span == (30.0, 40.0)


def test_generate_candidates_and_merge():
    T = 20
    timestamps = np.arange(T, dtype=np.float32)
    rel_logits = np.full(T, -2.0)
    rel_logits[5:10] = 3.0  # Clear peak between 5 and 10

    start_logits = np.full(T, -2.0)
    start_logits[5] = 5.0

    end_logits = np.full(T, -2.0)
    end_logits[10] = 5.0

    preds = generate_candidates_and_merge(
        rel_logits=rel_logits,
        start_logits=start_logits,
        end_logits=end_logits,
        timestamps=timestamps,
        video_duration=20.0,
        final_top_k=3
    )

    assert len(preds) > 0
    assert preds[0]["rank"] == 1
    # Check top candidate covers the peak
    assert 4.0 <= preds[0]["start"] <= 6.0
    assert 9.0 <= preds[0]["end"] <= 11.0


def test_interval_reranker():
    reranker = IntervalReranker(d_model=256, device="cpu")
    candidates = [
        {"query_id": "q1", "video_id": "v1", "rank": 1, "start": 5.0, "end": 10.0, "score": 0.8},
        {"query_id": "q1", "video_id": "v1", "rank": 2, "start": 15.0, "end": 20.0, "score": 0.7}
    ]
    frame_feats = np.random.randn(30, 256).astype(np.float32)
    query_vec = np.random.randn(256).astype(np.float32)
    timestamps = np.arange(30, dtype=np.float32)

    reranked = reranker.rerank_candidates(candidates, frame_feats, query_vec, timestamps, top_k=2)
    assert len(reranked) == 2
    assert "combined_score" in reranked[0]
    assert reranked[0]["rank"] == 1
    assert reranked[1]["rank"] == 2
