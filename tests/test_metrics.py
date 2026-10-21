"""Unit tests for metrics, tIoU, and paired bootstrap significance testing."""

import pytest
from mc.eval.metrics import compute_tiou, evaluate
from mc.eval.bootstrap import paired_bootstrap_test, compute_metric_ci
from mc.eval.query_types import classify_query, QueryType


def test_compute_tiou_exact_match():
    assert compute_tiou((5.0, 15.0), (5.0, 15.0)) == 1.0


def test_compute_tiou_disjoint():
    assert compute_tiou((0.0, 5.0), (6.0, 10.0)) == 0.0
    assert compute_tiou((10.0, 12.0), (0.0, 5.0)) == 0.0


def test_compute_tiou_partial_overlap():
    # Intersection = 5.0 (from 5 to 10), Union = 15.0 (from 0 to 15) -> 5/15 = 1/3
    iou = compute_tiou((0.0, 10.0), (5.0, 15.0))
    assert pytest.approx(iou, rel=1e-3) == 1.0 / 3.0


def test_compute_tiou_nested():
    # Span (2, 8) inside (0, 10). Intersection = 6, Union = 10 -> 0.6
    iou = compute_tiou((2.0, 8.0), (0.0, 10.0))
    assert pytest.approx(iou, rel=1e-3) == 0.6


def test_evaluate_gate_metrics():
    gts = {
        "q1": (10.0, 20.0),
        "q2": (30.0, 40.0),
        "q3": (50.0, 60.0),
        "q4": (70.0, 80.0)
    }

    # Preds:
    # q1: exact match at rank 1 (IoU 1.0)
    # q2: partial overlap at rank 1 (35, 45 -> IoU 5/15 = 0.333, passes >= 0.3, fails >= 0.5)
    # q3: poor overlap at rank 1, good overlap at rank 2 (51, 59 -> IoU 8/10 = 0.8)
    # q4: complete miss
    preds = [
        {"query_id": "q1", "video_id": "v1", "rank": 1, "start": 10.0, "end": 20.0, "score": 0.95},
        {"query_id": "q2", "video_id": "v2", "rank": 1, "start": 35.0, "end": 45.0, "score": 0.85},
        {"query_id": "q3", "video_id": "v3", "rank": 1, "start": 0.0, "end": 5.0, "score": 0.80},
        {"query_id": "q3", "video_id": "v3", "rank": 2, "start": 51.0, "end": 59.0, "score": 0.75},
        {"query_id": "q4", "video_id": "v4", "rank": 1, "start": 0.0, "end": 10.0, "score": 0.70},
    ]

    results = evaluate(preds, gts, tiou_thresholds=(0.3, 0.5, 0.7), top_k=(1, 5))

    # Top-1: q1 has IoU 1.0 (passes 0.3, 0.5, 0.7)
    #        q2 has IoU 0.333 (passes 0.3, fails 0.5, 0.7)
    #        q3 rank 1 has 0.0 (fails all)
    #        q4 has 0.0 (fails all)
    # Total queries: 4. R@1 @ 0.3 = 2/4 = 0.50. R@1 @ 0.5 = 1/4 = 0.25. R@1 @ 0.7 = 1/4 = 0.25
    assert results["num_queries"] == 4
    assert pytest.approx(results["R@1_tiou@0.3"], rel=1e-2) == 0.50
    assert pytest.approx(results["R@1_tiou@0.5"], rel=1e-2) == 0.25
    assert pytest.approx(results["R@1_tiou@0.7"], rel=1e-2) == 0.25

    # Top-5: q3 rank 2 has 0.8 (passes 0.3, 0.5, 0.7)
    # So R@5 @ 0.5 = (q1 + q3) / 4 = 2/4 = 0.50
    assert pytest.approx(results["R@5_tiou@0.5"], rel=1e-2) == 0.50


def test_paired_bootstrap_test():
    # Model A clearly superior to Model B
    scores_a = {f"q{i}": 0.85 for i in range(50)}
    scores_b = {f"q{i}": 0.35 for i in range(50)}

    res = paired_bootstrap_test(scores_a, scores_b, num_resamples=500, seed=123)
    assert res["is_significant"] is True
    assert res["obs_delta"] > 0.4
    assert res["ci_lower"] > 0.3


def test_query_types_classification():
    assert classify_query("person opens the refrigerator door") == QueryType.SUBJECT_OBJECT
    assert classify_query("first the person stood up and then smiled") == QueryType.MULTI_EVENT_ORDER
    assert classify_query("a red towel on the table") == QueryType.OBJECT_ATTRIBUTE
    assert classify_query("person is smiling") == QueryType.SINGLE_ACTION
