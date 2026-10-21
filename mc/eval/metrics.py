"""Core metrics calculation for temporal video grounding: tIoU, R@K, mIoU."""

import json
from typing import List, Dict, Tuple, Any, Union


def compute_tiou(
    pred_span: Union[Tuple[float, float], List[float]],
    gt_span: Union[Tuple[float, float], List[float]]
) -> float:
    """
    Compute Temporal Intersection-over-Union (tIoU) between two time segments.
    Each span is (start, end).
    """
    s_p, e_p = pred_span[0], pred_span[1]
    s_g, e_g = gt_span[0], gt_span[1]

    # Ensure start <= end
    s_p, e_p = min(s_p, e_p), max(s_p, e_p)
    s_g, e_g = min(s_g, e_g), max(s_g, e_g)

    inter_start = max(s_p, s_g)
    inter_end = min(e_p, e_g)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(s_p, s_g)
    union_end = max(e_p, e_g)
    union = max(0.0, union_end - union_start)

    if union <= 1e-8:
        return 0.0
    return float(intersection / union)


def evaluate(
    preds: List[Dict[str, Any]],
    gts: Dict[str, Tuple[float, float]],
    tiou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    top_k: Tuple[int, ...] = (1, 5),
    query_types: Dict[str, str] = None
) -> Dict[str, float]:
    """
    Evaluate temporal grounding predictions against ground truth spans.

    Args:
        preds: List of prediction dicts:
               [{'query_id': str, 'video_id': str, 'rank': int, 'start': float, 'end': float, 'score': float}, ...]
        gts: Dict mapping query_id -> (start, end)
        tiou_thresholds: Sequence of tIoU thresholds (default: 0.3, 0.5, 0.7)
        top_k: Sequence of K thresholds (default: 1, 5)
        query_types: Optional dict mapping query_id -> type_category

    Returns:
        dict containing:
          - R@{k}_tiou@{iou} for each pair
          - mIoU (mean top-1 IoU)
          - per_query_tiou: dict of top-1 tIoU per query_id (for bootstrap CI)
          - breakdown by query_type if provided
    """
    # Group and sort predictions by query_id
    grouped_preds: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        qid = str(p["query_id"])
        grouped_preds.setdefault(qid, []).append(p)

    for qid in grouped_preds:
        grouped_preds[qid].sort(key=lambda x: (x.get("rank", 999), -x.get("score", 0.0)))

    # Metrics accumulators
    num_queries = len(gts)
    if num_queries == 0:
        return {"num_queries": 0, "mIoU": 0.0}

    # Track hits: hits[(k, th)] = count
    hits = {(k, th): 0 for k in top_k for th in tiou_thresholds}
    top1_ious = {}

    type_counts = {}
    type_hits = {}

    for qid, gt_span in gts.items():
        q_preds = grouped_preds.get(qid, [])
        q_type = query_types.get(qid, "all") if query_types else "all"

        type_counts[q_type] = type_counts.get(q_type, 0) + 1

        # Calculate max tIoU achieved in top-k
        best_top1_iou = 0.0
        if q_preds:
            best_top1_iou = compute_tiou((q_preds[0]["start"], q_preds[0]["end"]), gt_span)

        top1_ious[qid] = best_top1_iou

        for k in top_k:
            sub_preds = q_preds[:k]
            ious_in_k = [compute_tiou((p["start"], p["end"]), gt_span) for p in sub_preds]
            max_iou_k = max(ious_in_k) if ious_in_k else 0.0

            for th in tiou_thresholds:
                if max_iou_k >= th:
                    hits[(k, th)] += 1
                    # Track for type breakdown
                    key = (q_type, k, th)
                    type_hits[key] = type_hits.get(key, 0) + 1

    results = {
        "num_queries": num_queries,
        "mIoU": round(sum(top1_ious.values()) / num_queries, 4)
    }

    for k in top_k:
        for th in tiou_thresholds:
            metric_name = f"R@{k}_tiou@{th}"
            recall = hits[(k, th)] / num_queries
            results[metric_name] = round(recall, 4)

    # Per-type breakdown
    if query_types:
        for q_type, count in type_counts.items():
            if count == 0:
                continue
            for th in [0.3, 0.5]:
                key = (q_type, 1, th)
                r1 = type_hits.get(key, 0) / count
                results[f"{q_type}_R@1_tiou@{th}"] = round(r1, 4)

    # Attach raw top-1 IoU dictionary for paired bootstrap testing
    results["_per_query_top1_tiou"] = top1_ious

    return results


def evaluate_predictions_file(
    preds_jsonl_path: str,
    annotations_path: str,
    tiou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    top_k: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    """Helper to evaluate directly from a saved predictions.jsonl and annotations."""
    from mc.data.charades_sta import load_annotations
    records = load_annotations(annotations_path)
    gts = {r.query_id: (r.start, r.end) for r in records}

    preds = []
    with open(preds_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    return evaluate(preds, gts, tiou_thresholds=tiou_thresholds, top_k=top_k)
