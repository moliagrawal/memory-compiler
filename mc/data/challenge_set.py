"""Challenge set generator and inter-annotator agreement evaluator (Phase 7)."""

import json
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from mc.eval.query_types import QueryType, classify_query
from mc.eval.metrics import compute_tiou


@dataclass
class ChallengeRecord:
    query_id: str
    video_id: str
    query: str
    start: float
    end: float
    category: str
    objects: List[str]
    actions: List[str]
    is_ordered: bool
    alternate_intervals: List[Tuple[float, float]]
    annotator_1_category: Optional[str] = None
    annotator_2_category: Optional[str] = None
    annotator_2_start: Optional[float] = None
    annotator_2_end: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_cohens_kappa(labels_1: List[str], labels_2: List[str]) -> float:
    """
    Compute Cohen's Kappa score measuring inter-annotator agreement on categorical labels.
    """
    if len(labels_1) != len(labels_2) or len(labels_1) == 0:
        return 0.0

    n = len(labels_1)
    categories = sorted(list(set(labels_1) | set(labels_2)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    matrix = np.zeros((num_cats, num_cats), dtype=np.int64)
    for l1, l2 in zip(labels_1, labels_2):
        matrix[cat_to_idx[l1], cat_to_idx[l2]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected chance agreement
    row_sums = np.sum(matrix, axis=1) / n
    col_sums = np.sum(matrix, axis=2 if matrix.ndim > 2 else 0) / n
    p_e = np.sum(row_sums * col_sums)

    if 1.0 - p_e == 0:
        return 1.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(round(kappa, 4))


def compute_temporal_agreement(intervals_1: List[Tuple[float, float]], intervals_2: List[Tuple[float, float]]) -> float:
    """Compute mean tIoU agreement between two annotators over temporal intervals."""
    if not intervals_1 or len(intervals_1) != len(intervals_2):
        return 0.0
    tious = [compute_tiou(i1, i2) for i1, i2 in zip(intervals_1, intervals_2)]
    return float(round(np.mean(tious), 4))


class ChallengeSetGenerator:
    """
    Generates and manages the 200-300 query challenge benchmark with dual-annotator verification.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_synthetic_challenge_set(
        self,
        base_video_ids: List[str],
        num_queries: int = 240
    ) -> List[ChallengeRecord]:
        """
        Create structured challenge set distributed evenly across the 4 canonical categories:
        1. Object / Attribute
        2. Single Action
        3. Subject-Object Interaction
        4. Multi-Event / Temporal Order
        """
        records = []
        num_per_cat = max(1, num_queries // 4)

        categories_templates = {
            QueryType.OBJECT_ATTRIBUTE.value: [
                ("holding a red cup in the living room", ["red cup", "table"], ["hold"], False),
                ("looking at the dark wooden cabinet", ["wooden cabinet"], ["look"], False),
                ("white towel hanging near the bathroom door", ["white towel", "door"], ["hang"], False),
                ("a small blue pillow on the couch", ["blue pillow", "couch"], ["rest"], False),
            ],
            QueryType.SINGLE_ACTION.value: [
                ("person stands up from the floor", ["floor"], ["stand up"], False),
                ("person turns around and smiles", [], ["turn", "smile"], False),
                ("someone walks slowly towards the window", ["window"], ["walk"], False),
                ("person laughs while sitting down", ["chair"], ["laugh", "sit down"], False),
            ],
            QueryType.SUBJECT_OBJECT.value: [
                ("person opens the refrigerator and inspects inside", ["refrigerator"], ["open", "inspect"], False),
                ("person washes a glass cup in the kitchen sink", ["glass cup", "sink"], ["wash"], False),
                ("takes a book from the shelf and opens it", ["book", "shelf"], ["take", "open"], False),
                ("pours water into a transparent glass", ["water", "glass"], ["pour"], False),
            ],
            QueryType.MULTI_EVENT_ORDER.value: [
                ("first closes the door and then sits on the sofa", ["door", "sofa"], ["close", "sit"], True),
                ("first picks up the phone and then walks out", ["phone"], ["pick up", "walk"], True),
                ("after putting down the cup, person leaves the room", ["cup"], ["put down", "leave"], True),
                ("first opens the refrigerator and then pours a drink", ["refrigerator", "drink"], ["open", "pour"], True),
            ]
        }

        q_idx = 0
        for cat, templates in categories_templates.items():
            for i in range(num_per_cat):
                vid = base_video_ids[q_idx % len(base_video_ids)] if base_video_ids else f"video_{q_idx:04d}"
                tpl = templates[i % len(templates)]
                query_text, objs, acts, is_ordered = tpl

                start = round(self.rng.uniform(1.0, 15.0), 1)
                end = round(start + self.rng.uniform(3.0, 10.0), 1)

                # Simulated second annotator: slight jitter in boundaries and high category agreement (~92%)
                ann2_cat = cat if self.rng.random() < 0.92 else random.choice(list(categories_templates.keys()))
                ann2_start = round(max(0.0, start + self.rng.uniform(-0.5, 0.5)), 1)
                ann2_end = round(max(ann2_start + 1.0, end + self.rng.uniform(-0.6, 0.6)), 1)

                alt_intervals = [
                    (round(max(0.0, start - 1.0), 1), round(end + 1.0, 1))
                ]

                record = ChallengeRecord(
                    query_id=f"chal_{q_idx:04d}",
                    video_id=vid,
                    query=query_text,
                    start=start,
                    end=end,
                    category=cat,
                    objects=objs,
                    actions=acts,
                    is_ordered=is_ordered,
                    alternate_intervals=alt_intervals,
                    annotator_1_category=cat,
                    annotator_2_category=ann2_cat,
                    annotator_2_start=ann2_start,
                    annotator_2_end=ann2_end
                )
                records.append(record)
                q_idx += 1

        return records

    def compute_agreement_report(self, records: List[ChallengeRecord]) -> Dict[str, float]:
        """Compute inter-annotator Cohen's kappa and mean interval agreement (tIoU)."""
        cat1 = [r.annotator_1_category for r in records if r.annotator_1_category and r.annotator_2_category]
        cat2 = [r.annotator_2_category for r in records if r.annotator_1_category and r.annotator_2_category]

        int1 = [(r.start, r.end) for r in records if r.annotator_2_start is not None]
        int2 = [(r.annotator_2_start, r.annotator_2_end) for r in records if r.annotator_2_start is not None]

        kappa = compute_cohens_kappa(cat1, cat2)
        mean_tiou = compute_temporal_agreement(int1, int2)

        return {
            "num_evaluated": len(records),
            "cohens_kappa_category": kappa,
            "mean_temporal_agreement_tiou": mean_tiou
        }
