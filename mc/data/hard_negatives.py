"""Hard negative generation for contrastive ranking and challenge evaluation."""

import re
import random
from typing import List, Dict, Tuple, Optional, Any
from mc.eval.metrics import compute_tiou


ACTION_PAIRS = [
    (r"\bopen(s|ed|ing)?\b", "close"),
    (r"\bclose(s|d|ing)?\b", "open"),
    (r"\bpick(s|ed|ing)? up\b", "put down"),
    (r"\bput(s|ting)? down\b", "pick up"),
    (r"\btake(s|n|ing)?\b", "put"),
    (r"\bput(s|ting)?\b", "take"),
    (r"\bsit(s|ting)? down\b", "stand up"),
    (r"\bstand(s|ing)? up\b", "sit down"),
    (r"\bentered?\b", "left"),
    (r"\bleav(es|ing)|left\b", "entered"),
    (r"\bturn(s|ed|ing)? on\b", "turn off"),
    (r"\bturn(s|ed|ing)? off\b", "turn on"),
]

OBJECT_SWAPS = [
    (r"\bcup\b", "glass"),
    (r"\bglass\b", "cup"),
    (r"\bdoor\b", "window"),
    (r"\bwindow\b", "door"),
    (r"\bphone\b", "book"),
    (r"\bbook\b", "phone"),
    (r"\bshoes?\b", "jacket"),
    (r"\bjacket\b", "shoes"),
    (r"\bchair\b", "sofa"),
    (r"\bsofa\b", "chair"),
    (r"\btowel\b", "blanket"),
    (r"\bblanket\b", "towel"),
    (r"\bbag\b", "box"),
    (r"\bbox\b", "bag"),
    (r"\bfridge\b", "cabinet"),
    (r"\bcabinet\b", "fridge")
]


class HardNegativeMiner:
    """
    Mines and synthesizes hard negative queries and segments:
      (a) Action reversal
      (b) Recipient / object swap
      (c) Order reversal for two-step queries
      (d) Same-video wrong segment (temporally disjoint or low IoU)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_action_reversal(self, query: str) -> Optional[str]:
        """Invert verb action (e.g. 'person opens door' -> 'person closes door')."""
        for pattern, replacement in ACTION_PAIRS:
            if re.search(pattern, query, re.IGNORECASE):
                # Replace the match
                return re.sub(pattern, replacement, query, count=1, flags=re.IGNORECASE)
        return None

    def generate_object_swap(self, query: str) -> Optional[str]:
        """Swap target recipient/object with plausible alternate object."""
        for pattern, replacement in OBJECT_SWAPS:
            if re.search(pattern, query, re.IGNORECASE):
                return re.sub(pattern, replacement, query, count=1, flags=re.IGNORECASE)
        return None

    def generate_order_reversal(self, query: str) -> Optional[str]:
        """Reverse chronological order in multi-event queries."""
        # Check for 'first X and then Y' or 'X then Y' or 'after X, Y'
        patterns = [
            r"^(first\s+)?(.+?)\s+(and\s+then|then|afterwards)\s+(.+)$",
            r"^after\s+(.+?),\s*(.+)$"
        ]

        # Try 'X then Y'
        m = re.match(patterns[0], query.strip(), re.IGNORECASE)
        if m:
            action1 = m.group(2).strip()
            connector = m.group(3).strip()
            action2 = m.group(4).strip()
            return f"first {action2} {connector} {action1}"

        # Try 'after X, Y'
        m2 = re.match(patterns[1], query.strip(), re.IGNORECASE)
        if m2:
            action1 = m2.group(1).strip()
            action2 = m2.group(2).strip()
            return f"after {action2}, {action1}"

        return None

    def mine_same_video_negative_segment(
        self,
        gt_start: float,
        gt_end: float,
        video_duration: float,
        max_tiou: float = 0.1,
        max_tries: int = 30
    ) -> Tuple[float, float]:
        """
        Sample a segment within the same video that has low or zero overlap (tIoU < max_tiou).
        """
        gt_span = (gt_start, gt_end)
        seg_len = max(2.0, gt_end - gt_start)

        for _ in range(max_tries):
            max_start = max(0.1, video_duration - seg_len)
            candidate_start = self.rng.uniform(0.0, max_start)
            candidate_end = min(video_duration, candidate_start + seg_len)

            if compute_tiou((candidate_start, candidate_end), gt_span) <= max_tiou:
                return round(candidate_start, 2), round(candidate_end, 2)

        # Fallback: segment at opposite end
        if gt_start > video_duration / 2:
            return 0.0, min(seg_len, video_duration * 0.4)
        else:
            return max(0.0, video_duration - seg_len), video_duration

    def generate_all_negatives_for_query(
        self,
        query: str,
        gt_start: float,
        gt_end: float,
        video_duration: float
    ) -> Dict[str, Any]:
        """Return all synthesized negative variations for a grounding query."""
        results = {
            "original_query": query,
            "action_reversal": self.generate_action_reversal(query),
            "object_swap": self.generate_object_swap(query),
            "order_reversal": self.generate_order_reversal(query),
            "wrong_segment": self.mine_same_video_negative_segment(gt_start, gt_end, video_duration)
        }
        return results
