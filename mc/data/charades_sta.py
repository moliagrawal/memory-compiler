"""Charades-STA dataset parsing, validation, video-level splitting, and statistics."""

import os
import re
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any


COMMON_VERBS = {
    "open", "close", "take", "put", "pour", "drink", "eat", "wash", "hold",
    "sit", "stand", "walk", "run", "throw", "cut", "clean", "watch", "read",
    "turn", "wear", "tidy", "cook", "smile", "laugh", "talk", "play", "enter",
    "leave", "pick", "drop", "look", "wipe", "carry", "touch", "stir"
}


@dataclass
class AnnotationRecord:
    query_id: str
    video_id: str
    start: float
    end: float
    query: str
    duration: Optional[float] = None
    verb: Optional[str] = None

    @property
    def segment_duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


STOPWORDS = {"person", "someone", "a", "an", "the", "is", "are", "and", "then", "of", "to", "in", "on", "at"}


def extract_primary_verb(sentence: str) -> str:
    """Extract primary action verb from query string."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in COMMON_VERBS:
            return token
        # Check proper suffixes
        if token.endswith("ing"):
            stem = token[:-3]
            if stem in COMMON_VERBS:
                return stem
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in COMMON_VERBS:
                return stem[:-1]
        elif token.endswith("ed"):
            stem = token[:-2]
            if stem in COMMON_VERBS or (stem + "e") in COMMON_VERBS:
                return stem if stem in COMMON_VERBS else (stem + "e")
        elif token.endswith("s"):
            stem = token[:-1]
            if stem in COMMON_VERBS:
                return stem

    # Second pass: first non-stopword
    for token in tokens:
        if token not in STOPWORDS:
            return token

    return "action"


def parse_annotation_line(line: str, line_idx: int = 0) -> Optional[AnnotationRecord]:
    """
    Parse a single Charades-STA annotation line.
    Format: 'video_id start end ## query text'
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if "##" in line:
        meta_part, query_part = line.split("##", 1)
        query = query_part.strip()
    else:
        # Fallback if delimiter is different
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            return None
        meta_part = f"{parts[0]} {parts[1]} {parts[2]}"
        query = parts[3].strip()

    tokens = meta_part.strip().split()
    if len(tokens) < 3:
        return None

    video_id = tokens[0].strip()
    try:
        start = float(tokens[1])
        end = float(tokens[2])
    except ValueError:
        return None

    if start < 0 or end <= start:
        return None

    verb = extract_primary_verb(query)
    query_id = f"{video_id}_{start:.2f}_{end:.2f}_{line_idx}"

    return AnnotationRecord(
        query_id=query_id,
        video_id=video_id,
        start=round(start, 2),
        end=round(end, 2),
        query=query,
        verb=verb
    )


def load_annotations(filepath: str) -> List[AnnotationRecord]:
    """Load and parse all annotations from a text file, with deduplication."""
    records = []
    seen = set()
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = parse_annotation_line(line, line_idx=idx)
            if rec is None:
                continue
            key = (rec.video_id, rec.start, rec.end, rec.query.lower())
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
    return records


def split_train_val(
    records: List[AnnotationRecord],
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[AnnotationRecord], List[AnnotationRecord]]:
    """
    Carve validation set strictly by VIDEO ID (seeded).
    Never split by query to prevent video leakage between train and val!
    """
    rng = random.Random(seed)
    video_ids = sorted(list({r.video_id for r in records}))
    rng.shuffle(video_ids)

    n_val = max(1, int(len(video_ids) * val_ratio))
    val_vids = set(video_ids[:n_val])

    train_records = [r for r in records if r.video_id not in val_vids]
    val_records = [r for r in records if r.video_id in val_vids]

    return train_records, val_records


def compute_dataset_stats(records: List[AnnotationRecord]) -> Dict[str, Any]:
    """Generate comprehensive statistical report for annotations."""
    if not records:
        return {
            "total_queries": 0,
            "unique_videos": 0,
            "mean_duration": 0.0,
            "mean_query_words": 0.0,
            "verbs_histogram": {}
        }

    durations = [r.segment_duration for r in records]
    query_lengths = [len(r.query.split()) for r in records]
    unique_videos = {r.video_id for r in records}
    verbs = [r.verb for r in records if r.verb]

    verb_counts = Counter(verbs).most_common(20)

    stats = {
        "total_queries": len(records),
        "unique_videos": len(unique_videos),
        "mean_duration": round(sum(durations) / len(durations), 2),
        "min_duration": round(min(durations), 2),
        "max_duration": round(max(durations), 2),
        "mean_query_words": round(sum(query_lengths) / len(query_lengths), 2),
        "min_query_words": min(query_lengths),
        "max_query_words": max(query_lengths),
        "top_verbs": dict(verb_counts)
    }
    return stats


class CharadesSTADataset:
    """Dataset wrapper for Charades-STA splits."""
    def __init__(self, records: List[AnnotationRecord]):
        self.records = records
        self._by_video = {}
        for r in records:
            self._by_video.setdefault(r.video_id, []).append(r)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> AnnotationRecord:
        return self.records[idx]

    def get_by_video(self, video_id: str) -> List[AnnotationRecord]:
        return self._by_video.get(video_id, [])

    def all_video_ids(self) -> List[str]:
        return list(self._by_video.keys())

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "CharadesSTADataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [AnnotationRecord(**item) for item in data]
        return cls(records)
