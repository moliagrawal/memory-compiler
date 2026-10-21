"""Unit tests for Charades-STA parsing, validation, and video-level splitting."""

import pytest
from mc.data.charades_sta import (
    parse_annotation_line,
    split_train_val,
    compute_dataset_stats,
    AnnotationRecord,
    CharadesSTADataset
)


def test_parse_valid_annotation_line():
    line = "46GG8 12.3 18.5 ## person is opening the refrigerator"
    rec = parse_annotation_line(line, 0)
    assert rec is not None
    assert rec.video_id == "46GG8"
    assert rec.start == 12.3
    assert rec.end == 18.5
    assert rec.query == "person is opening the refrigerator"
    assert rec.verb in ["open", "opening"]


def test_parse_invalid_annotation_line():
    # End before start
    assert parse_annotation_line("46GG8 18.5 12.3 ## invalid", 0) is None
    # Negative start
    assert parse_annotation_line("46GG8 -1.0 12.3 ## invalid", 0) is None
    # Empty
    assert parse_annotation_line("", 0) is None
    # Comment
    assert parse_annotation_line("# this is a comment", 0) is None


def test_video_level_split_no_leakage():
    # Create 30 records across 10 unique videos
    records = []
    for vid_num in range(10):
        vid = f"vid_{vid_num:03d}"
        for q_idx in range(3):
            records.append(AnnotationRecord(
                query_id=f"{vid}_q{q_idx}",
                video_id=vid,
                start=float(q_idx * 5),
                end=float(q_idx * 5 + 4),
                query=f"person does action {q_idx}"
            ))

    train_recs, val_recs = split_train_val(records, val_ratio=0.20, seed=42)

    train_vids = {r.video_id for r in train_recs}
    val_vids = {r.video_id for r in val_recs}

    # CRITICAL: Verify zero video overlap between train and val
    assert len(train_vids & val_vids) == 0
    assert len(val_vids) == 2
    assert len(train_vids) == 8


def test_dataset_stats_report():
    records = [
        AnnotationRecord("q1", "v1", 0.0, 5.0, "person walks"),
        AnnotationRecord("q2", "v2", 2.0, 8.0, "person opens refrigerator door")
    ]
    stats = compute_dataset_stats(records)
    assert stats["total_queries"] == 2
    assert stats["unique_videos"] == 2
    assert stats["mean_duration"] == 5.5
