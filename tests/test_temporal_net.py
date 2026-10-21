"""Unit tests for Temporal Retrieval Network architecture and frozen interface."""

import torch
import pytest
from mc.models.temporal_net import TemporalRetrievalNet
from mc.models.losses import MultiObjectiveGroundingLoss


def test_temporal_net_forward_frozen_interface():
    B = 2
    T = 16
    D = 512

    model = TemporalRetrievalNet(d_in=D, d_hidden=256, num_heads=4, num_layers=2)
    frames = torch.randn(B, T, D)
    query = torch.randn(B, D)
    mask = torch.ones(B, T)

    out = model(frames, query, mask=mask)

    # Check frozen interface
    assert "rel" in out
    assert "start" in out
    assert "end" in out

    assert out["rel"].shape == (B, T)
    assert out["start"].shape == (B, T)
    assert out["end"].shape == (B, T)


def test_multi_objective_loss():
    B = 2
    T = 10
    criterion = MultiObjectiveGroundingLoss()

    rel_logits = torch.randn(B, T)
    start_logits = torch.randn(B, T)
    end_logits = torch.randn(B, T)

    gt_rel = torch.zeros(B, T)
    gt_rel[:, 2:6] = 1.0
    gt_start = torch.tensor([2, 3], dtype=torch.long)
    gt_end = torch.tensor([6, 7], dtype=torch.long)

    loss, loss_dict = criterion(
        rel_logits=rel_logits,
        start_logits=start_logits,
        end_logits=end_logits,
        gt_rel_mask=gt_rel,
        gt_start_idx=gt_start,
        gt_end_idx=gt_end
    )

    assert loss.item() > 0.0
    assert "loss_total" in loss_dict
    assert "loss_rel" in loss_dict
    assert "loss_loc" in loss_dict
    assert "loss_rank" in loss_dict
