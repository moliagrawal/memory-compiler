"""Grounding loss formulation: relevance BCE, Gaussian label smoothed boundaries, and ranking margin."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gaussian_smoothed_targets(
    indices: torch.Tensor,
    seq_len: int,
    sigma: float = 1.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate Gaussian-smoothed label probability distribution over timesteps [0, seq_len - 1].
    Args:
        indices: [B] ground truth integer index for each batch item
        seq_len: total number of timesteps T
        sigma: standard deviation of Gaussian curve
    Returns:
        targets: [B, seq_len] normalized probability distributions
    """
    B = indices.shape[0]
    timesteps = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    center = indices.unsqueeze(1).float()

    dist_sq = (timesteps - center) ** 2
    gaussian = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))

    # Normalize across sequence length
    targets = gaussian / (gaussian.sum(dim=-1, keepdim=True) + 1e-8)
    return targets


class MultiObjectiveGroundingLoss(nn.Module):
    """
    Multi-objective grounding loss:
      L = w_rel * L_rel + w_loc * (L_start + L_end) + w_rank * L_rank
    """

    def __init__(
        self,
        w_rel: float = 1.0,
        w_loc: float = 2.0,
        w_rank: float = 0.5,
        label_smoothing_sigma: float = 1.5,
        margin: float = 0.2
    ):
        super().__init__()
        self.w_rel = w_rel
        self.w_loc = w_loc
        self.w_rank = w_rank
        self.sigma = label_smoothing_sigma
        self.margin = margin

    def forward(
        self,
        rel_logits: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        gt_rel_mask: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            rel_logits: [B, T]
            start_logits: [B, T]
            end_logits: [B, T]
            gt_rel_mask: [B, T] (1.0 for timesteps inside gt segment, 0.0 outside)
            gt_start_idx: [B] integer start index
            gt_end_idx: [B] integer end index
            pad_mask: [B, T] (1 for valid frame, 0 for padding)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of component loss values
        """
        B, T = rel_logits.shape
        device = rel_logits.device

        # 1. Relevance Loss (BCEWithLogits)
        if pad_mask is not None:
            valid_mask = pad_mask.bool()
            bce_loss = F.binary_cross_entropy_with_logits(
                rel_logits[valid_mask],
                gt_rel_mask[valid_mask].float()
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(rel_logits, gt_rel_mask.float())

        # 2. Localization Loss: KLDiv / CE with Gaussian Smoothed Targets
        start_targets = make_gaussian_smoothed_targets(gt_start_idx, T, self.sigma, device=device)
        end_targets = make_gaussian_smoothed_targets(gt_end_idx, T, self.sigma, device=device)

        log_p_start = F.log_softmax(start_logits, dim=-1)
        log_p_end = F.log_softmax(end_logits, dim=-1)

        loss_start = F.kl_div(log_p_start, start_targets, reduction="batchmean")
        loss_end = F.kl_div(log_p_end, end_targets, reduction="batchmean")
        loc_loss = (loss_start + loss_end) / 2.0

        # 3. Ranking Margin Loss: Segment inside GT vs outside GT
        probs = torch.sigmoid(rel_logits)
        pos_scores = (probs * gt_rel_mask).sum(dim=-1) / (gt_rel_mask.sum(dim=-1) + 1e-8)

        neg_mask = (1.0 - gt_rel_mask)
        if pad_mask is not None:
            neg_mask = neg_mask * pad_mask

        neg_scores = (probs * neg_mask).sum(dim=-1) / (neg_mask.sum(dim=-1) + 1e-8)

        # Margin ranking: pos should exceed neg by self.margin
        rank_loss = F.relu(self.margin - (pos_scores - neg_scores)).mean()

        total_loss = self.w_rel * bce_loss + self.w_loc * loc_loss + self.w_rank * rank_loss

        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_rel": float(bce_loss.item()),
            "loss_loc": float(loc_loss.item()),
            "loss_rank": float(rank_loss.item())
        }
