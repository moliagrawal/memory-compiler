"""Multi-task grounding heads: relevance, start boundary, and end boundary logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskGroundingHeads(nn.Module):
    """
    Multi-task heads attached to temporal representations [B, T, D_hidden]:
      1. Relevance Head: MLP predicting binary segment relevance logit per timestep [B, T]
      2. Start Boundary Head: MLP predicting start probability distribution over T [B, T]
      3. End Boundary Head: MLP predicting end probability distribution over T [B, T]
    """

    def __init__(self, d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_hidden = d_hidden

        # Relevance Head: [B, T, D] -> [B, T]
        self.rel_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

        # Start Boundary Head: [B, T, D] -> [B, T]
        self.start_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

        # End Boundary Head: [B, T, D] -> [B, T]
        self.end_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1)
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: [B, T, D_hidden]
            mask: [B, T] (1 for valid frames, 0 for padding)

        Returns:
            rel_logits: [B, T]
            start_logits: [B, T]
            end_logits: [B, T]
        """
        B, T, _ = features.shape

        rel_logits = self.rel_head(features).squeeze(-1)    # [B, T]
        start_logits = self.start_head(features).squeeze(-1)  # [B, T]
        end_logits = self.end_head(features).squeeze(-1)      # [B, T]

        if mask is not None:
            fill_val = -1e4
            rel_logits = rel_logits.masked_fill(~mask.bool(), fill_val)
            start_logits = start_logits.masked_fill(~mask.bool(), fill_val)
            end_logits = end_logits.masked_fill(~mask.bool(), fill_val)

        return rel_logits, start_logits, end_logits
