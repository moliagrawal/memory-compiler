"""Temporal Retrieval Network (B3): Lightweight Transformer with query-conditioned fusion."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from mc.models.heads import MultiTaskGroundingHeads


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class QueryConditionedFusion(nn.Module):
    """
    Query-conditioned multimodal fusion:
      s_t = MLP([h_t; q; h_t * q])
    Concatenates visual representation, query representation, and their element-wise (Hadamard) product.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # In: [h_t; q; h_t * q] has dimension 3 * d_model
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, h_frames: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_frames: [B, T, D]
            q_emb: [B, D]
        Returns:
            fused: [B, T, D]
        """
        B, T, D = h_frames.shape
        q_expanded = q_emb.unsqueeze(1).expand(B, T, D)
        hadamard = h_frames * q_expanded
        cat_feat = torch.cat([h_frames, q_expanded, hadamard], dim=-1)  # [B, T, 3*D]
        return self.fusion_mlp(cat_feat)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion variant for ablation comparisons."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_frames: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        # q_emb: [B, D] -> [B, 1, D]
        q_seq = q_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=h_frames, key=q_seq, value=q_seq)
        return self.norm(h_frames + attn_out)


class TemporalRetrievalNet(nn.Module):
    """
    Temporal Retrieval Network (B3):
      1. Linear input projection D_in (512) -> D_hidden (256)
      2. Positional encoding
      3. Transformer Encoder (3 layers, 4 heads, dropout 0.1)
      4. Query-conditioned fusion (Hadamard + MLP, or Cross-Attention)
      5. Multi-task grounding heads: relevance, start, end logits
    """

    def __init__(
        self,
        d_in: int = 512,
        d_hidden: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        cross_attention: bool = False,
        use_transformer: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.use_transformer = use_transformer

        # Input projection
        self.proj_frames = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_query = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_hidden, max_len=max_seq_len)

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_hidden,
                nhead=num_heads,
                dim_feedforward=d_hidden * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Ablation A1: Simple frame-level MLP without temporal attention
            self.transformer = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        if cross_attention:
            self.fusion = CrossAttentionFusion(d_hidden, num_heads=num_heads, dropout=dropout)
        else:
            self.fusion = QueryConditionedFusion(d_hidden, dropout=dropout)

        self.heads = MultiTaskGroundingHeads(d_hidden=d_hidden, dropout=dropout)

    def forward(
        self,
        frames: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Frozen interface:
        forward(frames[B, T, D], query[B, D], mask[B, T]) -> {rel[B, T], start[B, T], end[B, T]}
        """
        # Linear projections
        h_frames = self.proj_frames(frames)  # [B, T, D_hidden]
        q_emb = self.proj_query(query)       # [B, D_hidden]

        # Add positional encoding
        h_frames = self.pos_encoder(h_frames)

        # Temporal Context Modeling
        if self.use_transformer:
            # key_padding_mask: True for positions that should be masked out
            key_padding_mask = (~mask.bool()) if mask is not None else None
            h_temporal = self.transformer(h_frames, src_key_padding_mask=key_padding_mask)
        else:
            h_temporal = self.transformer(h_frames)

        # Multimodal Query-Conditioned Fusion
        fused = self.fusion(h_temporal, q_emb)

        # Multi-task heads
        rel_logits, start_logits, end_logits = self.heads(fused, mask=mask)

        return {
            "rel": rel_logits,
            "start": start_logits,
            "end": end_logits,
            "features": fused
        }
