import torch
import torch.nn as nn
from typing import Tuple


class TDACountingHead(nn.Module):
    """
    Transformer Decoder Attractor (TDA) Counting Head

    Input
    -----
    M : [B, T, D]

    Outputs
    -------
    presence_logits : [B, Qmax]
    presence_probs  : [B, Qmax]
    predicted_count : [B]
    attractors      : [B, Qmax, D]
    """

    def __init__(
        self,
        dim: int = 256,
        qmax: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.qmax = qmax
        self.dim = dim

        # Learnable speaker queries
        self.speaker_queries = nn.Parameter(
            torch.randn(qmax, dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.presence_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

        mask = torch.triu(
            torch.ones(qmax, qmax),
            diagonal=1,
        ).bool()

        self.register_buffer(
            "tgt_mask",
            mask,
            persistent=False,
        )

    def forward(
        self,
        M: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        B = M.size(0)

        queries = self.speaker_queries.unsqueeze(0).expand(
            B,
            -1,
            -1,
        )

        attractors = self.decoder(
            tgt=queries,
            memory=M,
            tgt_mask=self.tgt_mask,
        )

        presence_logits = self.presence_head(
            attractors
        ).squeeze(-1)

        presence_probs = torch.sigmoid(
            presence_logits
        )

        predicted_count = (
            presence_probs > 0.5
        ).sum(dim=1)

        predicted_count = predicted_count.clamp(
            min=1,
            max=self.qmax,
        )

        return (
            presence_logits,
            presence_probs,
            predicted_count,
            attractors,
        )