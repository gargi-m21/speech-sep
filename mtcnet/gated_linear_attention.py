import math
import torch
import torch.nn as nn

from mtcnet.rotary_embedding import RotaryEmbedding


class GatedLinearAttention(nn.Module):
    """
    MossFormer2-style Gated Linear Attention (GSA)

    Input:
        x : [B, T, D]

    Output:
        y : [B, T, D]
    """

    def __init__(
        self,
        dim: int = 256,
        expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.rope = RotaryEmbedding(dim)

        self.output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def feature_map(x: torch.Tensor):

        # Positive kernel feature map
        return torch.nn.functional.elu(x) + 1.0

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        """
        Args
        ----
        x : [B,T,D]

        Returns
        -------
        y : [B,T,D]
        """

        residual = x

        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        gate = self.gate_proj(x)

        # Rotary Position Embedding
        q, k = self.rope(q, k)

        # Linear Attention Feature Map
        q = self.feature_map(q)
        k = self.feature_map(k)

        #
        # Linear Attention
        #

        kv = torch.einsum(
            "btd,bte->bde",
            k,
            v,
        )

        k_sum = k.sum(
            dim=1,
        )

        numerator = torch.einsum(
            "btd,bde->bte",
            q,
            kv,
        )

        denominator = torch.einsum(
            "btd,bd->bt",
            q,
            k_sum,
        )

        denominator = denominator.unsqueeze(-1)

        y = numerator / (
            denominator + 1e-6
        )

        # Gating
        y = gate * y

        y = self.output(y)

        y = self.dropout(y)

        return residual + y