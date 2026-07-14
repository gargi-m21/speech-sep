import torch
import torch.nn as nn

from mtcnet.gsa_block import GSABlock


class MossFormerBlock(nn.Module):
    """
    MossFormer2 Block

    Architecture

        Input
          │
          ▼
     LayerNorm
          │
          ▼
    Gated Linear Attention
          │
          ▼
       Residual
          │
          ▼
          FSMN
          │
          ▼
       Residual
          │
          ▼
     Gated FeedForward
          │
          ▼
       Residual
          │
          ▼
        Output

    Input:
        [B, T, D]

    Output:
        [B, T, D]
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,      # kept for config compatibility
        ffn_dim: int = 1024,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # num_heads is intentionally unused because
        # MossFormer2 uses single-head gated linear attention.
        _ = num_heads

        self.block = GSABlock(
            dim=dim,
            ffn_dim=ffn_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        x : [B, T, D]

        Returns
        -------
        x : [B, T, D]
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape [B, T, D], got {tuple(x.shape)}"
            )

        return self.block(x)