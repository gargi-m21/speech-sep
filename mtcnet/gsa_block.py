import torch
import torch.nn as nn

from mtcnet.gated_linear_attention import GatedLinearAttention
from mtcnet.fsmn import FSMN


class FeedForward(nn.Module):
    """
    Gated Feed Forward Network
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            dim,
            hidden_dim * 2
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            dim
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x1, gate = self.fc1(x).chunk(
            2,
            dim=-1
        )

        x = x1 * torch.sigmoid(gate)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x



class GSABlock(nn.Module):
    """
    MossFormer2 Gated Single-head Attention Block

    Expected shape everywhere:

        [B,T,D]
    """


    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 1024,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()


        self.dim = dim


        self.norm1 = nn.LayerNorm(dim)


        self.attention = GatedLinearAttention(
            dim=dim,
            dropout=dropout,
        )


        self.norm2 = nn.LayerNorm(dim)


        self.fsmn = FSMN(
            dim=dim,
            num_groups=num_groups,
            dropout=dropout,
        )


        self.norm3 = nn.LayerNorm(dim)


        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=ffn_dim,
            dropout=dropout,
        )



    def _check_shape(self, x, name):

        print(
            f"{name}: {tuple(x.shape)}"
        )

        if x.dim() != 3:
            raise RuntimeError(
                f"{name} must be [B,T,D], got {x.shape}"
            )

        if x.shape[-1] != self.dim:
            raise RuntimeError(
                f"{name} last dimension must be {self.dim}, got {x.shape}"
            )



    def forward(self, x):


        # self._check_shape(
        #     x,
        #     "GSA INPUT"
        # )


        # -------------------------
        # Attention
        # -------------------------

        residual = x


        x = self.norm1(x)


        x = self.attention(x)


        # self._check_shape(
        #     x,
        #     "AFTER ATTENTION"
        # )


        x = residual + x



        # -------------------------
        # FSMN
        # -------------------------

        residual = x


        x = self.norm2(x)


        # self._check_shape(
        #     x,
        #     "BEFORE FSMN"
        # )


        x = self.fsmn(x)


        # self._check_shape(
        #     x,
        #     "AFTER FSMN"
        # )


        x = residual + x



        # -------------------------
        # FFN
        # -------------------------

        residual = x


        x = self.norm3(x)


        x = self.ffn(x)


        # self._check_shape(
        #     x,
        #     "AFTER FFN"
        # )


        x = residual + x


        return x