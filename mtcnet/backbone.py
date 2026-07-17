import torch
import torch.nn as nn

from mtcnet.mossformer_block import MossFormerBlock


class SharedMossFormerBackbone(nn.Module):

    def __init__(
        self,
        num_blocks: int = 16,
        in_dim: int = 256,
        dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.dim = dim

        if in_dim != dim:
            self.input_proj = nn.Linear(
                in_dim,
                dim
            )
        else:
            self.input_proj = nn.Identity()


        self.shared_block = MossFormerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_groups=num_groups,
            dropout=dropout,
        )

        self.output_norm = nn.LayerNorm(dim)



    def forward(
        self,
        E: torch.Tensor
    ):

        if E.dim() != 3:
            raise ValueError(
                f"Expected [B,T,D], got {E.shape}"
            )


        # Fix accidental Conv1D format
        # [B,D,T] -> [B,T,D]
        if E.shape[-1] != self.input_feature_dim():

            if E.shape[1] == self.input_feature_dim():
                E = E.transpose(1,2)


        x = self.input_proj(E)


        for _ in range(self.num_blocks):
            x = self.shared_block(x)


        return self.output_norm(x)



    def input_feature_dim(self):

        if isinstance(self.input_proj, nn.Linear):
            return self.input_proj.in_features

        return self.dim