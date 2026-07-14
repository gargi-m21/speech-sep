import torch
import torch.nn as nn
from typing import List


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)

    Feature : [B,T,D]
    Attractor : [B,D]

    Output : [B,T,D]
    """

    def __init__(self, dim: int):
        super().__init__()

        self.gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.beta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        attractor: torch.Tensor,
    ):

        gamma = self.gamma(attractor).unsqueeze(1)
        beta = self.beta(attractor).unsqueeze(1)

        return (1 + gamma) * features + beta


class MaskingSlot(nn.Module):
    """
    Attractor-conditioned Mask Prediction Block.

    Inputs
    ------
    features   : [B,T,D]
    attractor  : [B,D]

    Output
    ------
    mask : [B,T,D]
    """

    def __init__(
        self,
        dim=256,
        out_dim=256,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.condition = FiLMConditioning(dim)

        self.local_conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

        self.norm1 = nn.LayerNorm(dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm3 = nn.LayerNorm(dim)

        self.mask_projection = nn.Linear(
            dim,
            out_dim,
        )

        self.mask_activation = nn.Sigmoid()

    def forward(
        self,
        features: torch.Tensor,
        attractor: torch.Tensor,
    ):
        """
        features : [B,T,D]
        attractor : [B,D]
        """

        x = self.condition(
            features,
            attractor,
        )

        residual = x

        y = x.transpose(1, 2)

        y = self.local_conv(y)

        y = y.transpose(1, 2)

        x = residual + y

        residual = x

        h = self.norm1(x)

        h, _ = self.cross_attention(
            h,
            h,
            h,
            need_weights=False,
        )

        x = residual + h

        residual = x

        h = self.norm2(x)

        h = self.ffn(h)

        x = residual + h

        x = self.norm3(x)

        mask = self.mask_projection(x)

        mask = self.mask_activation(mask)

        return mask


class DynamicMossFormerMaskingHead(nn.Module):
    """
    Attractor-conditioned Dynamic Mask Prediction Head.

    Inputs
    ------
    M          : [B,T,D]
    attractors : [B,Qmax,D]
    counts     : [B]

    Training
    --------
    Computes masks for all Qmax attractors.

    Inference
    ---------
    Computes masks only for predicted speakers.
    """

    def __init__(
        self,
        dim: int = 256,
        out_dim: int = 256,
        qmax: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.qmax = qmax

        self.slots = nn.ModuleList(
            [
                MaskingSlot(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(qmax)
            ]
        )

    def forward(
        self,
        M: torch.Tensor,
        attractors: torch.Tensor,
        counts: torch.Tensor = None,
    ) -> List[torch.Tensor]:

        B = M.size(0)

        outputs = []

        # =====================================================
        # TRAINING
        # =====================================================

        if self.training:

            for b in range(B):

                sample_masks = []

                features = M[b].unsqueeze(0)

                for i in range(self.qmax):

                    attractor = attractors[
                        b,
                        i,
                    ].unsqueeze(0)

                    mask = self.slots[i](
                        features,
                        attractor,
                    )

                    sample_masks.append(
                        mask.squeeze(0)
                    )

                sample_masks = torch.stack(
                    sample_masks,
                    dim=0,
                )

                outputs.append(
                    sample_masks
                )

            return outputs

        # =====================================================
        # INFERENCE
        # =====================================================

        if counts is None:

            counts = torch.full(
                (B,),
                self.qmax,
                dtype=torch.long,
                device=M.device,
            )

        for b in range(B):

            num_spk = int(
                counts[b].item()
            )

            num_spk = max(
                1,
                min(
                    num_spk,
                    self.qmax,
                ),
            )

            sample_masks = []

            features = M[
                b
            ].unsqueeze(0)
            for i in range(num_spk):

                attractor = attractors[
                    b,
                    i,
                ].unsqueeze(0)

                mask = self.slots[i](
                    features,
                    attractor,
                )

                sample_masks.append(
                    mask.squeeze(0)
                )

            sample_masks = torch.stack(
                sample_masks,
                dim=0,
            )

            outputs.append(
                sample_masks
            )
        return outputs