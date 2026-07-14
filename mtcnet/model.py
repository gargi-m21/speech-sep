import torch
import torch.nn as nn
from typing import Dict, Any

from mtcnet.config import MTCNetConfig
from mtcnet.encoder import Conv1DEncoder
from mtcnet.decoder import Conv1DDecoder
from mtcnet.backbone import SharedMossFormerBackbone
from mtcnet.tda_counting_head import TDACountingHead
from mtcnet.masking_head import DynamicMossFormerMaskingHead


class MTCNet(nn.Module):

    def __init__(
        self,
        config: MTCNetConfig
    ):
        super().__init__()

        self.config = config


        # Encoder

        self.encoder = Conv1DEncoder(
            kernel_size=config.encoder.kernel_size,
            stride=config.encoder.kernel_size // 2,
            out_channels=config.encoder.out_channels,
        )


        # Backbone

        self.backbone = SharedMossFormerBackbone(
            num_blocks=config.backbone.num_blocks,
            in_dim=config.encoder.out_channels,
            dim=config.backbone.dim,
            num_heads=config.backbone.num_heads,
            ffn_dim=config.backbone.ffn_dim,
            num_groups=config.backbone.num_groups,
        )


        # Counting Head

        self.counting_head = TDACountingHead(
            dim=config.backbone.dim,
            qmax=config.counting_head.qmax,
            num_layers=config.counting_head.num_layers,
            num_heads=config.counting_head.num_heads,
            ffn_dim=config.counting_head.ffn_dim,
        )


        # Mask Head

        self.masking_head = DynamicMossFormerMaskingHead(
            dim=config.backbone.dim,
            out_dim=config.masking_head.out_dim,
            qmax=config.masking_head.qmax,
        )


        # Decoder

        self.decoder = Conv1DDecoder(
            kernel_size=config.encoder.kernel_size,
            stride=config.encoder.kernel_size // 2,
            in_channels=config.encoder.out_channels,
        )



    def forward(
        self,
        mixture: torch.Tensor,
    ) -> Dict[str, Any]:


        if mixture.dim() != 2:
            raise ValueError(
                f"Expected [B,L], got {tuple(mixture.shape)}"
            )


        B, L = mixture.shape



        # ============================
        # Encoder
        # ============================

        encoder_features = self.encoder(
            mixture
        )

        # Encoder output:
        # [B,C,T]

        # Backbone input:
        # [B,T,C]

        # encoder_features = encoder_features.permute(
        #     0,
        #     2,
        #     1
        # )



        # ============================
        # MossFormer Backbone
        # ============================

        backbone_features = self.backbone(
            encoder_features
        )



        # ============================
        # Speaker Counting
        # ============================

        (
            presence_logits,
            presence_probs,
            predicted_count,
            attractors,

        ) = self.counting_head(
            backbone_features
        )



        # ============================
        # Mask Prediction
        # ============================

        masks = self.masking_head(
    backbone_features,
    attractors,
    predicted_count,
)



        # ============================
        # Separation
        # ============================

        separated_waveforms = []

        for b in range(B):

            # Encoder features for one sample
            # [T,D] -> [1,D,T]

            encoded = (
                encoder_features[b]
                .transpose(0, 1)
                .unsqueeze(0)
            )

            sample_masks = masks[b]

            # [N,T,D] -> [N,D,T]
            sample_masks = sample_masks.transpose(1, 2)

            # Broadcast encoder features
            # encoded      : [1,D,T]
            # sample_masks : [N,D,T]

            masked_features = sample_masks * encoded

            decoded = self.decoder(
                masked_features.transpose(1, 2),
                target_length=L,
            )

            separated_waveforms.append(decoded)

        return {

            "waveforms": separated_waveforms,

            "presence_logits": presence_logits,

            "presence_probs": presence_probs,

            "predicted_count": predicted_count,

            "attractors": attractors,

        }