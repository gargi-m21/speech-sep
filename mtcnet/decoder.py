import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DDecoder(nn.Module):
    """
    Conv-TasNet style decoder.

    Input:
        Z : [B, T, D]

    Output:
        y : [B, L]
    """

    def __init__(
        self,
        kernel_size: int = 16,
        stride: int = 8,
        in_channels: int = 256,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels

        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(
        self,
        z: torch.Tensor,
        target_length: int = None,
    ) -> torch.Tensor:
        """
        Args
        ----
        z : [B, T, D]
        target_length : original waveform length

        Returns
        -------
        y : [B, L]
        """

        if z.dim() != 3:
            raise ValueError(
                f"Expected input shape [B, T, D], got {tuple(z.shape)}"
            )

        # Convert to channel-first
        z = z.transpose(1, 2)  # [B,D,T]

        # Decode
        y = self.deconv(z)     # [B,1,L]
        y = y.squeeze(1)       # [B,L]

        if target_length is not None:

            current_length = y.size(-1)

            if current_length > target_length:
                y = y[..., :target_length]

            elif current_length < target_length:
                pad = target_length - current_length
                y = F.pad(y, (0, pad))

        return y