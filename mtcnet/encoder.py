import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DEncoder(nn.Module):
    """
    Conv-TasNet style encoder.

    Input:
        x : [B, L]

    Output:
        E : [B, T, D]
    """

    def __init__(
        self,
        kernel_size: int = 16,
        stride: int = 8,
        out_channels: int = 256,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : [B, L]

        Returns
        -------
        E : [B, T, D]
        """

        if x.dim() != 2:
            raise ValueError(
                f"Expected input shape [B, L], got {tuple(x.shape)}"
            )

        B, L = x.shape

        # Add channel dimension
        x = x.unsqueeze(1)  # [B,1,L]

        # Pad so Conv1D always works correctly
        if L < self.kernel_size:
            pad = self.kernel_size - L
        else:
            remainder = (L - self.kernel_size) % self.stride
            pad = (self.stride - remainder) % self.stride

        if pad > 0:
            x = F.pad(x, (0, pad))

        # Encode
        E = self.conv(x)              # [B,D,T]
        E = self.activation(E)

        # Channel-last for transformer
        E = E.transpose(1, 2)         # [B,T,D]

        return E