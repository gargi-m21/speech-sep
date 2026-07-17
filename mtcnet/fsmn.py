import torch
import torch.nn as nn


class ConvUBlock(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.linear = nn.Linear(
            dim,
            dim
        )

        self.act = nn.SiLU()

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.linear(x)

        x = self.act(x)


        # [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)

        x = self.conv(x)

        # [B,D,T] -> [B,T,D]
        x = x.transpose(1, 2)

        x = self.dropout(x)

        return residual + x



class DilatedFSMNBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_groups=32,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )


        self.memory_conv = nn.ModuleList()

        for i in range(num_layers):

            self.memory_conv.append(
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=2 ** i,
                    dilation=2 ** i,
                    groups=dim
                )
            )


        self.norm = nn.LayerNorm(dim)



    def forward(self, x):

        residual = x

        x = self.ffn(x)


        # [B,T,D] -> [B,D,T]
        y = x.transpose(1, 2)


        for conv in self.memory_conv:
            y = y + conv(y)


        # [B,D,T] -> [B,T,D]
        y = y.transpose(1, 2)


        y = self.norm(y)


        return residual + y





class FSMN(nn.Module):

    def __init__(
        self,
        dim,
        num_groups=32,
        bottleneck_dim=None,
        kernel_size=3,
        num_memory_layers=4,
        dropout=0.1,
    ):
        super().__init__()


        if bottleneck_dim is None:
            bottleneck_dim = dim



        self.input_norm = nn.LayerNorm(dim)


        # FIXED:
        # Removed PReLU because it expects [B,C,T]
        self.input_proj = nn.Linear(
            dim,
            bottleneck_dim
        )


        self.conv_u = ConvUBlock(
            bottleneck_dim,
            kernel_size,
            dropout
        )


        self.conv_v = ConvUBlock(
            bottleneck_dim,
            kernel_size,
            dropout
        )


        self.memory = DilatedFSMNBlock(
            bottleneck_dim,
            num_groups,
            num_memory_layers,
            dropout
        )


        self.output_norm = nn.LayerNorm(
            bottleneck_dim
        )


        self.output_proj = nn.Linear(
            bottleneck_dim,
            dim
        )


        self.dropout = nn.Dropout(dropout)



    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Input:
            x: [B,T,D]

        Output:
            y: [B,T,D]
        """

        if x.dim() != 3:
            raise ValueError(
                f"FSMN expected [B,T,D], got {x.shape}"
            )


        residual = x


        # [B,T,D]
        h = self.input_norm(x)


        # Linear projection
        h = self.input_proj(h)


        h = torch.nn.functional.gelu(h)


        # Local convolution blocks
        h = self.conv_u(h)

        h = self.conv_v(h)


        # FSMN long memory
        h = self.memory(h)


        h = self.output_norm(h)


        h = self.output_proj(h)


        h = self.dropout(h)


        return residual + h