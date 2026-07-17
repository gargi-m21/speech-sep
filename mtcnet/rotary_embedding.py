import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Input:
        x : [B, T, D]

    Output:
        x : [B, T, D]
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "RotaryEmbedding requires an even dimension."
            )

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def _build_cache(
        self,
        seq_len: int,
        device,
    ):

        t = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32,
        )

        freqs = torch.einsum(
            "i,j->ij",
            t,
            self.inv_freq,
        )

        emb = torch.cat(
            (
                freqs,
                freqs,
            ),
            dim=-1,
        )

        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]

        return cos, sin

    @staticmethod
    def rotate_half(
        x: torch.Tensor,
    ):

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        y = torch.stack(
            (
                -x2,
                x1,
            ),
            dim=-1,
        )

        return y.flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ):

        """
        Args
        ----
        q : [B,T,D]
        k : [B,T,D]

        Returns
        -------
        q_rot
        k_rot
        """

        seq_len = q.size(1)

        cos, sin = self._build_cache(
            seq_len,
            q.device,
        )

        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin

        return q, k