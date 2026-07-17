import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio.

    Args
    ----
    estimate : [L]
    target   : [L]
    """

    estimate = estimate - estimate.mean()
    target = target - target.mean()

    target_energy = torch.sum(target ** 2) + eps

    projection = (
        torch.sum(estimate * target)
        * target
        / target_energy
    )

    noise = estimate - projection

    ratio = (
        torch.sum(projection ** 2)
        + eps
    ) / (
        torch.sum(noise ** 2)
        + eps
    )

    return 10 * torch.log10(ratio)


class PITLoss(nn.Module):
    """
    Permutation Invariant Training Loss
    using SI-SNR.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> torch.Tensor:

        device = predictions[0].device

        batch_loss = torch.zeros(
            1,
            device=device,
        )

        B = len(predictions)

        for b in range(B):

            pred = predictions[b]
            tgt = targets[b]

            num_spk = tgt.shape[0]

            pred = pred[:num_spk]

            perms = list(
                itertools.permutations(
                    range(num_spk)
                )
            )

            best_loss = None

            for perm in perms:

                loss = 0.0

                for i, j in enumerate(perm):

                    loss -= si_snr(
                        pred[j],
                        tgt[i],
                    )

                loss = loss / num_spk

                if (
                    best_loss is None
                    or loss < best_loss
                ):
                    best_loss = loss

            batch_loss += best_loss

        return batch_loss / B


class MTCNetLoss(nn.Module):
    """
    Total Loss

    L =
        PIT Loss
        +
        lambda * Counting Loss
    """

    def __init__(
        self,
        lambda_count: float = 1.0,
        qmax: int = 5,
    ):
        super().__init__()

        self.lambda_count = lambda_count
        self.qmax = qmax

        self.pit = PITLoss()

    def forward(
        self,
        outputs: dict,
        targets: List[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        device = outputs["presence_logits"].device

        separation_loss = self.pit(
            outputs["waveforms"],
            targets,
        )

        B = len(targets)

        target_presence = torch.zeros(
            B,
            self.qmax,
            device=device,
        )

        for b in range(B):

            n = min(
                targets[b].shape[0],
                self.qmax,
            )

            target_presence[
                b,
                :n,
            ] = 1.0

        counting_loss = F.binary_cross_entropy_with_logits(
            outputs["presence_logits"],
            target_presence,
        )

        total_loss = (
            separation_loss
            + self.lambda_count * counting_loss
        )

        return (
            total_loss,
            separation_loss,
            counting_loss,
        )