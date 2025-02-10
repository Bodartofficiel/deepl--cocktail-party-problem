import itertools

import torch
import torch.nn.functional as F
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def pit_mse_loss(predictions, targets):
    pit = PermutationInvariantTraining(F.mse_loss, eval_func="min").to(
        predictions.device
    )
    return pit(predictions, targets)


def pit_spectral_loss(pred_mel, target_mel, eps=1e-8):
    loss = lambda pred_mel, target_mel: torch.norm(pred_mel - target_mel, p="fro") / (
        torch.norm(target_mel, p="fro") + eps
    )
    pit = PermutationInvariantTraining(loss, eval_func="min").to(pred_mel.device)
    return pit(pred_mel, target_mel)


def pit_si_sdr_loss(predictions, targets):
    neg_si_sdr = lambda x, y: -1 * scale_invariant_signal_distortion_ratio(x, y)
    pit = PermutationInvariantTraining(neg_si_sdr, eval_func="min").to(
        predictions.device
    )
    return pit(predictions, targets)


if __name__ == "__main__":
    batch_size, n_sources, seq_len = 4, 2, 1000
    predictions = torch.rand(batch_size, n_sources, seq_len)  # Prédictions aléatoires
    targets = torch.rand(batch_size, n_sources, seq_len)  # Cibles aléatoires
