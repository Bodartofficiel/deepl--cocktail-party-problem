import numpy as np
import torch
from torchmetrics.functional import permutation_invariant_training
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from loss import pit_mse_loss


def preprocess_function(batch):
    input_ids = torch.tensor(batch["mixed_audio"])
    labels = torch.stack(
        [torch.tensor(batch["audio1"]), torch.tensor(batch["audio2"])], dim=1
    )
    return {"input_ids": input_ids, "labels": labels}


def compute_metrics(eval_pred):
    predictions, targets = eval_pred
    predictions, targets = torch.tensor(predictions), torch.tensor(targets)
    loss = pit_mse_loss(predictions, targets)
    si_sdr = permutation_invariant_training(
        predictions, targets, scale_invariant_signal_distortion_ratio
    )[0].mean()

    return {"loss": loss.item(), "si_sdr": si_sdr}
