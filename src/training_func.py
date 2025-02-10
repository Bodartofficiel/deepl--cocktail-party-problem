import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from torchmetrics.audio.pit import permutation_invariant_training
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from loss import pit_mse_loss


def preprocess_function(batch):
    #     rate = 32000  # Sample rate of the audio
    #     win_length = 40 * 1e-3  # Size of FFT
    #     win_step = 10 * 1e-3  # Step size between FFT windows
    #     n_mels = 20  # Number of Mel bins

    #     transform = WaveformProcessor(rate, win_length, win_step, n_mels, False, None)

    input_ids = torch.tensor(batch["mixed_audio"])

    #     audio1 = transform()
    #     audio2 = transform()

    labels = torch.stack(
        [torch.tensor(batch["audio1"]), torch.tensor(batch["audio2"])], dim=1
    )
    #     print(input_ids.shape)
    #     exit()

    return {"input_ids": input_ids, "labels": labels}


def compute_metrics(eval_pred):
    predictions, targets = eval_pred
    predictions, targets = torch.tensor(predictions), torch.tensor(targets)
    loss = pit_mse_loss(predictions, targets)
    loss_values, _ = permutation_invariant_training(
        predictions.view(predictions.size(0), predictions.size(1), -1),
        targets.view(targets.size(0), targets.size(1), -1),
        scale_invariant_signal_distortion_ratio,
    )

    si_sdr = loss_values.mean()

    return {"loss": loss.item(), "si_sdr": si_sdr}
