import torch
from loss import pit_si_sdr_loss, pit_si_snr_loss

def compute_metrics(eval_pred):
    predictions, targets = eval_pred
    predictions, targets = torch.tensor(predictions), torch.tensor(targets)
    si_sdr = pit_si_sdr_loss(predictions, targets)
    si_snr = pit_si_snr_loss(predictions, targets)

    return {"si_snr": si_snr, "si_sdr": si_sdr}