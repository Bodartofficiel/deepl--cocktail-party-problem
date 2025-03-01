import torch
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, scale_invariant_signal_noise_ratio

def pit_si_sdr_loss(predictions, targets):
    neg_si_sdr = lambda x, y: -1 * scale_invariant_signal_distortion_ratio(x, y)
    pit = PermutationInvariantTraining(neg_si_sdr, eval_func="min").to(
        predictions.device
    )
    return pit(predictions, targets)

def pit_si_snr_loss(predictions, targets):
    neg_si_snr = lambda x, y: -1 * scale_invariant_signal_noise_ratio(x, y)
    pit = PermutationInvariantTraining(neg_si_snr, eval_func="min").to(
        predictions.device
    )
    return pit(predictions, targets)

def pit_si_snr_plus_si_sdr_loss(predictions, targets):
    neg_si_snr = lambda x, y: -1 * scale_invariant_signal_noise_ratio(x, y)
    neg_si_sdr = lambda x, y: -1 * scale_invariant_signal_distortion_ratio(x, y)
    sum_si_snr_sdr = lambda x, y: neg_si_snr(x,y) + neg_si_sdr(x,y)
    pit = PermutationInvariantTraining(sum_si_snr_sdr, eval_func="min").to(
        predictions.device
    )
    return pit(predictions, targets)

if __name__ == "__main__":
    batch_size, n_sources, seq_len = 4, 2, 1000
    predictions = torch.rand(batch_size, n_sources, seq_len)  # Prédictions aléatoires
    targets = torch.rand(batch_size, n_sources, seq_len)  # Cibles aléatoires
    print("PIT SI-SDR:",pit_si_sdr_loss(predictions, targets))
    print("PIT SI-SNR:",pit_si_snr_loss(predictions, targets))
    print("PIT SI-SNR + SI-SDR:",pit_si_snr_plus_si_sdr_loss(predictions, targets))