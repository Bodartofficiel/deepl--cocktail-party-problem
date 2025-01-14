import torch
import torchaudio


def augment_data(track_1: torch.Tensor, track_2: torch.Tensor, ratio: float):
    assert 0 < ratio < 1, "ratio for weighted mean should be between 0 and 1"
    track_1 = torch.mul(track_1, ratio)
    track_2 = torch.mul(track_2, (1 - ratio))

    return track_1 + track_2
