from typing import Tuple

import torch
import torch.nn as nn
from torchaudio.transforms import (
    AmplitudeToDB,
    FrequencyMasking,
    MelSpectrogram,
    TimeMasking,
)


class WaveformProcessor(object):
    def __init__(
        self,
        rate: float,
        win_length: float,
        win_step: float,
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float],
    ):
        """
        Args:
            rate: the sampling rate of the waveform
            win_length: the length in second of the window for the STFT
            win_step: the length in second of the step size of the STFT window
            nmels:  the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
        """
        self.nfft = int(win_length * rate)
        self.nstep = int(win_step * rate)
        self.spectro_normalization = spectro_normalization

        modules = [
            MelSpectrogram(
                sample_rate=rate, n_fft=self.nfft, hop_length=self.nstep, n_mels=nmels
            ),
            AmplitudeToDB(),
        ]
        self.transform_tospectro = nn.Sequential(*modules)

        self.transform_augment = None
        if augment:
            time_mask_duration = 0.1  # s.
            time_mask_nsamples = int(time_mask_duration / win_step)
            nmel_mask = nmels // 4

            modules = [FrequencyMasking(nmel_mask), TimeMasking(time_mask_nsamples)]
            self.transform_augment = nn.Sequential(*modules)

    def get_spectro_length(self, waveform_length: int):
        """
        Computes the length of the spectrogram given the length
        of the waveform

        Args:
            waveform_lengths: the number of samples of the waveform

        Returns:
            int: the number of time samples in the spectrogram
        """
        return waveform_length // self.nstep + 1

    def __call__(self, waveforms: torch.Tensor):
        """
        Apply the transformation on the input waveform tensor
        The time dimension is smalled because of the hop_length given
        to the MelSpectrogram object.

        Args:
            waveforms(torch.Tensor) : (B, Tx) waveform
        Returns:
            spectrograms(torch.Tensor):
        """
        # Compute the spectorgram
        spectro = self.transform_tospectro(waveforms)  # (B, n_mels, T)

        # Normalize the spectrogram
        if self.spectro_normalization is not None:
            spectro = (
                spectro - self.spectro_normalization[0]
            ) / self.spectro_normalization[1]

        # Apply data augmentation
        if self.transform_augment is not None:
            spectro = self.transform_augment(spectro)

        # spectrograms is (B, n_mel, T)
        # we permute it to be (B, T, n_mel)
        return spectro.permute(0, 2, 1)
