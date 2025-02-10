import pathlib

import datasets
import datasets.search
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from .data_augmentation import augment_data
from .utils import compress_audio_tensor
from .waveform_processor import WaveformProcessor

torch.random.manual_seed(1)


class MixedDatasetConfig(datasets.BuilderConfig):
    def __init__(
        self,
        number_of_train,
        number_of_test,
        max_tensor_size=131072,
        win_length=0.04,
        win_steps=0.01,
        n_mels=20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.number_of_train = number_of_train
        self.number_of_test = number_of_test
        self.max_tensor_size = max_tensor_size
        self.win_length = win_length
        self.win_steps = win_steps
        self.n_mels = n_mels


class MixedDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MixedDatasetConfig

    def __init__(self, *args, **config_kwargs):
        super().__init__(*args, **config_kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description="Data augmented dataset for cocktail party problem based on CommonVoice dataset from Mozilla",
            features=datasets.Features(
                {
                    "audio1": datasets.Sequence(
                        datasets.Sequence(
                            feature=datasets.Value(dtype="float32"),
                        )
                    ),
                    "audio2": datasets.Sequence(
                        datasets.Sequence(
                            feature=datasets.Value(dtype="float32"),
                        )
                    ),
                    "mixed_audio": datasets.Sequence(
                        datasets.Sequence(
                            feature=datasets.Value(dtype="float32"),
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="---",
            citation="---",
        )

    def _split_generators(self, dl_manager):
        data_dir = pathlib.Path(__name__).parent.parent / "data" / "clips"
        files = list(data_dir.glob("*"))
        train_files = files[: int(len(files) * 0.8)]
        test_files = files[int(len(files) * 0.8) :]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files_path": train_files, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files_path": test_files, "split": "test"},
            ),
        ]

    def _generate_examples(self, files_path: list, split: str):
        transform = WaveformProcessor(
            32000,
            self.config.win_length,
            self.config.win_steps,
            self.config.n_mels,
            False,
            None,
        )
        i = 0
        break_point = (
            self.config.number_of_test
            if split == "test"
            else self.config.number_of_train
        )

        for file1 in files_path:
            track_1 = torchaudio.load(str(file1), format="mp3")[0]
            assert torchaudio.load(str(file1), format="mp3")[1] == 32000

            # Skip if too big
            if track_1.shape[1] > self.config.max_tensor_size:
                continue
            # Pad to max_tensor_size
            track_1 = torch.nn.functional.pad(
                track_1, [0, self.config.max_tensor_size - track_1.size(1)], value=0
            )
            transformed_track_1 = transform(track_1).squeeze()

            for file2 in files_path:
                if file1 != file2:
                    if i == break_point:
                        break
                    track_2 = torchaudio.load(str(file2), format="mp3")[0]
                    assert torchaudio.load(str(file2), format="mp3")[1] == 32000

                    # Skip if too big
                    if track_2.shape[1] > self.config.max_tensor_size:
                        continue
                    track_2 = torch.nn.functional.pad(
                        track_2,
                        [0, self.config.max_tensor_size - track_2.size(1)],
                        value=0,
                    )
                    ratio = float(torch.rand(1))
                    mixed = augment_data(track_1, track_2, ratio)

                    # Process STFT and MelSpectrogram

                    transformed_track_2 = transform(track_2).squeeze()
                    mixed = transform(mixed).squeeze()

                    yield i, {
                        "audio1": transformed_track_1,
                        "audio2": transformed_track_2,
                        "mixed_audio": mixed,
                    }
                    i += 1
            if i == break_point:
                break
