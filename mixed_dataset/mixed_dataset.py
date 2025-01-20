import pathlib

import datasets
import numpy as np
import torch
import torchaudio

from .data_augmentation import augment_data

torch.random.manual_seed(1)


class MixedDatasetConfig(datasets.BuilderConfig):
    def __init__(
        self, number_of_train, number_of_test, max_tensor_size=100000, **kwargs
    ):
        super().__init__(**kwargs)
        self.number_of_train = number_of_train
        self.number_of_test = number_of_test
        self.max_tensor_size = max_tensor_size


class MixedDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = MixedDatasetConfig

    def __init__(self, *args, **config_kwargs):
        super().__init__(*args, **config_kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description="Data augmented dataset for cocktail party problem based on Common Voice dataset from Mozilla",
            features=datasets.Features(
                {
                    "audio1": datasets.Array2D(
                        (1, self.config.max_tensor_size), "float32"
                    ),
                    "audio2": datasets.Array2D(
                        (1, self.config.max_tensor_size), "float32"
                    ),
                    "mixed_audio": datasets.Array2D(
                        (1, self.config.max_tensor_size), "float32"
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

    def _generate_examples(self, files_path: pathlib.Path, split):
        i = 0
        break_point = (
            self.config.number_of_test
            if split == "test"
            else self.config.number_of_train
        )
        for file1 in files_path:
            track_1 = torchaudio.load(str(file1), format="mp3")[0]
            # Skip if too big
            if track_1.shape[1] > self.config.max_tensor_size:
                continue
            for file2 in files_path:
                if file1 != file2:
                    if i == break_point:
                        break
                    track_2 = torchaudio.load(str(file2), format="mp3")[0]
                    # Skip if too big
                    if track_2.shape[1] > self.config.max_tensor_size:
                        continue

                    track_1 = torch.nn.functional.pad(
                        track_1, (0, self.config.max_tensor_size - track_1.shape[1])
                    )
                    track_2 = torch.nn.functional.pad(
                        track_2, (0, self.config.max_tensor_size - track_2.shape[1])
                    )

                    ratio = float(torch.rand(1))

                    yield i, {
                        "audio1": track_1,
                        "audio2": track_2,
                        "mixed_audio": augment_data(track_1, track_2, ratio),
                    }
                    i += 1
            if i == break_point:
                break