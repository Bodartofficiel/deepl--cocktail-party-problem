import pathlib

# import ffmpeg
import numpy as np
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_transformer_cnn import HybridTransformerCNN

# path = pathlib.Path("./data/clips")

# iterator = path.glob("*")
# shapes = []
# for audio_file in tqdm(iterator):
#     tensor, _ = torchaudio.load(str(audio_file))
#     shapes.append(tensor.shape[0])


# median_shape = int(np.median(shapes))
# std_shape = int(np.std(shapes))
# print(f"Standard deviation of the second dimension: {std_shape}")
# print(f"Median shape of the second dimension: {median_shape}")
# # Median shape of the second dimension: 186624
# exit()


# we use GPU if available, otherwise CPU
# NB: with several GPUs, "cuda" --> "cuda:0" or "cuda:1"...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f'({torch.cuda.get_device_name(device)})' if torch.cuda.is_available() else '')


# import dataset
number_of_train = 4
number_of_test = 1

dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{number_of_train}_{number_of_test}",
    number_of_train=number_of_train,
    number_of_test=number_of_test,
    trust_remote_code=True,
)

train_set = dataset["train"]
test_set = dataset["test"]


# define data loaders
batch_size = 100
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('total training batch number: {}'.format(len(train_loader)))
print('total testing batch number: {}'.format(len(test_loader)))

model = HybridTransformerCNN()
