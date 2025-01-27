import pathlib

# import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from tqdm import tqdm

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
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)


# import dataset
number_of_train = 4
number_of_test = 2
max_tensor_size = 100000
batch_size = 2

dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{number_of_train}_{number_of_test}",
    number_of_train=number_of_train,
    number_of_test=number_of_test,
    max_tensor_size=max_tensor_size,
    trust_remote_code=True,
)

train_set = dataset["train"].batch(batch_size)
test_set = dataset["test"].batch(batch_size)


# # define data loaders
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_set, batch_size=batch_size, shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(
#     dataset=test_set, batch_size=batch_size, shuffle=False
# )

# print("total training batch number: {}".format(len(train_loader)))
# print("total testing batch number: {}".format(len(test_loader)))


model = HybridTransformerCNN(max_tensor_size, device, 2)

model.to(device)  # puts model on GPU / CPU

# optimization hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # try lr=0.01, momentum=0.9
loss_fn = PermutationInvariantTraining(
    metric_func=scale_invariant_signal_distortion_ratio
).to(device=device)

# main loop (train+test)
for epoch in tqdm(range(10)):
    # training
    model.train()  # mode "train" agit sur "dropout" ou "batchnorm"
    for batch_idx, element in enumerate(train_set):
        audio1, audio2, x = (
            torch.tensor(element["audio1"], device=device),
            torch.tensor(element["audio2"], device=device),
            torch.tensor([element["mixed_audio"]], device=device).permute(1, 0, 2),
        )
        target = torch.stack((audio1, audio2))
        optimizer.zero_grad()
        # fonctionne jusqu'ici

        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "epoch {:2d} batch {:3d} [{:5d}/{:5d}] training loss: {:0.4f}".format(
                    epoch,
                    batch_idx,
                    batch_idx * len(x),
                    len(train_set),
                    loss.item(),
                )
            )
    # testing
    model.eval()
    correct = 0
    with torch.no_grad():
        sum_loss = 0
        for batch_idx, element in enumerate(test_set):
            audio1, audio2, x = (
                torch.tensor(element["audio1"], device=device),
                torch.tensor(element["audio2"], device=device),
                torch.tensor([element["mixed_audio"]], device=device).permute(1, 0, 2),
            )
            target = torch.stack((audio1, audio2))
            out = model(x)
            loss = loss_fn(out, target)
            sum_loss += loss.item()
    print("Average loss:", sum_loss / (batch_idx + 1))
    # torch.set_printoptions(sci_mode=False)
