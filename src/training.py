import pathlib

# import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional import scale_invariant_signal_distortion_ratio
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

dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{number_of_train}_{number_of_test}",
    number_of_train=number_of_train,
    number_of_test=number_of_test,
    max_tensor_size=max_tensor_size,
    trust_remote_code=True,
)

train_set = dataset["train"]
test_set = dataset["test"]


# define data loaders
batch_size = 4
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False
)

print("total training batch number: {}".format(len(train_loader)))
print("total testing batch number: {}".format(len(test_loader)))


model = HybridTransformerCNN(max_tensor_size, 512, 2)

model.to(device)  # puts model on GPU / CPU

# optimization hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # try lr=0.01, momentum=0.9
loss_fn = PermutationInvariantTraining(scale_invariant_signal_distortion_ratio)

# main loop (train+test)
for epoch in tqdm(range(10)):
    # training
    model.train()  # mode "train" agit sur "dropout" ou "batchnorm"
    print(train_loader)
    exit()
    for batch_idx, element in enumerate(train_loader):
        print(element)
        exit()
        audio1, audio2, mixed_audio = (
            element["audio1"][0],
            element["audio2"][0],
            element["mixed_audio"][0],
        )
        x = mixed_audio
        print(x)
        exit()
        target = (audio1, audio2)
        optimizer.zero_grad()
        x = torch.stack([tensor.float().to(device) for tensor in x])
        target = (
            [t.float().to(device) for t in target[0]],
            [t.float().to(device) for t in target[1]],
        )
        # fonctionne jusqu'ici
        print(x)
        print(x._nnz())
        exit()
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
                    len(train_loader.dataset),
                    loss.item(),
                )
            )
    # testing
    model.eval()
    correct = 0
    with torch.no_grad():
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            # _, prediction = torch.max(out.data, 1)
            prediction = out.argmax(
                dim=1, keepdim=True
            )  # index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            # since 'prediction' and 'target' may be on the GPU memory
            # thus (i,j) are on the GPU as well. They must be transfered
            # to the CPU, where 'confusion' has been allocated
            for i, j in zip(prediction, target):
                confusion[i.to("cpu"), j.to("cpu")] += 1
    taux_classif = 100.0 * correct / len(test_loader.dataset)
    print(
        "Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)".format(
            correct, len(test_loader.dataset), taux_classif, 100.0 - taux_classif
        )
    )
    torch.set_printoptions(sci_mode=False)
    print("Confusion matrix:")
    print(confusion.int().numpy())  # or e.g print(confusion.to(torch.int16))
