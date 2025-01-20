import pathlib

import ffmpeg
import numpy as np
import torch.nn.modules as nn
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

number_of_train = 400
number_of_test = 100
max_tensor_size = 100000

# path = pathlib.Path("./data/clips")

# iterator = list(path.glob("*"))[0]
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


## Vizualize the audio source
# import matplotlib.pyplot as plt

# # Charger le fichier sans normalisation
# audio, sample_rate = torchaudio.load(str(iterator), normalize=False)

# # Afficher les premiers échantillons
# plt.plot(audio[0].numpy())  # Seulement les 1000 premiers échantillons
# plt.title("Valeurs des échantillons audio bruts")
# plt.xlabel("Échantillons")
# plt.ylabel("Amplitude")
# plt.show()

# exit()

dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{number_of_train}_{number_of_test}",
    max_tensor_size=max_tensor_size,
    number_of_train=number_of_train,
    number_of_test=number_of_test,
    trust_remote_code=True,
)
