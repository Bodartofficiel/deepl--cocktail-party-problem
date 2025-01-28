import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from hybrid_transformer_hf import HybridTransformerCNN, HybridTransformerCNNConfig
from training_func import compute_metrics, preprocess_function

# Configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

# Paramètres d'entraînement
number_of_train = 800
number_of_test = 200
max_tensor_size = 131072
compression_factor = 4
input_size = max_tensor_size // compression_factor
batch_size = 4
num_epochs = 10
learning_rate = 0.05

# Chargement du dataset
dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{number_of_train}_{number_of_test}",
    number_of_train=number_of_train,
    number_of_test=number_of_test,
    max_tensor_size=max_tensor_size,
    compression_factor=compression_factor,
    trust_remote_code=True,
)

train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=["mixed_audio", "audio1", "audio2"],
)
test_dataset = dataset["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=["mixed_audio", "audio1", "audio2"],
)


model_config = HybridTransformerCNNConfig(
    input_size=input_size, output_size=2, loss="pit_si_sdr"
)
model = HybridTransformerCNN(model_config)
model.to(device)


training_args = TrainingArguments(
    output_dir="./results_si_sdr",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=None,  # Pas de tokenizer dans ce cas
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
