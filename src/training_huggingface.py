import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from hybrid_transformer_hf import HybridTransformerCNN, HybridTransformerCNNConfig
from parameters import *
from training_func import compute_metrics, preprocess_function

# Configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)


# Chargement du dataset
dataset = load_dataset(
    path="./mixed_dataset",
    name=f"audio_deepl_{NUMBER_OF_TRAIN}_{NUMBER_OF_TEST}",
    number_of_train=NUMBER_OF_TRAIN,
    number_of_test=NUMBER_OF_TEST,
    max_tensor_size=MAX_TENSOR_SIZE,
    win_length=DEFAULT_WIN_LENGTH,
    win_steps=DEFAULT_WIN_STEPS,
    n_mels=DEFAULT_N_MELS,
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
    d_model=D_MODEL,
    n_head=N_HEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    loss=loss,
).from_pretrained("res_lr0.001_dm128_nh4_nel2_ndl2/checkpoint-2000")

model = HybridTransformerCNN(model_config)
model.to(device)

print(
    f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
)


training_args = TrainingArguments(
    output_dir=f"./res_lr{LEARNING_RATE}_dm{D_MODEL}_nh{N_HEAD}_nel{NUM_ENCODER_LAYERS}_ndl{NUM_DECODER_LAYERS}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    max_grad_norm=1.0,
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

# Test the model on one example from the train dataset
example = train_dataset[0]
inputs = {
    key: torch.tensor([value]).to(device)
    for key, value in example.items()
    if key != "labels"
}
labels = torch.tensor([example["labels"]]).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    torchaudio.save("data/output/output.wav", outputs, 16000)
