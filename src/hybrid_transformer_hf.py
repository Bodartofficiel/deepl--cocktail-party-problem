from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from loss import pit_mse_loss, pit_si_sdr_loss, pit_spectral_loss


class HybridTransformerCNNConfig(PretrainedConfig):
    def __init__(
        self,
        d_model=64,
        n_head=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        loss: Literal["pit_mse", "pit_si_sdr", "pit_spectral"] = "pit_mse",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.loss = loss


class HybridTransformerCNN(PreTrainedModel):
    config_class = HybridTransformerCNNConfig

    def __init__(self, config: HybridTransformerCNNConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.loss = {
            "pit_mse": pit_mse_loss,
            "pit_si_sdr": pit_si_sdr_loss,
            "pit_spectral": pit_spectral_loss,
        }[config.loss]

        # *************************************************
        # ****************** Encoding *********************
        # *************************************************

        self.conv1 = nn.Conv1d(
            in_channels=20,
            out_channels=self.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers,
        )

        # *************************************************
        # ****************** Decoding *********************
        # *************************************************

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_head, batch_first=True
        )
        self.transformer_decoder1 = nn.TransformerEncoder(
            decoder_layer, num_layers=self.num_decoder_layers
        )
        self.transformer_decoder2 = nn.TransformerEncoder(
            decoder_layer, num_layers=self.num_decoder_layers
        )

        self.fc1 = nn.Linear(self.d_model, 20)
        self.fc2 = nn.Linear(self.d_model, 20)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        (B, T, F) = input_ids.shape

        x = input_ids.permute(0, 2, 1)

        x_conv = self.conv1(x)
        x_conv = self.relu(x_conv)

        x_transf = x_conv.permute(0, 2, 1)
        encoded = self.transformer_encoder(x_transf)

        # Two sources
        encoded1 = encoded.clone()
        encoded2 = encoded.clone()

        dec1 = self.transformer_decoder1(encoded1)
        dec2 = self.transformer_decoder2(encoded2)

        out1 = self.fc1(dec1)
        out2 = self.fc2(dec2)

        out = torch.stack([out1, out2], dim=1)

        loss = None
        if labels is not None:
            # Calculer la perte avec MSE
            loss = self.loss(out, labels)

        return {"logits": out, "loss": loss}
