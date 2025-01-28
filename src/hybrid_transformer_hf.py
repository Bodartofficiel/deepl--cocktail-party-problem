import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from loss import pit_mse_loss, pit_si_sdr_loss


class HybridTransformerCNNConfig(PretrainedConfig):
    def __init__(self, input_size=131072, output_size=2, loss="pit_mse", **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.loss = loss


class HybridTransformerCNN(PreTrainedModel):
    config_class = HybridTransformerCNNConfig

    def __init__(self, config: HybridTransformerCNNConfig):
        super().__init__(config)
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.loss = {
            "pit_mse": pit_mse_loss,
            "pit_si_sdr": pit_si_sdr_loss,
        }[config.loss]

        self.mp1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3
        )
        self.mp2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=6, stride=2, padding=2
        )
        self.mp3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=6, stride=2, padding=2
        )
        self.mp4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(
            in_channels=16, out_channels=64, kernel_size=6, stride=2, padding=2
        )

        self.transformer = nn.Transformer(
            d_model=64, batch_first=True  # Embedding size and batch format
        )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        input_ids: Tensor de forme (batch_size, input_size).
        labels: Tensor de forme (batch_size, output_size, input_size), utilisé pour la perte.

        Retourne :
            - Un dictionnaire avec les clés :
              - logits : la sortie du modèle.
              - loss : la perte si les labels sont fournis.
        """
        x = input_ids.unsqueeze(1)  # Ajouter une dimension pour in_channels
        out = self.mp1(x)
        out = self.conv1(out)
        out = self.mp2(out)
        out = self.conv2(out)
        out = self.mp3(out)
        out = self.conv3(out)
        out = self.mp4(out)
        out = self.conv4(out)

        # Transformer Layer
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)
        out = self.transformer(
            out,
            torch.zeros(
                (out.size(0), self.input_size * self.output_size // 64, 64),
                device=out.device,
            ),
        )

        # Projection finale
        logits = out.view(-1, self.output_size, self.input_size)

        loss = None
        if labels is not None:
            # Calculer la perte avec MSE
            loss = self.loss(logits, labels)

        return {"logits": logits, "loss": loss}
