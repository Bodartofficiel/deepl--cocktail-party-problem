import torch
import torch.nn.modules as nn
import torchaudio


class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, device, output_size):
        super(HybridTransformerCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.mp1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 1, input_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3
        )
        # 16, input_size // 2
        self.mp2 = nn.MaxPool1d(2, 2)
        # 16, input_size // 4
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=6, stride=2, padding=2
        )
        # 16, input_size // 8

        self.mp3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 16, input_size // 16
        self.conv3 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=6, stride=2, padding=2
        )
        # 16, input_size // 32
        self.mp4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 16, input_size // 64
        self.conv4 = nn.Conv1d(
            in_channels=16, out_channels=64, kernel_size=6, stride=2, padding=2
        )
        self.transformer = nn.Transformer(
            d_model=64,  # Embedding size
            batch_first=True,  # Ensures (batch_size, seq_len, embed_dim) format
        )
        # self.fc = nn.Linear(781 * 16, output_size * input_size)

    def forward(self, input: torch.Tensor):
        out = self.mp1(input)

        # N, 1, 50000
        out = self.conv1(out)

        out = self.mp2(out)

        out = self.conv2(out)

        out = self.mp3(out)

        out = self.conv3(out)

        out = self.mp4(out)

        out = self.conv4(out)

        # N, 1, 50000 - 7 + 2 * 3 + 1  = 500000
        out = out.permute(0, 2, 1)
        out = self.transformer(
            out,
            torch.zeros(
                (out.size(0), self.input_size * self.output_size // 64, 64),
                device=self.device,
            ),
        )

        out = out.view(-1, self.output_size, self.input_size)
        return out
