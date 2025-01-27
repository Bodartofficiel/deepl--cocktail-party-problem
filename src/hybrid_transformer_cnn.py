import torch
import torch.nn.modules as nn
import torchaudio


class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super(HybridTransformerCNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(input_size, hidden_layer, 1), nn.ReLU())
        self.transformer = nn.Transformer(d_model=hidden_layer, batch_first=True)
        self.fc = nn.Linear(hidden_layer, output_size * input_size)

    def forward(self, input: torch.Tensor):

        out = input.permute(0, 2, 1)
        out = self.cnn(out)
        out = out.permute(0, 2, 1)
        out = self.transformer(out, out)
        out: torch.Tensor = self.fc(out)
        out = out.view(out.size(0), -1, input.size(-1))
        return out
