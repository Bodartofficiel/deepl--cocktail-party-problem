import torch.nn.modules as nn


class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        self.cnn = nn.Sequential(nn.Conv1d(input_size, hidden_layer, 1), nn.ReLU())
        self.transformer = nn.Transformer()
        self.fc = nn.Linear(hidden_layer, output_size)

    def forward(self, input):
        out = self.cnn(input)
        out = self.transformer(out)
        out = self.fc(out)
        return out
