import torch
import torch.nn.modules as nn
import torchaudio

short_to = 100000
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


class HybridTransformerCNN(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super(HybridTransformerCNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(input_size, hidden_layer, 1), nn.ReLU())
        self.transformer = nn.Transformer(d_model=hidden_layer, batch_first=True)
        self.fc = nn.Linear(hidden_layer, output_size * input_size)

    def forward(self, input: torch.Tensor):
        if len(input.shape) == 2:  # No batch
            input = torch.stack([input])

        out = input.permute(0, 2, 1)
        out = self.cnn(out)
        out = out.permute(0, 2, 1)
        out = self.transformer(out, out)
        out: torch.Tensor = self.fc(out)
        out = out.view(out.size(0), -1, input.size(-1))
        return out

if __name__=='__main__':
    model = HybridTransformerCNN(short_to, 512, 2).to(device)

    audio = torchaudio.load("data/clips/common_voice_en_41236242.mp3")[0].to(device)


    audio = audio[:, :short_to]
    # audio = torch.stack(list([audio for i in range(5)]))

    output = model(audio)
