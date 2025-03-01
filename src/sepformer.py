import torch
import torch.nn as nn


class SepFormer(nn.Module):
    def __init__(self, F0=256, L=16, H=4, F=64, R=4, J=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.F0 = F0
        # self.L = L
        # self.H = H
        # self.F = F
        self.R = R

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, F0, kernel_size=L, stride=H), nn.GELU()
        )  # (I - L) // H + 1 = 32765

        self.input_layer = nn.Sequential(nn.Linear(F0, F), nn.LayerNorm(F))

        self.sep_encoder = SeparationEncoder(F)

        self.speaker_split = SpeakerSplit(F, J)

    def forward(self, x):
        x = self.audio_encoder(x)
        print(f"{torch.mps.current_allocated_memory():,}")
        x = x.transpose(2, 1)
        x = self.input_layer(x)
        print(f"{torch.mps.current_allocated_memory():,}")
        x = x.transpose(2, 1)
        for _ in range(self.R // 2):
            x = self.sep_encoder(x)
            print(f"{torch.mps.current_allocated_memory():,}")
        x = self.speaker_split(x)
        return x


class SeparationEncoder(nn.Module):
    def __init__(self, F, n_head=4, reduction=2, K=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(F, F, 7, 4, 3, groups=F)
        self.batch_norm = nn.BatchNorm1d(F)
        self.gelu = nn.GELU()
        self.global_transformer = GlobalTransformer(F, reduction, n_head)
        self.local_transformer = LocalTransformer(F, K)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.gelu(x)
        x = x.transpose(2, 1)
        x = self.global_transformer(x)
        x = self.local_transformer(x)
        x = x.transpose(2, 1)
        return x


class SpeakerSplit(nn.Module):
    def __init__(self, F, J=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.J = J
        self.F = F
        self.conv1 = nn.Conv1d(F, 2 * J * F, 1)
        self.conv2 = nn.Conv1d(F, 2 * J * F, 1)
        self.glu = nn.GLU(dim=1)
        self.conv3 = nn.Conv1d(2 * J * F, J * F, 1)
        self.layernorm = nn.LayerNorm(J * F)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.glu(torch.cat((x1, x2), dim=1))
        x = self.conv3(x)
        x = x.transpose(2, 1)
        x = self.layernorm(x)
        x = x.transpose(2, 1)
        x = x.reshape(x.size(0), self.J, self.F, -1)
        return x


class GlobalTransformer(nn.Module):
    def __init__(self, F, reduction, n_head=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm1 = nn.LayerNorm(F)

        self.ega = EGA(F, reduction, n_head)

        self.layer_norm2 = nn.LayerNorm(F)

        self.gcfn = GCFN(F)

    def forward(self, x):
        # (B, T, F)
        x1 = self.layer_norm1(x)
        x1 = self.ega(x1)

        x += x1

        x1 = self.layer_norm2(x)

        x1 = self.gcfn(x1)

        x += x1

        return x


class LocalTransformer(nn.Module):
    def __init__(self, F, K, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm1 = nn.LayerNorm(F)
        self.cla = CLA(F, K)
        self.layer_norm2 = nn.LayerNorm(F)
        self.gcfn = GCFN(F)

    def forward(self, x):
        # (B, T, F)
        x1 = self.layer_norm1(x)
        x1 = self.cla(x1)
        x += x1

        x1 = self.layer_norm2(x)
        x1 = self.gcfn(x)
        x += x1
        return x


class EGA(nn.Module):
    def __init__(self, F, reduction, n_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsampling = nn.AvgPool1d(reduction)

        self.multihead_self_attention = nn.MultiheadAttention(
            F // reduction, num_heads=n_head, batch_first=True
        )
        self.upsampling = nn.UpsamplingNearest2d(F)

        self.linear = nn.Linear(F, F)
        self.glu = nn.GLU(dim=1)

        self.dropout = nn.Dropout1d()

    def forward(self, x):
        x1 = self.downsampling(x)
        x1, _ = self.multihead_self_attention(x1, x1, x1)
        x1 = self.upsampling(x)

        x2 = self.linear(x)

        x = self.glu(torch.cat((x1, x2), dim=1)).transpose(2, 1)

        x = self.dropout(x)

        return x.transpose(2, 1)


class CLA(nn.Module):
    def __init__(self, F, K, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pconv1 = nn.Conv1d(F, F, 1)
        self.pconv2 = nn.Conv1d(F, F, 1)

        self.glu = nn.GLU(1)

        self.dconv1 = nn.Conv1d(F, F, K, padding=K // 2, groups=F)
        self.pconv3 = nn.Conv1d(F, 2 * F, 1)
        self.batch_norm = nn.BatchNorm1d(2 * F)

        self.gelu = nn.GELU()
        self.pconv4 = nn.Conv1d(2 * F, F, 1)
        self.dropout = nn.Dropout1d()

    def forward(self, x):
        x = x.transpose(2, 1)
        x1 = self.pconv1(x)
        x2 = self.pconv2(x)
        x = self.glu(torch.cat((x1, x2), dim=1))
        x = self.dconv1(x)
        x = self.pconv3(x)
        x = self.batch_norm(x)
        x = self.gelu(x)
        x = self.pconv4(x)
        x = self.dropout(x)
        return x.transpose(2, 1)


class GCFN(nn.Module):
    def __init__(self, F, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pconv1 = nn.Conv1d(F, 3 * F, 1)
        self.pconv2 = nn.Conv1d(F, 3 * F, 1)
        self.dconv1 = nn.Conv1d(3 * F, 3 * F, 3, padding=1, groups=3 * F)
        self.dconv2 = nn.Conv1d(3 * F, 3 * F, 3, padding=1, groups=3 * F)

        self.glu = nn.GLU(1)
        self.dropout1 = nn.Dropout1d()

        self.pconv3 = nn.Conv1d(3 * F, F, 1)
        self.dropout2 = nn.Dropout1d()

    def forward(self, x):
        x = x.transpose(2, 1)
        x1 = self.pconv1(x)
        x2 = self.pconv2(x)
        x1 = self.dconv1(x1)
        x2 = self.dconv2(x2)
        x = self.glu(torch.cat((x1, x2), dim=1))
        x = self.dropout1(x)
        x = self.pconv3(x)
        x = self.dropout2(x)
        return x.transpose(2, 1)


if __name__ == "__main__":
    x = torch.randn((1, 1, 131072), device="mps")
    sep = SepFormer().to("mps")
    total_params = sum(p.numel() for p in sep.parameters())
    print(f"Total number of parameters: {total_params:,}")
    print(sep(x))
