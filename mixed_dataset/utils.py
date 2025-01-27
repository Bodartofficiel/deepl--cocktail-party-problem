from torch import Tensor
from torch.nn.functional import avg_pool1d


def compress_audio_tensor(audio_tensor: Tensor, compression_factor: int):
    if compression_factor == 1:
        return audio_tensor
    else:
        return avg_pool1d(
            audio_tensor, kernel_size=compression_factor, stride=compression_factor
        )
