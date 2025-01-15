# SI-SDR
# PIT
from torch import tensor
from torchmetrics.functional.audio import permutation_invariant_training,pit_permutate,scale_invariant_signal_distortion_ratio

target1 = tensor([3.0, -0.5, 2.0, 7.0])
preds1 = tensor([2.5, 0.0, 2.0, 8.0])
target2 = tensor([1.0, 2.0, 3.0, 4.0])
preds2 = tensor([1.5, 2.5, 2.5, 3.5])
target = tensor([[target1.tolist(),target2.tolist()]])
preds = tensor([[preds2.tolist(), preds1.tolist()]])

best_metric, best_perm = permutation_invariant_training(
    preds, target, scale_invariant_signal_distortion_ratio,
    mode="speaker-wise", eval_func="max")
print(best_metric, best_perm)
print(pit_permutate(preds, best_perm))

# PESQ
SAMPLING_FREQUENCE = 8000
BANDWIDTH = "nb" #narrow-band or wb wide-band

from torch import randn
from torchmetrics.audio import PerceptualEvaluationSpeechQuality #requires torchmetrics[audio]

target = tensor([[randn(8000).tolist(),randn(8000).tolist()]])
preds = tensor([[randn(8000).tolist(), randn(8000).tolist()]])
pesq = PerceptualEvaluationSpeechQuality(SAMPLING_FREQUENCE, BANDWIDTH)
print(pesq(preds, target))
# print(perceptual_evaluation_speech_quality(preds, target, 16000, 'wb'))

