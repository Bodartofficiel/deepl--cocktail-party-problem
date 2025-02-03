# SI-SDR
# PIT
from torch import tensor
from torchmetrics.functional.audio import permutation_invariant_training,pit_permutate
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

target1 = tensor([3.0, -0.5, 2.0, 7.0])
prediction1 = tensor([2.5, 0.0, 2.0, 8.0])
target2 = tensor([1.0, 2.0, 3.0, 4.0])
prediction2 = tensor([1.5, 2.5, 2.5, 3.5])
targets = tensor([[target1.tolist(),target2.tolist()]])
predictions = tensor([[prediction2.tolist(), prediction1.tolist()]])

best_metric, best_perm = permutation_invariant_training(
    predictions, targets, scale_invariant_signal_distortion_ratio,
    mode="speaker-wise", eval_func="max")
print(best_metric, best_perm)
print(pit_permutate(predictions, best_perm))

loss_fn = PermutationInvariantTraining(scale_invariant_signal_distortion_ratio)
loss = loss_fn(predictions,target)
print(loss)

# PESQ
SAMPLING_FREQUENCE = 8000
BANDWIDTH = "nb" #narrow-band or wb wide-band

from torch import randn
from torchmetrics.audio import PerceptualEvaluationSpeechQuality #requires torchmetrics[audio]

targets = tensor([[randn(8000).tolist(),randn(8000).tolist()]])
predictions = tensor([[randn(8000).tolist(), randn(8000).tolist()]])
pesq = PerceptualEvaluationSpeechQuality(SAMPLING_FREQUENCE, BANDWIDTH)
print(pesq(predictions, targets))
# print(perceptual_evaluation_speech_quality(preds, target, 16000, 'wb'))

