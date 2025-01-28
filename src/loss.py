import itertools

import torch
import torch.nn.functional as F


def pit_mse_loss(predictions, targets):
    """
    PIT (Permutation Invariant Training) avec MSE pour la descente de gradient.

    :param predictions: Tensor des prédictions, shape (batch_size, n_sources, seq_len)
    :param targets: Tensor des cibles, shape (batch_size, n_sources, seq_len)
    :return: Perte scalaire Tensor différentiable
    """
    batch_size, n_sources, seq_len = predictions.size()

    # Étendre les dimensions pour comparer chaque prédiction avec chaque cible
    predictions = predictions.unsqueeze(2)  # Shape: (batch_size, n_sources, 1, seq_len)
    targets = targets.unsqueeze(1)  # Shape: (batch_size, 1, n_sources, seq_len)
    predictions, targets = torch.broadcast_tensors(predictions, targets)
    # Calculer la MSE entre toutes les permutations possibles
    mse_matrix = F.mse_loss(
        predictions, targets, reduction="none"
    )  # Shape: (batch_size, n_sources, n_sources, seq_len)

    mse_matrix = mse_matrix.mean(
        dim=-1
    )  # Moyenne sur la dimension seq_len -> Shape: (batch_size, n_sources, n_sources)

    # Trouver la permutation minimale pour chaque batch
    # Utiliser l'algorithme de Kuhn-Munkres (Hungarian Algorithm) pour minimiser
    from scipy.optimize import linear_sum_assignment

    losses = []
    for batch in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(
            mse_matrix[batch].detach().cpu().numpy()
        )
        optimal_mse = mse_matrix[batch, row_ind, col_ind].mean()
        losses.append(optimal_mse)

    # Retourner la moyenne des pertes sur le batch
    return torch.stack(losses).mean()


import torch
import torch.nn.functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio


def pit_si_sdr_loss(predictions, targets):
    """
    Compute the PIT SI-SDR loss.

    Args:
        predictions (torch.Tensor): Predicted audio sources of shape (batch_size, n_sources, seq_len).
        targets (torch.Tensor): Ground-truth audio sources of shape (batch_size, n_sources, seq_len).

    Returns:
        torch.Tensor: The PIT SI-SDR loss (a single scalar).
    """
    batch_size, n_sources, seq_len = predictions.shape

    # Expand dimensions to compute pairwise SI-SDR for all permutations
    predictions = predictions.unsqueeze(2)  # (batch_size, n_sources, 1, seq_len)
    targets = targets.unsqueeze(1)  # (batch_size, 1, n_sources, seq_len)
    predictions, targets = torch.broadcast_tensors(predictions, targets)
    # Compute pairwise SI-SDR scores for all permutations
    si_sdr_matrix = scale_invariant_signal_distortion_ratio(
        predictions,
        targets,
    )
    # si_sdr_matrix: (batch_size, n_sources, n_sources)

    # Compute all possible permutations
    permutations = torch.tensor(
        list(itertools.permutations(range(n_sources))), device=predictions.device
    )
    n_permutations = permutations.shape[0]

    # Calculate SI-SDR loss for each permutation
    losses = torch.zeros(batch_size, n_permutations, device=predictions.device)
    for i, perm in enumerate(permutations):
        # Permute the target sources according to the current permutation
        permuted_targets = targets[:, :, perm, :]
        # Sum SI-SDR over all sources for the given permutation
        losses[:, i] = -si_sdr_matrix.gather(
            2, perm.unsqueeze(0).unsqueeze(-1).expand(batch_size, n_sources, seq_len)
        ).mean(dim=(1, 2))

    # Take the minimum loss across all permutations
    min_loss, _ = losses.min(dim=1)

    # Return the mean loss across the batch
    return min_loss.mean()


if __name__ == "__main__":
    batch_size, n_sources, seq_len = 4, 1, 1000
    predictions = torch.rand(batch_size, n_sources, seq_len)  # Prédictions aléatoires
    targets = torch.rand(batch_size, n_sources, seq_len)  # Cibles aléatoires

    loss = pit_mse_loss(predictions, targets)
    print(f"Perte PIT MSE : {loss.item()}")
