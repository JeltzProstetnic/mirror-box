"""Lempel-Ziv complexity of state trajectories.

Measures the dynamical richness of the explicit-model state sequence.
Higher complexity = richer dynamics (desirable in wake/dream).
Lower complexity = simpler dynamics (expected in sleep).
"""

from __future__ import annotations

import math

import torch


def _binarize_trajectory(trajectory: torch.Tensor, threshold: float = 0.0) -> str:
    """Convert a tensor trajectory to a binary string for LZ analysis.

    Each element above the threshold becomes '1', otherwise '0'.

    Args:
        trajectory: Flattened state trajectory. Shape: (T * state_dim,) or (T, state_dim).
        threshold: Binarization threshold (default: median-split at 0).

    Returns:
        Binary string of '0' and '1' characters.
    """
    flat = trajectory.flatten()
    return "".join("1" if x > threshold else "0" for x in flat.tolist())


def lempel_ziv_complexity(binary_string: str) -> float:
    """Compute normalized Lempel-Ziv complexity.

    Uses the sequential LZ76 algorithm. Normalized by the theoretical
    maximum for a random binary string of the same length:
        C_max ~ n / log2(n)

    Args:
        binary_string: String of '0' and '1' characters.

    Returns:
        Normalized complexity in [0, ~1]. Values near 1 indicate
        maximal complexity (random-like). Values near 0 indicate
        highly regular (compressible) sequences.
    """
    n = len(binary_string)
    if n <= 1:
        return 0.0

    # LZ76 complexity count
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    while i + k <= n:
        substring = binary_string[i + 1 : i + k + 1] if i + k < n else ""
        if binary_string[i + k - 1 : i + k] not in binary_string[0 : i + k - 1]:
            complexity += 1
            i += k
            k = 1
        else:
            k += 1
        if k > k_max:
            k_max = k
        # Safety: prevent infinite loop on degenerate input
        if k > n - i:
            break

    # Normalize by theoretical maximum
    log2_n = math.log2(n) if n > 1 else 1.0
    c_max = n / log2_n
    return complexity / c_max


def trajectory_complexity(
    states: list[torch.Tensor],
    threshold: float = 0.0,
) -> float:
    """Compute LZ complexity of a sequence of state vectors.

    Args:
        states: List of state vectors, each shape (state_dim,).
        threshold: Binarization threshold.

    Returns:
        Normalized LZ complexity in [0, ~1].
    """
    if len(states) < 2:
        return 0.0
    trajectory = torch.stack(states)
    binary = _binarize_trajectory(trajectory, threshold)
    return lempel_ziv_complexity(binary)
