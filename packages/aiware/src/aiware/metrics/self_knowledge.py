"""Self-knowledge metric R and self-surprise computation.

R = 1 - H(actual | predicted) / H(actual)

Where H is a differential-entropy proxy computed from the variance
of the prediction residual vs. the variance of the actual signal.
"""

from __future__ import annotations

import torch


def compute_R(predicted: torch.Tensor, actual: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute self-knowledge metric R in [0, 1].

    Uses variance-ratio as entropy proxy:
        R = 1 - Var(actual - predicted) / max(Var(actual), eps)

    Clamped to [0, 1] to handle numerical edge cases.
    Computation is done in float32 to avoid FP16 precision issues.

    Args:
        predicted: Predicted ESM state. Shape: (state_dim,).
        actual: Actual ESM state. Shape: (state_dim,).
        eps: Floor to prevent division by zero.

    Returns:
        R value in [0.0, 1.0].
    """
    residual = (actual.float() - predicted.float())
    var_residual = torch.var(residual).item()
    var_actual = max(torch.var(actual.float()).item(), eps)
    r = 1.0 - var_residual / var_actual
    return float(max(0.0, min(1.0, r)))


def compute_surprise(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    """Compute self-surprise as L2 norm of prediction error.

    Computation in float32 to avoid FP16 overflow in norm.

    Args:
        predicted: Predicted ESM state. Shape: (state_dim,).
        actual: Actual ESM state. Shape: (state_dim,).

    Returns:
        Non-negative scalar: ||actual - predicted||_2.
    """
    return float(torch.norm(actual.float() - predicted.float(), p=2).item())
