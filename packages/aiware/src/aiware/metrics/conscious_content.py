"""Total conscious content proxy C(t).

C(t) is a scalar summarizing how much "conscious content" the explicit
layer is producing. High C = rich simulation (wake). C ~ 0 = collapsed
simulation (sleep).

C(t) = sum of activation norms above a threshold, normalized by state_dim.
"""

from __future__ import annotations

import torch


def compute_C(
    ewm_state: torch.Tensor,
    esm_state: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """Compute total conscious content proxy.

    C = (||e_w||_2 + ||e_s||_2) if above threshold, else thresholded components.

    More precisely: C = relu(||e_w|| - threshold) + relu(||e_s|| - threshold),
    normalized by the state dimensionality for cross-architecture comparability.

    Args:
        ewm_state: EWM state vector e_w(t). Shape: (state_dim,).
        esm_state: ESM state vector e_s(t). Shape: (state_dim,).
        threshold: Minimum activation norm to count as "conscious content".

    Returns:
        C(t) >= 0. Dimensionless, normalized by state_dim.
    """
    dim = float(ewm_state.shape[0])
    norm_w = torch.norm(ewm_state.float(), p=2).item()
    norm_s = torch.norm(esm_state.float(), p=2).item()
    c = max(0.0, norm_w - threshold) + max(0.0, norm_s - threshold)
    return c / dim
