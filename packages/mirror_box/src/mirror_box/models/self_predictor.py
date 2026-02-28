"""Self-predictor MLP: predicts next ESM state from (prev_esm, gated_self).

The self-predictor is a lightweight 3-layer MLP trained online (one gradient
step per conscious frame). Its prediction error is the self-surprise signal;
the predictability ratio is the self-knowledge metric R.

Architecture:
    LayerNorm -> Linear(state_dim * 2, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, state_dim)

Input: concat(prev_esm, gated_self) -- shape (state_dim * 2,)
Output: predicted e_s(t+1) -- shape (state_dim,)

The input LayerNorm stabilizes online learning when fed high-magnitude
LLM hidden states (norm ~50+). Without it, a single Adam step per frame
cannot reduce MSE loss because the gradient landscape is too steep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aiware.interfaces.protocols import SelfPredictor
from aiware.metrics.self_knowledge import compute_R as _compute_R
from aiware.metrics.self_knowledge import compute_surprise as _compute_surprise
from mirror_box.config import MirrorBoxConfig


class MLPSelfPredictor(SelfPredictor, nn.Module):
    """Self-predictor MLP: predicts next ESM state from (prev_esm, gated_self).

    The predictor is trained online -- one gradient step per conscious frame.
    predict() runs without gradients (inference only) and caches the
    concatenated input. update() re-runs the forward pass with gradients
    enabled using the cached input, computes MSE loss against the actual
    ESM state (detached), and takes one Adam step.

    Input normalization (LayerNorm) is applied to stabilize online learning
    with high-magnitude LLM hidden states.

    Args:
        config: MirrorBoxConfig providing state_dim, hidden size, and learning rate.
    """

    def __init__(self, config: MirrorBoxConfig) -> None:
        nn.Module.__init__(self)

        state_dim = config.state_dim
        hidden = config.self_predictor_hidden_size
        lr = config.self_predictor_lr
        device = config.hardware.resolve_device()

        self._state_dim = state_dim
        self._device = device
        self._cached_input: torch.Tensor | None = None

        # 3-layer MLP with input LayerNorm for stable online learning.
        # Always float32 for numerical stability (inputs are cast on entry).
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        ).to(device=device, dtype=torch.float32)

        self._optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        self._loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # SelfPredictor ABC implementation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        prev_esm: torch.Tensor,
        gated_self: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next ESM state (inference only, no gradients).

        Caches the concatenated input so that update() can re-forward with
        gradients enabled for the training step.

        Args:
            prev_esm: Previous ESM state e_s(t-1). Shape: (state_dim,).
            gated_self: Current gated self-state. Shape: (state_dim,).

        Returns:
            Predicted ESM state. Shape: (state_dim,).
        """
        x = torch.cat([prev_esm.float(), gated_self.float()], dim=-1)
        self._cached_input = x.detach().clone()
        return self.net(x)

    def compute_R(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Compute self-knowledge metric R in [0, 1].

        Delegates to aiware.metrics.self_knowledge.compute_R.
        """
        return _compute_R(predicted, actual)

    def compute_surprise(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Compute self-surprise (L2 norm of prediction error).

        Delegates to aiware.metrics.self_knowledge.compute_surprise.
        """
        return _compute_surprise(predicted, actual)

    def update(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """One gradient step: re-forward with grad, MSE against actual.detach().

        The ``predicted`` tensor from predict() was computed under
        torch.no_grad() and has no gradient graph. This method re-runs the
        forward pass using the cached input (stored by predict()) with
        gradients enabled, then computes MSE loss and backpropagates.

        Args:
            predicted: Predicted ESM state (from predict(); not used for
                gradient computation, only as fallback if no cached input).
            actual: Actual ESM state (will be detached as target).

        Returns:
            Training loss value (float).
        """
        if self._cached_input is not None:
            # Re-run forward WITH gradients using cached input
            pred = self.net(self._cached_input)
        else:
            # Fallback: if predict() was not called first, use provided tensor.
            # This path should not occur in normal operation.
            pred = predicted

        target = actual.detach().float()
        loss = self._loss_fn(pred, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vectors this predictor operates on."""
        return self._state_dim
