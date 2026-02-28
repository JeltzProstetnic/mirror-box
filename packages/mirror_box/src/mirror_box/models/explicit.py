"""Explicit models (EWM + ESM) — the conscious simulation layer.

Uses small MLPs (not the LLM) to transform gated substrate signals into
explicit state vectors:

1. **EWM (Explicit World Model)**: A 2-layer MLP that transforms gated
   world-state into an explicit world representation e_w(t).

2. **ESM (Explicit Self Model)**: A 2-layer MLP with self-referential
   input. Takes concat(gated_self, prev_esm) and produces e_s(t).
   The concatenation of previous ESM output creates the self-referential
   loop: the system models the process of self-modeling.

The LLM is used separately for TEXT generation (introspection module),
not for state vector computation here. The MLPs are lightweight so the
explicit layer can run at ~20 Hz without bottlenecking on LLM inference.

Operates at ~20 Hz (conscious simulation rate).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from aiware.interfaces.data_types import ExplicitState, GateOutput
from aiware.interfaces.protocols import ExplicitGenerator
from mirror_box.config import MirrorBoxConfig

logger = logging.getLogger(__name__)


class ExplicitModels(ExplicitGenerator):
    """Concrete ExplicitGenerator: MLP-based EWM and ESM state generators.

    EWM: ewm_mlp(gated_world) -> e_w(t)
    ESM: esm_mlp(concat(gated_self, prev_esm)) -> e_s(t)

    Both MLPs are 2-layer networks with GELU activation and layer
    normalization. The ESM MLP takes double the input dimension
    (state_dim * 2) to accommodate the self-referential concatenation.

    Args:
        config: MirrorBoxConfig with state_dim and device settings.
        backend: LLMBackend (used only for device placement, not inference).
    """

    def __init__(self, config: MirrorBoxConfig, backend: Any = None) -> None:
        self._config = config
        self._state_dim = config.state_dim

        # Resolve device from backend or config
        if backend is not None:
            self._device = backend.device
        else:
            self._device = config.hardware.resolve_device()

        # Hidden dimension for the MLPs (2x state_dim for capacity)
        hidden_dim = self._state_dim * 2
        dtype = config.hardware.resolve_dtype()

        # EWM MLP: gated_world (state_dim,) -> e_w(t) (state_dim,)
        # Output LayerNorm bounds the state magnitude for stable dynamics.
        self._ewm_mlp = nn.Sequential(
            nn.Linear(self._state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self._state_dim),
            nn.LayerNorm(self._state_dim),
        ).to(device=self._device, dtype=dtype)

        # ESM MLP: concat(gated_self, prev_esm) (state_dim*2,) -> delta (state_dim,)
        # Produces the *update* to prev_esm; residual connection applied in generate_esm().
        self._esm_mlp = nn.Sequential(
            nn.Linear(self._state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self._state_dim),
        ).to(device=self._device, dtype=dtype)

        # Post-residual LayerNorm: bounds ESM state magnitude for stable recurrence.
        self._esm_residual_norm = nn.LayerNorm(self._state_dim).to(
            device=self._device, dtype=dtype
        )

        # Initialize weights with small values for stability
        self._init_weights()

        logger.info(
            "ExplicitModels initialized: state_dim=%d, hidden_dim=%d, device=%s",
            self._state_dim,
            hidden_dim,
            self._device,
        )

    def _init_weights(self) -> None:
        """Initialize MLP weights with Xavier uniform for stability."""
        for mlp in (self._ewm_mlp, self._esm_mlp):
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # ExplicitGenerator interface
    # ------------------------------------------------------------------

    def generate_ewm(
        self,
        gated_world: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate explicit world-model state e_w(t).

        Transforms the gated world-state through a 2-layer MLP.
        The optional context is reserved for future use (e.g.,
        conversation history integration).

        Args:
            gated_world: Gated world-state from permeability gate.
                Shape: (state_dim,).
            context: Optional additional context (unused in v1).

        Returns:
            EWM state e_w(t). Shape: (state_dim,).
        """
        gated_world = gated_world.to(self._device)
        return self._ewm_mlp(gated_world)

    def generate_esm(
        self,
        gated_self: torch.Tensor,
        prev_esm: torch.Tensor,
    ) -> torch.Tensor:
        """Generate explicit self-model state e_s(t) with self-referential closure.

        Uses a residual connection: e_s(t) = LayerNorm(prev_esm + MLP(input)).
        This ensures temporal autocorrelation — e_s(t) evolves smoothly from
        e_s(t-1) rather than jumping chaotically. Smooth dynamics are essential
        for the self-predictor to learn meaningful patterns.

        Args:
            gated_self: Gated self-state from permeability gate.
                Shape: (state_dim,).
            prev_esm: Previous ESM state e_s(t-1). Shape: (state_dim,).
                On first frame, this should be a zero vector.

        Returns:
            ESM state e_s(t). Shape: (state_dim,).
        """
        gated_self = gated_self.to(self._device)
        prev_esm = prev_esm.to(self._device)

        # Self-referential concatenation: current gated self + previous self-model
        combined = torch.cat([gated_self, prev_esm], dim=-1)  # (state_dim * 2,)
        # Residual connection: smooth evolution, not chaotic jumps
        delta = self._esm_mlp(combined)
        return self._esm_residual_norm(prev_esm + delta)

    def step(
        self,
        gate_output: GateOutput,
        prev_esm: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> ExplicitState:
        """Execute one simulation frame: generate EWM and ESM.

        Args:
            gate_output: Output from the permeability gate.
            prev_esm: Previous ESM state for self-referential feedback.
            context: Optional additional context.

        Returns:
            ExplicitState containing e_w(t), e_s(t), timestamp, metadata.
        """
        ewm_state = self.generate_ewm(gate_output.gated_world, context=context)
        esm_state = self.generate_esm(gate_output.gated_self, prev_esm)

        return ExplicitState(
            ewm_state=ewm_state,
            esm_state=esm_state,
            timestamp=time.monotonic(),
            metadata={
                "ewm_norm": ewm_state.norm().item(),
                "esm_norm": esm_state.norm().item(),
            },
        )

    def state_dim(self) -> int:
        """Dimensionality of the explicit state vectors."""
        return self._state_dim
