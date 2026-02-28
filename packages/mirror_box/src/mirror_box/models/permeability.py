"""Sigmoid permeability gate — substrate to simulation boundary.

Controls information flow from the implicit (substrate) layer to the
explicit (simulation) layer. The gate's behavior changes with the
system's operating state via StateProfiles:

- **Wake**: High permeability (alpha~1.0, beta~0.0) -> most information passes.
- **Sleep**: Very low permeability (alpha~0.1, beta~-3.0) -> nearly disconnected.
- **Dream**: Medium permeability, high noise (alpha~0.0, beta~-0.5, noise~0.5).
- **Altered**: Increased permeability (alpha~0.5, beta~2.0) -> content flooding.

Gate equation:
    g(t) = sigmoid(alpha * x(t) + beta + N(0, noise_std))
    gated_output = g(t) * x(t)

World and self channels are gated independently with their own alpha
parameters (alpha_world, alpha_self) but share beta and noise_std.
"""

from __future__ import annotations

import logging

import torch

from aiware.config.base import StateProfile
from aiware.interfaces.data_types import ConsciousState, GateOutput, SubstrateState
from aiware.interfaces.protocols import PermeabilityGate
from mirror_box.config import MirrorBoxConfig

logger = logging.getLogger(__name__)


class SigmoidPermeabilityGate(PermeabilityGate):
    """Concrete PermeabilityGate using sigmoid activation with additive noise.

    Each ConsciousState maps to a StateProfile that controls the gate
    parameters. Profiles can be overridden at runtime for experiments.

    Args:
        config: MirrorBoxConfig with state_profiles and device settings.
    """

    def __init__(self, config: MirrorBoxConfig) -> None:
        self._config = config
        self._device = config.hardware.resolve_device()

        # Copy profiles so runtime overrides don't mutate the config
        self._profiles: dict[str, StateProfile] = {}
        for state_name, profile in config.state_profiles.items():
            self._profiles[state_name] = profile.model_copy()

        logger.info(
            "SigmoidPermeabilityGate initialized with profiles: %s",
            list(self._profiles.keys()),
        )

    def _get_profile(self, state: ConsciousState) -> StateProfile:
        """Look up the StateProfile for a given ConsciousState."""
        profile = self._profiles.get(state.value)
        if profile is None:
            logger.warning(
                "No profile for state %s, falling back to wake.", state.value
            )
            profile = self._profiles["wake"]
        return profile

    # ------------------------------------------------------------------
    # PermeabilityGate interface
    # ------------------------------------------------------------------

    def gate(
        self, substrate_state: SubstrateState, state: ConsciousState
    ) -> GateOutput:
        """Apply sigmoid gating to substrate output.

        g_world(t) = sigmoid(alpha_world * w(t) + beta + noise)
        g_self(t)  = sigmoid(alpha_self  * s(t) + beta + noise)
        gated_world = g_world(t) * w(t)
        gated_self  = g_self(t)  * s(t)

        The gate activations are element-wise sigmoid values; the reported
        activation scalars are their means (useful for monitoring).

        Args:
            substrate_state: Raw substrate output containing w(t) and s(t).
            state: Current operating state (determines gate parameters).

        Returns:
            GateOutput with gated world/self signals and mean gate activations.
        """
        profile = self._get_profile(state)

        w = substrate_state.world_state.to(self._device)
        s = substrate_state.self_state.to(self._device)

        # World channel gating
        noise_w = torch.randn_like(w) * profile.noise_std
        gate_w = torch.sigmoid(profile.alpha_world * w + profile.beta + noise_w)
        gated_world = gate_w * w

        # Self channel gating
        noise_s = torch.randn_like(s) * profile.noise_std
        gate_s = torch.sigmoid(profile.alpha_self * s + profile.beta + noise_s)
        gated_self = gate_s * s

        return GateOutput(
            gated_world=gated_world,
            gated_self=gated_self,
            gate_world_activation=gate_w.mean().item(),
            gate_self_activation=gate_s.mean().item(),
            timestamp=substrate_state.timestamp,
        )

    def set_profile(self, state: ConsciousState, **params: float) -> None:
        """Override gate parameters for a given state at runtime.

        Only updates the specified parameters; others remain unchanged.

        Args:
            state: Which state profile to modify.
            **params: Gate-specific parameters. Valid keys:
                alpha_world, alpha_self, beta, noise_std,
                substrate_rate_hz. Boolean fields (external_input,
                self_ref_loop) can be passed as 0.0/1.0.
        """
        profile = self._profiles.get(state.value)
        if profile is None:
            logger.warning("Cannot set_profile: no profile for state %s", state.value)
            return

        # Build update dict, converting boolean fields
        update = {}
        for key, value in params.items():
            if key in ("external_input", "self_ref_loop"):
                update[key] = bool(value)
            else:
                update[key] = value

        # Create updated profile
        self._profiles[state.value] = profile.model_copy(update=update)

        logger.info(
            "Profile updated for %s: %s",
            state.value,
            {k: getattr(self._profiles[state.value], k) for k in params},
        )
