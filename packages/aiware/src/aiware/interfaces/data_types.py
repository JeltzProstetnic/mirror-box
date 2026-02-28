"""Shared data types for inter-component communication.

These are the data structures that flow between components. They use
dataclasses (not Pydantic) because they carry tensors and must be
lightweight — created thousands of times per second at 20-40 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class ConsciousState(str, Enum):
    """Operating states of an AC system.

    State transitions follow FMT predictions:
    - WAKE: At criticality, all four models active, full external input.
    - SLEEP: Subcritical, simulation collapsed, substrate maintenance.
    - DREAM: Near-critical, no external input, EWM/ESM run on internal content.
    - ALTERED: Increased permeability, content flooding from implicit to explicit.
    """

    WAKE = "wake"
    SLEEP = "sleep"
    DREAM = "dream"
    ALTERED = "altered"


@dataclass(frozen=True, slots=True)
class SubstrateState:
    """Output of the substrate layer (implicit models).

    Produced at ~40 Hz by the SubstrateEncoder.

    Attributes:
        world_state: w(t) world-state vector from IWM encoding.
            Shape: (state_dim,). Design 16: 512-dim from LLM hidden layer 8.
            Design 15: population spike-rate readout from SNN.
        self_state: s(t) self-state vector from ISM encoding.
            Shape: (state_dim,). Encodes system telemetry / interoceptive data.
        timestamp: Monotonic time in seconds when this state was produced.
        metadata: Diagnostic data (latency, activation norms, etc.).
    """

    world_state: torch.Tensor
    self_state: torch.Tensor
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.world_state.shape != self.self_state.shape:
            raise ValueError(
                f"world_state shape {self.world_state.shape} != "
                f"self_state shape {self.self_state.shape}"
            )


@dataclass(frozen=True, slots=True)
class GateOutput:
    """Output of the permeability gate.

    The gate modulates what flows from substrate (implicit) to
    simulation (explicit). State-dependent parameters control how much
    information passes through.

    Attributes:
        gated_world: Gated world-state. Shape: (state_dim,).
        gated_self: Gated self-state. Shape: (state_dim,).
        gate_world_activation: Raw gate activation for world channel [0, 1].
        gate_self_activation: Raw gate activation for self channel [0, 1].
        timestamp: From the input SubstrateState.
    """

    gated_world: torch.Tensor
    gated_self: torch.Tensor
    gate_world_activation: float
    gate_self_activation: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class ExplicitState:
    """Output of the explicit model layer (virtual/conscious simulation).

    Produced at ~20 Hz by the ExplicitGenerator.

    Attributes:
        ewm_state: e_w(t) explicit world-model state. Shape: (state_dim,).
        esm_state: e_s(t) explicit self-model state. Shape: (state_dim,).
            Includes self-referential feedback from previous frame.
        timestamp: Monotonic time in seconds.
        metadata: Diagnostic data (generation latency, token entropy, etc.).
    """

    ewm_state: torch.Tensor
    esm_state: torch.Tensor
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Metrics computed per conscious frame (~20 Hz).

    Collected by MetricsCollector and consumed by dashboard, experiment
    tracking, and state-transition detection.

    Attributes:
        R: Self-knowledge metric [0, 1]. How well the system predicts its
            own next ESM state. R=0: no self-knowledge. R=1: perfect.
        self_surprise: Magnitude of self-prediction error. High values mean
            the system did something it did not expect.
        C: Total conscious content proxy. Sum of explicit-model activation
            norms above a threshold. C~0 in sleep, C>0 in wake.
        complexity: Lempel-Ziv complexity of recent state trajectory.
            Measures dynamical richness.
        state: Current operating state.
        frame_number: Monotonic frame counter.
        timestamp: When this frame was computed.
        extras: Implementation-specific additional metrics.
    """

    R: float
    self_surprise: float
    C: float
    complexity: float
    state: ConsciousState
    frame_number: int
    timestamp: float
    extras: dict[str, float] = field(default_factory=dict)
