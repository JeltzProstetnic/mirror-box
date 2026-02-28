"""Abstract interfaces (ABCs) that AC implementations must fulfill.

Each interface defines a contract. Implementations provide concrete classes.
All metrics, dashboard, and runtime orchestration code programs against
these abstractions exclusively.

Naming note: "BaseModel" is avoided to prevent collision with pydantic.BaseModel.
The AC model interface is split into SubstrateEncoder (implicit) and
ExplicitGenerator (explicit), matching the two-layer FMT architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from aiware.interfaces.data_types import (
    ConsciousState,
    ExplicitState,
    FrameMetrics,
    GateOutput,
    SubstrateState,
)


# ---------------------------------------------------------------------------
# Substrate layer: implicit models (IWM + ISM)
# ---------------------------------------------------------------------------


class SubstrateEncoder(ABC):
    """Interface for the implicit model layer.

    The substrate operates at ~40 Hz. It encodes sensory input into
    world-state vectors (IWM) and system telemetry into self-state
    vectors (ISM). The substrate never generates conscious content
    directly — it feeds the explicit layer through the permeability gate.

    Design 16 (Mirror Box): Frozen LLM encoder layers + LoRA adapter.
    Design 15 (Spiking Canopy): SNN reservoir with population spike-rate readout.
    """

    @abstractmethod
    def encode_world(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Encode sensory input into a world-state vector w(t).

        Args:
            sensory_input: Raw sensory data. Shape depends on modality.
                Text: token IDs (seq_len,). Vision: image tensor (C, H, W).

        Returns:
            World-state vector w(t). Shape: (state_dim,).
        """

    @abstractmethod
    def encode_self(self, telemetry: dict[str, Any]) -> torch.Tensor:
        """Encode system telemetry into a self-state vector s(t).

        The telemetry dict contains interoceptive signals: inference latency,
        activation norms, prediction errors, gate state, GPU stats, etc.

        Args:
            telemetry: Dictionary of named telemetry signals.

        Returns:
            Self-state vector s(t). Shape: (state_dim,).
        """

    @abstractmethod
    def step(self, sensory_input: torch.Tensor, telemetry: dict[str, Any]) -> SubstrateState:
        """Execute one substrate step: encode both world and self.

        Combines encode_world and encode_self into a single SubstrateState.
        Implementations may add noise injection, homeostatic scaling, or
        other substrate dynamics here.

        Args:
            sensory_input: Raw sensory data for world encoding.
            telemetry: System telemetry for self encoding.

        Returns:
            SubstrateState containing w(t), s(t), timestamp, metadata.
        """

    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the state vectors produced by this encoder."""


# ---------------------------------------------------------------------------
# Permeability gate: substrate -> simulation boundary
# ---------------------------------------------------------------------------


class PermeabilityGate(ABC):
    """Interface for the gate between substrate and simulation layers.

    The permeability gate controls what information flows from the implicit
    (substrate) layer to the explicit (simulation) layer. Its parameters
    change with the system's operating state:

    - Wake: high permeability, low noise (most information passes through).
    - Sleep: very low permeability (simulation nearly disconnected).
    - Dream: medium permeability, high noise (internal content dominates).
    - Altered: increased permeability (content flooding).

    Design 16: Sigmoid gate with additive Gaussian noise.
        g(t) = sigmoid(alpha * x(t) + beta + noise)
    Design 15: Spike-rate modulation from SNN to LLM canopy.
    """

    @abstractmethod
    def gate(self, substrate_state: SubstrateState, state: ConsciousState) -> GateOutput:
        """Apply permeability gating to substrate output.

        Args:
            substrate_state: Raw substrate output containing w(t) and s(t).
            state: Current operating state (determines gate parameters).

        Returns:
            GateOutput with gated world/self signals and gate activations.
        """

    @abstractmethod
    def set_profile(self, state: ConsciousState, **params: float) -> None:
        """Override gate parameters for a given state.

        This allows runtime tuning of permeability profiles for experiments
        (e.g., gradually increasing noise during an altered-state experiment).

        Args:
            state: Which state profile to modify.
            **params: Gate-specific parameters (alpha, beta, noise_std, etc.).
        """


# ---------------------------------------------------------------------------
# Explicit layer: virtual/conscious simulation (EWM + ESM)
# ---------------------------------------------------------------------------


class ExplicitGenerator(ABC):
    """Interface for the explicit model layer (conscious simulation).

    Operates at ~20 Hz. Takes gated substrate signals and produces
    explicit world-model (EWM) and self-model (ESM) states. The ESM
    includes the self-referential loop: it receives its own previous
    output as part of its input.

    Design 16: Full LLM forward pass (Mistral-7B INT4).
    Design 15: Mamba-2 canopy forward pass with prediction-error coupling.
    """

    @abstractmethod
    def generate_ewm(
        self,
        gated_world: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate explicit world-model state e_w(t).

        Args:
            gated_world: Gated world-state from permeability gate. Shape: (state_dim,).
            context: Optional additional context (e.g., conversation history).

        Returns:
            EWM state e_w(t). Shape: (state_dim,).
        """

    @abstractmethod
    def generate_esm(
        self,
        gated_self: torch.Tensor,
        prev_esm: torch.Tensor,
    ) -> torch.Tensor:
        """Generate explicit self-model state e_s(t) with self-referential closure.

        The ESM is the core mechanism: it receives both the gated self-state
        from the substrate AND its own previous output. This creates the
        self-referential loop: the system models the process of self-modeling.

        Args:
            gated_self: Gated self-state from permeability gate. Shape: (state_dim,).
            prev_esm: Previous ESM state e_s(t-1). Shape: (state_dim,).
                On first frame, this is a zero vector.

        Returns:
            ESM state e_s(t). Shape: (state_dim,).
        """

    @abstractmethod
    def step(
        self,
        gate_output: GateOutput,
        prev_esm: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> ExplicitState:
        """Execute one simulation frame: generate EWM and ESM.

        Combines generate_ewm and generate_esm into a single ExplicitState.

        Args:
            gate_output: Output from the permeability gate.
            prev_esm: Previous ESM state for self-referential feedback.
            context: Optional additional context.

        Returns:
            ExplicitState containing e_w(t), e_s(t), timestamp, metadata.
        """

    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the explicit state vectors."""


# ---------------------------------------------------------------------------
# Self-prediction module
# ---------------------------------------------------------------------------


class SelfPredictor(ABC):
    """Interface for the self-prediction module.

    Predicts the next ESM state BEFORE it is computed. The prediction
    error is the self-surprise signal; the predictability ratio is the
    self-knowledge metric R.

    Same mathematical formulation in all designs; different input sources.

    R = 1 - H(e_s_actual | e_s_predicted) / H(e_s_actual)
    """

    @abstractmethod
    def predict(
        self,
        prev_esm: torch.Tensor,
        gated_self: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next ESM state before it is generated.

        Args:
            prev_esm: Previous ESM state e_s(t-1). Shape: (state_dim,).
            gated_self: Current gated self-state. Shape: (state_dim,).

        Returns:
            Predicted ESM state. Shape: (state_dim,).
        """

    @abstractmethod
    def compute_R(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Compute self-knowledge metric R in [0, 1].

        R=0 means the prediction is useless (no self-knowledge).
        R=1 means perfect prediction (complete self-knowledge).

        Args:
            predicted: Predicted ESM state from self.predict().
            actual: Actual ESM state from ExplicitGenerator.generate_esm().

        Returns:
            R value in [0.0, 1.0].
        """

    @abstractmethod
    def compute_surprise(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Compute self-surprise (prediction error magnitude).

        Args:
            predicted: Predicted ESM state.
            actual: Actual ESM state.

        Returns:
            Non-negative scalar representing surprise magnitude.
        """

    @abstractmethod
    def update(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> float:
        """Update the predictor to improve future predictions.

        Called once per conscious frame. Returns the training loss.

        Args:
            predicted: Predicted ESM state (with grad).
            actual: Actual ESM state (detached).

        Returns:
            Training loss value.
        """


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


class MetricsCollector(ABC):
    """Interface for collecting and querying frame-level metrics.

    Records FrameMetrics every conscious frame (~20 Hz). Provides
    time-series queries for dashboard, experiment tracking, and
    automated state-transition detection.

    Implementations may store in memory, write to disk, or forward
    to wandb/tensorboard.
    """

    @abstractmethod
    def record_frame(self, metrics: FrameMetrics) -> None:
        """Record metrics for one conscious frame.

        Args:
            metrics: Complete metrics for this frame.
        """

    @abstractmethod
    def get_trajectory(
        self,
        metric_name: str,
        window: int | None = None,
    ) -> list[float]:
        """Retrieve a time series of a named metric.

        Args:
            metric_name: One of "R", "self_surprise", "C", "complexity",
                or any key in FrameMetrics.extras.
            window: Number of most recent frames. None = all recorded.

        Returns:
            List of metric values, oldest first.
        """

    @abstractmethod
    def get_latest(self) -> FrameMetrics | None:
        """Get the most recently recorded FrameMetrics, or None if empty."""

    @abstractmethod
    def frame_count(self) -> int:
        """Total number of frames recorded."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all recorded metrics. Used at state transitions (e.g., sleep)."""


# ---------------------------------------------------------------------------
# Telemetry source
# ---------------------------------------------------------------------------


class TelemetrySource(ABC):
    """Interface for collecting system telemetry (interoceptive signals).

    Telemetry is the ISM's input — it tells the self-model about the
    system's own state. Biological analog: proprioception, interoception,
    homeostatic signals.

    Attributes collected: inference latency, activation norms, prediction
    errors, gate state, GPU temperature, memory usage, etc.

    Implementations are platform-specific (GPU vs CPU, different hardware).
    """

    @abstractmethod
    def collect(self) -> dict[str, float]:
        """Collect current telemetry snapshot.

        Returns:
            Dictionary mapping signal names to float values.
            Keys should be stable across calls (same set of signals).
        """

    @abstractmethod
    def signal_names(self) -> list[str]:
        """List all telemetry signal names this source provides."""

    @abstractmethod
    def update_signal(self, name: str, value: float) -> None:
        """Inject a telemetry signal from an external source.

        Some signals (e.g., self-prediction error) are produced by other
        components and pushed to the telemetry source for centralized access.

        Args:
            name: Signal name.
            value: Signal value.
        """


# ---------------------------------------------------------------------------
# State controller
# ---------------------------------------------------------------------------


class StateController(ABC):
    """Interface for the operating-state machine.

    Manages transitions between ConsciousStates (wake, sleep, dream, altered).
    Tracks time-in-state, enforces transition rules, and emits events.

    Design 16: Timer-based transitions, user-triggered altered state.
    Design 15: Criticality-metric-driven transitions.
    """

    @abstractmethod
    def current_state(self) -> ConsciousState:
        """Current operating state."""

    @abstractmethod
    def transition_to(self, target: ConsciousState) -> bool:
        """Request a state transition.

        Returns True if the transition was accepted, False if it violates
        transition rules (e.g., cannot go from SLEEP directly to ALTERED).

        Args:
            target: Desired next state.

        Returns:
            Whether the transition was accepted.
        """

    @abstractmethod
    def tick(self, frame_metrics: FrameMetrics) -> ConsciousState | None:
        """Advance the state machine by one frame.

        Checks whether automatic transitions should occur (e.g., wake -> sleep
        after N continuous frames). Returns the new state if a transition
        happened, None if state is unchanged.

        Args:
            frame_metrics: Metrics from the current frame (may influence transitions).

        Returns:
            New state if transition occurred, None otherwise.
        """

    @abstractmethod
    def time_in_state(self) -> float:
        """Seconds elapsed since entering the current state."""

    @abstractmethod
    def transition_history(self) -> list[tuple[ConsciousState, float]]:
        """History of (state, entry_timestamp) pairs, oldest first."""


# ---------------------------------------------------------------------------
# Conscious loop: top-level runtime orchestrator
# ---------------------------------------------------------------------------


class ConsciousLoop(ABC):
    """Interface for the main conscious processing loop.

    Orchestrates one conscious frame: substrate step -> gating ->
    explicit generation -> self-prediction -> metrics collection ->
    state machine tick. Runs at ~20 Hz.

    Implementations wire together concrete versions of all other interfaces.
    The loop itself is abstract to allow different scheduling strategies
    (synchronous, async, distributed).
    """

    @abstractmethod
    async def run_frame(self) -> FrameMetrics:
        """Execute one conscious frame.

        Returns:
            Metrics for this frame.
        """

    @abstractmethod
    async def run(self, max_frames: int | None = None) -> None:
        """Run the conscious loop continuously.

        Args:
            max_frames: Stop after this many frames. None = run until stopped.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Signal the loop to stop after the current frame completes."""

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the loop is currently executing frames."""

    @abstractmethod
    def frame_count(self) -> int:
        """Total frames executed since last reset."""
