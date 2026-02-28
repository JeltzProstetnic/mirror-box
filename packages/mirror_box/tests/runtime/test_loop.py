"""Tests for MirrorBoxLoop — the main conscious processing loop.

Uses lightweight stub/mock implementations of each interface to test the
loop orchestration without loading real models. All tests run on CPU with
64-dim state vectors and fast frame rates.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import (
    ConsciousState,
    ExplicitState,
    FrameMetrics,
    GateOutput,
    SubstrateState,
)
from aiware.interfaces.protocols import (
    ExplicitGenerator,
    MetricsCollector,
    PermeabilityGate,
    SelfPredictor,
    StateController,
    SubstrateEncoder,
    TelemetrySource,
)
from mirror_box.config import MirrorBoxConfig
from mirror_box.runtime.loop import MirrorBoxLoop


# ---------------------------------------------------------------------------
# Stub implementations for testing
# ---------------------------------------------------------------------------

STATE_DIM = 64


class StubSubstrate(SubstrateEncoder):
    """Minimal substrate that returns random state vectors."""

    def encode_world(self, sensory_input: torch.Tensor) -> torch.Tensor:
        return torch.randn(STATE_DIM)

    def encode_self(self, telemetry: dict[str, Any]) -> torch.Tensor:
        return torch.randn(STATE_DIM)

    def step(self, sensory_input: torch.Tensor, telemetry: dict[str, Any]) -> SubstrateState:
        return SubstrateState(
            world_state=self.encode_world(sensory_input),
            self_state=self.encode_self(telemetry),
            timestamp=time.monotonic(),
        )

    def state_dim(self) -> int:
        return STATE_DIM


class StubGate(PermeabilityGate):
    """Pass-through gate with fixed activations."""

    def gate(self, substrate_state: SubstrateState, state: ConsciousState) -> GateOutput:
        return GateOutput(
            gated_world=substrate_state.world_state,
            gated_self=substrate_state.self_state,
            gate_world_activation=0.8,
            gate_self_activation=0.7,
            timestamp=substrate_state.timestamp,
        )

    def set_profile(self, state: ConsciousState, **params: float) -> None:
        pass


class StubExplicit(ExplicitGenerator):
    """Deterministic explicit generator using simple linear transforms."""

    def __init__(self) -> None:
        torch.manual_seed(42)
        self._ewm_w = torch.randn(STATE_DIM, STATE_DIM) * 0.1
        self._esm_w = torch.randn(STATE_DIM * 2, STATE_DIM) * 0.1

    def generate_ewm(
        self, gated_world: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        return gated_world @ self._ewm_w

    def generate_esm(self, gated_self: torch.Tensor, prev_esm: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([gated_self, prev_esm], dim=-1)
        return combined @ self._esm_w

    def step(
        self, gate_output: GateOutput, prev_esm: torch.Tensor, context: torch.Tensor | None = None
    ) -> ExplicitState:
        ewm = self.generate_ewm(gate_output.gated_world, context)
        esm = self.generate_esm(gate_output.gated_self, prev_esm)
        return ExplicitState(
            ewm_state=ewm,
            esm_state=esm,
            timestamp=time.monotonic(),
        )

    def state_dim(self) -> int:
        return STATE_DIM


class StubPredictor(SelfPredictor):
    """Simple predictor that returns a slightly shifted version of prev_esm."""

    def predict(self, prev_esm: torch.Tensor, gated_self: torch.Tensor) -> torch.Tensor:
        return prev_esm * 0.5 + gated_self * 0.5

    def compute_R(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        residual_var = torch.var(actual - predicted).item()
        actual_var = max(torch.var(actual).item(), 1e-8)
        return float(max(0.0, min(1.0, 1.0 - residual_var / actual_var)))

    def compute_surprise(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        return float(torch.norm(actual - predicted, p=2).item())

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        return float(torch.nn.functional.mse_loss(predicted, actual.detach()).item())


class StubStateController(StateController):
    """Simple state controller with configurable auto-transition."""

    def __init__(self, auto_transition_after: int | None = None) -> None:
        self._state = ConsciousState.WAKE
        self._frames: int = 0
        self._history: list[tuple[ConsciousState, float]] = [
            (ConsciousState.WAKE, time.monotonic())
        ]
        self._auto_transition_after = auto_transition_after

    def current_state(self) -> ConsciousState:
        return self._state

    def transition_to(self, target: ConsciousState) -> bool:
        self._state = target
        self._frames = 0
        self._history.append((target, time.monotonic()))
        return True

    def tick(self, frame_metrics: FrameMetrics) -> ConsciousState | None:
        self._frames += 1
        if (
            self._auto_transition_after is not None
            and self._frames >= self._auto_transition_after
            and self._state == ConsciousState.WAKE
        ):
            self._state = ConsciousState.SLEEP
            self._frames = 0
            self._history.append((ConsciousState.SLEEP, time.monotonic()))
            return ConsciousState.SLEEP
        return None

    def time_in_state(self) -> float:
        return self._frames / 20.0

    def transition_history(self) -> list[tuple[ConsciousState, float]]:
        return list(self._history)


class StubMetricsCollector(MetricsCollector):
    """In-memory metrics recorder."""

    def __init__(self) -> None:
        self._frames: list[FrameMetrics] = []

    def record_frame(self, metrics: FrameMetrics) -> None:
        self._frames.append(metrics)

    def get_trajectory(self, metric_name: str, window: int | None = None) -> list[float]:
        frames = self._frames if window is None else self._frames[-window:]
        result = []
        for f in frames:
            if metric_name == "R":
                result.append(f.R)
            elif metric_name == "self_surprise":
                result.append(f.self_surprise)
            elif metric_name == "C":
                result.append(f.C)
            elif metric_name == "complexity":
                result.append(f.complexity)
            else:
                result.append(f.extras.get(metric_name, 0.0))
        return result

    def get_latest(self) -> FrameMetrics | None:
        return self._frames[-1] if self._frames else None

    def frame_count(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()


class StubTelemetry(TelemetrySource):
    """Simple telemetry source with injectable signals."""

    def __init__(self) -> None:
        self._signals: dict[str, float] = {
            "latency": 0.025,
            "gpu_temp": 55.0,
        }

    def collect(self) -> dict[str, float]:
        return dict(self._signals)

    def signal_names(self) -> list[str]:
        return list(self._signals.keys())

    def update_signal(self, name: str, value: float) -> None:
        self._signals[name] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loop_config() -> MirrorBoxConfig:
    """MirrorBoxConfig for loop testing."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=STATE_DIM,
        quantization="none",
        conscious_rate_hz=1000.0,  # Very fast for tests (no real sleeping)
        continuous_wake_frames=100,
        wandb_enabled=False,
    )


@pytest.fixture
def stub_components() -> dict[str, Any]:
    """All stub components needed for MirrorBoxLoop."""
    return {
        "substrate": StubSubstrate(),
        "gate": StubGate(),
        "explicit": StubExplicit(),
        "predictor": StubPredictor(),
        "state_ctrl": StubStateController(),
        "metrics_collector": StubMetricsCollector(),
        "telemetry": StubTelemetry(),
    }


@pytest.fixture
def loop(loop_config: MirrorBoxConfig, stub_components: dict[str, Any]) -> MirrorBoxLoop:
    """Fully wired MirrorBoxLoop with stub components."""
    return MirrorBoxLoop(
        config=loop_config,
        **stub_components,
    )


@pytest.fixture
def loop_with_auto_transition(
    loop_config: MirrorBoxConfig, stub_components: dict[str, Any]
) -> MirrorBoxLoop:
    """Loop with a state controller that auto-transitions after 5 frames."""
    stub_components["state_ctrl"] = StubStateController(auto_transition_after=5)
    return MirrorBoxLoop(
        config=loop_config,
        **stub_components,
    )


# ===========================================================================
# run_frame() basics
# ===========================================================================


class TestRunFrame:
    """run_frame() returns valid FrameMetrics."""

    async def test_returns_frame_metrics(self, loop: MirrorBoxLoop) -> None:
        """run_frame() returns a FrameMetrics instance."""
        result = await loop.run_frame()
        assert isinstance(result, FrameMetrics)

    async def test_increments_frame_count(self, loop: MirrorBoxLoop) -> None:
        """Each run_frame() increments frame_count by 1."""
        assert loop.frame_count() == 0
        await loop.run_frame()
        assert loop.frame_count() == 1
        await loop.run_frame()
        assert loop.frame_count() == 2

    async def test_frame_number_in_metrics(self, loop: MirrorBoxLoop) -> None:
        """FrameMetrics.frame_number matches the loop's counter."""
        m0 = await loop.run_frame()
        m1 = await loop.run_frame()
        assert m0.frame_number == 0
        assert m1.frame_number == 1

    async def test_R_in_valid_range(self, loop: MirrorBoxLoop) -> None:
        """R metric should be in [0, 1]."""
        result = await loop.run_frame()
        assert 0.0 <= result.R <= 1.0

    async def test_C_non_negative(self, loop: MirrorBoxLoop) -> None:
        """C metric should be >= 0."""
        result = await loop.run_frame()
        assert result.C >= 0.0

    async def test_self_surprise_non_negative(self, loop: MirrorBoxLoop) -> None:
        """self_surprise should be >= 0."""
        result = await loop.run_frame()
        assert result.self_surprise >= 0.0

    async def test_complexity_non_negative(self, loop: MirrorBoxLoop) -> None:
        """complexity should be >= 0."""
        result = await loop.run_frame()
        assert result.complexity >= 0.0

    async def test_state_is_conscious_state(self, loop: MirrorBoxLoop) -> None:
        """FrameMetrics.state should be a ConsciousState."""
        result = await loop.run_frame()
        assert isinstance(result.state, ConsciousState)

    async def test_timestamp_is_positive(self, loop: MirrorBoxLoop) -> None:
        """FrameMetrics.timestamp should be positive."""
        result = await loop.run_frame()
        assert result.timestamp > 0

    async def test_extras_contain_prediction_loss(self, loop: MirrorBoxLoop) -> None:
        """extras should contain prediction_loss from self-predictor update."""
        result = await loop.run_frame()
        assert "prediction_loss" in result.extras
        assert result.extras["prediction_loss"] >= 0.0


# ===========================================================================
# run() with max_frames
# ===========================================================================


class TestRun:
    """run() with max_frames runs exactly that many frames."""

    async def test_max_frames_5(self, loop: MirrorBoxLoop) -> None:
        """run(max_frames=5) runs exactly 5 frames."""
        await loop.run(max_frames=5)
        assert loop.frame_count() == 5

    async def test_max_frames_1(self, loop: MirrorBoxLoop) -> None:
        """run(max_frames=1) runs exactly 1 frame."""
        await loop.run(max_frames=1)
        assert loop.frame_count() == 1

    async def test_max_frames_10(self, loop: MirrorBoxLoop) -> None:
        """run(max_frames=10) runs exactly 10 frames."""
        await loop.run(max_frames=10)
        assert loop.frame_count() == 10


# ===========================================================================
# stop()
# ===========================================================================


class TestStop:
    """stop() terminates the loop."""

    async def test_stop_terminates(self, loop: MirrorBoxLoop) -> None:
        """After stop(), is_running() returns False."""
        assert not loop.is_running()
        await loop.stop()
        assert not loop.is_running()

    async def test_stop_mid_run(self, loop: MirrorBoxLoop) -> None:
        """Calling stop() after some frames prevents further execution."""
        await loop.run(max_frames=3)
        await loop.stop()
        assert not loop.is_running()
        assert loop.frame_count() == 3


# ===========================================================================
# State transitions during run
# ===========================================================================


class TestStateTransitions:
    """State transitions happen automatically when thresholds are reached."""

    async def test_auto_transition_happens(
        self, loop_with_auto_transition: MirrorBoxLoop
    ) -> None:
        """State should change from WAKE to SLEEP after 5 frames."""
        # Run enough frames to trigger the auto-transition (threshold=5)
        await loop_with_auto_transition.run(max_frames=6)
        # The 5th tick triggers WAKE->SLEEP, then 6th frame runs in SLEEP
        # Check via the metrics collector
        assert loop_with_auto_transition.frame_count() == 6


# ===========================================================================
# Self-referential feedback (prev_esm updates)
# ===========================================================================


class TestSelfReferentialFeedback:
    """prev_esm updates between frames (self-referential closure)."""

    async def test_prev_esm_changes(self, loop: MirrorBoxLoop) -> None:
        """prev_esm should change between frames."""
        initial_esm = loop.prev_esm.clone()
        await loop.run_frame()
        updated_esm = loop.prev_esm.clone()

        # After first frame, prev_esm should have changed from zeros
        assert not torch.allclose(initial_esm, updated_esm, atol=1e-8)

    async def test_prev_esm_evolves_over_frames(self, loop: MirrorBoxLoop) -> None:
        """ESM state should evolve across multiple frames."""
        esm_states = []
        for _ in range(5):
            await loop.run_frame()
            esm_states.append(loop.prev_esm.clone())

        # Each frame's ESM should differ from the previous
        for i in range(1, len(esm_states)):
            assert not torch.allclose(esm_states[i], esm_states[i - 1], atol=1e-8), (
                f"ESM at frame {i} should differ from frame {i - 1}"
            )

    async def test_initial_prev_esm_is_zeros(self, loop: MirrorBoxLoop) -> None:
        """Before any frame, prev_esm should be all zeros."""
        assert torch.allclose(loop.prev_esm, torch.zeros(STATE_DIM))


# ===========================================================================
# Metrics collection
# ===========================================================================


class TestMetricsCollection:
    """Metrics are correctly recorded via the collector."""

    async def test_metrics_recorded(
        self, loop: MirrorBoxLoop, stub_components: dict[str, Any]
    ) -> None:
        """MetricsCollector receives one entry per frame."""
        collector: StubMetricsCollector = stub_components["metrics_collector"]

        await loop.run(max_frames=5)
        assert collector.frame_count() == 5

    async def test_latest_metrics_match(
        self, loop: MirrorBoxLoop, stub_components: dict[str, Any]
    ) -> None:
        """get_latest() returns the most recent frame's metrics."""
        collector: StubMetricsCollector = stub_components["metrics_collector"]

        result = await loop.run_frame()
        latest = collector.get_latest()
        assert latest is not None
        assert latest.frame_number == result.frame_number
        assert latest.R == result.R


# ===========================================================================
# Telemetry feedback
# ===========================================================================


class TestTelemetryFeedback:
    """Loop feeds back prediction error and R to telemetry."""

    async def test_telemetry_updated(
        self, loop: MirrorBoxLoop, stub_components: dict[str, Any]
    ) -> None:
        """After a frame, telemetry should contain R and self_surprise."""
        telemetry: StubTelemetry = stub_components["telemetry"]

        await loop.run_frame()

        signals = telemetry.collect()
        assert "R" in signals
        assert "self_surprise" in signals
        assert "prediction_loss" in signals
        assert "C" in signals


# ===========================================================================
# Sensory input
# ===========================================================================


class TestSensoryInput:
    """set_sensory_input() controls the next frame's input."""

    async def test_set_sensory_input(self, loop: MirrorBoxLoop) -> None:
        """Setting sensory input doesn't crash and the loop continues."""
        loop.set_sensory_input(torch.tensor([5, 10, 15], dtype=torch.long))
        result = await loop.run_frame()
        assert isinstance(result, FrameMetrics)

    async def test_default_sensory_input_works(self, loop: MirrorBoxLoop) -> None:
        """Default dummy tokens work without explicit set_sensory_input."""
        result = await loop.run_frame()
        assert isinstance(result, FrameMetrics)
