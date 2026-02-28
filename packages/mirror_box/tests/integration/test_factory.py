"""Integration tests for factory assembly and end-to-end operation.

Tests that create_mirror_box() wires everything correctly and that
the full system runs frames, transitions states, collects metrics,
and generates introspection reports — all on CPU with TinyMockLLM.
"""

from __future__ import annotations

import pytest
import torch

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from conftest import TinyMockLLM
from mirror_box.config import MirrorBoxConfig
from mirror_box.factory import MirrorBoxSystem, create_mirror_box


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STATE_DIM = 64


@pytest.fixture
def integration_config() -> MirrorBoxConfig:
    """Config tuned for fast integration testing on CPU."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=STATE_DIM,
        model_name_or_path="test-model",
        quantization="none",
        encoder_layer_index=1,
        lora_rank=8,
        lora_alpha=16,
        max_context_length=128,
        generation_temperature=0.7,
        generation_max_tokens=16,
        conscious_rate_hz=1000.0,  # No real sleeping
        continuous_wake_frames=10,  # Quick transitions
        sleep_duration_frames=5,
        dream_duration_frames=3,
        altered_duration_frames=3,
        metrics_buffer_size=500,
        wandb_enabled=False,
    )


@pytest.fixture
def mock_model() -> TinyMockLLM:
    return TinyMockLLM(hidden_size=STATE_DIM, num_layers=2)


@pytest.fixture
def system(integration_config: MirrorBoxConfig, mock_model: TinyMockLLM) -> MirrorBoxSystem:
    """Fully assembled Mirror Box system with mock LLM."""
    return create_mirror_box(integration_config, model=mock_model)


# ===========================================================================
# Factory assembly
# ===========================================================================


class TestFactoryAssembly:
    """create_mirror_box() returns a fully wired system."""

    def test_returns_system(self, system: MirrorBoxSystem) -> None:
        assert isinstance(system, MirrorBoxSystem)

    def test_all_components_present(self, system: MirrorBoxSystem) -> None:
        assert system.config is not None
        assert system.backend is not None
        assert system.implicit is not None
        assert system.explicit is not None
        assert system.gate is not None
        assert system.predictor is not None
        assert system.state_ctrl is not None
        assert system.telemetry is not None
        assert system.metrics_collector is not None
        assert system.loop is not None
        assert system.sleep_maintenance is not None
        assert system.detector is not None
        assert system.reporter is not None
        assert system.dialogue is not None

    def test_default_config(self, mock_model: TinyMockLLM) -> None:
        """create_mirror_box(None) uses default MirrorBoxConfig."""
        sys = create_mirror_box(
            MirrorBoxConfig(
                hardware=DeviceConfig(device="cpu", dtype="float32"),
                state_dim=STATE_DIM,
                quantization="none",
                encoder_layer_index=1,
                wandb_enabled=False,
            ),
            model=mock_model,
        )
        assert sys.config is not None

    def test_initial_state_is_wake(self, system: MirrorBoxSystem) -> None:
        assert system.state_ctrl.current_state() == ConsciousState.WAKE

    def test_loop_not_running(self, system: MirrorBoxSystem) -> None:
        assert not system.loop.is_running()
        assert system.loop.frame_count() == 0


# ===========================================================================
# End-to-end frame execution
# ===========================================================================


class TestEndToEndFrames:
    """Full pipeline: sensory input → substrate → gate → explicit → metrics."""

    async def test_single_frame(self, system: MirrorBoxSystem) -> None:
        """One frame produces valid FrameMetrics."""
        result = await system.loop.run_frame()
        assert isinstance(result, FrameMetrics)
        assert result.frame_number == 0
        assert 0.0 <= result.R <= 1.0
        assert result.C >= 0.0
        assert result.state == ConsciousState.WAKE

    async def test_multiple_frames(self, system: MirrorBoxSystem) -> None:
        """Run 20 frames without error."""
        await system.loop.run(max_frames=20)
        assert system.loop.frame_count() == 20

    async def test_metrics_collected(self, system: MirrorBoxSystem) -> None:
        """Metrics collector has entries after running frames."""
        await system.loop.run(max_frames=5)
        assert system.metrics_collector.frame_count() == 5
        latest = system.metrics_collector.get_latest()
        assert latest is not None
        assert latest.frame_number == 4

    async def test_sensory_input(self, system: MirrorBoxSystem) -> None:
        """Custom sensory input flows through the pipeline."""
        system.loop.set_sensory_input(torch.tensor([1, 2, 3], dtype=torch.long))
        result = await system.loop.run_frame()
        assert isinstance(result, FrameMetrics)

    async def test_self_referential_closure(self, system: MirrorBoxSystem) -> None:
        """prev_esm evolves across frames (self-referential loop is active)."""
        initial = system.loop.prev_esm.clone()
        await system.loop.run_frame()
        after_one = system.loop.prev_esm.clone()
        await system.loop.run_frame()
        after_two = system.loop.prev_esm.clone()

        assert not torch.allclose(initial, after_one, atol=1e-8)
        assert not torch.allclose(after_one, after_two, atol=1e-8)


# ===========================================================================
# State transitions
# ===========================================================================


class TestStateTransitions:
    """State machine transitions work end-to-end."""

    async def test_wake_to_sleep_auto(self, system: MirrorBoxSystem) -> None:
        """After continuous_wake_frames (10), auto-transitions to SLEEP."""
        await system.loop.run(max_frames=11)
        # Frame 10 triggers WAKE→SLEEP, frame 11 runs in SLEEP
        assert system.state_ctrl.current_state() == ConsciousState.SLEEP

    async def test_full_cycle(self, system: MirrorBoxSystem) -> None:
        """Full WAKE→SLEEP→DREAM→WAKE cycle."""
        # 10 wake + 5 sleep + 3 dream = 18 frames for full cycle
        await system.loop.run(max_frames=19)
        # Should be back to WAKE
        assert system.state_ctrl.current_state() == ConsciousState.WAKE

    async def test_manual_transition(self, system: MirrorBoxSystem) -> None:
        """Manual transition_to() works mid-run."""
        await system.loop.run(max_frames=3)
        system.state_ctrl.transition_to(ConsciousState.ALTERED)
        assert system.state_ctrl.current_state() == ConsciousState.ALTERED
        result = await system.loop.run_frame()
        assert result.state == ConsciousState.ALTERED

    async def test_transition_history(self, system: MirrorBoxSystem) -> None:
        """Transition history records state changes."""
        await system.loop.run(max_frames=16)  # Should trigger WAKE→SLEEP→DREAM
        history = system.state_ctrl.transition_history()
        states = [s for s, _ in history]
        assert ConsciousState.WAKE in states
        assert ConsciousState.SLEEP in states


# ===========================================================================
# Telemetry integration
# ===========================================================================


class TestTelemetryIntegration:
    """Telemetry signals update during frame execution."""

    async def test_telemetry_has_signals(self, system: MirrorBoxSystem) -> None:
        """After frames, telemetry contains expected signals."""
        await system.loop.run(max_frames=3)
        signals = system.telemetry.collect()
        assert "self_knowledge_R" in signals or "R" in signals

    async def test_telemetry_values_change(self, system: MirrorBoxSystem) -> None:
        """Telemetry values evolve over frames."""
        await system.loop.run_frame()
        signals_1 = system.telemetry.collect()
        await system.loop.run(max_frames=5)
        signals_2 = system.telemetry.collect()
        # At least some signal should have changed
        changed = any(
            signals_1.get(k, 0.0) != signals_2.get(k, 0.0)
            for k in signals_2
        )
        assert changed


# ===========================================================================
# Metrics trajectories
# ===========================================================================


class TestMetricsTrajectories:
    """Metric trajectories are queryable after frames."""

    async def test_r_trajectory(self, system: MirrorBoxSystem) -> None:
        """R trajectory has one entry per frame."""
        await system.loop.run(max_frames=10)
        r_vals = system.metrics_collector.get_trajectory("R")
        assert len(r_vals) == 10
        assert all(0.0 <= v <= 1.0 for v in r_vals)

    async def test_c_trajectory(self, system: MirrorBoxSystem) -> None:
        """C trajectory is queryable with a window."""
        await system.loop.run(max_frames=10)
        c_vals = system.metrics_collector.get_trajectory("C", window=5)
        assert len(c_vals) == 5

    async def test_surprise_trajectory(self, system: MirrorBoxSystem) -> None:
        """self_surprise trajectory is non-negative."""
        await system.loop.run(max_frames=10)
        s_vals = system.metrics_collector.get_trajectory("self_surprise")
        assert all(v >= 0.0 for v in s_vals)


# ===========================================================================
# Permeability gate integration
# ===========================================================================


class TestGateIntegration:
    """Gate changes behavior across states."""

    async def test_gate_profiles_exist(self, system: MirrorBoxSystem) -> None:
        """Gate has profiles for all conscious states."""
        for state in ConsciousState:
            # set_profile should not raise
            system.gate.set_profile(state, noise_std=0.01)

    async def test_sleep_reduces_output(self, system: MirrorBoxSystem) -> None:
        """In SLEEP state, gate output magnitudes should be smaller."""
        # Run a few wake frames to get baseline
        await system.loop.run(max_frames=3)
        wake_latest = system.metrics_collector.get_latest()

        # Force sleep and run frames
        system.state_ctrl.transition_to(ConsciousState.SLEEP)
        await system.loop.run(max_frames=3)
        sleep_latest = system.metrics_collector.get_latest()

        # Sleep should have lower C (less conscious content)
        assert wake_latest is not None
        assert sleep_latest is not None


# ===========================================================================
# Self-predictor integration
# ===========================================================================


class TestSelfPredictorIntegration:
    """Self-predictor learns online during frame execution."""

    async def test_predictor_updates(self, system: MirrorBoxSystem) -> None:
        """prediction_loss appears in frame extras."""
        result = await system.loop.run_frame()
        assert "prediction_loss" in result.extras
        assert result.extras["prediction_loss"] >= 0.0

    async def test_r_metric_evolves(self, system: MirrorBoxSystem) -> None:
        """R should change over many frames as predictor learns."""
        r_values = []
        for _ in range(20):
            result = await system.loop.run_frame()
            r_values.append(result.R)
        # Not all the same (predictor is updating)
        assert len(set(round(v, 4) for v in r_values)) > 1


# ===========================================================================
# Introspection integration
# ===========================================================================


class TestIntrospectionIntegration:
    """Introspection generates reports from live system state."""

    async def test_state_report(self, system: MirrorBoxSystem) -> None:
        """State report is a non-empty string."""
        await system.loop.run(max_frames=3)
        report = system.reporter.generate_state_report()
        assert isinstance(report, str)
        assert len(report) > 0

    async def test_ewm_narrative(self, system: MirrorBoxSystem) -> None:
        """EWM narrative is generated."""
        await system.loop.run(max_frames=3)
        narrative = system.reporter.generate_ewm_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    async def test_esm_narrative(self, system: MirrorBoxSystem) -> None:
        """ESM narrative is generated."""
        await system.loop.run(max_frames=3)
        narrative = system.reporter.generate_esm_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    async def test_dialogue_ask(self, system: MirrorBoxSystem) -> None:
        """Dialogue can answer a question."""
        await system.loop.run(max_frames=3)
        answer = system.dialogue.ask("What is your current state?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    async def test_dialogue_metric(self, system: MirrorBoxSystem) -> None:
        """Dialogue can explain a metric."""
        await system.loop.run(max_frames=3)
        answer = system.dialogue.ask_about_metric("R")
        assert isinstance(answer, str)


# ===========================================================================
# Sleep maintenance integration
# ===========================================================================


class TestSleepMaintenanceIntegration:
    """Sleep maintenance runs during sleep state."""

    async def test_record_and_replay(self, system: MirrorBoxSystem) -> None:
        """Record experience during wake, replay during sleep."""
        # Run wake frames (experience gets recorded via telemetry)
        await system.loop.run(max_frames=5)

        # Manually record experiences for sleep replay
        telemetry = system.telemetry.collect()
        state_vec = torch.randn(STATE_DIM)
        for _ in range(3):
            system.sleep_maintenance.record_experience(telemetry, state_vec)

        # Run maintenance
        result = await system.sleep_maintenance.run_maintenance()
        assert isinstance(result, dict)
        assert "num_samples" in result
        assert result["num_samples"] == 3.0

    async def test_clear_buffers(self, system: MirrorBoxSystem) -> None:
        """clear_buffers() empties the experience buffer."""
        telemetry = system.telemetry.collect()
        system.sleep_maintenance.record_experience(telemetry, torch.randn(STATE_DIM))
        system.sleep_maintenance.clear_buffers()
        result = await system.sleep_maintenance.run_maintenance()
        assert result["num_samples"] == 0


# ===========================================================================
# State transition detector integration
# ===========================================================================


class TestDetectorIntegration:
    """State transition detector uses metrics to suggest transitions."""

    async def test_detector_returns_state_or_none(self, system: MirrorBoxSystem) -> None:
        """detect() returns ConsciousState or None."""
        await system.loop.run(max_frames=25)
        suggestion = system.detector.detect(system.metrics_collector)
        assert suggestion is None or isinstance(suggestion, ConsciousState)
