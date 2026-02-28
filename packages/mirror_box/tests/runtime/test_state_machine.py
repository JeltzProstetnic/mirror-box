"""Tests for MirrorBoxStateController — timer-based state machine.

Verifies automatic transitions (WAKE->SLEEP->DREAM->WAKE), manual
overrides via transition_to(), transition history tracking, and
time_in_state() computation.
"""

from __future__ import annotations

import time

import pytest
import torch

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from mirror_box.config import MirrorBoxConfig
from mirror_box.runtime.state_machine import MirrorBoxStateController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_config() -> MirrorBoxConfig:
    """MirrorBoxConfig with very short durations for fast tests."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=64,
        quantization="none",
        conscious_rate_hz=20.0,
        continuous_wake_frames=10,
        sleep_duration_frames=5,
        dream_duration_frames=3,
        altered_duration_frames=4,
        wandb_enabled=False,
    )


@pytest.fixture
def state_ctrl(short_config: MirrorBoxConfig) -> MirrorBoxStateController:
    """State controller with short durations."""
    return MirrorBoxStateController(config=short_config)


@pytest.fixture
def dummy_metrics() -> FrameMetrics:
    """Dummy FrameMetrics for tick() calls."""
    return FrameMetrics(
        R=0.5,
        self_surprise=1.0,
        C=0.5,
        complexity=0.4,
        state=ConsciousState.WAKE,
        frame_number=0,
        timestamp=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Helper: advance N frames
# ---------------------------------------------------------------------------


def _tick_n(ctrl: MirrorBoxStateController, metrics: FrameMetrics, n: int) -> ConsciousState | None:
    """Tick the state controller N times, return last transition result."""
    result = None
    for _ in range(n):
        r = ctrl.tick(metrics)
        if r is not None:
            result = r
    return result


# ===========================================================================
# Initial state
# ===========================================================================


class TestInitialState:
    """State controller starts in WAKE."""

    def test_starts_in_wake(self, state_ctrl: MirrorBoxStateController) -> None:
        assert state_ctrl.current_state() == ConsciousState.WAKE

    def test_initial_time_in_state_zero(self, state_ctrl: MirrorBoxStateController) -> None:
        assert state_ctrl.time_in_state() == 0.0

    def test_initial_history_has_wake(self, state_ctrl: MirrorBoxStateController) -> None:
        history = state_ctrl.transition_history()
        assert len(history) == 1
        assert history[0][0] == ConsciousState.WAKE

    def test_initial_frames_in_state_zero(self, state_ctrl: MirrorBoxStateController) -> None:
        assert state_ctrl.frames_in_state == 0


# ===========================================================================
# Automatic transitions
# ===========================================================================


class TestAutoTransitionWakeToSleep:
    """WAKE -> SLEEP after continuous_wake_frames."""

    def test_no_transition_before_threshold(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """Should remain in WAKE before reaching the threshold."""
        for _ in range(9):  # config: continuous_wake_frames=10
            result = state_ctrl.tick(dummy_metrics)
            assert result is None
        assert state_ctrl.current_state() == ConsciousState.WAKE

    def test_transitions_at_threshold(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """Should transition to SLEEP at exactly continuous_wake_frames."""
        result = _tick_n(state_ctrl, dummy_metrics, 10)
        assert result == ConsciousState.SLEEP
        assert state_ctrl.current_state() == ConsciousState.SLEEP


class TestAutoTransitionSleepToDream:
    """SLEEP -> DREAM after sleep_duration_frames."""

    def test_sleep_to_dream(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        # First: WAKE -> SLEEP
        _tick_n(state_ctrl, dummy_metrics, 10)
        assert state_ctrl.current_state() == ConsciousState.SLEEP

        # Then: SLEEP -> DREAM (5 frames)
        result = _tick_n(state_ctrl, dummy_metrics, 5)
        assert result == ConsciousState.DREAM
        assert state_ctrl.current_state() == ConsciousState.DREAM


class TestAutoTransitionDreamToWake:
    """DREAM -> WAKE after dream_duration_frames."""

    def test_dream_to_wake(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        # WAKE -> SLEEP -> DREAM
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
        _tick_n(state_ctrl, dummy_metrics, 5)   # SLEEP -> DREAM
        assert state_ctrl.current_state() == ConsciousState.DREAM

        # DREAM -> WAKE (3 frames)
        result = _tick_n(state_ctrl, dummy_metrics, 3)
        assert result == ConsciousState.WAKE
        assert state_ctrl.current_state() == ConsciousState.WAKE


class TestAutoTransitionAlteredToWake:
    """ALTERED -> WAKE after altered_duration_frames."""

    def test_altered_auto_returns_to_wake(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        state_ctrl.transition_to(ConsciousState.ALTERED)
        assert state_ctrl.current_state() == ConsciousState.ALTERED

        result = _tick_n(state_ctrl, dummy_metrics, 4)
        assert result == ConsciousState.WAKE
        assert state_ctrl.current_state() == ConsciousState.WAKE


class TestFullCycle:
    """Full cycle: WAKE -> SLEEP -> DREAM -> WAKE completes."""

    def test_full_cycle(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        # WAKE (10 frames) -> SLEEP
        _tick_n(state_ctrl, dummy_metrics, 10)
        assert state_ctrl.current_state() == ConsciousState.SLEEP

        # SLEEP (5 frames) -> DREAM
        _tick_n(state_ctrl, dummy_metrics, 5)
        assert state_ctrl.current_state() == ConsciousState.DREAM

        # DREAM (3 frames) -> WAKE
        _tick_n(state_ctrl, dummy_metrics, 3)
        assert state_ctrl.current_state() == ConsciousState.WAKE

    def test_two_full_cycles(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """Two complete cycles should work without issues."""
        for _ in range(2):
            _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
            _tick_n(state_ctrl, dummy_metrics, 5)   # SLEEP -> DREAM
            _tick_n(state_ctrl, dummy_metrics, 3)   # DREAM -> WAKE

        assert state_ctrl.current_state() == ConsciousState.WAKE


# ===========================================================================
# Manual transitions (transition_to)
# ===========================================================================


class TestManualTransitions:
    """transition_to() works for manual overrides."""

    def test_wake_to_altered(self, state_ctrl: MirrorBoxStateController) -> None:
        result = state_ctrl.transition_to(ConsciousState.ALTERED)
        assert result is True
        assert state_ctrl.current_state() == ConsciousState.ALTERED

    def test_sleep_to_wake_manual(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
        assert state_ctrl.current_state() == ConsciousState.SLEEP

        result = state_ctrl.transition_to(ConsciousState.WAKE)
        assert result is True
        assert state_ctrl.current_state() == ConsciousState.WAKE

    def test_manual_resets_frame_counter(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """Manual transition resets the frames_in_state counter."""
        _tick_n(state_ctrl, dummy_metrics, 5)
        assert state_ctrl.frames_in_state == 5

        state_ctrl.transition_to(ConsciousState.ALTERED)
        assert state_ctrl.frames_in_state == 0

    def test_transition_to_same_state_is_noop(
        self, state_ctrl: MirrorBoxStateController
    ) -> None:
        """Transitioning to the current state is a no-op that returns True."""
        result = state_ctrl.transition_to(ConsciousState.WAKE)
        assert result is True
        assert state_ctrl.current_state() == ConsciousState.WAKE
        # History should still have just 1 entry (initial)
        assert len(state_ctrl.transition_history()) == 1

    def test_any_state_reachable_manually(self, state_ctrl: MirrorBoxStateController) -> None:
        """All states are reachable via transition_to."""
        for target in ConsciousState:
            result = state_ctrl.transition_to(target)
            assert result is True
            assert state_ctrl.current_state() == target


# ===========================================================================
# Transition history
# ===========================================================================


class TestTransitionHistory:
    """transition_history() records all transitions."""

    def test_history_records_manual_transitions(
        self, state_ctrl: MirrorBoxStateController
    ) -> None:
        state_ctrl.transition_to(ConsciousState.ALTERED)
        state_ctrl.transition_to(ConsciousState.SLEEP)

        history = state_ctrl.transition_history()
        assert len(history) == 3  # WAKE (initial) + ALTERED + SLEEP
        assert history[0][0] == ConsciousState.WAKE
        assert history[1][0] == ConsciousState.ALTERED
        assert history[2][0] == ConsciousState.SLEEP

    def test_history_records_automatic_transitions(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP

        history = state_ctrl.transition_history()
        assert len(history) == 2
        assert history[0][0] == ConsciousState.WAKE
        assert history[1][0] == ConsciousState.SLEEP

    def test_history_timestamps_increase(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """Timestamps in history should be monotonically non-decreasing."""
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
        _tick_n(state_ctrl, dummy_metrics, 5)   # SLEEP -> DREAM

        history = state_ctrl.transition_history()
        timestamps = [ts for _, ts in history]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


# ===========================================================================
# time_in_state
# ===========================================================================


class TestTimeInState:
    """time_in_state() returns correct value based on frames and rate."""

    def test_time_after_ticks(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """After 5 ticks at 20 Hz, time_in_state should be 0.25s."""
        _tick_n(state_ctrl, dummy_metrics, 5)
        expected = 5 / 20.0  # 0.25
        assert state_ctrl.time_in_state() == pytest.approx(expected)

    def test_time_resets_after_transition(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """time_in_state resets to 0 after a transition."""
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
        assert state_ctrl.time_in_state() == 0.0

    def test_time_accumulates_in_new_state(
        self, state_ctrl: MirrorBoxStateController, dummy_metrics: FrameMetrics
    ) -> None:
        """After transition + more ticks, time accumulates in new state."""
        _tick_n(state_ctrl, dummy_metrics, 10)  # WAKE -> SLEEP
        _tick_n(state_ctrl, dummy_metrics, 3)   # 3 frames in SLEEP
        expected = 3 / 20.0
        assert state_ctrl.time_in_state() == pytest.approx(expected)
