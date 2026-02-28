"""Tests for MirrorBoxTelemetry — system interoceptive signal collector.

Verifies signal initialization, collection, update, auto-registration of new
signals, and bulk update from FrameMetrics.
"""

from __future__ import annotations

import time

import pytest

from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from mirror_box.runtime.telemetry import MirrorBoxTelemetry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry() -> MirrorBoxTelemetry:
    """Fresh MirrorBoxTelemetry instance."""
    return MirrorBoxTelemetry()


# ---------------------------------------------------------------------------
# Expected signal names (canonical list from the docstring)
# ---------------------------------------------------------------------------

EXPECTED_SIGNALS = sorted([
    "inference_latency_ms",
    "self_prediction_error",
    "self_knowledge_R",
    "gate_activation_world",
    "gate_activation_self",
    "frame_rate_hz",
    "esm_activation_norm",
    "ewm_activation_norm",
    "time_since_sleep_s",
    "self_surprise",
    "frames_in_state",
])


# ===========================================================================
# Initial state
# ===========================================================================


class TestInitialState:
    """All signals start at 0.0 and are correctly enumerated."""

    def test_initial_values_all_zero(self, telemetry: MirrorBoxTelemetry) -> None:
        """All initial signal values must be 0.0."""
        snapshot = telemetry.collect()
        for name, value in snapshot.items():
            assert value == 0.0, f"Signal {name!r} should be 0.0, got {value}"

    def test_signal_names_returns_sorted(self, telemetry: MirrorBoxTelemetry) -> None:
        """signal_names() returns a sorted list."""
        names = telemetry.signal_names()
        assert names == sorted(names)

    def test_signal_names_stable(self, telemetry: MirrorBoxTelemetry) -> None:
        """Two calls to signal_names() produce the same list."""
        assert telemetry.signal_names() == telemetry.signal_names()

    def test_collect_returns_all_expected_signals(self, telemetry: MirrorBoxTelemetry) -> None:
        """collect() returns a dict containing all expected signal names."""
        snapshot = telemetry.collect()
        for name in EXPECTED_SIGNALS:
            assert name in snapshot, f"Missing signal {name!r} in collect() output"

    def test_signal_names_match_expected(self, telemetry: MirrorBoxTelemetry) -> None:
        """signal_names() matches the canonical signal set."""
        assert telemetry.signal_names() == EXPECTED_SIGNALS


# ===========================================================================
# collect()
# ===========================================================================


class TestCollect:
    """collect() returns correct snapshot."""

    def test_collect_returns_dict(self, telemetry: MirrorBoxTelemetry) -> None:
        """collect() returns a plain dict."""
        snapshot = telemetry.collect()
        assert isinstance(snapshot, dict)

    def test_collect_values_are_floats(self, telemetry: MirrorBoxTelemetry) -> None:
        """All values in the snapshot are floats."""
        for value in telemetry.collect().values():
            assert isinstance(value, float)

    def test_collect_is_a_copy(self, telemetry: MirrorBoxTelemetry) -> None:
        """Mutating the returned dict does not affect internal state."""
        snapshot = telemetry.collect()
        snapshot["inference_latency_ms"] = 999.0
        assert telemetry.collect()["inference_latency_ms"] == 0.0


# ===========================================================================
# update_signal()
# ===========================================================================


class TestUpdateSignal:
    """update_signal() changes signal values and auto-registers new ones."""

    def test_update_changes_value(self, telemetry: MirrorBoxTelemetry) -> None:
        """Updating a known signal changes its value."""
        telemetry.update_signal("inference_latency_ms", 42.5)
        assert telemetry.collect()["inference_latency_ms"] == 42.5

    def test_update_multiple_signals(self, telemetry: MirrorBoxTelemetry) -> None:
        """Multiple updates are reflected independently."""
        telemetry.update_signal("self_knowledge_R", 0.85)
        telemetry.update_signal("frame_rate_hz", 19.5)
        snapshot = telemetry.collect()
        assert snapshot["self_knowledge_R"] == 0.85
        assert snapshot["frame_rate_hz"] == 19.5

    def test_update_overwrites_previous(self, telemetry: MirrorBoxTelemetry) -> None:
        """A second update replaces the first."""
        telemetry.update_signal("self_surprise", 1.0)
        telemetry.update_signal("self_surprise", 2.0)
        assert telemetry.collect()["self_surprise"] == 2.0

    def test_update_accepts_new_signal(self, telemetry: MirrorBoxTelemetry) -> None:
        """Updating a non-existent signal auto-registers it."""
        telemetry.update_signal("custom_metric", 3.14)
        assert telemetry.collect()["custom_metric"] == 3.14

    def test_new_signal_appears_in_signal_names(self, telemetry: MirrorBoxTelemetry) -> None:
        """Auto-registered signal appears in signal_names()."""
        telemetry.update_signal("gpu_temperature", 75.0)
        assert "gpu_temperature" in telemetry.signal_names()

    def test_signal_names_still_sorted_after_new_signal(
        self, telemetry: MirrorBoxTelemetry
    ) -> None:
        """signal_names() remains sorted after adding a new signal."""
        telemetry.update_signal("aaa_first", 1.0)
        telemetry.update_signal("zzz_last", 2.0)
        names = telemetry.signal_names()
        assert names == sorted(names)


# ===========================================================================
# update_from_frame()
# ===========================================================================


class TestUpdateFromFrame:
    """update_from_frame() bulk-updates from FrameMetrics + gate activations."""

    @pytest.fixture
    def frame_metrics(self) -> FrameMetrics:
        """FrameMetrics with known values."""
        return FrameMetrics(
            R=0.72,
            self_surprise=1.5,
            C=0.88,
            complexity=0.6,
            state=ConsciousState.WAKE,
            frame_number=100,
            timestamp=time.monotonic(),
            extras={"prediction_loss": 0.05},
        )

    def test_R_updated(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """self_knowledge_R is set from FrameMetrics.R."""
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        assert telemetry.collect()["self_knowledge_R"] == 0.72

    def test_surprise_updated(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """self_surprise is set from FrameMetrics.self_surprise."""
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        assert telemetry.collect()["self_surprise"] == 1.5

    def test_self_prediction_error_updated(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """self_prediction_error is set from FrameMetrics.self_surprise."""
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        assert telemetry.collect()["self_prediction_error"] == 1.5

    def test_gate_activations_updated(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """Gate activations are set from the explicit parameters."""
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        snapshot = telemetry.collect()
        assert snapshot["gate_activation_world"] == 0.9
        assert snapshot["gate_activation_self"] == 0.7

    def test_frames_in_state_updated(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """frames_in_state is set from FrameMetrics.frame_number."""
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        assert telemetry.collect()["frames_in_state"] == 100.0

    def test_other_signals_unchanged(
        self, telemetry: MirrorBoxTelemetry, frame_metrics: FrameMetrics
    ) -> None:
        """Signals not covered by update_from_frame remain at their previous values."""
        telemetry.update_signal("inference_latency_ms", 50.0)
        telemetry.update_from_frame(frame_metrics, gate_world=0.9, gate_self=0.7)
        assert telemetry.collect()["inference_latency_ms"] == 50.0
