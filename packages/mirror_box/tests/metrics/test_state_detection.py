"""Tests for StateTransitionDetector — automated state detection from metrics.

Verifies detection of sleep (low C), wake (high C), stability (no transition),
and insufficient data handling.
"""

from __future__ import annotations

import time

import pytest

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from mirror_box.config import MirrorBoxConfig
from mirror_box.metrics.collector import InMemoryMetricsCollector
from mirror_box.metrics.state_detection import StateTransitionDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MirrorBoxConfig:
    """Config with large buffer for detector tests."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=64,
        quantization="none",
        metrics_buffer_size=1000,
        wandb_enabled=False,
    )


@pytest.fixture
def collector(config: MirrorBoxConfig) -> InMemoryMetricsCollector:
    return InMemoryMetricsCollector(config)


@pytest.fixture
def detector() -> StateTransitionDetector:
    """Detector with default thresholds, window=20."""
    return StateTransitionDetector(
        c_sleep_threshold=0.05,
        c_wake_threshold=0.3,
        r_volatility_threshold=0.2,
        window=20,
    )


def _make_frame(
    *,
    R: float = 0.5,
    self_surprise: float = 1.0,
    C: float = 0.5,
    complexity: float = 0.4,
    state: ConsciousState = ConsciousState.WAKE,
    frame_number: int = 0,
) -> FrameMetrics:
    """Factory for FrameMetrics with configurable values."""
    return FrameMetrics(
        R=R,
        self_surprise=self_surprise,
        C=C,
        complexity=complexity,
        state=state,
        frame_number=frame_number,
        timestamp=time.monotonic(),
    )


# ===========================================================================
# Insufficient data
# ===========================================================================


class TestInsufficientData:
    """Detector returns None when not enough data is available."""

    def test_empty_collector(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """No data -> None."""
        assert detector.detect(collector) is None

    def test_too_few_frames(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """Fewer frames than detector window -> None."""
        for i in range(5):  # window=20, so 5 is insufficient
            collector.record_frame(_make_frame(C=0.01, frame_number=i))
        assert detector.detect(collector) is None


# ===========================================================================
# Sleep detection (low C)
# ===========================================================================


class TestSleepDetection:
    """C(t) consistently below c_sleep_threshold -> SLEEP."""

    def test_all_low_c_returns_sleep(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """20 frames with C < 0.05 -> SLEEP."""
        for i in range(20):
            collector.record_frame(_make_frame(C=0.01, frame_number=i))
        result = detector.detect(collector)
        assert result == ConsciousState.SLEEP

    def test_mostly_low_c_returns_sleep(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """If the mean of recent C values is below threshold -> SLEEP."""
        for i in range(20):
            # Occasional small blip but mostly low
            c = 0.01 if i % 5 != 0 else 0.04
            collector.record_frame(_make_frame(C=c, frame_number=i))
        result = detector.detect(collector)
        assert result == ConsciousState.SLEEP


# ===========================================================================
# Wake detection (high C)
# ===========================================================================


class TestWakeDetection:
    """C(t) above c_wake_threshold -> WAKE."""

    def test_all_high_c_returns_wake(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """20 frames with C > 0.3 -> WAKE."""
        for i in range(20):
            collector.record_frame(_make_frame(C=0.8, frame_number=i))
        result = detector.detect(collector)
        assert result == ConsciousState.WAKE


# ===========================================================================
# Stable / no transition
# ===========================================================================


class TestStableState:
    """When metrics are in the middle range, no state transition is detected."""

    def test_middle_c_returns_none(
        self,
        detector: StateTransitionDetector,
        collector: InMemoryMetricsCollector,
    ) -> None:
        """C values between sleep and wake thresholds -> None (ambiguous)."""
        for i in range(20):
            collector.record_frame(_make_frame(C=0.15, frame_number=i))
        result = detector.detect(collector)
        assert result is None
