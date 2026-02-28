"""Tests for MirrorBoxDashboard -- Gradio visualization dashboard.

Uses stub/mock implementations of loop, metrics, state controller, and
reporter. Tests verify that the dashboard can be instantiated, built,
and that its data helper methods return valid results.

Skips gracefully when gradio is not installed.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from aiware.interfaces.protocols import ConsciousLoop, MetricsCollector, StateController

gr = pytest.importorskip("gradio")


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class StubLoop(ConsciousLoop):
    """Minimal loop stub for dashboard testing."""

    def __init__(self) -> None:
        self._running = False
        self._frames = 0
        self._sensory_input: Any = None

    async def run_frame(self) -> FrameMetrics:
        self._frames += 1
        return FrameMetrics(
            R=0.5, self_surprise=1.0, C=0.6, complexity=0.3,
            state=ConsciousState.WAKE, frame_number=self._frames - 1,
            timestamp=time.monotonic(),
        )

    async def run(self, max_frames: int | None = None) -> None:
        pass

    async def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def frame_count(self) -> int:
        return self._frames

    def set_sensory_input(self, token_ids: Any) -> None:
        self._sensory_input = token_ids


class StubMetrics(MetricsCollector):
    """In-memory metrics stub with injectable data."""

    def __init__(self) -> None:
        self._frames: list[FrameMetrics] = []

    def record_frame(self, metrics: FrameMetrics) -> None:
        self._frames.append(metrics)

    def get_trajectory(self, metric_name: str, window: int | None = None) -> list[float]:
        frames = self._frames if window is None else self._frames[-window:]
        named = {"R", "self_surprise", "C", "complexity"}
        if metric_name in named:
            return [getattr(f, metric_name) for f in frames]
        has = any(metric_name in f.extras for f in frames)
        if not has:
            return []
        return [f.extras.get(metric_name, 0.0) for f in frames]

    def get_latest(self) -> FrameMetrics | None:
        return self._frames[-1] if self._frames else None

    def frame_count(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()


class StubStateCtrl(StateController):
    """Minimal state controller stub."""

    def __init__(self) -> None:
        self._state = ConsciousState.WAKE
        self._frames = 0
        self._history: list[tuple[ConsciousState, float]] = [
            (ConsciousState.WAKE, time.monotonic()),
        ]

    def current_state(self) -> ConsciousState:
        return self._state

    def transition_to(self, target: ConsciousState) -> bool:
        old = self._state
        self._state = target
        self._frames = 0
        self._history.append((target, time.monotonic()))
        return True

    def tick(self, frame_metrics: FrameMetrics) -> ConsciousState | None:
        self._frames += 1
        return None

    def time_in_state(self) -> float:
        return self._frames / 20.0

    def transition_history(self) -> list[tuple[ConsciousState, float]]:
        return list(self._history)


class StubReporter:
    """Minimal reporter stub for dashboard testing."""

    def generate_ewm_narrative(self) -> str:
        return "[EWM] World perception is nominal."

    def generate_esm_narrative(self) -> str:
        return "[ESM] Self-model is stable."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_loop() -> StubLoop:
    return StubLoop()


@pytest.fixture
def stub_metrics() -> StubMetrics:
    return StubMetrics()


@pytest.fixture
def stub_state_ctrl() -> StubStateCtrl:
    return StubStateCtrl()


@pytest.fixture
def stub_reporter() -> StubReporter:
    return StubReporter()


@pytest.fixture
def dashboard(
    stub_loop: StubLoop,
    stub_metrics: StubMetrics,
    stub_state_ctrl: StubStateCtrl,
    stub_reporter: StubReporter,
) -> Any:
    """MirrorBoxDashboard with stub components."""
    from mirror_box.visualization.dashboard import MirrorBoxDashboard

    return MirrorBoxDashboard(
        loop=stub_loop,
        metrics=stub_metrics,
        state_ctrl=stub_state_ctrl,
        reporter=stub_reporter,
    )


@pytest.fixture
def dashboard_no_reporter(
    stub_loop: StubLoop,
    stub_metrics: StubMetrics,
    stub_state_ctrl: StubStateCtrl,
) -> Any:
    """MirrorBoxDashboard without a reporter."""
    from mirror_box.visualization.dashboard import MirrorBoxDashboard

    return MirrorBoxDashboard(
        loop=stub_loop,
        metrics=stub_metrics,
        state_ctrl=stub_state_ctrl,
        reporter=None,
    )


@pytest.fixture
def sample_metrics(stub_metrics: StubMetrics) -> StubMetrics:
    """Populate stub_metrics with sample frame data."""
    for i in range(10):
        stub_metrics.record_frame(
            FrameMetrics(
                R=0.1 * i,
                self_surprise=1.0 - 0.05 * i,
                C=0.3 + 0.02 * i,
                complexity=0.2 + 0.01 * i,
                state=ConsciousState.WAKE,
                frame_number=i,
                timestamp=time.monotonic(),
                extras={
                    "gate_world_activation": 0.8,
                    "gate_self_activation": 0.7,
                    "prediction_loss": 0.1,
                },
            )
        )
    return stub_metrics


# ===========================================================================
# Instantiation
# ===========================================================================


class TestInstantiation:
    """MirrorBoxDashboard can be instantiated."""

    def test_create_with_reporter(self, dashboard: Any) -> None:
        """Dashboard can be created with a reporter."""
        assert dashboard is not None

    def test_create_without_reporter(self, dashboard_no_reporter: Any) -> None:
        """Dashboard can be created without a reporter."""
        assert dashboard_no_reporter is not None


# ===========================================================================
# build() returns a gr.Blocks
# ===========================================================================


class TestBuild:
    """build() returns a valid gr.Blocks object."""

    def test_build_returns_blocks(self, dashboard: Any) -> None:
        """build() returns a gr.Blocks instance."""
        result = dashboard.build()
        assert isinstance(result, gr.Blocks)

    def test_build_without_reporter(self, dashboard_no_reporter: Any) -> None:
        """build() works without a reporter."""
        result = dashboard_no_reporter.build()
        assert isinstance(result, gr.Blocks)

    def test_build_idempotent(self, dashboard: Any) -> None:
        """build() can be called multiple times."""
        result1 = dashboard.build()
        result2 = dashboard.build()
        assert isinstance(result1, gr.Blocks)
        assert isinstance(result2, gr.Blocks)


# ===========================================================================
# _get_metric_plot
# ===========================================================================


class TestGetMetricPlot:
    """_get_metric_plot returns valid DataFrames."""

    def test_empty_metrics(self, dashboard: Any) -> None:
        """Returns empty DataFrame when no metrics recorded."""
        import pandas as pd

        result = dashboard._get_metric_plot("R")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "frame" in result.columns
        assert "R" in result.columns

    def test_with_data(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """Returns populated DataFrame when metrics exist."""
        import pandas as pd

        result = dashboard._get_metric_plot("R", window=200)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "frame" in result.columns
        assert "R" in result.columns

    def test_window_limits(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """Window parameter limits number of returned points."""
        result = dashboard._get_metric_plot("R", window=5)
        assert len(result) == 5

    def test_all_metric_names(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """All four primary metrics return data."""
        for name in ["R", "C", "self_surprise", "complexity"]:
            result = dashboard._get_metric_plot(name)
            assert len(result) == 10, f"Expected 10 rows for {name}"

    def test_extras_metric(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """Extras metrics (gate activations) are accessible."""
        result = dashboard._get_metric_plot("gate_world_activation")
        assert len(result) == 10

    def test_unknown_metric_empty(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """Unknown metric name returns empty DataFrame."""
        result = dashboard._get_metric_plot("nonexistent_metric")
        assert len(result) == 0


# ===========================================================================
# _get_state_display
# ===========================================================================


class TestGetStateDisplay:
    """_get_state_display returns formatted state text."""

    def test_returns_string(self, dashboard: Any) -> None:
        """Returns a non-empty string."""
        result = dashboard._get_state_display()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_state_name(self, dashboard: Any) -> None:
        """Display contains the current state name."""
        result = dashboard._get_state_display()
        assert "WAKE" in result

    def test_contains_loop_status(self, dashboard: Any) -> None:
        """Display contains loop running status."""
        result = dashboard._get_state_display()
        assert "Loop running" in result

    def test_reflects_state_change(
        self, dashboard: Any, stub_state_ctrl: StubStateCtrl
    ) -> None:
        """Display updates when state changes."""
        stub_state_ctrl.transition_to(ConsciousState.SLEEP)
        result = dashboard._get_state_display()
        assert "SLEEP" in result


# ===========================================================================
# _trigger_state_transition
# ===========================================================================


class TestTriggerStateTransition:
    """_trigger_state_transition changes the system state."""

    def test_valid_transition(
        self, dashboard: Any, stub_state_ctrl: StubStateCtrl
    ) -> None:
        """Valid target state changes the controller state."""
        result = dashboard._trigger_state_transition("sleep")
        assert "SLEEP" in result
        assert stub_state_ctrl.current_state() == ConsciousState.SLEEP

    def test_same_state_noop(self, dashboard: Any) -> None:
        """Transitioning to the current state is a no-op."""
        result = dashboard._trigger_state_transition("wake")
        assert "Already" in result

    def test_invalid_state(self, dashboard: Any) -> None:
        """Invalid state name returns error message."""
        result = dashboard._trigger_state_transition("invalid_state")
        assert "Invalid" in result

    def test_all_states(
        self, dashboard: Any, stub_state_ctrl: StubStateCtrl
    ) -> None:
        """All four states can be targeted."""
        for state_name in ["sleep", "dream", "altered", "wake"]:
            result = dashboard._trigger_state_transition(state_name)
            assert stub_state_ctrl.current_state() == ConsciousState(state_name)


# ===========================================================================
# Gate data
# ===========================================================================


class TestGateData:
    """_get_gate_data returns valid DataFrames."""

    def test_empty_gate_data(self, dashboard: Any) -> None:
        """Returns zero activations when no metrics recorded."""
        import pandas as pd

        result = dashboard._get_gate_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["activation"]) == [0.0, 0.0]

    def test_populated_gate_data(
        self, dashboard: Any, sample_metrics: StubMetrics
    ) -> None:
        """Returns gate activations from latest frame."""
        import pandas as pd

        result = dashboard._get_gate_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["activation"].iloc[0] == pytest.approx(0.8)
        assert result["activation"].iloc[1] == pytest.approx(0.7)


# ===========================================================================
# Narratives
# ===========================================================================


class TestNarratives:
    """EWM/ESM narrative methods return text."""

    def test_ewm_with_reporter(self, dashboard: Any) -> None:
        """EWM narrative uses the reporter."""
        result = dashboard._get_ewm_narrative()
        assert "EWM" in result

    def test_esm_with_reporter(self, dashboard: Any) -> None:
        """ESM narrative uses the reporter."""
        result = dashboard._get_esm_narrative()
        assert "ESM" in result

    def test_ewm_without_reporter(self, dashboard_no_reporter: Any) -> None:
        """EWM returns placeholder when no reporter."""
        result = dashboard_no_reporter._get_ewm_narrative()
        assert "No introspection reporter" in result

    def test_esm_without_reporter(self, dashboard_no_reporter: Any) -> None:
        """ESM returns placeholder when no reporter."""
        result = dashboard_no_reporter._get_esm_narrative()
        assert "No introspection reporter" in result


# ===========================================================================
# _refresh_all
# ===========================================================================


class TestRefreshAll:
    """_refresh_all returns the correct tuple of outputs."""

    def test_returns_tuple_of_8(self, dashboard: Any) -> None:
        """Refresh returns exactly 8 outputs."""
        result = dashboard._refresh_all()
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_with_data(self, dashboard: Any, sample_metrics: StubMetrics) -> None:
        """Refresh works with populated metrics."""
        import pandas as pd

        result = dashboard._refresh_all()
        assert len(result) == 8
        # First 4 are DataFrames (metric plots)
        for i in range(4):
            assert isinstance(result[i], pd.DataFrame)
            assert len(result[i]) == 10
        # 5th is state display (str)
        assert isinstance(result[4], str)
        # 6th is gate data (DataFrame)
        assert isinstance(result[5], pd.DataFrame)
        # 7th and 8th are narrative strings
        assert isinstance(result[6], str)
        assert isinstance(result[7], str)


# ===========================================================================
# Lazy import guard
# ===========================================================================


class TestLazyImport:
    """get_dashboard_class() works as a lazy import guard."""

    def test_get_dashboard_class(self) -> None:
        """get_dashboard_class returns the MirrorBoxDashboard class."""
        from mirror_box.visualization import get_dashboard_class

        cls = get_dashboard_class()
        assert cls.__name__ == "MirrorBoxDashboard"
