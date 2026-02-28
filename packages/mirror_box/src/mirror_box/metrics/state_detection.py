"""Automated state-transition detection from metric trajectories.

Uses threshold-based detection on C(t) and R trajectories to infer
the operating state the system should be in:

- C(t) drops below threshold -> entering sleep
- C(t) rises above threshold -> waking up (or dreaming if no external input)
- R volatility exceeds threshold -> potentially altered state

This is a *detection* module, not a *controller*. It reads metrics and
reports what state the metrics suggest, but does not trigger transitions.
The state machine (MirrorBoxStateController) decides whether to act on
the detector's output.
"""

from __future__ import annotations

import statistics

from aiware.interfaces.data_types import ConsciousState
from aiware.interfaces.protocols import MetricsCollector


class StateTransitionDetector:
    """Detects state transitions from metric trajectories.

    Uses threshold-based detection on C(t) and R trajectories:

    - C(t) average below ``c_sleep_threshold`` over the window -> SLEEP
    - C(t) average above ``c_wake_threshold`` over the window -> WAKE
    - R standard deviation above ``r_volatility_threshold`` -> ALTERED
    - Otherwise -> None (no clear state detected)

    Args:
        c_sleep_threshold: Mean C(t) below this -> SLEEP. Default 0.05.
        c_wake_threshold: Mean C(t) above this -> WAKE. Default 0.3.
        r_volatility_threshold: Std(R) above this -> ALTERED. Default 0.2.
        window: Number of recent frames to analyze. Default 20.
    """

    def __init__(
        self,
        c_sleep_threshold: float = 0.05,
        c_wake_threshold: float = 0.3,
        r_volatility_threshold: float = 0.2,
        window: int = 20,
    ) -> None:
        self._c_sleep_threshold = c_sleep_threshold
        self._c_wake_threshold = c_wake_threshold
        self._r_volatility_threshold = r_volatility_threshold
        self._window = window

    def detect(self, collector: MetricsCollector) -> ConsciousState | None:
        """Analyze recent metrics and return the detected state.

        Returns None if there is insufficient data (fewer than ``window``
        frames) or if the metrics are ambiguous (C between thresholds,
        R volatility below threshold).

        Args:
            collector: MetricsCollector to query for trajectories.

        Returns:
            Detected ConsciousState, or None if no clear state detected.
        """
        if collector.frame_count() < self._window:
            return None

        c_trajectory = collector.get_trajectory("C", window=self._window)
        r_trajectory = collector.get_trajectory("R", window=self._window)

        if not c_trajectory or not r_trajectory:
            return None

        mean_c = statistics.mean(c_trajectory)

        # Check R volatility for altered state detection
        if len(r_trajectory) >= 2:
            r_std = statistics.stdev(r_trajectory)
            if r_std > self._r_volatility_threshold:
                return ConsciousState.ALTERED

        # C-based detection
        if mean_c < self._c_sleep_threshold:
            return ConsciousState.SLEEP

        if mean_c > self._c_wake_threshold:
            return ConsciousState.WAKE

        # Ambiguous: C is between thresholds and R is stable
        return None
