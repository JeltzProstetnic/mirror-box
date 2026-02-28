"""System telemetry collector — the ISM's interoceptive input.

Collects signals analogous to biological interoception: inference latency,
self-prediction error, gate activations, frame rate, activation norms,
and timing information. These signals are the raw material from which the
implicit self-model (ISM) builds its encoding of the system's own state.

Implements the TelemetrySource ABC from aiware.interfaces.protocols.
"""

from __future__ import annotations

from aiware.interfaces.data_types import FrameMetrics
from aiware.interfaces.protocols import TelemetrySource

# Canonical signal names and their initial values.
# Sorted alphabetically for stable enumeration.
_DEFAULT_SIGNALS: dict[str, float] = {
    "esm_activation_norm": 0.0,
    "ewm_activation_norm": 0.0,
    "frame_rate_hz": 0.0,
    "frames_in_state": 0.0,
    "gate_activation_self": 0.0,
    "gate_activation_world": 0.0,
    "inference_latency_ms": 0.0,
    "self_knowledge_R": 0.0,
    "self_prediction_error": 0.0,
    "self_surprise": 0.0,
    "time_since_sleep_s": 0.0,
}


class MirrorBoxTelemetry(TelemetrySource):
    """System telemetry collector -- the ISM's input.

    Collects signals analogous to biological interoception:

    - inference_latency_ms: LLM forward pass time (neural processing speed)
    - self_prediction_error: R metric from last frame (error monitoring)
    - self_knowledge_R: R value from last frame
    - gate_activation_world: permeability gate state for world channel
    - gate_activation_self: permeability gate state for self channel
    - frame_rate_hz: actual achieved frame rate
    - esm_activation_norm: L2 norm of ESM state vector
    - ewm_activation_norm: L2 norm of EWM state vector
    - time_since_sleep_s: seconds since last sleep cycle
    - self_surprise: surprise metric from last frame
    - frames_in_state: how long in current state

    New signals can be added at runtime via update_signal() with a
    previously unseen name. This enables components to push custom
    telemetry without modifying this class.
    """

    def __init__(self) -> None:
        self._signals: dict[str, float] = dict(_DEFAULT_SIGNALS)

    # ------------------------------------------------------------------
    # TelemetrySource ABC implementation
    # ------------------------------------------------------------------

    def collect(self) -> dict[str, float]:
        """Return a snapshot of all current signal values.

        Returns a copy so callers cannot mutate internal state.
        """
        return dict(self._signals)

    def signal_names(self) -> list[str]:
        """Return sorted list of all signal names (including auto-registered)."""
        return sorted(self._signals.keys())

    def update_signal(self, name: str, value: float) -> None:
        """Update a signal value.

        If ``name`` is not a known signal, it is auto-registered with
        the given value. This allows other components to push custom
        telemetry signals at runtime.

        Args:
            name: Signal name.
            value: Signal value.
        """
        self._signals[name] = float(value)

    # ------------------------------------------------------------------
    # Convenience: bulk update from a FrameMetrics
    # ------------------------------------------------------------------

    def update_from_frame(
        self,
        frame_metrics: FrameMetrics,
        gate_world: float,
        gate_self: float,
    ) -> None:
        """Bulk-update telemetry from a completed frame's metrics.

        Sets self_knowledge_R, self_surprise, self_prediction_error,
        gate activations, and frames_in_state from the provided data.
        Signals not covered here (e.g., inference_latency_ms) are left
        at their previous values.

        Args:
            frame_metrics: Metrics from the most recent conscious frame.
            gate_world: World-channel gate activation from the permeability gate.
            gate_self: Self-channel gate activation from the permeability gate.
        """
        self._signals["self_knowledge_R"] = float(frame_metrics.R)
        self._signals["self_surprise"] = float(frame_metrics.self_surprise)
        self._signals["self_prediction_error"] = float(frame_metrics.self_surprise)
        self._signals["gate_activation_world"] = float(gate_world)
        self._signals["gate_activation_self"] = float(gate_self)
        self._signals["frames_in_state"] = float(frame_metrics.frame_number)
