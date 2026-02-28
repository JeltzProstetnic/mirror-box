"""In-memory ring buffer for frame-level metrics.

Stores up to ``config.metrics_buffer_size`` FrameMetrics in a deque.
Provides efficient time-series queries for dashboard, experiment
tracking, and automated state-transition detection.

Implements the MetricsCollector ABC from aiware.interfaces.protocols.
"""

from __future__ import annotations

import collections

from aiware.interfaces.data_types import FrameMetrics
from aiware.interfaces.protocols import MetricsCollector
from mirror_box.config import MirrorBoxConfig

# Fields of FrameMetrics that can be read by get_trajectory().
_NAMED_FIELDS = frozenset({"R", "self_surprise", "C", "complexity"})


class InMemoryMetricsCollector(MetricsCollector):
    """In-memory ring buffer for frame metrics.

    Stores up to ``config.metrics_buffer_size`` frames. Provides efficient
    time-series queries for dashboard and state detection.

    Args:
        config: MirrorBoxConfig (only ``metrics_buffer_size`` is used).
    """

    def __init__(self, config: MirrorBoxConfig) -> None:
        self._buffer: collections.deque[FrameMetrics] = collections.deque(
            maxlen=config.metrics_buffer_size,
        )

    # ------------------------------------------------------------------
    # MetricsCollector ABC implementation
    # ------------------------------------------------------------------

    def record_frame(self, metrics: FrameMetrics) -> None:
        """Append a frame's metrics to the ring buffer.

        If the buffer is full, the oldest frame is silently dropped.

        Args:
            metrics: Complete metrics for this frame.
        """
        self._buffer.append(metrics)

    def get_trajectory(
        self,
        metric_name: str,
        window: int | None = None,
    ) -> list[float]:
        """Retrieve a time series of a named metric.

        Resolution order:
        1. If ``metric_name`` is a known FrameMetrics field
           ("R", "self_surprise", "C", "complexity"), read the attribute.
        2. Otherwise, look in ``FrameMetrics.extras``. Frames that do
           not have the key in extras contribute 0.0.
        3. If ``metric_name`` is not a known field AND no frame in the
           buffer has it in extras, return an empty list.

        Args:
            metric_name: One of the known field names, or a key in extras.
            window: Number of most recent frames to include.
                ``None`` means all recorded frames.

        Returns:
            List of metric values, oldest first.
        """
        if not self._buffer:
            return []

        # Determine which frames to examine
        if window is not None:
            frames = list(self._buffer)[-window:]
        else:
            frames = list(self._buffer)

        if metric_name in _NAMED_FIELDS:
            return [getattr(f, metric_name) for f in frames]

        # Check if any frame has this key in extras
        has_key = any(metric_name in f.extras for f in frames)
        if not has_key:
            return []

        return [f.extras.get(metric_name, 0.0) for f in frames]

    def get_latest(self) -> FrameMetrics | None:
        """Get the most recently recorded FrameMetrics, or None if empty."""
        if self._buffer:
            return self._buffer[-1]
        return None

    def frame_count(self) -> int:
        """Total number of frames currently stored in the buffer.

        Note: this reflects the current buffer occupancy, not the total
        number of frames ever recorded (after overflow, it caps at
        ``metrics_buffer_size``).
        """
        return len(self._buffer)

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._buffer.clear()
