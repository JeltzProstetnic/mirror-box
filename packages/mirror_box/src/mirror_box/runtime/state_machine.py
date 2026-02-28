"""State machine for conscious-state transitions (wake/sleep/dream/altered).

Implements the StateController ABC with timer-based automatic transitions
and manual override. The state machine tracks frames in the current state
and triggers transitions when configured duration thresholds are reached.

Transition rules (automatic):
    WAKE   -> SLEEP   : after continuous_wake_frames frames
    SLEEP  -> DREAM   : after sleep_duration_frames frames
    DREAM  -> WAKE    : after dream_duration_frames frames
    ALTERED -> WAKE   : after altered_duration_frames frames

Manual transitions via transition_to() always succeed and override automatic
timing. This allows user-triggered altered states and emergency resets.
"""

from __future__ import annotations

import logging
import time

from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from aiware.interfaces.protocols import StateController
from mirror_box.config import MirrorBoxConfig

logger = logging.getLogger(__name__)


class MirrorBoxStateController(StateController):
    """Timer-based state machine for wake/sleep/dream/altered transitions.

    Each tick() increments the frame counter for the current state and checks
    whether an automatic transition should fire. transition_to() provides
    manual override (always succeeds).

    Args:
        config: MirrorBoxConfig with timing thresholds and conscious_rate_hz.
    """

    def __init__(self, config: MirrorBoxConfig) -> None:
        self._config = config
        self._state = ConsciousState.WAKE
        self._frames_in_state: int = 0
        self._entry_time: float = time.monotonic()

        # History: list of (state, entry_timestamp) pairs, oldest first
        self._history: list[tuple[ConsciousState, float]] = [
            (ConsciousState.WAKE, self._entry_time),
        ]

        # Automatic transition thresholds: state -> (target, frame_threshold)
        self._auto_transitions: dict[ConsciousState, tuple[ConsciousState, int]] = {
            ConsciousState.WAKE: (
                ConsciousState.SLEEP,
                config.continuous_wake_frames,
            ),
            ConsciousState.SLEEP: (
                ConsciousState.DREAM,
                config.sleep_duration_frames,
            ),
            ConsciousState.DREAM: (
                ConsciousState.WAKE,
                config.dream_duration_frames,
            ),
            ConsciousState.ALTERED: (
                ConsciousState.WAKE,
                config.altered_duration_frames,
            ),
        }

        logger.info(
            "MirrorBoxStateController initialized in %s. "
            "Thresholds: wake=%d, sleep=%d, dream=%d, altered=%d",
            self._state.value,
            config.continuous_wake_frames,
            config.sleep_duration_frames,
            config.dream_duration_frames,
            config.altered_duration_frames,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_transition(self, target: ConsciousState) -> None:
        """Execute a state transition (shared by tick and transition_to)."""
        old = self._state
        self._state = target
        self._frames_in_state = 0
        self._entry_time = time.monotonic()
        self._history.append((target, self._entry_time))

        logger.info(
            "State transition: %s -> %s (after %.2fs in %s)",
            old.value,
            target.value,
            self.time_in_state(),
            old.value,
        )

    # ------------------------------------------------------------------
    # StateController ABC implementation
    # ------------------------------------------------------------------

    def current_state(self) -> ConsciousState:
        """Current operating state."""
        return self._state

    def transition_to(self, target: ConsciousState) -> bool:
        """Manual state transition (always succeeds).

        Resets the frame counter and records the transition in history.

        Args:
            target: Desired next state.

        Returns:
            Always True (manual override always accepted).
        """
        if target == self._state:
            logger.debug("transition_to(%s) is a no-op (already in state)", target.value)
            return True

        self._do_transition(target)
        return True

    def tick(self, frame_metrics: FrameMetrics) -> ConsciousState | None:
        """Advance by one frame, check automatic transition rules.

        Increments the frame counter, then checks if the current state's
        automatic transition threshold has been reached. If so, transitions
        and returns the new state. Otherwise returns None.

        Args:
            frame_metrics: Metrics from the current frame (available for
                future criticality-driven transitions, currently unused).

        Returns:
            New state if transition occurred, None otherwise.
        """
        self._frames_in_state += 1

        rule = self._auto_transitions.get(self._state)
        if rule is not None:
            target, threshold = rule
            if self._frames_in_state >= threshold:
                self._do_transition(target)
                return target

        return None

    def time_in_state(self) -> float:
        """Seconds elapsed since entering the current state.

        Computed from frames and conscious_rate_hz for frame-accurate timing.
        """
        return self._frames_in_state / self._config.conscious_rate_hz

    def transition_history(self) -> list[tuple[ConsciousState, float]]:
        """History of (state, entry_timestamp) pairs, oldest first."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def frames_in_state(self) -> int:
        """Number of frames spent in the current state."""
        return self._frames_in_state
