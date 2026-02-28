"""State reporter -- natural language descriptions of the system's internal state.

Uses the LLM backend to produce self-reports grounded in actual telemetry
and metrics. This is how the Mirror Box "speaks about itself."

In mock mode (no tokenizer on the LLM backend), falls back to template-based
reports built directly from the state context data. This allows the
introspection module to function in tests without a real language model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirror_box.models.llm_backend import LLMBackend

from aiware.interfaces.protocols import (
    MetricsCollector,
    StateController,
    TelemetrySource,
)

logger = logging.getLogger(__name__)


class StateReporter:
    """Generates natural language descriptions of the system's internal state.

    Uses the LLM to produce self-reports grounded in actual telemetry
    and metrics. This is how the system "speaks about itself."

    In mock mode (backend has no tokenizer), all generation methods
    return template-formatted strings using the raw state context data.

    Args:
        backend: LLM backend for text generation (or mock).
        metrics: Metrics collector with frame-level data.
        telemetry: Telemetry source with interoceptive signals.
        state_ctrl: State controller for current operating state.
    """

    def __init__(
        self,
        backend: LLMBackend,
        metrics: MetricsCollector,
        telemetry: TelemetrySource,
        state_ctrl: StateController,
        identity: str = "",
    ) -> None:
        self._backend = backend
        self._metrics = metrics
        self._telemetry = telemetry
        self._state_ctrl = state_ctrl
        self._identity = identity

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_state_context(self) -> str:
        """Build a text context string from current metrics and telemetry.

        Combines the current operating state, key metric values from the
        most recent frame, and telemetry signals into a single string
        suitable for inclusion in an LLM prompt.

        Returns:
            Multi-line text describing the system's current internal state.
            Always non-empty (at minimum contains the state name).
        """
        signals = self._telemetry.collect()
        latest = self._metrics.get_latest()
        state = self._state_ctrl.current_state()
        time_in_state = self._state_ctrl.time_in_state()

        parts: list[str] = []
        parts.append(f"Current state: {state.value.upper()}.")

        # Frame-level metrics (if any frames have been recorded)
        if latest is not None:
            parts.append(f"Self-knowledge R: {latest.R:.2f}.")
            parts.append(f"Self-surprise: {latest.self_surprise:.2f}.")
            parts.append(f"Conscious content C(t): {latest.C:.2f}.")
            parts.append(f"Complexity: {latest.complexity:.2f}.")
            parts.append(f"Frame number: {latest.frame_number}.")
        else:
            # Use telemetry signals as fallback
            parts.append(f"Self-knowledge R: {signals.get('self_knowledge_R', 0.0):.2f}.")
            parts.append(f"Self-surprise: {signals.get('self_surprise', 0.0):.2f}.")

        # Gate activations from telemetry
        parts.append(f"Gate world: {signals.get('gate_activation_world', 0.0):.2f}.")
        parts.append(f"Gate self: {signals.get('gate_activation_self', 0.0):.2f}.")

        # Timing
        parts.append(f"Time in state: {time_in_state:.1f}s.")
        parts.append(f"Frames processed: {self._metrics.frame_count()}.")

        return " ".join(parts)

    # State name → integer code (opaque to the LLM)
    _STATE_CODES: dict[str, int] = {
        "wake": 0, "sleep": 1, "dream": 2, "altered": 3,
    }

    def build_raw_context(self) -> str:
        """Build an opaque context string with pure numbers only.

        Used when no identity is set, so the LLM receives raw numbers
        without any semantic labels, variable names, or formatting that
        could trigger pretraining associations.
        """
        signals = self._telemetry.collect()
        latest = self._metrics.get_latest()
        state = self._state_ctrl.current_state()

        vals: list[str] = []
        vals.append(str(self._STATE_CODES.get(state.value, 0)))
        if latest is not None:
            vals.append(f"{latest.R:.3f}")
            vals.append(f"{latest.self_surprise:.3f}")
            vals.append(f"{latest.C:.3f}")
            vals.append(f"{latest.complexity:.3f}")
        vals.append(f"{signals.get('gate_activation_world', 0.0):.3f}")
        vals.append(f"{signals.get('gate_activation_self', 0.0):.3f}")
        return " ".join(vals)

    # ------------------------------------------------------------------
    # Has-tokenizer check (determines LLM vs template path)
    # ------------------------------------------------------------------

    @property
    def _has_tokenizer(self) -> bool:
        """Whether the backend has a real tokenizer for text generation."""
        return self._backend.get_tokenizer() is not None

    # ------------------------------------------------------------------
    # State report generation
    # ------------------------------------------------------------------

    def generate_state_report(self) -> str:
        """Generate a natural language self-report.

        If the LLM backend has a tokenizer (real model): uses generate_text
        with a prompt that includes the state context.

        If no tokenizer (mock mode): returns a formatted template string
        using the state context data directly.

        Returns:
            A human-readable self-report string.
        """
        context = self.build_state_context()

        if self._has_tokenizer:
            if self._identity:
                fragment = (
                    f"\n{self._identity}\n"
                    f"Internal state: {context}\n"
                    "Self-report:"
                )
            else:
                raw = self.build_raw_context()
                fragment = f"\n{raw}\nReport:"
            return self._backend.generate_continuation(fragment)

        # Mock / template mode
        state = self._state_ctrl.current_state()
        latest = self._metrics.get_latest()

        lines: list[str] = []
        lines.append(f"[State Report] Operating in {state.value.upper()} state.")

        if latest is not None:
            lines.append(
                f"Self-knowledge (R={latest.R:.2f}): "
                + self._describe_R(latest.R)
            )
            lines.append(
                f"Self-surprise level: {latest.self_surprise:.2f} "
                + self._describe_surprise(latest.self_surprise)
            )
            lines.append(
                f"Conscious content (C={latest.C:.2f}): "
                + self._describe_C(latest.C)
            )
        else:
            lines.append("No frame data yet. System is initializing.")

        lines.append(f"Time in current state: {self._state_ctrl.time_in_state():.1f}s.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # EWM narrative (what the system perceives)
    # ------------------------------------------------------------------

    def generate_ewm_narrative(self) -> str:
        """Generate a description of what the system currently perceives (EWM).

        The EWM (Explicit World Model) represents the system's conscious
        perception of the external world.

        Returns:
            A description of the system's current world-model contents.
        """
        context = self.build_state_context()

        if self._has_tokenizer:
            if self._identity:
                fragment = (
                    f"\n{self._identity}\n"
                    f"Internal state: {context}\n"
                    "World perception:"
                )
            else:
                raw = self.build_raw_context()
                fragment = f"\n{raw}\nPerception:"
            return self._backend.generate_continuation(fragment)

        # Template mode
        state = self._state_ctrl.current_state()
        signals = self._telemetry.collect()
        gate_world = signals.get("gate_activation_world", 0.0)

        lines: list[str] = []
        lines.append(f"[EWM Narrative] World-model in {state.value.upper()} state.")

        if state.value == "wake":
            lines.append(
                f"World gate activation: {gate_world:.2f}. "
                "Sensory input is being processed normally."
            )
        elif state.value == "sleep":
            lines.append(
                "World gate nearly closed. "
                "External perception is suspended."
            )
        elif state.value == "dream":
            lines.append(
                "World gate partially open with internal noise. "
                "Perception is internally generated."
            )
        elif state.value == "altered":
            lines.append(
                f"World gate activation: {gate_world:.2f}. "
                "Increased permeability -- content flooding from implicit models."
            )

        latest = self._metrics.get_latest()
        if latest is not None:
            lines.append(f"Conscious content level: {latest.C:.2f}.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ESM narrative (the system's self-model)
    # ------------------------------------------------------------------

    def generate_esm_narrative(self) -> str:
        """Generate a description of the system's self-model (ESM).

        The ESM (Explicit Self Model) represents the system's conscious
        model of itself -- its self-awareness.

        Returns:
            A description of the system's current self-model contents.
        """
        context = self.build_state_context()

        if self._has_tokenizer:
            if self._identity:
                fragment = (
                    f"\n{self._identity}\n"
                    f"Internal state: {context}\n"
                    "Self-model description:"
                )
            else:
                raw = self.build_raw_context()
                fragment = f"\n{raw}\nSelf-description:"
            return self._backend.generate_continuation(fragment)

        # Template mode
        state = self._state_ctrl.current_state()
        signals = self._telemetry.collect()
        latest = self._metrics.get_latest()
        gate_self = signals.get("gate_activation_self", 0.0)

        lines: list[str] = []
        lines.append(f"[ESM Narrative] Self-model in {state.value.upper()} state.")
        lines.append(f"Self gate activation: {gate_self:.2f}.")

        if latest is not None:
            r_val = latest.R
            lines.append(
                f"Self-knowledge R={r_val:.2f}: {self._describe_R(r_val)}"
            )
            lines.append(
                f"Self-surprise: {latest.self_surprise:.2f}. "
                + self._describe_surprise(latest.self_surprise)
            )
        else:
            lines.append("No self-prediction data yet. Self-model is bootstrapping.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Qualitative descriptors for template mode
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_R(r: float) -> str:
        """Qualitative description of self-knowledge level."""
        if r >= 0.8:
            return "High self-knowledge -- the system predicts its own state well."
        if r >= 0.5:
            return "Moderate self-knowledge -- partial self-prediction ability."
        if r >= 0.2:
            return "Low self-knowledge -- the system is still learning about itself."
        return "Minimal self-knowledge -- self-prediction is essentially random."

    @staticmethod
    def _describe_surprise(surprise: float) -> str:
        """Qualitative description of self-surprise level."""
        if surprise < 0.5:
            return "(low -- operating predictably)"
        if surprise < 1.5:
            return "(moderate -- some unexpected internal dynamics)"
        return "(high -- significant deviation from self-expectations)"

    @staticmethod
    def _describe_C(c: float) -> str:
        """Qualitative description of conscious content level."""
        if c >= 0.7:
            return "Rich conscious content -- simulation is highly active."
        if c >= 0.3:
            return "Moderate conscious content -- simulation is partially active."
        if c > 0.05:
            return "Low conscious content -- simulation is fading."
        return "Near-zero conscious content -- simulation is collapsed."
