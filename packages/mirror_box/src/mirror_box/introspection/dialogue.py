"""Introspection dialogue -- Q&A about the system's internal state.

Composes prompts from user questions and the current state context,
then generates responses via the LLM. In mock mode (no tokenizer),
produces template-based answers grounded in real metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirror_box.models.llm_backend import LLMBackend

from mirror_box.introspection.reporter import StateReporter

logger = logging.getLogger(__name__)

# Mapping from common metric short names to their descriptions and
# telemetry/frame field names.
_METRIC_INFO: dict[str, tuple[str, str]] = {
    "R": (
        "self_knowledge_R",
        "Self-knowledge (R) measures how well the system predicts "
        "its own next ESM state. R=0 means no self-knowledge; R=1 "
        "means perfect self-prediction.",
    ),
    "self_knowledge_R": (
        "self_knowledge_R",
        "Self-knowledge (R) measures how well the system predicts "
        "its own next ESM state. R=0 means no self-knowledge; R=1 "
        "means perfect self-prediction.",
    ),
    "C": (
        "C",
        "Conscious content (C) is a proxy for total explicit-model "
        "activation. C near 0 indicates collapsed simulation (sleep); "
        "C > 0 indicates active conscious processing.",
    ),
    "self_surprise": (
        "self_surprise",
        "Self-surprise measures the magnitude of self-prediction error. "
        "High values mean the system did something it did not expect.",
    ),
    "complexity": (
        "complexity",
        "Complexity (Lempel-Ziv) measures the dynamical richness of "
        "the recent state trajectory. Higher values indicate more "
        "complex dynamics.",
    ),
    "gate_world": (
        "gate_activation_world",
        "World gate activation controls how much external sensory "
        "information flows into the conscious simulation.",
    ),
    "gate_activation_world": (
        "gate_activation_world",
        "World gate activation controls how much external sensory "
        "information flows into the conscious simulation.",
    ),
    "gate_self": (
        "gate_activation_self",
        "Self gate activation controls how much interoceptive "
        "information flows into the self-model.",
    ),
    "gate_activation_self": (
        "gate_activation_self",
        "Self gate activation controls how much interoceptive "
        "information flows into the self-model.",
    ),
}


class IntrospectionDialogue:
    """Allows asking questions about the system's internal state.

    Composes a prompt from the question + current state context,
    then generates a response via the LLM backend.

    In mock mode (no tokenizer): returns template-based answers
    grounded in the real state context data.

    Args:
        reporter: StateReporter that provides state context.
        backend: LLM backend for text generation.
    """

    def __init__(
        self,
        reporter: StateReporter,
        backend: LLMBackend,
        identity: str = "",
    ) -> None:
        self._reporter = reporter
        self._backend = backend
        self._identity = identity

    # ------------------------------------------------------------------
    # Has-tokenizer check
    # ------------------------------------------------------------------

    @property
    def _has_tokenizer(self) -> bool:
        """Whether the backend has a real tokenizer for text generation."""
        return self._backend.get_tokenizer() is not None

    # ------------------------------------------------------------------
    # General Q&A
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Ask the system a question about its own state.

        Uses persistent KV cache so the model remembers previous
        exchanges within the session (working memory). Each call
        appends to the ongoing stream rather than starting fresh.

        Args:
            question: A natural language question about internal state.

        Returns:
            A response string grounded in actual system metrics.
        """
        context = self._reporter.build_state_context()

        if self._has_tokenizer:
            raw = self._reporter.build_raw_context()
            if self._identity:
                fragment = (
                    f"\n{self._identity}\n"
                    f"Internal state: {context}\n"
                    f"Q: {question}\nA:"
                )
            else:
                fragment = (
                    f"\n{raw}\n"
                    f"Q: {question}\nA:"
                )
            return self._backend.generate_continuation(fragment)

        # Mock / template mode -- produce a grounded answer
        return self._mock_answer(question, context)

    # ------------------------------------------------------------------
    # Metric-specific Q&A
    # ------------------------------------------------------------------

    def ask_about_metric(self, metric_name: str) -> str:
        """Ask specifically about a metric value and its meaning.

        Looks up the metric in both frame-level data and telemetry,
        then produces a response that includes the current value and
        an explanation of what the metric represents.

        Args:
            metric_name: Short name of the metric (e.g., "R", "C",
                "self_surprise", "gate_world").

        Returns:
            A response describing the metric's current value and meaning.
        """
        context = self._reporter.build_state_context()

        if self._has_tokenizer:
            raw = self._reporter.build_raw_context()
            if self._identity:
                fragment = (
                    f"\n{self._identity}\n"
                    f"Internal state: {context}\n"
                    f"Q: What is {metric_name}?\nA:"
                )
            else:
                fragment = (
                    f"\n{raw}\n"
                    f"Q: What is {metric_name}?\nA:"
                )
            return self._backend.generate_continuation(fragment)

        # Mock / template mode
        return self._mock_metric_answer(metric_name)

    # ------------------------------------------------------------------
    # Template-based mock answers
    # ------------------------------------------------------------------

    def _mock_answer(self, question: str, context: str) -> str:
        """Generate a template-based answer in mock mode.

        Attempts keyword matching on the question to provide a
        relevant, context-grounded response.
        """
        q_lower = question.lower()
        lines: list[str] = []

        # Detect question topic and respond accordingly
        if any(word in q_lower for word in ["experiencing", "experience", "perceive", "feel"]):
            lines.append(
                "Based on my current internal state, "
                "here is what I am experiencing:"
            )
            lines.append(context)

        elif any(word in q_lower for word in ["know yourself", "self-knowledge", "self knowledge"]):
            signals = self._reporter._telemetry.collect()
            r_val = signals.get("self_knowledge_R", 0.0)
            latest = self._reporter._metrics.get_latest()
            if latest is not None:
                r_val = latest.R
            lines.append(
                f"My self-knowledge metric R is currently {r_val:.2f}. "
                + StateReporter._describe_R(r_val)
            )

        elif any(word in q_lower for word in ["state", "mode", "operating"]):
            state = self._reporter._state_ctrl.current_state()
            time_s = self._reporter._state_ctrl.time_in_state()
            lines.append(
                f"I am currently in {state.value.upper()} state, "
                f"and have been for {time_s:.1f} seconds."
            )

        elif any(word in q_lower for word in ["surprise", "surprised", "unexpected"]):
            signals = self._reporter._telemetry.collect()
            surprise = signals.get("self_surprise", 0.0)
            latest = self._reporter._metrics.get_latest()
            if latest is not None:
                surprise = latest.self_surprise
            lines.append(
                f"My current self-surprise level is {surprise:.2f}. "
                + StateReporter._describe_surprise(surprise)
            )

        else:
            # Generic fallback: return the full state context
            lines.append(
                "Here is my current internal state, which may help "
                "answer your question:"
            )
            lines.append(context)

        return "\n".join(lines)

    def _mock_metric_answer(self, metric_name: str) -> str:
        """Generate a template-based metric explanation in mock mode."""
        info = _METRIC_INFO.get(metric_name)

        # Look up the current value from frame data or telemetry
        value = self._resolve_metric_value(metric_name)

        if info is not None:
            _signal_name, description = info
            return (
                f"Metric '{metric_name}' current value: {value:.2f}. "
                f"{description}"
            )

        # Unknown metric -- still try to report a value
        return (
            f"Metric '{metric_name}' current value: {value:.2f}. "
            f"No detailed description available for this metric."
        )

    def _resolve_metric_value(self, metric_name: str) -> float:
        """Resolve a metric's current value from frame data or telemetry.

        Resolution order:
        1. If a known FrameMetrics field (R, C, self_surprise, complexity),
           read from the latest frame.
        2. Map common short names to telemetry signal names.
        3. Fall back to raw telemetry lookup.
        4. Default to 0.0 if nothing found.
        """
        latest = self._reporter._metrics.get_latest()

        # Direct FrameMetrics fields
        frame_fields = {"R", "C", "self_surprise", "complexity"}
        if metric_name in frame_fields and latest is not None:
            return float(getattr(latest, metric_name))

        # Map through _METRIC_INFO to get the telemetry signal name
        info = _METRIC_INFO.get(metric_name)
        if info is not None:
            signal_name = info[0]
            # Try frame data first for R/C fields
            if signal_name in frame_fields and latest is not None:
                return float(getattr(latest, signal_name, 0.0))
            # Then telemetry
            signals = self._reporter._telemetry.collect()
            return signals.get(signal_name, 0.0)

        # Raw telemetry lookup
        signals = self._reporter._telemetry.collect()
        return signals.get(metric_name, 0.0)
