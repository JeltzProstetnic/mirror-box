"""Tests for the introspection module -- StateReporter and IntrospectionDialogue.

All tests run with TinyMockLLM which has no tokenizer. This exercises the
template/context-building path (mock mode). Tests verify that state context
is correctly assembled from telemetry and metrics, and that generated reports
and dialogue responses are grounded in actual system data.
"""

from __future__ import annotations

import time

import pytest

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from conftest import TinyMockLLM
from mirror_box.config import MirrorBoxConfig
from mirror_box.introspection.dialogue import IntrospectionDialogue
from mirror_box.introspection.reporter import StateReporter
from mirror_box.metrics.collector import InMemoryMetricsCollector
from mirror_box.models.llm_backend import LLMBackend
from mirror_box.runtime.state_machine import MirrorBoxStateController
from mirror_box.runtime.telemetry import MirrorBoxTelemetry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config() -> MirrorBoxConfig:
    """MirrorBoxConfig for testing: CPU, no quantization, small state_dim."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=64,
        model_name_or_path="test-model",
        quantization="none",
        encoder_layer_index=1,
        lora_rank=8,
        lora_alpha=16,
        max_context_length=128,
        generation_temperature=0.7,
        generation_max_tokens=16,
        wandb_enabled=False,
        conscious_rate_hz=10.0,
    )


@pytest.fixture
def backend(tiny_config: MirrorBoxConfig) -> LLMBackend:
    """LLMBackend wrapping TinyMockLLM (no tokenizer)."""
    model = TinyMockLLM(hidden_size=64, num_layers=2)
    return LLMBackend(config=tiny_config, model=model)


@pytest.fixture
def telemetry() -> MirrorBoxTelemetry:
    """Fresh MirrorBoxTelemetry with some signal values set."""
    t = MirrorBoxTelemetry()
    t.update_signal("self_knowledge_R", 0.45)
    t.update_signal("self_surprise", 1.23)
    t.update_signal("gate_activation_world", 0.85)
    t.update_signal("gate_activation_self", 0.72)
    t.update_signal("frame_rate_hz", 20.0)
    return t


@pytest.fixture
def metrics(tiny_config: MirrorBoxConfig) -> InMemoryMetricsCollector:
    """InMemoryMetricsCollector with a few recorded frames."""
    collector = InMemoryMetricsCollector(config=tiny_config)
    # Record several frames so we have data
    for i in range(5):
        frame = FrameMetrics(
            R=0.40 + i * 0.02,  # 0.40, 0.42, 0.44, 0.46, 0.48
            self_surprise=1.20 + i * 0.01,
            C=0.75 + i * 0.01,
            complexity=0.50 + i * 0.01,
            state=ConsciousState.WAKE,
            frame_number=i + 1,
            timestamp=time.monotonic(),
        )
        collector.record_frame(frame)
    return collector


@pytest.fixture
def state_ctrl(tiny_config: MirrorBoxConfig) -> MirrorBoxStateController:
    """MirrorBoxStateController starting in WAKE state."""
    return MirrorBoxStateController(config=tiny_config)


@pytest.fixture
def reporter(
    backend: LLMBackend,
    metrics: InMemoryMetricsCollector,
    telemetry: MirrorBoxTelemetry,
    state_ctrl: MirrorBoxStateController,
) -> StateReporter:
    """StateReporter wired up with mock backend and real metrics/telemetry."""
    return StateReporter(
        backend=backend,
        metrics=metrics,
        telemetry=telemetry,
        state_ctrl=state_ctrl,
    )


@pytest.fixture
def dialogue(reporter: StateReporter, backend: LLMBackend) -> IntrospectionDialogue:
    """IntrospectionDialogue wired up with the reporter and mock backend."""
    return IntrospectionDialogue(reporter=reporter, backend=backend)


# ===========================================================================
# StateReporter -- build_state_context()
# ===========================================================================


class TestBuildStateContext:
    """Tests for the state context builder."""

    def test_returns_non_empty_string(self, reporter: StateReporter) -> None:
        """build_state_context() returns a non-empty string."""
        ctx = reporter.build_state_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_includes_state_name(self, reporter: StateReporter) -> None:
        """Context includes the current state name."""
        ctx = reporter.build_state_context()
        assert "WAKE" in ctx

    def test_includes_R_value(self, reporter: StateReporter) -> None:
        """Context includes the self-knowledge R metric."""
        ctx = reporter.build_state_context()
        # The latest frame has R=0.48
        assert "R:" in ctx or "R =" in ctx
        assert "0.48" in ctx

    def test_includes_C_value(self, reporter: StateReporter) -> None:
        """Context includes the conscious content C metric."""
        ctx = reporter.build_state_context()
        # The latest frame has C=0.79
        assert "C" in ctx
        assert "0.79" in ctx

    def test_includes_gate_world(self, reporter: StateReporter) -> None:
        """Context includes the world gate activation from telemetry."""
        ctx = reporter.build_state_context()
        assert "Gate world:" in ctx
        assert "0.85" in ctx

    def test_includes_gate_self(self, reporter: StateReporter) -> None:
        """Context includes the self gate activation from telemetry."""
        ctx = reporter.build_state_context()
        assert "Gate self:" in ctx
        assert "0.72" in ctx

    def test_includes_frame_count(self, reporter: StateReporter) -> None:
        """Context includes the number of frames processed."""
        ctx = reporter.build_state_context()
        assert "Frames processed: 5" in ctx

    def test_includes_time_in_state(self, reporter: StateReporter) -> None:
        """Context includes time in state."""
        ctx = reporter.build_state_context()
        assert "Time in state:" in ctx

    def test_no_frames_fallback(
        self,
        backend: LLMBackend,
        tiny_config: MirrorBoxConfig,
        telemetry: MirrorBoxTelemetry,
        state_ctrl: MirrorBoxStateController,
    ) -> None:
        """Context still works when no frames have been recorded."""
        empty_metrics = InMemoryMetricsCollector(config=tiny_config)
        reporter = StateReporter(
            backend=backend,
            metrics=empty_metrics,
            telemetry=telemetry,
            state_ctrl=state_ctrl,
        )
        ctx = reporter.build_state_context()
        assert "WAKE" in ctx
        assert len(ctx) > 0
        # Should use telemetry fallback for R
        assert "0.45" in ctx


# ===========================================================================
# StateReporter -- generate_state_report()
# ===========================================================================


class TestGenerateStateReport:
    """Tests for mock-mode state report generation."""

    def test_returns_non_empty_string(self, reporter: StateReporter) -> None:
        """generate_state_report() returns a non-empty string in mock mode."""
        report = reporter.generate_state_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_includes_state(self, reporter: StateReporter) -> None:
        """Report mentions the current state."""
        report = reporter.generate_state_report()
        assert "WAKE" in report

    def test_includes_R_description(self, reporter: StateReporter) -> None:
        """Report includes R value and qualitative description."""
        report = reporter.generate_state_report()
        assert "R=" in report or "R:" in report
        assert "0.48" in report

    def test_includes_surprise(self, reporter: StateReporter) -> None:
        """Report includes self-surprise information."""
        report = reporter.generate_state_report()
        assert "surprise" in report.lower()

    def test_includes_conscious_content(self, reporter: StateReporter) -> None:
        """Report includes conscious content C."""
        report = reporter.generate_state_report()
        assert "C=" in report or "C:" in report


# ===========================================================================
# StateReporter -- generate_ewm_narrative()
# ===========================================================================


class TestGenerateEwmNarrative:
    """Tests for EWM (world-model) narrative generation."""

    def test_returns_non_empty_string(self, reporter: StateReporter) -> None:
        """generate_ewm_narrative() returns a non-empty string."""
        narrative = reporter.generate_ewm_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_mentions_world_model(self, reporter: StateReporter) -> None:
        """EWM narrative references world perception."""
        narrative = reporter.generate_ewm_narrative()
        assert "EWM" in narrative or "World" in narrative or "world" in narrative.lower()

    def test_mentions_state(self, reporter: StateReporter) -> None:
        """EWM narrative mentions the current state."""
        narrative = reporter.generate_ewm_narrative()
        assert "WAKE" in narrative

    def test_sleep_state_narrative(
        self,
        backend: LLMBackend,
        metrics: InMemoryMetricsCollector,
        telemetry: MirrorBoxTelemetry,
        tiny_config: MirrorBoxConfig,
    ) -> None:
        """EWM narrative changes content in SLEEP state."""
        state_ctrl = MirrorBoxStateController(config=tiny_config)
        state_ctrl.transition_to(ConsciousState.SLEEP)
        reporter = StateReporter(
            backend=backend,
            metrics=metrics,
            telemetry=telemetry,
            state_ctrl=state_ctrl,
        )
        narrative = reporter.generate_ewm_narrative()
        assert "SLEEP" in narrative
        assert "suspend" in narrative.lower() or "closed" in narrative.lower()


# ===========================================================================
# StateReporter -- generate_esm_narrative()
# ===========================================================================


class TestGenerateEsmNarrative:
    """Tests for ESM (self-model) narrative generation."""

    def test_returns_non_empty_string(self, reporter: StateReporter) -> None:
        """generate_esm_narrative() returns a non-empty string."""
        narrative = reporter.generate_esm_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_mentions_self_model(self, reporter: StateReporter) -> None:
        """ESM narrative references self-model."""
        narrative = reporter.generate_esm_narrative()
        assert "ESM" in narrative or "Self" in narrative or "self" in narrative.lower()

    def test_includes_R_value(self, reporter: StateReporter) -> None:
        """ESM narrative includes self-knowledge R value."""
        narrative = reporter.generate_esm_narrative()
        assert "R=" in narrative or "R:" in narrative
        assert "0.48" in narrative

    def test_includes_surprise(self, reporter: StateReporter) -> None:
        """ESM narrative includes self-surprise information."""
        narrative = reporter.generate_esm_narrative()
        assert "surprise" in narrative.lower()


# ===========================================================================
# IntrospectionDialogue -- ask()
# ===========================================================================


class TestDialogueAsk:
    """Tests for the general Q&A interface."""

    def test_returns_non_empty_string(self, dialogue: IntrospectionDialogue) -> None:
        """ask() returns a non-empty string."""
        response = dialogue.ask("What state are you in?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_state_question_references_state(
        self, dialogue: IntrospectionDialogue
    ) -> None:
        """Asking about state references the actual state name."""
        response = dialogue.ask("What state are you in?")
        assert "WAKE" in response

    def test_experience_question(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about experience returns context-grounded response."""
        response = dialogue.ask("What are you currently experiencing?")
        assert len(response) > 0
        # Should contain some of the state context
        assert "WAKE" in response or "state" in response.lower()

    def test_self_knowledge_question(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about self-knowledge returns R-grounded response."""
        response = dialogue.ask("How well do you know yourself?")
        assert "R" in response or "self-knowledge" in response.lower()

    def test_surprise_question(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about surprise returns surprise-grounded response."""
        response = dialogue.ask("Are you surprised right now?")
        assert "surprise" in response.lower()

    def test_unknown_question_returns_context(
        self, dialogue: IntrospectionDialogue
    ) -> None:
        """Unknown questions fall back to returning the full state context."""
        response = dialogue.ask("What is the meaning of life?")
        assert len(response) > 0
        # Should contain some state information as fallback
        assert "WAKE" in response or "state" in response.lower()


# ===========================================================================
# IntrospectionDialogue -- ask_about_metric()
# ===========================================================================


class TestDialogueAskAboutMetric:
    """Tests for metric-specific Q&A."""

    def test_R_metric_returns_value(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about R returns a string mentioning R and its value."""
        response = dialogue.ask_about_metric("R")
        assert "R" in response
        # Latest frame R is 0.48
        assert "0.48" in response

    def test_R_metric_includes_description(
        self, dialogue: IntrospectionDialogue
    ) -> None:
        """Asking about R includes a description of what R means."""
        response = dialogue.ask_about_metric("R")
        assert "self-knowledge" in response.lower() or "predict" in response.lower()

    def test_C_metric(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about C returns value and description."""
        response = dialogue.ask_about_metric("C")
        assert "C" in response
        # Latest frame C is 0.79
        assert "0.79" in response

    def test_self_surprise_metric(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about self_surprise returns value and description."""
        response = dialogue.ask_about_metric("self_surprise")
        assert "surprise" in response.lower()

    def test_gate_world_metric(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about gate_world returns value from telemetry."""
        response = dialogue.ask_about_metric("gate_world")
        assert "0.85" in response

    def test_unknown_metric(self, dialogue: IntrospectionDialogue) -> None:
        """Asking about an unknown metric still returns a response."""
        response = dialogue.ask_about_metric("nonexistent_metric")
        assert isinstance(response, str)
        assert len(response) > 0
        assert "nonexistent_metric" in response


# ===========================================================================
# Module-level imports
# ===========================================================================


class TestModuleExports:
    """Verify that __init__.py exports work correctly."""

    def test_import_state_reporter(self) -> None:
        """StateReporter is importable from the introspection package."""
        from mirror_box.introspection import StateReporter as SR

        assert SR is StateReporter

    def test_import_introspection_dialogue(self) -> None:
        """IntrospectionDialogue is importable from the introspection package."""
        from mirror_box.introspection import IntrospectionDialogue as ID

        assert ID is IntrospectionDialogue
