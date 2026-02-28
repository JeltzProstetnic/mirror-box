"""Factory function for assembling a complete Mirror Box system.

Given a MirrorBoxConfig, instantiates and wires all components in
dependency order and returns a MirrorBoxSystem dataclass holding
every component reference needed for operation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from mirror_box.config import MirrorBoxConfig
from mirror_box.introspection import IntrospectionDialogue, StateReporter
from mirror_box.metrics import InMemoryMetricsCollector, StateTransitionDetector
from mirror_box.models import (
    ExplicitModels,
    ImplicitModels,
    LLMBackend,
    MLPSelfPredictor,
    SigmoidPermeabilityGate,
)
from mirror_box.runtime import (
    MirrorBoxLoop,
    MirrorBoxStateController,
    MirrorBoxTelemetry,
    SleepMaintenance,
)


@dataclasses.dataclass(frozen=True)
class MirrorBoxSystem:
    """All components of an assembled Mirror Box, accessible by name."""

    config: MirrorBoxConfig
    backend: LLMBackend
    implicit: ImplicitModels
    explicit: ExplicitModels
    gate: SigmoidPermeabilityGate
    predictor: MLPSelfPredictor
    state_ctrl: MirrorBoxStateController
    telemetry: MirrorBoxTelemetry
    metrics_collector: InMemoryMetricsCollector
    loop: MirrorBoxLoop
    sleep_maintenance: SleepMaintenance
    detector: StateTransitionDetector
    reporter: StateReporter
    dialogue: IntrospectionDialogue


def create_mirror_box(
    config: MirrorBoxConfig | None = None,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> MirrorBoxSystem:
    """Assemble a complete Mirror Box from config.

    Args:
        config: System configuration. Uses defaults if None.
        model: Optional pre-loaded model (for testing with TinyMockLLM).
        tokenizer: Optional pre-loaded tokenizer (paired with custom model).

    Returns:
        MirrorBoxSystem with all components wired and ready.
    """
    if config is None:
        config = MirrorBoxConfig()

    # 1. LLM backend (foundation for everything)
    backend = LLMBackend(config, model=model, tokenizer=tokenizer)

    # 2. Implicit models (substrate: IWM + ISM)
    implicit = ImplicitModels(config, backend)

    # 3. Explicit models (simulation: EWM + ESM)
    explicit = ExplicitModels(config, backend)

    # 4. Permeability gate (implicit → explicit boundary)
    gate = SigmoidPermeabilityGate(config)

    # 5. Self-predictor (ESM prediction for R metric)
    predictor = MLPSelfPredictor(config)

    # 6. State machine (wake/sleep/dream/altered transitions)
    state_ctrl = MirrorBoxStateController(config)

    # 7. Telemetry (signal collection)
    telemetry = MirrorBoxTelemetry()

    # 8. Metrics collector (frame history ring buffer)
    metrics_collector = InMemoryMetricsCollector(config)

    # 9. Main conscious loop (orchestrates all per-frame processing)
    loop = MirrorBoxLoop(
        config=config,
        substrate=implicit,
        gate=gate,
        explicit=explicit,
        predictor=predictor,
        state_ctrl=state_ctrl,
        metrics_collector=metrics_collector,
        telemetry=telemetry,
    )

    # 10. Sleep maintenance (ISM LoRA replay during sleep)
    sleep_maintenance = SleepMaintenance(config, implicit)

    # 11. State transition detector (metrics-based detection)
    detector = StateTransitionDetector()

    # 12. Introspection: reporter + dialogue
    reporter = StateReporter(
        backend, metrics_collector, telemetry, state_ctrl,
        identity=config.system_identity,
    )
    dialogue = IntrospectionDialogue(reporter, backend, identity=config.system_identity)

    return MirrorBoxSystem(
        config=config,
        backend=backend,
        implicit=implicit,
        explicit=explicit,
        gate=gate,
        predictor=predictor,
        state_ctrl=state_ctrl,
        telemetry=telemetry,
        metrics_collector=metrics_collector,
        loop=loop,
        sleep_maintenance=sleep_maintenance,
        detector=detector,
        reporter=reporter,
        dialogue=dialogue,
    )


def gpu_config(
    state_dim: int = 2048,
    quantization: str = "none",
    **overrides: Any,
) -> MirrorBoxConfig:
    """Config preset for RTX 4090 (24GB VRAM).

    Mistral-7B FP16 (~14GB) + state_dim=2048 + larger predictor.
    Total VRAM ~16-18GB, leaving headroom for activations.
    """
    from aiware.config.base import DeviceConfig

    defaults = dict(
        hardware=DeviceConfig(device="auto", dtype="float16"),
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
        quantization=quantization,
        state_dim=state_dim,
        encoder_layer_index=8,
        lora_rank=16,
        lora_alpha=32,
        self_predictor_hidden_size=2048,
        self_predictor_lr=3e-3,
        conscious_rate_hz=20.0,
        continuous_wake_frames=6000,
        sleep_duration_frames=600,
        dream_duration_frames=300,
        altered_duration_frames=600,
        wandb_enabled=False,
    )
    defaults.update(overrides)
    return MirrorBoxConfig(**defaults)
