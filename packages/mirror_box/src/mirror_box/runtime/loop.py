"""Main conscious processing loop for the Mirror Box (Design 16).

Orchestrates one conscious frame at ~20 Hz:

1. Collect telemetry from TelemetrySource
2. Substrate step: ImplicitModels.step(sensory_input, telemetry)
3. Permeability gating: SigmoidPermeabilityGate.gate(substrate_state, state)
4. Self-prediction: MLPSelfPredictor.predict(prev_esm, gated_self) BEFORE generation
5. Explicit generation: ExplicitModels.step(gate_output, prev_esm)
6. Compute metrics: R, surprise, C(t), complexity
7. Update self-predictor: MLPSelfPredictor.update(predicted, actual_esm)
8. Record metrics via MetricsCollector
9. State machine tick
10. Update telemetry with prediction error, R, etc.
"""

from __future__ import annotations

import asyncio
import logging
import time

import torch

from aiware.interfaces.data_types import ConsciousState, FrameMetrics
from aiware.interfaces.protocols import (
    ConsciousLoop,
    ExplicitGenerator,
    MetricsCollector,
    PermeabilityGate,
    SelfPredictor,
    StateController,
    SubstrateEncoder,
    TelemetrySource,
)
from aiware.metrics.complexity import trajectory_complexity
from aiware.metrics.conscious_content import compute_C
from mirror_box.config import MirrorBoxConfig

logger = logging.getLogger(__name__)


class MirrorBoxLoop(ConsciousLoop):
    """Main conscious processing loop for the Mirror Box.

    Wires together all components (substrate, gate, explicit, predictor,
    state machine, metrics, telemetry) and runs the 10-step frame sequence
    at the configured conscious_rate_hz.

    Args:
        config: MirrorBoxConfig with timing and architecture settings.
        substrate: Implicit model layer (IWM + ISM).
        gate: Permeability gate (substrate -> simulation boundary).
        explicit: Explicit model layer (EWM + ESM).
        predictor: Self-prediction module.
        state_ctrl: State machine for wake/sleep/dream/altered.
        metrics_collector: Frame-level metrics recorder.
        telemetry: System telemetry source.
    """

    def __init__(
        self,
        config: MirrorBoxConfig,
        substrate: SubstrateEncoder,
        gate: PermeabilityGate,
        explicit: ExplicitGenerator,
        predictor: SelfPredictor,
        state_ctrl: StateController,
        metrics_collector: MetricsCollector,
        telemetry: TelemetrySource,
    ) -> None:
        self._config = config
        self._substrate = substrate
        self._gate = gate
        self._explicit = explicit
        self._predictor = predictor
        self._state_ctrl = state_ctrl
        self._metrics = metrics_collector
        self._telemetry = telemetry

        # State
        self._device = config.hardware.resolve_device()
        self._dtype = config.hardware.resolve_dtype()
        self._prev_esm = torch.zeros(config.state_dim, device=self._device, dtype=self._dtype)
        self._frame_counter: int = 0
        self._running: bool = False

        # State history buffer for complexity computation (last N ESM states)
        self._complexity_window: int = 100
        self._state_history: list[torch.Tensor] = []

        # Last substrate state (for sleep experience recording)
        self._last_substrate_self_state: torch.Tensor | None = None

        # Sensory input: default dummy tokens, overridable via set_sensory_input
        self._sensory_input = torch.tensor(
            [1, 42, 100, 7], dtype=torch.long, device=self._device
        )

        logger.info(
            "MirrorBoxLoop initialized: state_dim=%d, rate=%.1f Hz, device=%s",
            config.state_dim,
            config.conscious_rate_hz,
            self._device,
        )

    # ------------------------------------------------------------------
    # Sensory input control
    # ------------------------------------------------------------------

    def set_sensory_input(self, token_ids: torch.Tensor) -> None:
        """Set the sensory input for the next frame.

        Args:
            token_ids: Token IDs for world encoding. Shape: (seq_len,).
        """
        self._sensory_input = token_ids.to(self._device)

    # ------------------------------------------------------------------
    # ConsciousLoop ABC implementation
    # ------------------------------------------------------------------

    async def run_frame(self) -> FrameMetrics:
        """Execute one conscious frame (the 10-step sequence).

        Returns:
            FrameMetrics for this frame.
        """
        frame_start = time.monotonic()
        current_state = self._state_ctrl.current_state()

        # Step 1: Collect telemetry
        telemetry_data = self._telemetry.collect()

        # Step 2: Substrate step (IWM + ISM encoding)
        substrate_state = self._substrate.step(self._sensory_input, telemetry_data)

        # Expose ISM self-state for sleep experience recording
        self._last_substrate_self_state = substrate_state.self_state.detach()

        # Step 3: Permeability gating
        gate_output = self._gate.gate(substrate_state, current_state)

        # Step 4: Self-prediction BEFORE generation
        predicted_esm = self._predictor.predict(self._prev_esm, gate_output.gated_self)

        # Step 5: Explicit generation (EWM + ESM)
        explicit_state = self._explicit.step(gate_output, self._prev_esm)

        # Step 6: Compute metrics
        actual_esm = explicit_state.esm_state.detach()

        R = self._predictor.compute_R(predicted_esm, actual_esm)
        surprise = self._predictor.compute_surprise(predicted_esm, actual_esm)
        C = compute_C(
            explicit_state.ewm_state.detach(),
            actual_esm,
            threshold=self._config.C_threshold,
        )

        # Trajectory complexity (over sliding window)
        self._state_history.append(actual_esm.clone())
        if len(self._state_history) > self._complexity_window:
            self._state_history = self._state_history[-self._complexity_window :]
        complexity = trajectory_complexity(self._state_history)

        # Step 7: Update self-predictor
        prediction_loss = self._predictor.update(predicted_esm, actual_esm)

        # Build FrameMetrics
        frame_time = time.monotonic()
        frame_metrics = FrameMetrics(
            R=R,
            self_surprise=surprise,
            C=C,
            complexity=complexity,
            state=current_state,
            frame_number=self._frame_counter,
            timestamp=frame_time,
            extras={
                "prediction_loss": prediction_loss,
                "gate_world_activation": gate_output.gate_world_activation,
                "gate_self_activation": gate_output.gate_self_activation,
                "frame_duration_ms": (frame_time - frame_start) * 1000.0,
            },
        )

        # Step 8: Record metrics
        self._metrics.record_frame(frame_metrics)

        # Step 9: State machine tick
        new_state = self._state_ctrl.tick(frame_metrics)
        if new_state is not None:
            logger.info("State transitioned to %s at frame %d", new_state.value, self._frame_counter)

        # Step 10: Update telemetry with feedback signals
        self._telemetry.update_signal("R", R)
        self._telemetry.update_signal("self_surprise", surprise)
        self._telemetry.update_signal("prediction_loss", prediction_loss)
        self._telemetry.update_signal("C", C)

        # Update self-referential state for next frame
        self._prev_esm = actual_esm
        self._frame_counter += 1

        return frame_metrics

    async def run(self, max_frames: int | None = None) -> None:
        """Run the conscious loop continuously at ~conscious_rate_hz.

        Args:
            max_frames: Stop after this many frames. None = run until stop().
        """
        self._running = True
        target_period = 1.0 / self._config.conscious_rate_hz
        frames_run = 0

        logger.info(
            "Conscious loop starting (max_frames=%s, rate=%.1f Hz)",
            max_frames,
            self._config.conscious_rate_hz,
        )

        while self._running:
            frame_start = time.monotonic()

            await self.run_frame()
            frames_run += 1

            if max_frames is not None and frames_run >= max_frames:
                break

            # Sleep to maintain target rate
            elapsed = time.monotonic() - frame_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._running = False
        logger.info("Conscious loop stopped after %d frames", frames_run)

    async def stop(self) -> None:
        """Signal the loop to stop after the current frame completes."""
        self._running = False

    def is_running(self) -> bool:
        """Whether the loop is currently executing frames."""
        return self._running

    def frame_count(self) -> int:
        """Total frames executed since initialization."""
        return self._frame_counter

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def prev_esm(self) -> torch.Tensor:
        """Current ESM state (self-referential feedback for next frame)."""
        return self._prev_esm

    @property
    def last_substrate_self_state(self) -> torch.Tensor | None:
        """Last ISM self-state s(t) from the most recent frame."""
        return self._last_substrate_self_state
