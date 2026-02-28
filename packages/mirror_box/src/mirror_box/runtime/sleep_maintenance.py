"""Sleep cycle maintenance: experience replay and ISM LoRA update.

During wake, the system buffers (telemetry, self_state) pairs. When the
state machine transitions to SLEEP, the SleepMaintenance module replays
these experiences through the ISM encoder to update its LoRA adapter weights.

This is the Mirror Box equivalent of sleep's restorative function in FMT:
consolidating self-model accuracy through offline replay.

Functions:
    record_experience: Buffer telemetry+self_state pairs during wake.
    run_maintenance: Replay experiences and update ISM (called in sleep).
    clear_buffers: Reset after maintenance completes.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from mirror_box.config import MirrorBoxConfig
from mirror_box.models.implicit import ImplicitModels

logger = logging.getLogger(__name__)


class SleepMaintenance:
    """Experience replay and ISM LoRA update during sleep cycles.

    During wake, records (telemetry, self_state) pairs into a circular
    buffer. When run_maintenance() is called (during sleep), it replays
    the buffered experiences through the ISM encoder, computes reconstruction
    loss, and updates the LoRA adapter weights via gradient descent.

    Args:
        config: MirrorBoxConfig with learning rate and buffer settings.
        implicit: ImplicitModels instance whose ISM LoRA will be updated.
    """

    def __init__(self, config: MirrorBoxConfig, implicit: ImplicitModels) -> None:
        self._config = config
        self._implicit = implicit

        # Experience buffer: list of (telemetry_dict, self_state_tensor)
        self._experience_buffer: list[tuple[dict[str, Any], torch.Tensor]] = []
        self._max_buffer_size: int = 1000

        # Loss function for ISM reconstruction
        self._loss_fn = nn.MSELoss()

        # Optimizer targeting only LoRA parameters
        self._optimizer = self._setup_optimizer()

        # Cumulative stats across all sleep cycles
        self._total_cycles: int = 0

        logger.info(
            "SleepMaintenance initialized: max_buffer_size=%d, lr=%g, epochs=%d",
            self._max_buffer_size,
            config.sleep_consolidation_lr,
            config.sleep_consolidation_epochs,
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer | None:
        """Create Adam optimizer for LoRA parameters only.

        Returns None if no LoRA adapter is attached (degraded mode).

        Note: LoRA adapters start disabled (requires_grad=False), so we
        identify LoRA params by name pattern (lora_A/lora_B) and force
        requires_grad=True for optimizer tracking. The enable/disable
        cycle during maintenance handles gradient flow correctly.
        """
        lora_adapter = self._implicit._lora_adapter
        if lora_adapter is None:
            logger.warning("No LoRA adapter — sleep consolidation will be no-op.")
            return None

        # Identify LoRA params by name (they start disabled after setup)
        lora_params = []
        for name, p in lora_adapter.named_parameters():
            if "lora_" in name and ("lora_A" in name or "lora_B" in name):
                lora_params.append(p)

        if not lora_params:
            logger.warning("No LoRA A/B parameters found.")
            return None

        logger.info(
            "Sleep optimizer: %d LoRA params (%d elements), lr=%g",
            len(lora_params),
            sum(p.numel() for p in lora_params),
            self._config.sleep_consolidation_lr,
        )
        return torch.optim.Adam(
            lora_params,
            lr=self._config.sleep_consolidation_lr,
        )

    def record_experience(self, telemetry: dict[str, Any], self_state: torch.Tensor) -> None:
        """Buffer a (telemetry, self_state) pair observed during wake.

        If the buffer is full, the oldest experience is dropped (FIFO).

        Args:
            telemetry: Telemetry dict from that frame.
            self_state: ISM self-state vector s(t) from that frame.
        """
        self._experience_buffer.append((telemetry, self_state.detach().clone()))

        if len(self._experience_buffer) > self._max_buffer_size:
            self._experience_buffer = self._experience_buffer[-self._max_buffer_size :]

    async def run_maintenance(
        self,
        progress_callback: Callable[[int, int, float], Any] | None = None,
    ) -> dict[str, float]:
        """Replay buffered experiences through ISM and update LoRA weights.

        Called when the system enters SLEEP state. For each epoch, shuffles
        the experience buffer and replays each sample: re-encodes telemetry
        through ISM (LoRA active), computes MSE loss against the original
        self_state target, backprops, and steps the optimizer.

        Args:
            progress_callback: Optional async callable(epoch, total_epochs, avg_loss)
                called after each epoch completes, for UI progress reporting.

        Returns:
            Maintenance statistics:
                num_samples: Number of unique samples in buffer.
                epochs: Number of passes completed.
                total_steps: Total optimizer steps (samples * epochs).
                initial_loss: Average loss on first epoch.
                final_loss: Average loss on last epoch.
        """
        if not self._experience_buffer:
            logger.info("No experiences to replay during sleep maintenance.")
            return {
                "num_samples": 0,
                "epochs": 0,
                "total_steps": 0,
                "initial_loss": 0.0,
                "final_loss": 0.0,
            }

        if self._optimizer is None:
            # No LoRA — just measure loss without training
            return await self._run_measurement_only()

        num_samples = len(self._experience_buffer)
        epochs = self._config.sleep_consolidation_epochs
        initial_loss = 0.0
        final_loss = 0.0
        total_steps = 0

        logger.info(
            "Starting sleep consolidation: %d samples, %d epochs",
            num_samples,
            epochs,
        )

        # Enable LoRA for training
        lora_model = self._implicit._lora_adapter
        if lora_model is not None:
            lora_model.enable_adapter_layers()

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                # Shuffle replay order (biological: stochastic replay)
                indices = list(range(num_samples))
                random.shuffle(indices)

                for idx in indices:
                    telemetry, target = self._experience_buffer[idx]

                    # Forward: re-encode telemetry through ISM via the PEFT model
                    # (must go through lora_model, not backend._model, for LoRA path)
                    token_ids = self._implicit._telemetry_to_token_ids(telemetry)
                    if token_ids.dim() == 1:
                        token_ids = token_ids.unsqueeze(0)
                    out = lora_model(
                        token_ids.to(self._implicit._device),
                        output_hidden_states=True,
                    )
                    layer_idx = min(
                        self._config.encoder_layer_index,
                        len(out.hidden_states) - 1,
                    )
                    h = out.hidden_states[layer_idx]
                    reconstructed = h.mean(dim=1).squeeze(0)

                    # Project to state_dim if needed
                    proj = self._implicit._backend._projection
                    if proj is not None:
                        reconstructed = proj(reconstructed)

                    # Loss against stored self-state
                    loss = self._loss_fn(reconstructed, target)

                    # Backward + step
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    epoch_loss += loss.item()
                    total_steps += 1

                avg_epoch_loss = epoch_loss / num_samples
                if epoch == 0:
                    initial_loss = avg_epoch_loss
                if epoch == epochs - 1:
                    final_loss = avg_epoch_loss

                logger.info(
                    "  Epoch %d/%d: avg_loss=%.6f",
                    epoch + 1,
                    epochs,
                    avg_epoch_loss,
                )

                if progress_callback is not None:
                    await progress_callback(epoch + 1, epochs, avg_epoch_loss)
        finally:
            # Always disable LoRA when done
            if lora_model is not None:
                lora_model.disable_adapter_layers()

        self._total_cycles += 1
        improvement = (initial_loss - final_loss) / max(initial_loss, 1e-8)

        logger.info(
            "Sleep consolidation complete: cycle=%d, steps=%d, "
            "loss %.6f -> %.6f (%.1f%% improvement)",
            self._total_cycles,
            total_steps,
            initial_loss,
            final_loss,
            improvement * 100,
        )

        return {
            "num_samples": float(num_samples),
            "epochs": float(epochs),
            "total_steps": float(total_steps),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }

    async def _run_measurement_only(self) -> dict[str, float]:
        """Measure reconstruction loss without training (no LoRA fallback)."""
        total_loss = 0.0
        num = len(self._experience_buffer)

        for telemetry, target in self._experience_buffer:
            with torch.no_grad():
                reconstructed = self._implicit.encode_self(telemetry)
                loss = self._loss_fn(reconstructed, target)
                total_loss += loss.item()

        avg = total_loss / num if num > 0 else 0.0
        return {
            "num_samples": float(num),
            "epochs": 0.0,
            "total_steps": 0.0,
            "initial_loss": avg,
            "final_loss": avg,
        }

    def clear_buffers(self) -> None:
        """Clear the experience buffer after maintenance completes."""
        count = len(self._experience_buffer)
        self._experience_buffer.clear()
        logger.info("Experience buffer cleared (%d entries removed)", count)

    @property
    def buffer_size(self) -> int:
        """Current number of buffered experiences."""
        return len(self._experience_buffer)

    @property
    def max_buffer_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_buffer_size

    @property
    def total_cycles(self) -> int:
        """Number of completed sleep consolidation cycles."""
        return self._total_cycles
