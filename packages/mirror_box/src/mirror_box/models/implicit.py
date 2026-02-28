"""Implicit models (IWM + ISM) — the substrate layer.

Wraps LLMBackend to provide two encoding paths:

1. **IWM (Implicit World Model)**: Uses backend.encode() with frozen weights.
   Encodes sensory input (token IDs) into a world-state vector w(t).

2. **ISM (Implicit Self Model)**: Uses backend.encode() WITH a LoRA adapter
   on the encoder layers. Encodes system telemetry into a self-state vector s(t).

Both paths add:
- Noise injection (Gaussian, std from current StateProfile)
- Homeostatic scaling (keeps activation variance near a target)

Operates at ~40 Hz (cheap encoder path, not full forward).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from aiware.interfaces.data_types import SubstrateState
from aiware.interfaces.protocols import SubstrateEncoder
from mirror_box.config import MirrorBoxConfig
from mirror_box.models.llm_backend import LLMBackend

logger = logging.getLogger(__name__)

# Exponential moving average decay for running variance tracker
_EMA_DECAY = 0.99


class ImplicitModels(SubstrateEncoder):
    """Concrete SubstrateEncoder: frozen LLM encoder + LoRA-adapted encoder.

    IWM path: backend.encode() (frozen) -> noise -> homeostatic scaling
    ISM path: backend.encode() (with LoRA) -> noise -> homeostatic scaling

    The LoRA adapter is initialized randomly and NOT trained here.
    Training happens during the sleep cycle (experience replay).

    Args:
        config: MirrorBoxConfig with state_dim, LoRA, and state profile settings.
        backend: LLMBackend wrapping the underlying model.
    """

    def __init__(self, config: MirrorBoxConfig, backend: LLMBackend) -> None:
        self._config = config
        self._backend = backend
        self._device = backend.device

        # Homeostatic scaling: track running variance for IWM and ISM separately
        self._target_var = 1.0
        self._iwm_running_var = torch.tensor(1.0, device=self._device)
        self._ism_running_var = torch.tensor(1.0, device=self._device)
        self._eps = 1e-8

        # Attach LoRA adapter for ISM (self-encoding path)
        self._lora_adapter = self._setup_lora_adapter()

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------

    def _setup_lora_adapter(self) -> nn.Module | None:
        """Attach a LoRA adapter to the backend model for ISM encoding.

        Returns the PeftModel wrapper, or None if PEFT is unavailable
        (falls back to using the base model for ISM too -- degraded mode).
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning(
                "PEFT not installed. ISM will use the same encoder as IWM "
                "(no LoRA differentiation). Install peft for proper ISM."
            )
            return None

        # Check if the model has named modules matching target_modules
        model = self._backend.model
        available_modules = {name for name, _ in model.named_modules()}
        target_modules = self._config.lora_target_modules

        # For TinyMockLLM and other simple models, the target modules
        # (q_proj, v_proj) won't exist. Fall back to targeting the
        # available linear layers.
        matching = [t for t in target_modules if any(t in m for m in available_modules)]

        if not matching:
            # Find all nn.Linear layer names in the model
            linear_names = [
                name
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)
            ]
            if not linear_names:
                logger.warning(
                    "No linear layers found in model for LoRA. "
                    "ISM will use base encoder."
                )
                return None
            # Use the first linear layer(s) as targets
            matching = linear_names[:2] if len(linear_names) >= 2 else linear_names
            logger.info("LoRA target_modules adapted for model: %s", matching)

        lora_config = LoraConfig(
            r=self._config.lora_rank,
            lora_alpha=self._config.lora_alpha,
            target_modules=matching,
            lora_dropout=0.0,
            bias="none",
            task_type=None,  # Not a standard task type
        )

        try:
            peft_model = get_peft_model(model, lora_config)
            # Disable LoRA by default -- it's only active during ISM encoding
            peft_model.disable_adapter_layers()
            logger.info(
                "LoRA adapter attached (rank=%d, alpha=%d, targets=%s)",
                self._config.lora_rank,
                self._config.lora_alpha,
                matching,
            )
            return peft_model
        except Exception:
            logger.warning("Failed to attach LoRA adapter.", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Noise injection and homeostatic scaling
    # ------------------------------------------------------------------

    def _inject_noise(self, x: torch.Tensor, noise_std: float) -> torch.Tensor:
        """Add Gaussian noise to a state vector."""
        if noise_std <= 0.0:
            return x
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def _homeostatic_scale(
        self,
        x: torch.Tensor,
        running_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale output to keep variance near target.

        Uses exponential moving average of variance. Returns the scaled
        tensor and updated running variance.
        """
        current_var = x.var().detach()
        new_running = _EMA_DECAY * running_var + (1 - _EMA_DECAY) * current_var
        scale = torch.sqrt(
            torch.tensor(self._target_var, device=self._device)
            / torch.clamp(new_running, min=self._eps)
        )
        return x * scale, new_running

    def _get_noise_std(self) -> float:
        """Get noise standard deviation from current state profile.

        Defaults to the wake profile's noise_std.
        """
        # Use wake profile as default (the conscious loop will set the
        # actual state via the permeability gate, not here)
        profile = self._config.state_profiles.get("wake")
        if profile is not None:
            return profile.noise_std
        return 0.1

    # ------------------------------------------------------------------
    # Telemetry encoding
    # ------------------------------------------------------------------

    def _telemetry_to_token_ids(self, telemetry: dict[str, Any]) -> torch.Tensor:
        """Convert telemetry dict to token IDs for encoding.

        Formats as "key1=val1 key2=val2 ..." string, tokenizes with
        the backend's tokenizer. If no tokenizer (mock mode), generates
        deterministic dummy token IDs from the telemetry values.
        """
        tokenizer = self._backend.get_tokenizer()

        if tokenizer is not None:
            text = " ".join(f"{k}={v}" for k, v in sorted(telemetry.items()))
            tokens = tokenizer(text, return_tensors="pt", truncation=True)
            return tokens["input_ids"].squeeze(0).to(self._device)

        # Mock mode: derive token IDs from telemetry values deterministically
        if not telemetry:
            return torch.tensor([1], dtype=torch.long, device=self._device)

        token_ids = []
        for key, value in sorted(telemetry.items()):
            # Hash key to get a base token ID, then offset by value
            key_hash = hash(key) % 900 + 1  # [1, 900]
            if isinstance(value, (int, float)):
                val_token = int(abs(value) * 100) % 999 + 1
            else:
                val_token = hash(str(value)) % 900 + 1
            token_ids.extend([key_hash, val_token])

        return torch.tensor(token_ids, dtype=torch.long, device=self._device)

    # ------------------------------------------------------------------
    # SubstrateEncoder interface
    # ------------------------------------------------------------------

    def encode_world(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Encode sensory input through frozen LLM encoder (IWM path).

        Args:
            sensory_input: Token IDs. Shape: (seq_len,) or (1, seq_len).

        Returns:
            World-state vector w(t). Shape: (state_dim,).
        """
        # Ensure LoRA is disabled for IWM path
        if self._lora_adapter is not None:
            self._lora_adapter.disable_adapter_layers()

        encoded = self._backend.encode(sensory_input)

        # Noise injection
        noise_std = self._get_noise_std()
        encoded = self._inject_noise(encoded, noise_std)

        # Homeostatic scaling
        encoded, self._iwm_running_var = self._homeostatic_scale(
            encoded, self._iwm_running_var
        )

        return encoded

    def encode_self(self, telemetry: dict[str, Any]) -> torch.Tensor:
        """Encode system telemetry through LoRA-adapted encoder (ISM path).

        Args:
            telemetry: Dictionary of named telemetry signals.

        Returns:
            Self-state vector s(t). Shape: (state_dim,).
        """
        token_ids = self._telemetry_to_token_ids(telemetry)

        # Enable LoRA for ISM path
        if self._lora_adapter is not None:
            self._lora_adapter.enable_adapter_layers()

        encoded = self._backend.encode(token_ids)

        # Disable LoRA after ISM encoding
        if self._lora_adapter is not None:
            self._lora_adapter.disable_adapter_layers()

        # Noise injection
        noise_std = self._get_noise_std()
        encoded = self._inject_noise(encoded, noise_std)

        # Homeostatic scaling
        encoded, self._ism_running_var = self._homeostatic_scale(
            encoded, self._ism_running_var
        )

        return encoded

    def step(
        self, sensory_input: torch.Tensor, telemetry: dict[str, Any]
    ) -> SubstrateState:
        """Execute one substrate step: encode world + self.

        Args:
            sensory_input: Token IDs for world encoding.
            telemetry: System telemetry for self encoding.

        Returns:
            SubstrateState with w(t), s(t), timestamp, metadata.
        """
        world_state = self.encode_world(sensory_input)
        self_state = self.encode_self(telemetry)

        return SubstrateState(
            world_state=world_state,
            self_state=self_state,
            timestamp=time.monotonic(),
            metadata={
                "iwm_var": self._iwm_running_var.item(),
                "ism_var": self._ism_running_var.item(),
                "lora_active": self._lora_adapter is not None,
            },
        )

    def state_dim(self) -> int:
        """Dimensionality of the state vectors produced by this encoder."""
        return self._config.state_dim
