"""Base configuration classes shared across all AC implementations.

Implementations extend these with their own model-specific settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DeviceConfig(BaseModel):
    """Hardware / device configuration."""

    device: str = Field(
        default="auto",
        description="Device: 'auto' (prefer CUDA), 'cuda', 'cuda:N', 'cpu', 'mps'.",
    )
    dtype: str = Field(
        default="float32",
        description="Default tensor dtype: 'float32', 'float16', 'bfloat16'.",
    )

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' to an actual device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        """Resolve dtype string to torch.dtype."""
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[self.dtype]


class StateProfile(BaseModel):
    """Permeability gate parameters for one operating state."""

    alpha_world: float = Field(description="World attention gain")
    alpha_self: float = Field(description="Self attention gain")
    beta: float = Field(description="Global permeability bias")
    noise_std: float = Field(ge=0.0, description="Stochastic noise standard deviation")
    external_input: bool = Field(description="Whether external sensory input is active")
    self_ref_loop: bool = Field(description="Whether the self-referential loop is active")
    substrate_rate_hz: float = Field(gt=0, description="Substrate update rate in Hz")


class AIwareConfig(BaseSettings):
    """Base configuration for any AC implementation.

    Subclasses add model-specific fields. All implementations share
    the device config and state profiles.
    """

    # -- Hardware --
    hardware: DeviceConfig = Field(default_factory=DeviceConfig)

    # -- Architecture --
    state_dim: int = Field(default=512, gt=0, description="Dimensionality of state vectors")
    conscious_rate_hz: float = Field(default=20.0, gt=0, description="Conscious frame rate in Hz")

    # -- State profiles (FMT Table 4.2) --
    state_profiles: dict[str, StateProfile] = Field(
        default_factory=lambda: {
            "wake": StateProfile(
                alpha_world=1.0,
                alpha_self=1.0,
                beta=0.0,
                noise_std=0.1,
                external_input=True,
                self_ref_loop=True,
                substrate_rate_hz=40.0,
            ),
            "sleep": StateProfile(
                alpha_world=0.1,
                alpha_self=0.1,
                beta=-3.0,
                noise_std=0.01,
                external_input=False,
                self_ref_loop=False,
                substrate_rate_hz=10.0,
            ),
            "dream": StateProfile(
                alpha_world=0.0,
                alpha_self=0.3,
                beta=-0.5,
                noise_std=0.5,
                external_input=False,
                self_ref_loop=True,
                substrate_rate_hz=20.0,
            ),
            "altered": StateProfile(
                alpha_world=0.5,
                alpha_self=0.5,
                beta=2.0,
                noise_std=1.0,
                external_input=True,
                self_ref_loop=True,
                substrate_rate_hz=40.0,
            ),
        }
    )

    # -- Metrics --
    metrics_buffer_size: int = Field(
        default=10000, gt=0, description="Max frames to keep in memory"
    )
    C_threshold: float = Field(
        default=0.01, ge=0.0, description="Activation threshold for C(t) computation"
    )

    # -- Experiment tracking --
    wandb_project: str = Field(default="aiware", description="W&B project name")
    wandb_enabled: bool = Field(default=False, description="Enable W&B logging")

    # -- Paths --
    config_dir: Path = Field(default=Path("configs"), description="Config file directory")

    model_config = {"env_prefix": "AIWARE_", "env_nested_delimiter": "__"}
