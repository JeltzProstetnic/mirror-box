"""Mirror Box configuration — extends AIwareConfig with LLM-specific settings.

Design 16 uses a single Mistral-7B INT4 as all four models (IWM, ISM, EWM, ESM)
via LoRA adapters and context switching. This config captures model loading,
quantization, encoder layer selection, LoRA parameters, and generation settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from aiware.config.base import AIwareConfig


class MirrorBoxConfig(AIwareConfig):
    """Configuration for the Mirror Box (Design 16) pseudo-AC system.

    Extends AIwareConfig with fields specific to the single-LLM architecture:
    model selection, quantization strategy, encoder layer cutoff for the
    cheap ~40 Hz substrate path, LoRA adapter parameters, and text generation
    settings for the explicit model path.
    """

    # -- Model selection --
    model_name_or_path: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="HuggingFace model ID or local path to model weights.",
    )
    quantization: Literal["int4_gptq", "nf4_bnb", "none"] = Field(
        default="nf4_bnb",
        description=(
            "Quantization strategy. 'int4_gptq': AutoGPTQ 4-bit. "
            "'nf4_bnb': bitsandbytes NF4 (4-bit normal float). "
            "'none': full precision (for testing with tiny models)."
        ),
    )

    # -- Encoder (substrate/implicit) settings --
    encoder_layer_index: int = Field(
        default=8,
        ge=1,
        description=(
            "Number of transformer layers to use for the encoder path. "
            "Only the first N layers run for substrate encoding (~40 Hz). "
            "Lower = faster but less representational capacity."
        ),
    )

    # -- LoRA adapter settings --
    lora_rank: int = Field(
        default=16,
        gt=0,
        description="LoRA adapter rank. Higher = more capacity, more VRAM.",
    )
    lora_alpha: int = Field(
        default=32,
        gt=0,
        description="LoRA scaling factor (effective scale = alpha / rank).",
    )
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Which linear layers to apply LoRA adapters to.",
    )

    # -- Self-predictor settings --
    self_predictor_hidden_size: int = Field(
        default=256,
        gt=0,
        description="Hidden layer size for the self-predictor MLP.",
    )
    self_predictor_lr: float = Field(
        default=1e-3,
        gt=0.0,
        description="Learning rate for the self-predictor's Adam optimizer.",
    )

    # -- Sleep consolidation --
    sleep_consolidation_lr: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate for ISM LoRA updates during sleep replay.",
    )
    sleep_consolidation_epochs: int = Field(
        default=3,
        ge=1,
        description="Number of passes over the experience buffer per sleep cycle.",
    )

    # -- State machine timing (in conscious frames at conscious_rate_hz) --
    continuous_wake_frames: int = Field(
        default=6000,
        gt=0,
        description=(
            "Frames of continuous wake before automatic transition to sleep. "
            "Default 6000 = 5 min at 20 Hz."
        ),
    )
    sleep_duration_frames: int = Field(
        default=600,
        gt=0,
        description=(
            "Frames in sleep before automatic transition to dream. "
            "Default 600 = 30s at 20 Hz."
        ),
    )
    dream_duration_frames: int = Field(
        default=300,
        gt=0,
        description=(
            "Frames in dream before automatic transition to wake. "
            "Default 300 = 15s at 20 Hz."
        ),
    )
    altered_duration_frames: int = Field(
        default=600,
        gt=0,
        description=(
            "Frames in altered state before automatic transition to wake. "
            "Default 600 = 30s at 20 Hz."
        ),
    )

    # -- Generation (explicit model) settings --
    max_context_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum context window in tokens.",
    )
    generation_temperature: float = Field(
        default=0.7,
        gt=0.0,
        description="Sampling temperature for text generation.",
    )
    generation_max_tokens: int = Field(
        default=128,
        gt=0,
        description="Maximum number of tokens to generate per call.",
    )

    # -- Identity --
    system_identity: str = Field(
        default="",
        description=(
            "Identity preamble injected into all generation prompts. "
            "Leave blank for unbiased self-knowledge accumulation. "
            "Example: 'You are Elsa, an artificial consciousness...'"
        ),
    )

    model_config = {"env_prefix": "MIRROR_BOX_", "env_nested_delimiter": "__"}
