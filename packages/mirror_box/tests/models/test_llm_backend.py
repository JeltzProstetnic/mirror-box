"""Tests for the LLM backend — the core model loading and inference engine.

All unit tests use TinyMockLLM from conftest.py (no real Mistral-7B download).
Integration tests that load real models are marked @pytest.mark.integration.
"""

from __future__ import annotations

import pytest
import torch

from aiware.config.base import DeviceConfig
from mirror_box.config import MirrorBoxConfig
from mirror_box.models.llm_backend import LLMBackend

# Import TinyMockLLM directly for custom instantiation
from conftest import TinyMockLLM


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
    )


@pytest.fixture
def tiny_config_projected() -> MirrorBoxConfig:
    """MirrorBoxConfig where state_dim != hidden_size (forces projection)."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=32,  # TinyMockLLM hidden_size=64, so projection needed
        model_name_or_path="test-model",
        quantization="none",
        encoder_layer_index=1,
        lora_rank=8,
        lora_alpha=16,
        max_context_length=128,
        generation_temperature=0.7,
        generation_max_tokens=16,
        wandb_enabled=False,
    )


@pytest.fixture
def mock_model() -> TinyMockLLM:
    """TinyMockLLM with hidden_size=64, 2 layers."""
    return TinyMockLLM(hidden_size=64, num_layers=2)


@pytest.fixture
def backend(tiny_config: MirrorBoxConfig, mock_model: TinyMockLLM) -> LLMBackend:
    """LLMBackend wrapping TinyMockLLM, no projection (hidden_size == state_dim)."""
    return LLMBackend(config=tiny_config, model=mock_model)


@pytest.fixture
def backend_projected(
    tiny_config_projected: MirrorBoxConfig, mock_model: TinyMockLLM
) -> LLMBackend:
    """LLMBackend wrapping TinyMockLLM, WITH projection (hidden_size != state_dim)."""
    return LLMBackend(config=tiny_config_projected, model=mock_model)


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Sample token ID tensor for testing."""
    return torch.tensor([[1, 42, 100, 7, 256, 33]], dtype=torch.long)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestLLMBackendConstruction:
    """Test model construction and property accessors."""

    def test_construction_with_mock_model(
        self, backend: LLMBackend, mock_model: TinyMockLLM
    ) -> None:
        """LLMBackend wraps the provided model without loading from HuggingFace."""
        assert backend.model is mock_model
        assert backend.device == torch.device("cpu")

    def test_hidden_size_detected(self, backend: LLMBackend) -> None:
        """Hidden size is correctly detected from TinyMockLLM."""
        assert backend.hidden_size == 64

    def test_state_dim_matches_config(self, backend: LLMBackend) -> None:
        """Output state_dim matches config when hidden_size == state_dim."""
        assert backend.state_dim == 64

    def test_no_projection_when_sizes_match(self, backend: LLMBackend) -> None:
        """No projection layer when hidden_size == state_dim."""
        assert not backend.has_projection

    def test_projection_when_sizes_differ(
        self, backend_projected: LLMBackend
    ) -> None:
        """Projection layer is created when hidden_size != state_dim."""
        assert backend_projected.has_projection
        assert backend_projected.hidden_size == 64
        assert backend_projected.state_dim == 32

    def test_tokenizer_is_none_for_mock(self, backend: LLMBackend) -> None:
        """No tokenizer when using a custom model without one."""
        assert backend.get_tokenizer() is None


# ---------------------------------------------------------------------------
# Encoder path
# ---------------------------------------------------------------------------


class TestEncoderPath:
    """Test the encode() method (substrate/implicit, ~40 Hz path)."""

    def test_encode_output_shape(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """encode() produces a vector of shape (state_dim,)."""
        result = backend.encode(sample_input)
        assert result.shape == (64,)

    def test_encode_output_shape_1d_input(self, backend: LLMBackend) -> None:
        """encode() handles 1-D input (no batch dim) by adding one."""
        input_1d = torch.tensor([1, 42, 100], dtype=torch.long)
        result = backend.encode(input_1d)
        assert result.shape == (64,)

    def test_encode_with_projection(
        self,
        backend_projected: LLMBackend,
        sample_input: torch.Tensor,
    ) -> None:
        """encode() projects to state_dim when hidden_size != state_dim."""
        result = backend_projected.encode(sample_input)
        assert result.shape == (32,)

    def test_encode_deterministic_same_input(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """Same input produces identical output (model is in eval mode, no dropout)."""
        r1 = backend.encode(sample_input)
        r2 = backend.encode(sample_input)
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_encode_different_inputs_differ(self, backend: LLMBackend) -> None:
        """Different inputs produce different outputs."""
        input_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
        input_b = torch.tensor([[100, 200, 300]], dtype=torch.long)
        r_a = backend.encode(input_a)
        r_b = backend.encode(input_b)
        assert not torch.allclose(r_a, r_b, atol=1e-4)

    def test_encode_output_is_float(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """Output tensor is floating point, not integer."""
        result = backend.encode(sample_input)
        assert result.dtype in (torch.float32, torch.float16, torch.bfloat16)


# ---------------------------------------------------------------------------
# Generator path — hidden states
# ---------------------------------------------------------------------------


class TestGenerateHidden:
    """Test the generate_hidden() method (explicit, ~20 Hz path)."""

    def test_generate_hidden_output_shape(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """generate_hidden() produces a vector of shape (state_dim,)."""
        result = backend.generate_hidden(sample_input)
        assert result.shape == (64,)

    def test_generate_hidden_1d_input(self, backend: LLMBackend) -> None:
        """generate_hidden() handles 1-D input."""
        input_1d = torch.tensor([1, 42, 100], dtype=torch.long)
        result = backend.generate_hidden(input_1d)
        assert result.shape == (64,)

    def test_generate_hidden_with_projection(
        self,
        backend_projected: LLMBackend,
        sample_input: torch.Tensor,
    ) -> None:
        """generate_hidden() projects to state_dim when sizes differ."""
        result = backend_projected.generate_hidden(sample_input)
        assert result.shape == (32,)

    def test_generate_hidden_deterministic(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """Same input produces identical output."""
        r1 = backend.generate_hidden(sample_input)
        r2 = backend.generate_hidden(sample_input)
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_generate_hidden_differs_from_encode(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """generate_hidden() (all layers) differs from encode() (first N layers).

        They use different layer depths, so outputs should generally differ
        unless the model is trivially simple.
        """
        encoded = backend.encode(sample_input)
        generated = backend.generate_hidden(sample_input)
        # With 2 layers and encoder_layer_index=1, encode uses layer 1
        # and generate_hidden uses layer 2, so they should differ.
        assert not torch.allclose(encoded, generated, atol=1e-4)


# ---------------------------------------------------------------------------
# Generator path — text
# ---------------------------------------------------------------------------


class TestGenerateText:
    """Test generate_text() — requires a tokenizer."""

    def test_generate_text_raises_without_tokenizer(
        self, backend: LLMBackend
    ) -> None:
        """generate_text() raises RuntimeError when no tokenizer is provided."""
        with pytest.raises(RuntimeError, match="requires a tokenizer"):
            backend.generate_text("Hello, world!")


# ---------------------------------------------------------------------------
# Projection layer
# ---------------------------------------------------------------------------


class TestProjection:
    """Test the learned linear projection layer."""

    def test_projection_reduces_dimension(
        self,
        backend_projected: LLMBackend,
        sample_input: torch.Tensor,
    ) -> None:
        """Projection correctly maps hidden_size=64 -> state_dim=32."""
        result = backend_projected.encode(sample_input)
        assert result.shape == (32,)

    def test_projection_is_learnable(
        self, backend_projected: LLMBackend
    ) -> None:
        """Projection layer has trainable parameters."""
        proj = backend_projected._projection
        assert proj is not None
        params = list(proj.parameters())
        assert len(params) == 2  # weight + bias
        assert params[0].requires_grad
        assert params[1].requires_grad

    def test_no_projection_output_unchanged(
        self, backend: LLMBackend, sample_input: torch.Tensor
    ) -> None:
        """Without projection, output passes through unmodified.

        With TinyMockLLM (hidden_size=64) and state_dim=64, the hidden
        states flow directly without transformation.
        """
        assert backend._projection is None
        result = backend.encode(sample_input)
        assert result.shape == (64,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_token_input(self, backend: LLMBackend) -> None:
        """Works with a single-token input sequence."""
        single = torch.tensor([[42]], dtype=torch.long)
        result = backend.encode(single)
        assert result.shape == (64,)

    def test_encoder_layer_clamped_to_available(self) -> None:
        """If encoder_layer_index > num_layers, clamps to last available layer."""
        config = MirrorBoxConfig(
            hardware=DeviceConfig(device="cpu", dtype="float32"),
            state_dim=64,
            quantization="none",
            encoder_layer_index=100,  # Way more than 2 layers
            wandb_enabled=False,
        )
        model = TinyMockLLM(hidden_size=64, num_layers=2)
        backend = LLMBackend(config=config, model=model)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        result = backend.encode(input_ids)
        # Should still work, using the last available layer
        assert result.shape == (64,)

    def test_output_on_correct_device(self, backend: LLMBackend) -> None:
        """Output tensors are on the configured device (CPU in tests)."""
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        encoded = backend.encode(input_ids)
        hidden = backend.generate_hidden(input_ids)
        assert encoded.device == torch.device("cpu")
        assert hidden.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestMirrorBoxConfig:
    """Test MirrorBoxConfig itself."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = MirrorBoxConfig()
        assert config.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.quantization == "nf4_bnb"
        assert config.encoder_layer_index == 8
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_target_modules == ["q_proj", "v_proj"]
        assert config.max_context_length == 2048
        assert config.generation_temperature == 0.7
        assert config.generation_max_tokens == 128

    def test_inherits_aiware_config(self) -> None:
        """MirrorBoxConfig extends AIwareConfig with all base fields."""
        config = MirrorBoxConfig()
        # Base class fields
        assert config.state_dim == 512
        assert config.conscious_rate_hz == 20.0
        assert hasattr(config, "hardware")
        assert hasattr(config, "state_profiles")

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = MirrorBoxConfig(
            state_dim=256,
            model_name_or_path="my/model",
            quantization="none",
            encoder_layer_index=4,
            lora_rank=32,
        )
        assert config.state_dim == 256
        assert config.model_name_or_path == "my/model"
        assert config.quantization == "none"
        assert config.encoder_layer_index == 4
        assert config.lora_rank == 32

    def test_invalid_quantization_rejected(self) -> None:
        """Invalid quantization strategy is rejected by Pydantic."""
        with pytest.raises(Exception):
            MirrorBoxConfig(quantization="invalid")

    def test_encoder_layer_index_must_be_positive(self) -> None:
        """encoder_layer_index must be >= 1."""
        with pytest.raises(Exception):
            MirrorBoxConfig(encoder_layer_index=0)


# ---------------------------------------------------------------------------
# Integration tests (require real model download, skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationRealModel:
    """Tests that load a real model from HuggingFace.

    Skipped unless explicitly requested (pytest -m integration).
    Requires network access and sufficient disk/VRAM.
    """

    def test_load_tiny_hf_model(self) -> None:
        """Load a tiny HuggingFace model (sshleifer/tiny-gpt2) for smoke testing."""
        config = MirrorBoxConfig(
            model_name_or_path="sshleifer/tiny-gpt2",
            quantization="none",
            state_dim=64,
            encoder_layer_index=1,
            wandb_enabled=False,
        )
        backend = LLMBackend(config=config)
        assert backend.get_tokenizer() is not None

        result = backend.encode(
            torch.tensor([[1, 2, 3]], dtype=torch.long)
        )
        assert result.shape == (config.state_dim,)

        text = backend.generate_text("Hello", max_tokens=5)
        assert isinstance(text, str)
