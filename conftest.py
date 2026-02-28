"""Root conftest.py — shared fixtures for CPU-mode testing.

All tests run on CPU by default. GPU tests are marked with @pytest.mark.gpu
and skipped unless CUDA is available. No real LLM is loaded in unit tests —
a tiny random model with the same interface is used instead.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from aiware.config.base import AIwareConfig, DeviceConfig
from aiware.interfaces.data_types import (
    ConsciousState,
    ExplicitState,
    FrameMetrics,
    GateOutput,
    SubstrateState,
)


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for all tests."""
    return torch.device("cpu")


@pytest.fixture
def state_dim() -> int:
    """Standard state vector dimensionality for tests (small for speed)."""
    return 64


@pytest.fixture
def test_dtype() -> torch.dtype:
    """Default dtype for test tensors."""
    return torch.float32


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------


@pytest.fixture
def random_state(state_dim: int, cpu_device: torch.device, test_dtype: torch.dtype):
    """Factory: create a random state vector on CPU."""

    def _make(seed: int = 42) -> torch.Tensor:
        gen = torch.Generator(device=cpu_device).manual_seed(seed)
        return torch.randn(state_dim, generator=gen, device=cpu_device, dtype=test_dtype)

    return _make


@pytest.fixture
def zero_state(state_dim: int, cpu_device: torch.device, test_dtype: torch.dtype) -> torch.Tensor:
    """A zero vector of state_dim on CPU."""
    return torch.zeros(state_dim, device=cpu_device, dtype=test_dtype)


# ---------------------------------------------------------------------------
# Data type fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_substrate_state(
    random_state,
) -> SubstrateState:
    """A sample SubstrateState with random vectors."""
    return SubstrateState(
        world_state=random_state(seed=1),
        self_state=random_state(seed=2),
        timestamp=time.monotonic(),
        metadata={"test": True},
    )


@pytest.fixture
def sample_gate_output(
    random_state,
) -> GateOutput:
    """A sample GateOutput with random gated vectors."""
    return GateOutput(
        gated_world=random_state(seed=3),
        gated_self=random_state(seed=4),
        gate_world_activation=0.85,
        gate_self_activation=0.72,
        timestamp=time.monotonic(),
    )


@pytest.fixture
def sample_explicit_state(
    random_state,
) -> ExplicitState:
    """A sample ExplicitState with random vectors."""
    return ExplicitState(
        ewm_state=random_state(seed=5),
        esm_state=random_state(seed=6),
        timestamp=time.monotonic(),
        metadata={"test": True},
    )


@pytest.fixture
def sample_frame_metrics() -> FrameMetrics:
    """A sample FrameMetrics with plausible values."""
    return FrameMetrics(
        R=0.45,
        self_surprise=1.23,
        C=0.78,
        complexity=0.55,
        state=ConsciousState.WAKE,
        frame_number=42,
        timestamp=time.monotonic(),
        extras={"prediction_loss": 0.03},
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config(state_dim: int) -> AIwareConfig:
    """AIwareConfig configured for CPU testing with small dimensions."""
    return AIwareConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=state_dim,
        conscious_rate_hz=10.0,  # Faster for tests
        wandb_enabled=False,
        metrics_buffer_size=1000,
    )


# ---------------------------------------------------------------------------
# Mock LLM (tiny random model for unit tests)
# ---------------------------------------------------------------------------


class TinyMockLLM(nn.Module):
    """Minimal model that mimics LLM encoder + generator interface.

    Has the same forward signature as a HuggingFace CausalLM but with
    2 layers and hidden_size=64. Produces deterministic output for
    reproducible tests. No tokenizer, no KV-cache, no generation — just
    forward passes that produce hidden states.

    This replaces the real Mistral-7B in unit tests so they run in <1s on CPU.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.projection = nn.Linear(hidden_size, hidden_size)
        # Make deterministic
        torch.manual_seed(0)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> MagicMock:
        """Mimics HuggingFace model forward, returning hidden states."""
        x = self.embed(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            hidden_states.append(x)

        result = MagicMock()
        result.last_hidden_state = x
        if output_hidden_states:
            result.hidden_states = tuple(hidden_states)
        return result

    def encode(self, input_ids: torch.Tensor, layer_index: int = -1) -> torch.Tensor:
        """Extract hidden state from a specific layer, mean-pooled to (hidden_size,)."""
        out = self.forward(input_ids, output_hidden_states=True)
        h = out.hidden_states[layer_index]  # (batch, seq, hidden)
        return h.mean(dim=1).squeeze(0)  # (hidden,)


@pytest.fixture
def mock_llm(state_dim: int) -> TinyMockLLM:
    """A tiny mock LLM with hidden_size matching state_dim."""
    return TinyMockLLM(hidden_size=state_dim, num_layers=2)


@pytest.fixture
def mock_token_ids(cpu_device: torch.device) -> torch.Tensor:
    """Sample token IDs for testing (short sequence)."""
    return torch.tensor([1, 42, 100, 7, 256, 33], device=cpu_device, dtype=torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# GPU skip marker
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip GPU tests when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
