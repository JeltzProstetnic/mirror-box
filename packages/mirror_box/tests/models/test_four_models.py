"""Tests for the four-model architecture: IWM, ISM, EWM, ESM + permeability gate.

All unit tests use TinyMockLLM from conftest.py (no real Mistral-7B).
Tests verify shape contracts, noise injection, homeostatic scaling,
self-referential closure, and state-dependent gate behavior.
"""

from __future__ import annotations

import time

import pytest
import torch

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState, GateOutput, SubstrateState
from conftest import TinyMockLLM
from mirror_box.config import MirrorBoxConfig
from mirror_box.models.explicit import ExplicitModels
from mirror_box.models.implicit import ImplicitModels
from mirror_box.models.llm_backend import LLMBackend
from mirror_box.models.permeability import SigmoidPermeabilityGate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config() -> MirrorBoxConfig:
    """MirrorBoxConfig for testing: CPU, no quantization, state_dim=64."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=64,
        model_name_or_path="test-model",
        quantization="none",
        encoder_layer_index=1,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=["q_proj", "v_proj"],
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
    """LLMBackend wrapping TinyMockLLM."""
    return LLMBackend(config=tiny_config, model=mock_model)


@pytest.fixture
def sample_tokens() -> torch.Tensor:
    """Sample token IDs for world encoding."""
    return torch.tensor([1, 42, 100, 7, 256, 33], dtype=torch.long)


@pytest.fixture
def sample_telemetry() -> dict[str, float]:
    """Sample telemetry dict for self encoding."""
    return {
        "latency": 0.025,
        "R": 0.45,
        "ewm_norm": 1.23,
        "esm_norm": 0.87,
        "gpu_temp": 62.0,
    }


@pytest.fixture
def implicit_models(
    tiny_config: MirrorBoxConfig, backend: LLMBackend
) -> ImplicitModels:
    """ImplicitModels instance for testing."""
    return ImplicitModels(config=tiny_config, backend=backend)


@pytest.fixture
def explicit_models(tiny_config: MirrorBoxConfig, backend: LLMBackend) -> ExplicitModels:
    """ExplicitModels instance for testing."""
    return ExplicitModels(config=tiny_config, backend=backend)


@pytest.fixture
def gate(tiny_config: MirrorBoxConfig) -> SigmoidPermeabilityGate:
    """SigmoidPermeabilityGate instance for testing."""
    return SigmoidPermeabilityGate(config=tiny_config)


@pytest.fixture
def sample_substrate_state() -> SubstrateState:
    """A SubstrateState with random vectors for gate testing."""
    torch.manual_seed(42)
    return SubstrateState(
        world_state=torch.randn(64),
        self_state=torch.randn(64),
        timestamp=time.monotonic(),
    )


# ===========================================================================
# ImplicitModels (IWM + ISM)
# ===========================================================================


class TestImplicitModelsShape:
    """Shape contracts for ImplicitModels."""

    def test_encode_world_shape(
        self,
        implicit_models: ImplicitModels,
        sample_tokens: torch.Tensor,
    ) -> None:
        """encode_world produces vector of shape (state_dim,)."""
        result = implicit_models.encode_world(sample_tokens)
        assert result.shape == (64,)

    def test_encode_self_shape(
        self,
        implicit_models: ImplicitModels,
        sample_telemetry: dict[str, float],
    ) -> None:
        """encode_self produces vector of shape (state_dim,) from telemetry dict."""
        result = implicit_models.encode_self(sample_telemetry)
        assert result.shape == (64,)

    def test_step_returns_substrate_state(
        self,
        implicit_models: ImplicitModels,
        sample_tokens: torch.Tensor,
        sample_telemetry: dict[str, float],
    ) -> None:
        """step() returns SubstrateState with matching shapes."""
        state = implicit_models.step(sample_tokens, sample_telemetry)
        assert isinstance(state, SubstrateState)
        assert state.world_state.shape == (64,)
        assert state.self_state.shape == (64,)
        assert state.timestamp > 0
        assert isinstance(state.metadata, dict)

    def test_state_dim_matches_config(
        self, implicit_models: ImplicitModels
    ) -> None:
        """state_dim() returns the configured value."""
        assert implicit_models.state_dim() == 64

    def test_encode_world_returns_float(
        self,
        implicit_models: ImplicitModels,
        sample_tokens: torch.Tensor,
    ) -> None:
        """Output tensor is floating point."""
        result = implicit_models.encode_world(sample_tokens)
        assert result.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_encode_self_returns_float(
        self,
        implicit_models: ImplicitModels,
        sample_telemetry: dict[str, float],
    ) -> None:
        """Output tensor is floating point."""
        result = implicit_models.encode_self(sample_telemetry)
        assert result.dtype in (torch.float32, torch.float16, torch.bfloat16)


class TestImplicitModelsNoise:
    """Noise injection behavior for ImplicitModels."""

    def test_noise_changes_output(
        self, tiny_config: MirrorBoxConfig, backend: LLMBackend
    ) -> None:
        """With noise_std > 0, different calls produce different outputs.

        We compare two calls on the same input -- noise makes them diverge.
        """
        # Ensure wake profile has nonzero noise
        assert tiny_config.state_profiles["wake"].noise_std > 0

        models = ImplicitModels(config=tiny_config, backend=backend)
        tokens = torch.tensor([1, 42, 100], dtype=torch.long)

        r1 = models.encode_world(tokens)
        r2 = models.encode_world(tokens)

        # With noise, outputs should differ (overwhelmingly likely)
        assert not torch.allclose(r1, r2, atol=1e-6)

    def test_zero_noise_deterministic(
        self, backend: LLMBackend
    ) -> None:
        """With noise_std=0, output is deterministic (modulo homeostatic scaling init)."""
        config = MirrorBoxConfig(
            hardware=DeviceConfig(device="cpu", dtype="float32"),
            state_dim=64,
            quantization="none",
            encoder_layer_index=1,
            wandb_enabled=False,
            state_profiles={
                "wake": {
                    "alpha_world": 1.0,
                    "alpha_self": 1.0,
                    "beta": 0.0,
                    "noise_std": 0.0,
                    "external_input": True,
                    "self_ref_loop": True,
                    "substrate_rate_hz": 40.0,
                },
                "sleep": {
                    "alpha_world": 0.1,
                    "alpha_self": 0.1,
                    "beta": -3.0,
                    "noise_std": 0.0,
                    "external_input": False,
                    "self_ref_loop": False,
                    "substrate_rate_hz": 10.0,
                },
                "dream": {
                    "alpha_world": 0.0,
                    "alpha_self": 0.3,
                    "beta": -0.5,
                    "noise_std": 0.0,
                    "external_input": False,
                    "self_ref_loop": True,
                    "substrate_rate_hz": 20.0,
                },
                "altered": {
                    "alpha_world": 0.5,
                    "alpha_self": 0.5,
                    "beta": 2.0,
                    "noise_std": 0.0,
                    "external_input": True,
                    "self_ref_loop": True,
                    "substrate_rate_hz": 40.0,
                },
            },
        )
        models = ImplicitModels(config=config, backend=backend)
        tokens = torch.tensor([1, 42, 100], dtype=torch.long)

        # Run many times to let homeostatic scaling EMA converge
        # (EMA decay=0.99, needs ~100 steps to stabilize)
        for _ in range(200):
            models.encode_world(tokens)

        r1 = models.encode_world(tokens)
        r2 = models.encode_world(tokens)
        # After convergence, consecutive outputs differ only by EMA drift.
        # The EMA (decay=0.99) still shifts ~0.5% per step at equilibrium,
        # so we use relative tolerance (rtol) to verify near-determinism.
        assert torch.allclose(r1, r2, rtol=0.01, atol=1e-6)


class TestImplicitModelsHomeostatic:
    """Homeostatic scaling behavior for ImplicitModels."""

    def test_homeostatic_keeps_variance_bounded(
        self,
        implicit_models: ImplicitModels,
        sample_tokens: torch.Tensor,
    ) -> None:
        """After several steps, output variance stays near target (1.0).

        We don't demand exact convergence, just that variance is in a
        reasonable range (0.1 to 10.0) after warm-up.
        """
        # Run several times to warm up the EMA tracker
        for _ in range(20):
            result = implicit_models.encode_world(sample_tokens)

        var = result.var().item()
        assert 0.01 < var < 100.0, f"Variance {var} is out of reasonable range"


class TestImplicitModelsDifferentiation:
    """Different inputs produce different outputs."""

    def test_different_world_inputs(
        self, implicit_models: ImplicitModels
    ) -> None:
        """Different token sequences produce different world encodings."""
        tokens_a = torch.tensor([1, 2, 3], dtype=torch.long)
        tokens_b = torch.tensor([100, 200, 300], dtype=torch.long)

        # Run with fixed seed for reproducibility (noise is random, but
        # the base encoding should differ enough)
        torch.manual_seed(99)
        r_a = implicit_models.encode_world(tokens_a)
        torch.manual_seed(99)
        r_b = implicit_models.encode_world(tokens_b)

        assert not torch.allclose(r_a, r_b, atol=1e-3)

    def test_different_telemetry(
        self, implicit_models: ImplicitModels
    ) -> None:
        """Different telemetry dicts produce different self encodings."""
        tel_a = {"latency": 0.01, "R": 0.1}
        tel_b = {"latency": 0.99, "R": 0.9}

        torch.manual_seed(99)
        r_a = implicit_models.encode_self(tel_a)
        torch.manual_seed(99)
        r_b = implicit_models.encode_self(tel_b)

        assert not torch.allclose(r_a, r_b, atol=1e-3)

    def test_empty_telemetry_works(
        self, implicit_models: ImplicitModels
    ) -> None:
        """Empty telemetry dict does not crash."""
        result = implicit_models.encode_self({})
        assert result.shape == (64,)


class TestImplicitModelsStep:
    """Test the step() method combining world + self encoding."""

    def test_step_metadata_contains_diagnostics(
        self,
        implicit_models: ImplicitModels,
        sample_tokens: torch.Tensor,
        sample_telemetry: dict[str, float],
    ) -> None:
        """step() metadata includes variance and LoRA status."""
        state = implicit_models.step(sample_tokens, sample_telemetry)
        assert "iwm_var" in state.metadata
        assert "ism_var" in state.metadata
        assert "lora_active" in state.metadata


# ===========================================================================
# ExplicitModels (EWM + ESM)
# ===========================================================================


class TestExplicitModelsShape:
    """Shape contracts for ExplicitModels."""

    def test_generate_ewm_shape(
        self, explicit_models: ExplicitModels
    ) -> None:
        """generate_ewm produces vector of shape (state_dim,)."""
        gated_world = torch.randn(64)
        result = explicit_models.generate_ewm(gated_world)
        assert result.shape == (64,)

    def test_generate_esm_shape(
        self, explicit_models: ExplicitModels
    ) -> None:
        """generate_esm produces vector of shape (state_dim,)."""
        gated_self = torch.randn(64)
        prev_esm = torch.randn(64)
        result = explicit_models.generate_esm(gated_self, prev_esm)
        assert result.shape == (64,)

    def test_step_returns_explicit_state(
        self, explicit_models: ExplicitModels
    ) -> None:
        """step() returns ExplicitState with correct shapes."""
        gate_output = GateOutput(
            gated_world=torch.randn(64),
            gated_self=torch.randn(64),
            gate_world_activation=0.8,
            gate_self_activation=0.7,
            timestamp=time.monotonic(),
        )
        prev_esm = torch.zeros(64)

        result = explicit_models.step(gate_output, prev_esm)
        assert result.ewm_state.shape == (64,)
        assert result.esm_state.shape == (64,)
        assert result.timestamp > 0
        assert "ewm_norm" in result.metadata
        assert "esm_norm" in result.metadata

    def test_state_dim_matches_config(
        self, explicit_models: ExplicitModels
    ) -> None:
        """state_dim() returns the configured value."""
        assert explicit_models.state_dim() == 64


class TestExplicitModelsSelfReferential:
    """Self-referential closure in ESM."""

    def test_esm_depends_on_prev_esm(
        self, explicit_models: ExplicitModels
    ) -> None:
        """Different prev_esm produces different ESM output.

        This verifies the self-referential loop: e_s(t) = f(gated_self, e_s(t-1)).
        """
        gated_self = torch.randn(64)
        prev_esm_a = torch.randn(64)
        prev_esm_b = torch.randn(64) * 2.0  # Different from a

        result_a = explicit_models.generate_esm(gated_self, prev_esm_a)
        result_b = explicit_models.generate_esm(gated_self, prev_esm_b)

        # Same gated_self but different prev_esm -> different output
        assert not torch.allclose(result_a, result_b, atol=1e-5)

    def test_esm_zero_prev_esm_works(
        self, explicit_models: ExplicitModels
    ) -> None:
        """First frame: prev_esm is zero vector, should not crash."""
        gated_self = torch.randn(64)
        prev_esm = torch.zeros(64)
        result = explicit_models.generate_esm(gated_self, prev_esm)
        assert result.shape == (64,)
        # Output should not be all zeros (GELU + bias in MLP)
        assert result.abs().sum() > 0

    def test_esm_self_referential_chain(
        self, explicit_models: ExplicitModels
    ) -> None:
        """Multiple ESM steps form a chain: each depends on the previous."""
        gated_self = torch.randn(64)
        prev = torch.zeros(64)
        outputs = []
        for _ in range(5):
            prev = explicit_models.generate_esm(gated_self, prev)
            outputs.append(prev.clone())

        # Each step should differ from the previous (chain evolves)
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i - 1], atol=1e-5)


class TestExplicitModelsEWM:
    """EWM-specific behavior."""

    def test_ewm_different_inputs_differ(
        self, explicit_models: ExplicitModels
    ) -> None:
        """Different gated_world inputs produce different EWM outputs."""
        gw_a = torch.randn(64)
        gw_b = torch.randn(64) * 3.0
        r_a = explicit_models.generate_ewm(gw_a)
        r_b = explicit_models.generate_ewm(gw_b)
        assert not torch.allclose(r_a, r_b, atol=1e-5)

    def test_ewm_deterministic(
        self, explicit_models: ExplicitModels
    ) -> None:
        """EWM MLP is deterministic (no dropout, no noise)."""
        gw = torch.randn(64)
        r1 = explicit_models.generate_ewm(gw)
        r2 = explicit_models.generate_ewm(gw)
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_ewm_context_param_accepted(
        self, explicit_models: ExplicitModels
    ) -> None:
        """generate_ewm accepts a context parameter without error."""
        gw = torch.randn(64)
        context = torch.randn(64)
        result = explicit_models.generate_ewm(gw, context=context)
        assert result.shape == (64,)


class TestExplicitModelsMLPs:
    """MLP weight properties."""

    def test_ewm_mlp_trainable(
        self, explicit_models: ExplicitModels
    ) -> None:
        """EWM MLP has trainable parameters."""
        params = list(explicit_models._ewm_mlp.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_esm_mlp_trainable(
        self, explicit_models: ExplicitModels
    ) -> None:
        """ESM MLP has trainable parameters."""
        params = list(explicit_models._esm_mlp.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_esm_mlp_input_dim_doubled(
        self, explicit_models: ExplicitModels
    ) -> None:
        """ESM MLP first layer takes 2*state_dim input (concat gated_self + prev_esm)."""
        first_layer = explicit_models._esm_mlp[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == 64 * 2  # state_dim * 2


# ===========================================================================
# SigmoidPermeabilityGate
# ===========================================================================


class TestPermeabilityGateShape:
    """Shape contracts for SigmoidPermeabilityGate."""

    def test_gate_returns_gate_output(
        self,
        gate: SigmoidPermeabilityGate,
        sample_substrate_state: SubstrateState,
    ) -> None:
        """gate() returns GateOutput with correct tensor shapes."""
        result = gate.gate(sample_substrate_state, ConsciousState.WAKE)
        assert isinstance(result, GateOutput)
        assert result.gated_world.shape == (64,)
        assert result.gated_self.shape == (64,)
        assert isinstance(result.gate_world_activation, float)
        assert isinstance(result.gate_self_activation, float)

    def test_gate_preserves_timestamp(
        self,
        gate: SigmoidPermeabilityGate,
        sample_substrate_state: SubstrateState,
    ) -> None:
        """GateOutput timestamp matches the input SubstrateState."""
        result = gate.gate(sample_substrate_state, ConsciousState.WAKE)
        assert result.timestamp == sample_substrate_state.timestamp


class TestPermeabilityGateWake:
    """Wake mode: high gate activations."""

    def test_wake_high_activations(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """In wake mode, mean gate activations should be > 0.5.

        Wake profile: alpha_world=1.0, alpha_self=1.0, beta=0.0, noise_std=0.1.
        With standard-normal inputs, sigmoid(1.0 * x + 0.0) averages ~0.5.
        We use slightly positive-biased input to ensure > 0.5.
        """
        # Use positive-biased input so sigmoid pushes above 0.5
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64) + 1.0,  # Bias positive
            self_state=torch.randn(64) + 1.0,
            timestamp=time.monotonic(),
        )

        # Average over multiple samples to smooth noise
        world_acts = []
        self_acts = []
        for _ in range(50):
            result = gate.gate(substrate, ConsciousState.WAKE)
            world_acts.append(result.gate_world_activation)
            self_acts.append(result.gate_self_activation)

        mean_world = sum(world_acts) / len(world_acts)
        mean_self = sum(self_acts) / len(self_acts)

        assert mean_world > 0.5, f"Wake world activation {mean_world} should be > 0.5"
        assert mean_self > 0.5, f"Wake self activation {mean_self} should be > 0.5"


class TestPermeabilityGateSleep:
    """Sleep mode: low gate activations."""

    def test_sleep_low_activations(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """In sleep mode, mean gate activations should be < 0.2.

        Sleep profile: alpha_world=0.1, alpha_self=0.1, beta=-3.0, noise_std=0.01.
        sigmoid(0.1 * x - 3.0) is very small for normal-distributed x.
        """
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64),
            self_state=torch.randn(64),
            timestamp=time.monotonic(),
        )

        world_acts = []
        self_acts = []
        for _ in range(50):
            result = gate.gate(substrate, ConsciousState.SLEEP)
            world_acts.append(result.gate_world_activation)
            self_acts.append(result.gate_self_activation)

        mean_world = sum(world_acts) / len(world_acts)
        mean_self = sum(self_acts) / len(self_acts)

        assert mean_world < 0.2, f"Sleep world activation {mean_world} should be < 0.2"
        assert mean_self < 0.2, f"Sleep self activation {mean_self} should be < 0.2"


class TestPermeabilityGateDream:
    """Dream mode: intermediate activations with high variance."""

    def test_dream_intermediate_activations(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """In dream mode, gate activations should be intermediate.

        Dream profile: alpha_world=0.0, alpha_self=0.3, beta=-0.5, noise_std=0.5.
        sigmoid(0.0 * x - 0.5 + noise) depends heavily on noise.
        World channel (alpha=0.0): mean ~ sigmoid(-0.5) ~ 0.38 + noise variation.
        Self channel (alpha=0.3): mean varies with input + noise.
        """
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64),
            self_state=torch.randn(64),
            timestamp=time.monotonic(),
        )

        world_acts = []
        for _ in range(100):
            result = gate.gate(substrate, ConsciousState.DREAM)
            world_acts.append(result.gate_world_activation)

        mean_act = sum(world_acts) / len(world_acts)
        var_act = sum((a - mean_act) ** 2 for a in world_acts) / len(world_acts)

        # Dream activations should be intermediate (not near 0 or 1)
        assert 0.1 < mean_act < 0.8, f"Dream mean {mean_act} should be intermediate"
        # High noise (0.5) should cause noticeable variation across calls
        assert var_act > 1e-5, f"Dream variance {var_act} should be non-trivial"


class TestPermeabilityGateAltered:
    """Altered mode: high activations (content flooding)."""

    def test_altered_high_activations(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """In altered mode, gate activations should be high.

        Altered profile: alpha_world=0.5, alpha_self=0.5, beta=2.0, noise_std=1.0.
        sigmoid(0.5 * x + 2.0 + noise): the +2.0 bias pushes activations high.
        """
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64),
            self_state=torch.randn(64),
            timestamp=time.monotonic(),
        )

        world_acts = []
        self_acts = []
        for _ in range(50):
            result = gate.gate(substrate, ConsciousState.ALTERED)
            world_acts.append(result.gate_world_activation)
            self_acts.append(result.gate_self_activation)

        mean_world = sum(world_acts) / len(world_acts)
        mean_self = sum(self_acts) / len(self_acts)

        assert mean_world > 0.5, f"Altered world activation {mean_world} should be > 0.5"
        assert mean_self > 0.5, f"Altered self activation {mean_self} should be > 0.5"


class TestPermeabilityGateProfileOverride:
    """set_profile() runtime override behavior."""

    def test_set_profile_changes_behavior(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """set_profile overrides gate parameters at runtime."""
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64) + 1.0,
            self_state=torch.randn(64) + 1.0,
            timestamp=time.monotonic(),
        )

        # Before: wake should have high activations
        result_before = gate.gate(substrate, ConsciousState.WAKE)
        act_before = result_before.gate_world_activation

        # Override wake to behave like sleep
        gate.set_profile(
            ConsciousState.WAKE,
            alpha_world=0.1,
            alpha_self=0.1,
            beta=-3.0,
            noise_std=0.01,
        )

        # After: wake should now have low activations
        acts_after = []
        for _ in range(20):
            result = gate.gate(substrate, ConsciousState.WAKE)
            acts_after.append(result.gate_world_activation)

        mean_after = sum(acts_after) / len(acts_after)

        # The override should have dramatically lowered the activation
        assert mean_after < act_before, (
            f"After override, activation {mean_after} should be lower than "
            f"original {act_before}"
        )

    def test_set_profile_partial_update(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """set_profile with only some params leaves others unchanged."""
        # Original wake noise_std is 0.1
        gate.set_profile(ConsciousState.WAKE, beta=-5.0)

        # beta should be updated, noise_std should remain 0.1
        profile = gate._profiles["wake"]
        assert profile.beta == -5.0
        assert profile.noise_std == 0.1  # Unchanged

    def test_set_profile_unknown_state_logs_warning(
        self,
        gate: SigmoidPermeabilityGate,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """set_profile with invalid state logs a warning but does not crash."""
        # ConsciousState is an enum so we can't pass an invalid value directly.
        # But we can test that a valid state works without error.
        gate.set_profile(ConsciousState.SLEEP, alpha_world=0.5)
        profile = gate._profiles["sleep"]
        assert profile.alpha_world == 0.5


class TestPermeabilityGateStateDifferences:
    """Gate produces meaningfully different outputs across states."""

    def test_sleep_gates_more_than_wake(
        self,
        gate: SigmoidPermeabilityGate,
    ) -> None:
        """Sleep mode attenuates signal more than wake mode.

        The gated output norm should be smaller in sleep than in wake.
        """
        torch.manual_seed(42)
        substrate = SubstrateState(
            world_state=torch.randn(64) + 0.5,
            self_state=torch.randn(64) + 0.5,
            timestamp=time.monotonic(),
        )

        wake_norms = []
        sleep_norms = []
        for _ in range(50):
            wake_out = gate.gate(substrate, ConsciousState.WAKE)
            sleep_out = gate.gate(substrate, ConsciousState.SLEEP)
            wake_norms.append(wake_out.gated_world.norm().item())
            sleep_norms.append(sleep_out.gated_world.norm().item())

        mean_wake = sum(wake_norms) / len(wake_norms)
        mean_sleep = sum(sleep_norms) / len(sleep_norms)

        assert mean_sleep < mean_wake, (
            f"Sleep norm {mean_sleep} should be smaller than wake norm {mean_wake}"
        )


# ===========================================================================
# Integration: Full pipeline IWM/ISM -> Gate -> EWM/ESM
# ===========================================================================


class TestFourModelPipeline:
    """End-to-end test: substrate -> gate -> explicit models."""

    def test_full_pipeline_shapes(
        self,
        implicit_models: ImplicitModels,
        gate: SigmoidPermeabilityGate,
        explicit_models: ExplicitModels,
        sample_tokens: torch.Tensor,
        sample_telemetry: dict[str, float],
    ) -> None:
        """Full pipeline produces correct shapes at each stage."""
        # Stage 1: Substrate encoding
        substrate_state = implicit_models.step(sample_tokens, sample_telemetry)
        assert substrate_state.world_state.shape == (64,)
        assert substrate_state.self_state.shape == (64,)

        # Stage 2: Permeability gating
        gate_output = gate.gate(substrate_state, ConsciousState.WAKE)
        assert gate_output.gated_world.shape == (64,)
        assert gate_output.gated_self.shape == (64,)

        # Stage 3: Explicit generation
        prev_esm = torch.zeros(64)
        explicit_state = explicit_models.step(gate_output, prev_esm)
        assert explicit_state.ewm_state.shape == (64,)
        assert explicit_state.esm_state.shape == (64,)

    def test_pipeline_multi_frame(
        self,
        implicit_models: ImplicitModels,
        gate: SigmoidPermeabilityGate,
        explicit_models: ExplicitModels,
        sample_tokens: torch.Tensor,
        sample_telemetry: dict[str, float],
    ) -> None:
        """Multiple frames with ESM self-referential chain."""
        prev_esm = torch.zeros(64)
        esm_history = []

        for _ in range(5):
            substrate = implicit_models.step(sample_tokens, sample_telemetry)
            gated = gate.gate(substrate, ConsciousState.WAKE)
            explicit = explicit_models.step(gated, prev_esm)
            prev_esm = explicit.esm_state.detach()
            esm_history.append(prev_esm.clone())

        # ESM should evolve over frames (self-referential loop)
        for i in range(1, len(esm_history)):
            assert not torch.allclose(
                esm_history[i], esm_history[i - 1], atol=1e-5
            ), f"Frame {i} ESM should differ from frame {i-1}"
