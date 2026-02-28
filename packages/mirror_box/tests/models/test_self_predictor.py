"""Tests for the MLPSelfPredictor — standalone MLP, CPU only, no LLM needed."""

from __future__ import annotations

import pytest
import torch

from aiware.config.base import DeviceConfig
from mirror_box.config import MirrorBoxConfig
from mirror_box.models.self_predictor import MLPSelfPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def predictor_config(state_dim: int) -> MirrorBoxConfig:
    """MirrorBoxConfig tuned for CPU testing with small state_dim."""
    return MirrorBoxConfig(
        hardware=DeviceConfig(device="cpu", dtype="float32"),
        state_dim=state_dim,
        quantization="none",
        self_predictor_hidden_size=128,
        self_predictor_lr=1e-3,
        wandb_enabled=False,
    )


@pytest.fixture
def predictor(predictor_config: MirrorBoxConfig) -> MLPSelfPredictor:
    """A fresh MLPSelfPredictor instance."""
    return MLPSelfPredictor(predictor_config)


# ---------------------------------------------------------------------------
# Shape and basic behaviour
# ---------------------------------------------------------------------------


class TestPredictShape:
    """predict() output shape and basic properties."""

    def test_output_shape(
        self, predictor: MLPSelfPredictor, random_state, state_dim: int
    ) -> None:
        """predict() returns a tensor of shape (state_dim,)."""
        prev_esm = random_state(seed=10)
        gated_self = random_state(seed=11)
        out = predictor.predict(prev_esm, gated_self)
        assert out.shape == (state_dim,)

    def test_output_dtype(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """predict() output is float32."""
        out = predictor.predict(random_state(seed=10), random_state(seed=11))
        assert out.dtype == torch.float32

    def test_different_inputs_give_different_outputs(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """Different inputs produce different predictions."""
        out_a = predictor.predict(random_state(seed=10), random_state(seed=11))
        out_b = predictor.predict(random_state(seed=20), random_state(seed=21))
        assert not torch.allclose(out_a, out_b)

    def test_deterministic_for_same_inputs(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """Same inputs produce identical outputs (no stochasticity in inference)."""
        prev = random_state(seed=10)
        gated = random_state(seed=11)
        out_1 = predictor.predict(prev, gated)
        out_2 = predictor.predict(prev, gated)
        assert torch.allclose(out_1, out_2)

    def test_no_grad_on_output(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """predict() output has no gradient graph (computed under no_grad)."""
        out = predictor.predict(random_state(seed=10), random_state(seed=11))
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# Metric delegation
# ---------------------------------------------------------------------------


class TestMetrics:
    """compute_R and compute_surprise delegate correctly."""

    def test_R_in_range(self, predictor: MLPSelfPredictor, random_state) -> None:
        """R is always in [0, 1]."""
        pred = random_state(seed=1)
        actual = random_state(seed=2)
        r = predictor.compute_R(pred, actual)
        assert 0.0 <= r <= 1.0

    def test_R_high_for_similar(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """R is high when predicted ~= actual."""
        actual = random_state(seed=1)
        # Small perturbation
        predicted = actual + torch.randn_like(actual) * 0.01
        r = predictor.compute_R(predicted, actual)
        assert r > 0.9

    def test_R_low_for_different(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """R is low when predicted and actual are very different."""
        predicted = random_state(seed=1)
        actual = random_state(seed=99) * 10.0  # Scale up for large divergence
        r = predictor.compute_R(predicted, actual)
        assert r < 0.5

    def test_surprise_non_negative(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """Surprise is always >= 0."""
        pred = random_state(seed=1)
        actual = random_state(seed=2)
        s = predictor.compute_surprise(pred, actual)
        assert s >= 0.0

    def test_surprise_zero_when_equal(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """Surprise is zero when predicted == actual."""
        v = random_state(seed=1)
        s = predictor.compute_surprise(v, v.clone())
        assert s == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Online training (update)
# ---------------------------------------------------------------------------


class TestUpdate:
    """update() trains the MLP online."""

    def test_returns_positive_loss(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """update() returns a loss > 0 for non-identical pred/actual."""
        prev_esm = random_state(seed=10)
        gated_self = random_state(seed=11)
        actual = random_state(seed=12)

        predicted = predictor.predict(prev_esm, gated_self)
        loss = predictor.update(predicted, actual)
        assert loss > 0.0

    def test_loss_decreases_after_update(
        self, predictor: MLPSelfPredictor, random_state
    ) -> None:
        """A single update step should reduce the loss for the same input."""
        prev_esm = random_state(seed=10)
        gated_self = random_state(seed=11)
        actual = random_state(seed=12)

        # First prediction + update
        pred_before = predictor.predict(prev_esm, gated_self)
        loss_before = predictor.update(pred_before, actual)

        # Second prediction on the same input (after one gradient step)
        pred_after = predictor.predict(prev_esm, gated_self)
        loss_after = predictor.update(pred_after, actual)

        assert loss_after < loss_before

    def test_training_loop_R_increases(
        self, predictor: MLPSelfPredictor, state_dim: int
    ) -> None:
        """Over 100 updates with correlated data, R should increase."""
        torch.manual_seed(42)

        # Generate a fixed linear mapping as the "true" ESM dynamics:
        # actual = W @ concat(prev_esm, gated_self) + b
        W = torch.randn(state_dim, state_dim * 2) * 0.1
        b = torch.randn(state_dim) * 0.01

        def true_dynamics(prev_esm: torch.Tensor, gated_self: torch.Tensor) -> torch.Tensor:
            x = torch.cat([prev_esm, gated_self], dim=-1)
            return W @ x + b

        # Measure R at start
        prev = torch.randn(state_dim)
        gated = torch.randn(state_dim)
        pred_initial = predictor.predict(prev, gated)
        actual_initial = true_dynamics(prev, gated)
        r_initial = predictor.compute_R(pred_initial, actual_initial)

        # Train for 100 steps
        for _ in range(100):
            prev = torch.randn(state_dim)
            gated = torch.randn(state_dim)
            actual = true_dynamics(prev, gated)
            predicted = predictor.predict(prev, gated)
            predictor.update(predicted, actual)

        # Measure R at end (average over several samples for stability)
        r_values = []
        for seed in range(200, 210):
            torch.manual_seed(seed)
            prev = torch.randn(state_dim)
            gated = torch.randn(state_dim)
            pred = predictor.predict(prev, gated)
            act = true_dynamics(prev, gated)
            r_values.append(predictor.compute_R(pred, act))

        r_final = sum(r_values) / len(r_values)
        assert r_final > r_initial, (
            f"R should increase with training: initial={r_initial:.4f}, final={r_final:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and robustness."""

    def test_zero_vectors_no_nan(
        self, predictor: MLPSelfPredictor, zero_state
    ) -> None:
        """Zero input vectors should not produce NaN outputs."""
        out = predictor.predict(zero_state, zero_state)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_zero_vectors_R(
        self, predictor: MLPSelfPredictor, zero_state
    ) -> None:
        """compute_R with zero vectors should not produce NaN."""
        r = predictor.compute_R(zero_state, zero_state)
        assert not (r != r)  # NaN check: NaN != NaN is True

    def test_zero_vectors_surprise(
        self, predictor: MLPSelfPredictor, zero_state
    ) -> None:
        """compute_surprise with zero vectors should return 0."""
        s = predictor.compute_surprise(zero_state, zero_state)
        assert s == pytest.approx(0.0, abs=1e-7)

    def test_update_with_zero_vectors(
        self, predictor: MLPSelfPredictor, zero_state
    ) -> None:
        """update() with zero vectors should not crash or produce NaN."""
        pred = predictor.predict(zero_state, zero_state)
        loss = predictor.update(pred, zero_state)
        assert not (loss != loss)  # NaN check

    def test_state_dim_property(
        self, predictor: MLPSelfPredictor, state_dim: int
    ) -> None:
        """state_dim property returns the configured value."""
        assert predictor.state_dim == state_dim

    def test_is_nn_module(self, predictor: MLPSelfPredictor) -> None:
        """MLPSelfPredictor is a proper nn.Module (parameters are trackable)."""
        params = list(predictor.parameters())
        assert len(params) > 0
        # LayerNorm (weight + bias) + 3 Linear layers x 2 (weight + bias) = 8
        assert len(params) == 8
