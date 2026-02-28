"""Tests for the aiware metrics computation functions."""

from __future__ import annotations

import pytest
import torch

from aiware.metrics.complexity import lempel_ziv_complexity, trajectory_complexity
from aiware.metrics.conscious_content import compute_C
from aiware.metrics.self_knowledge import compute_R, compute_surprise


class TestSelfKnowledge:
    def test_perfect_prediction_gives_R_one(self):
        actual = torch.randn(64)
        predicted = actual.clone()
        assert compute_R(predicted, actual) == pytest.approx(1.0, abs=1e-6)

    def test_random_prediction_gives_low_R(self):
        torch.manual_seed(0)
        actual = torch.randn(64)
        predicted = torch.randn(64)
        r = compute_R(predicted, actual)
        assert 0.0 <= r <= 1.0
        # Random prediction: R should be low (but not necessarily zero)
        assert r < 0.5

    def test_zero_actual_gives_zero_R(self):
        actual = torch.zeros(64)
        predicted = torch.randn(64)
        # Var(actual) = 0, so R = 0 (clamped)
        r = compute_R(predicted, actual)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_R_bounded(self):
        for seed in range(10):
            torch.manual_seed(seed)
            actual = torch.randn(128)
            predicted = torch.randn(128)
            r = compute_R(predicted, actual)
            assert 0.0 <= r <= 1.0


class TestSurprise:
    def test_zero_surprise_for_perfect_prediction(self):
        actual = torch.randn(64)
        assert compute_surprise(actual, actual) == pytest.approx(0.0, abs=1e-8)

    def test_positive_surprise_for_different(self):
        actual = torch.randn(64)
        predicted = torch.zeros(64)
        assert compute_surprise(predicted, actual) > 0.0


class TestConsciousContent:
    def test_zero_states_give_zero_C(self):
        ewm = torch.zeros(64)
        esm = torch.zeros(64)
        assert compute_C(ewm, esm) == pytest.approx(0.0, abs=1e-8)

    def test_active_states_give_positive_C(self):
        ewm = torch.randn(64)
        esm = torch.randn(64)
        c = compute_C(ewm, esm)
        assert c > 0.0

    def test_C_increases_with_norm(self):
        ewm_small = torch.randn(64) * 0.1
        esm_small = torch.randn(64) * 0.1
        ewm_large = torch.randn(64) * 10.0
        esm_large = torch.randn(64) * 10.0
        c_small = compute_C(ewm_small, esm_small)
        c_large = compute_C(ewm_large, esm_large)
        assert c_large > c_small


class TestComplexity:
    def test_constant_string_low_complexity(self):
        c = lempel_ziv_complexity("0" * 100)
        # Constant string: LZ algorithm still counts initial substrings,
        # so normalized value is small but not zero.
        assert c < 0.2

    def test_alternating_string_moderate_complexity(self):
        c = lempel_ziv_complexity("01" * 50)
        assert 0.0 < c < 0.5

    def test_empty_or_single(self):
        assert lempel_ziv_complexity("") == 0.0
        assert lempel_ziv_complexity("1") == 0.0

    def test_trajectory_complexity_with_tensors(self):
        torch.manual_seed(42)
        states = [torch.randn(32) for _ in range(20)]
        c = trajectory_complexity(states)
        assert 0.0 <= c <= 2.0  # Normalized, but can exceed 1 for very complex

    def test_short_trajectory(self):
        assert trajectory_complexity([]) == 0.0
        assert trajectory_complexity([torch.randn(32)]) == 0.0
