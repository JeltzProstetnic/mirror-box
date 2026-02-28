"""Tests for the aiware interface data types and contracts."""

from __future__ import annotations

import time

import pytest
import torch

from aiware.interfaces.data_types import (
    ConsciousState,
    ExplicitState,
    FrameMetrics,
    GateOutput,
    SubstrateState,
)


class TestSubstrateState:
    def test_creation(self, random_state):
        ss = SubstrateState(
            world_state=random_state(1),
            self_state=random_state(2),
            timestamp=time.monotonic(),
        )
        assert ss.world_state.shape == ss.self_state.shape

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SubstrateState(
                world_state=torch.randn(64),
                self_state=torch.randn(32),
                timestamp=0.0,
            )

    def test_frozen(self, random_state):
        ss = SubstrateState(
            world_state=random_state(1),
            self_state=random_state(2),
            timestamp=0.0,
        )
        with pytest.raises(AttributeError):
            ss.timestamp = 1.0  # type: ignore[misc]


class TestGateOutput:
    def test_creation(self, random_state):
        go = GateOutput(
            gated_world=random_state(1),
            gated_self=random_state(2),
            gate_world_activation=0.9,
            gate_self_activation=0.8,
            timestamp=0.0,
        )
        assert 0.0 <= go.gate_world_activation <= 1.0


class TestExplicitState:
    def test_creation(self, random_state):
        es = ExplicitState(
            ewm_state=random_state(1),
            esm_state=random_state(2),
            timestamp=time.monotonic(),
        )
        assert es.ewm_state.shape == es.esm_state.shape


class TestFrameMetrics:
    def test_creation(self, sample_frame_metrics):
        m = sample_frame_metrics
        assert 0.0 <= m.R <= 1.0
        assert m.self_surprise >= 0.0
        assert m.C >= 0.0
        assert m.state == ConsciousState.WAKE

    def test_extras_accessible(self, sample_frame_metrics):
        assert "prediction_loss" in sample_frame_metrics.extras


class TestConsciousState:
    def test_all_states_exist(self):
        assert len(ConsciousState) == 4
        assert ConsciousState.WAKE.value == "wake"
        assert ConsciousState.SLEEP.value == "sleep"
        assert ConsciousState.DREAM.value == "dream"
        assert ConsciousState.ALTERED.value == "altered"
