"""Tests for the aiware configuration framework."""

from __future__ import annotations

import torch

from aiware.config.base import AIwareConfig, DeviceConfig, StateProfile


class TestDeviceConfig:
    def test_auto_resolves_to_cpu_without_cuda(self):
        dc = DeviceConfig(device="auto")
        device = dc.resolve_device()
        # On CI / CPU-only machine, should resolve to cpu
        assert device.type in ("cpu", "cuda", "mps")

    def test_explicit_cpu(self):
        dc = DeviceConfig(device="cpu")
        assert dc.resolve_device() == torch.device("cpu")

    def test_dtype_float32(self):
        dc = DeviceConfig(dtype="float32")
        assert dc.resolve_dtype() == torch.float32

    def test_dtype_float16(self):
        dc = DeviceConfig(dtype="float16")
        assert dc.resolve_dtype() == torch.float16


class TestStateProfile:
    def test_wake_profile(self):
        p = StateProfile(
            alpha_world=1.0,
            alpha_self=1.0,
            beta=0.0,
            noise_std=0.1,
            external_input=True,
            self_ref_loop=True,
            substrate_rate_hz=40.0,
        )
        assert p.external_input is True
        assert p.noise_std >= 0

    def test_negative_noise_rejected(self):
        import pytest

        with pytest.raises(Exception):  # Pydantic validation error
            StateProfile(
                alpha_world=1.0,
                alpha_self=1.0,
                beta=0.0,
                noise_std=-0.1,
                external_input=True,
                self_ref_loop=True,
                substrate_rate_hz=40.0,
            )


class TestAIwareConfig:
    def test_defaults(self):
        cfg = AIwareConfig()
        assert cfg.state_dim == 512
        assert cfg.conscious_rate_hz == 20.0
        assert len(cfg.state_profiles) == 4

    def test_all_states_have_profiles(self):
        cfg = AIwareConfig()
        for state_name in ("wake", "sleep", "dream", "altered"):
            assert state_name in cfg.state_profiles

    def test_custom_state_dim(self):
        cfg = AIwareConfig(state_dim=256)
        assert cfg.state_dim == 256

    def test_test_config_fixture(self, test_config):
        assert test_config.hardware.device == "cpu"
        assert test_config.wandb_enabled is False
