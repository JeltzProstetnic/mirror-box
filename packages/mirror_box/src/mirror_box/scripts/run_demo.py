"""CLI entry point for Mirror Box demo scenarios.

Usage:
    mirror-box-demo --gpu                 # RTX 4090 preset (FP16, state_dim=2048)
    mirror-box-demo --gpu --frames 500    # Longer run
    mirror-box-demo --gpu --dashboard     # With Gradio dashboard
    mirror-box-demo --mock                # TinyMockLLM (no GPU needed)
    mirror-box-demo --scenario identity   # Identity disruption
    mirror-box-demo --scenario altered    # Altered state exploration
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from aiware.config.base import DeviceConfig
from aiware.interfaces.data_types import ConsciousState
from mirror_box.config import MirrorBoxConfig
from mirror_box.factory import create_mirror_box, gpu_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mirror-box-demo",
        description="Run Mirror Box consciousness demo scenarios.",
    )
    p.add_argument(
        "--scenario",
        choices=["wake-sleep", "identity", "altered"],
        default="wake-sleep",
        help="Demo scenario to run (default: wake-sleep)",
    )
    p.add_argument(
        "--frames", type=int, default=100, help="Number of frames to run (default: 100)"
    )
    p.add_argument(
        "--dashboard", action="store_true", help="Launch Gradio dashboard after run"
    )
    p.add_argument(
        "--mock", action="store_true", help="Use TinyMockLLM instead of real model"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="RTX 4090 preset: Mistral-7B FP16, state_dim=2048 (~18GB VRAM)",
    )
    p.add_argument(
        "--state-dim", type=int, default=None, help="State dimension (default: 512, or 64 for --mock)"
    )
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> MirrorBoxConfig:
    if args.mock:
        state_dim = args.state_dim or 64
        return MirrorBoxConfig(
            hardware=DeviceConfig(device="cpu", dtype="float32"),
            state_dim=state_dim,
            model_name_or_path="test-model",
            quantization="none",
            encoder_layer_index=1,
            lora_rank=8,
            lora_alpha=16,
            max_context_length=128,
            conscious_rate_hz=100.0,
            continuous_wake_frames=args.frames // 2,
            sleep_duration_frames=args.frames // 6,
            dream_duration_frames=args.frames // 8,
            altered_duration_frames=args.frames // 8,
            wandb_enabled=False,
        )
    if args.gpu:
        overrides = {}
        if args.state_dim is not None:
            overrides["state_dim"] = args.state_dim
        return gpu_config(**overrides)
    state_dim = args.state_dim or 512
    return MirrorBoxConfig(state_dim=state_dim)


async def _run_wake_sleep(system, frames: int) -> None:
    """Default scenario: natural wake-sleep-dream cycle."""
    import torch

    # Build a small corpus of sensory "scenes" — the system dwells on each
    # for multiple frames, allowing the self-predictor to learn the pattern,
    # then switches (surprise spike, R dip, then recovery).
    vocab_size = 32000
    n_scenes = max(5, frames // 20)  # switch scene every ~20 frames
    scenes = [torch.randint(1, vocab_size, (8,)) for _ in range(n_scenes)]

    print(f"Running wake-sleep cycle for {frames} frames ({n_scenes} scenes)...")
    for i in range(frames):
        scene_idx = min(i // 20, n_scenes - 1)
        system.loop.set_sensory_input(scenes[scene_idx])
        result = await system.loop.run_frame()
        if i % 10 == 0:
            state = system.state_ctrl.current_state()
            loss = result.extras.get("prediction_loss", 0.0)
            print(
                f"  Frame {result.frame_number:4d} | "
                f"State: {state.value:8s} | "
                f"R: {result.R:.3f} | "
                f"C: {result.C:.3f} | "
                f"Surprise: {result.self_surprise:.3f} | "
                f"Loss: {loss:.4f}"
            )
    print()
    print(system.reporter.generate_state_report())


async def _run_identity(system, frames: int) -> None:
    """Identity disruption: force altered state to test ESM redirection."""
    print(f"Running identity disruption for {frames} frames...")
    third = max(frames // 3, 1)

    # Phase 1: Normal wake
    print("Phase 1: Normal wake...")
    await system.loop.run(max_frames=third)
    print(f"  R after wake: {system.metrics_collector.get_latest().R:.3f}")
    print(f"  ESM narrative: {system.reporter.generate_esm_narrative()}")

    # Phase 2: Force altered state (increased self-permeability)
    print("\nPhase 2: Altered state (identity disruption)...")
    system.state_ctrl.transition_to(ConsciousState.ALTERED)
    await system.loop.run(max_frames=third)
    print(f"  R after altered: {system.metrics_collector.get_latest().R:.3f}")
    print(f"  ESM narrative: {system.reporter.generate_esm_narrative()}")

    # Phase 3: Return to wake
    print("\nPhase 3: Recovery (back to wake)...")
    system.state_ctrl.transition_to(ConsciousState.WAKE)
    await system.loop.run(max_frames=third)
    print(f"  R after recovery: {system.metrics_collector.get_latest().R:.3f}")
    print(f"  ESM narrative: {system.reporter.generate_esm_narrative()}")


async def _run_altered(system, frames: int) -> None:
    """Explore altered states: cycle through all states with introspection."""
    print(f"Running altered state exploration for {frames} frames...")
    quarter = max(frames // 4, 1)

    for state in [ConsciousState.WAKE, ConsciousState.ALTERED, ConsciousState.SLEEP, ConsciousState.DREAM]:
        print(f"\n--- {state.value.upper()} ---")
        system.state_ctrl.transition_to(state)
        await system.loop.run(max_frames=quarter)
        latest = system.metrics_collector.get_latest()
        if latest:
            print(f"  R: {latest.R:.3f} | C: {latest.C:.3f} | Surprise: {latest.self_surprise:.3f}")
        print(f"  Self-report: {system.reporter.generate_state_report()}")


async def _async_main(args: argparse.Namespace) -> None:
    config = _build_config(args)

    model = None
    if args.mock:
        from conftest import TinyMockLLM
        model = TinyMockLLM(hidden_size=config.state_dim, num_layers=2)

    mode = "mock" if args.mock else ("gpu" if args.gpu else "default")
    print(f"Assembling Mirror Box (mode={mode})...")
    system = create_mirror_box(config, model=model)
    print(f"  State dim: {config.state_dim}")
    print(f"  Device: {config.hardware.device}")
    print(f"  Quantization: {config.quantization}")
    print()

    scenarios = {
        "wake-sleep": _run_wake_sleep,
        "identity": _run_identity,
        "altered": _run_altered,
    }
    await scenarios[args.scenario](system, args.frames)

    if args.dashboard:
        try:
            from mirror_box.visualization import get_dashboard_class
            DashboardClass = get_dashboard_class()
            dashboard = DashboardClass(
                system.loop, system.metrics_collector, system.state_ctrl, system.reporter
            )
            print("\nLaunching dashboard...")
            dashboard.launch(share=False)
        except ImportError:
            print("\nDashboard requires gradio: pip install 'mirror-box[dashboard]'", file=sys.stderr)
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Run a Mirror Box demo scenario."""
    args = _parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
