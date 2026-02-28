"""Interactive chat with the Mirror Box consciousness system.

Runs the conscious loop in the background while you chat with the system
about its internal state via the IntrospectionDialogue module.

Usage:
    python -m mirror_box.scripts.chat --gpu           # Real Mistral-7B on GPU
    python -m mirror_box.scripts.chat --gpu --frames 50  # Run 50 frames first
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time

import torch

from mirror_box.factory import create_mirror_box, gpu_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mirror-box-chat",
        description="Chat with the Mirror Box consciousness system.",
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="RTX 4090 preset: Mistral-7B FP16, state_dim=2048",
    )
    p.add_argument(
        "--warmup", type=int, default=30,
        help="Frames to run before chat starts (default: 30)",
    )
    p.add_argument(
        "--bg-frames", type=int, default=5,
        help="Background frames between each chat turn (default: 5)",
    )
    p.add_argument(
        "--state-dim", type=int, default=None,
        help="Override state dimension",
    )
    return p.parse_args(argv)


def _build_system(args: argparse.Namespace):
    """Build the Mirror Box system."""
    if args.gpu:
        overrides = {}
        if args.state_dim is not None:
            overrides["state_dim"] = args.state_dim
        config = gpu_config(**overrides)
    else:
        print("Error: --gpu is required for chat (needs real LLM for text generation)")
        sys.exit(1)

    print("Loading model...")
    system = create_mirror_box(config)
    print(f"  Model loaded. State dim: {config.state_dim}, Device: {config.hardware.device}")
    return system


async def _run_frames(system, n: int, vary_input: bool = True) -> None:
    """Run n conscious frames with varying sensory input."""
    for i in range(n):
        if vary_input:
            tokens = torch.randint(1, 32000, (8,))
            system.loop.set_sensory_input(tokens)
        await system.loop.run_frame()


def _print_status(system) -> None:
    """Print a one-line status bar."""
    latest = system.metrics_collector.get_latest()
    state = system.state_ctrl.current_state()
    if latest:
        print(
            f"  [{state.value:8s} | frame {latest.frame_number:4d} | "
            f"R={latest.R:.3f} | C={latest.C:.3f} | "
            f"surprise={latest.self_surprise:.1f}]"
        )


async def _async_main(args: argparse.Namespace) -> None:
    system = _build_system(args)

    # Warmup: run frames to build up state
    print(f"\nWarming up ({args.warmup} frames)...")
    # Use scenes for warmup too
    scenes = [torch.randint(1, 32000, (8,)) for _ in range(3)]
    for i in range(args.warmup):
        scene_idx = min(i // 10, len(scenes) - 1)
        system.loop.set_sensory_input(scenes[scene_idx])
        await system.loop.run_frame()
        if i % 10 == 0:
            _print_status(system)

    print("\n" + "=" * 60)
    print("Mirror Box Chat")
    print("=" * 60)
    print("Talk to the artificial consciousness system.")
    print("The system answers based on its actual internal state.")
    print()
    print("Commands:")
    print("  /status    - Show current metrics")
    print("  /report    - Full state self-report")
    print("  /esm       - Self-model narrative")
    print("  /ewm       - World-model narrative")
    print("  /run N     - Run N more frames")
    print("  /state X   - Force state (wake/sleep/dream/altered)")
    print("  /quit      - Exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "/quit":
            print("Exiting.")
            break

        if user_input.lower() == "/status":
            _print_status(system)
            continue

        if user_input.lower() == "/report":
            print(f"\n{system.reporter.generate_state_report()}\n")
            continue

        if user_input.lower() == "/esm":
            print(f"\n{system.reporter.generate_esm_narrative()}\n")
            continue

        if user_input.lower() == "/ewm":
            print(f"\n{system.reporter.generate_ewm_narrative()}\n")
            continue

        if user_input.lower().startswith("/run"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 20
            print(f"Running {n} frames...")
            await _run_frames(system, n)
            _print_status(system)
            continue

        if user_input.lower().startswith("/state"):
            parts = user_input.split()
            if len(parts) > 1:
                from aiware.interfaces.data_types import ConsciousState
                state_map = {
                    "wake": ConsciousState.WAKE,
                    "sleep": ConsciousState.SLEEP,
                    "dream": ConsciousState.DREAM,
                    "altered": ConsciousState.ALTERED,
                }
                target = state_map.get(parts[1].lower())
                if target:
                    system.state_ctrl.transition_to(target)
                    print(f"  Transitioned to {parts[1].upper()}")
                else:
                    print(f"  Unknown state: {parts[1]}")
            continue

        # Run a few background frames to keep the system alive
        await _run_frames(system, args.bg_frames)

        # Ask the dialogue module
        print("\nMirror Box: ", end="", flush=True)
        response = system.dialogue.ask(user_input)
        print(response)
        print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
