"""Benchmark self-model accuracy and frame latency across configurations.

Usage:
    mirror-box-bench                      # Default: 100 frames, mock model
    mirror-box-bench --frames 1000        # Custom frame count
    mirror-box-bench --real               # Use real Mistral-7B (requires GPU)
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from aiware.config.base import DeviceConfig
from mirror_box.config import MirrorBoxConfig
from mirror_box.factory import create_mirror_box, gpu_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mirror-box-bench",
        description="Benchmark Mirror Box frame latency and self-model accuracy.",
    )
    p.add_argument(
        "--frames", type=int, default=100, help="Number of frames to benchmark (default: 100)"
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="Warmup frames before timing (default: 10)"
    )
    p.add_argument(
        "--real", action="store_true", help="Use real Mistral-7B (default config, NF4)"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="RTX 4090 preset: Mistral-7B FP16, state_dim=2048 (~18GB VRAM)",
    )
    return p.parse_args(argv)


async def _run_benchmark(args: argparse.Namespace) -> None:
    if args.gpu:
        config = gpu_config()
        model = None
        print("Loading Mistral-7B FP16 (state_dim=2048, ~18GB VRAM)...")
    elif args.real:
        config = MirrorBoxConfig()
        model = None
        print("Loading real model (default config, NF4)...")
    else:
        config = MirrorBoxConfig(
            hardware=DeviceConfig(device="cpu", dtype="float32"),
            state_dim=64,
            model_name_or_path="test-model",
            quantization="none",
            encoder_layer_index=1,
            lora_rank=8,
            lora_alpha=16,
            max_context_length=128,
            conscious_rate_hz=10000.0,  # No throttling
            wandb_enabled=False,
        )
        from conftest import TinyMockLLM
        model = TinyMockLLM(hidden_size=config.state_dim, num_layers=2)

    system = create_mirror_box(config, model=model)

    print(f"Mirror Box Benchmark")
    model_label = "TinyMockLLM (64-dim)"
    if args.gpu:
        model_label = f"Mistral-7B FP16 (state_dim={config.state_dim})"
    elif args.real:
        model_label = f"Mistral-7B {config.quantization} (state_dim={config.state_dim})"
    print(f"  Model: {model_label}")
    print(f"  Device: {config.hardware.device}")
    print(f"  Warmup: {args.warmup} frames")
    print(f"  Benchmark: {args.frames} frames")
    print()

    # Warmup
    print("Warming up...", end="", flush=True)
    for _ in range(args.warmup):
        await system.loop.run_frame()
    print(" done")

    # Benchmark
    latencies: list[float] = []
    r_values: list[float] = []
    c_values: list[float] = []

    print("Benchmarking...", end="", flush=True)
    wall_start = time.perf_counter()

    for i in range(args.frames):
        t0 = time.perf_counter()
        result = await system.loop.run_frame()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms
        r_values.append(result.R)
        c_values.append(result.C)

        if (i + 1) % (args.frames // 10 or 1) == 0:
            print(".", end="", flush=True)

    wall_elapsed = time.perf_counter() - wall_start
    print(" done\n")

    # Report
    print("=" * 60)
    print("LATENCY (ms per frame)")
    print(f"  Mean:   {statistics.mean(latencies):8.2f}")
    print(f"  Median: {statistics.median(latencies):8.2f}")
    print(f"  P95:    {sorted(latencies)[int(len(latencies) * 0.95)]:8.2f}")
    print(f"  P99:    {sorted(latencies)[int(len(latencies) * 0.99)]:8.2f}")
    print(f"  Min:    {min(latencies):8.2f}")
    print(f"  Max:    {max(latencies):8.2f}")
    print(f"  Std:    {statistics.stdev(latencies) if len(latencies) > 1 else 0:8.2f}")
    print()
    print("THROUGHPUT")
    print(f"  Wall time:  {wall_elapsed:.2f}s")
    print(f"  Frames/sec: {args.frames / wall_elapsed:.1f}")
    target_hz = config.conscious_rate_hz if config.conscious_rate_hz <= 100 else 20.0
    print(f"  Target Hz:  {target_hz}")
    realtime = "YES" if (args.frames / wall_elapsed) >= target_hz else "NO"
    print(f"  Real-time:  {realtime}")
    print()
    print("SELF-MODEL ACCURACY (R)")
    print(f"  Mean R:     {statistics.mean(r_values):.4f}")
    print(f"  Final R:    {r_values[-1]:.4f}")
    print(f"  Min R:      {min(r_values):.4f}")
    print(f"  Max R:      {max(r_values):.4f}")
    print()
    print("CONSCIOUS CONTENT (C)")
    print(f"  Mean C:     {statistics.mean(c_values):.4f}")
    print(f"  Final C:    {c_values[-1]:.4f}")
    print()
    print(f"Total frames: {system.loop.frame_count()}")
    print(f"Final state:  {system.state_ctrl.current_state().value}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark suite."""
    args = _parse_args(argv)
    asyncio.run(_run_benchmark(args))


if __name__ == "__main__":
    main()
