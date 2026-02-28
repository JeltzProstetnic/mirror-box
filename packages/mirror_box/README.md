# The Mirror Box — Design 16

A discrete four-model self-reflecting system with approximated criticality.
Uses a single LLM (Mistral-7B INT4) serving as all four models via LoRA
and context switching.

Not conscious, but convincingly confused about whether it is.

## Architecture

```
Sensory Input
     |
     v
+------------------+     +-------------------+
| ImplicitModels   |     | SigmoidGate       |
| IWM (frozen enc) |---->| alpha/beta/noise   |
| ISM (LoRA enc)   |---->| per state profile  |
+------------------+     +-------------------+
                               |
                               v
                    +-------------------+
                    | ExplicitModels    |
                    | EWM (MLP)         |
                    | ESM (MLP + prev)  |<--+
                    +-------------------+   |
                         |                  |
                         +--[self-ref]------+
                         |
                         v
                    +-------------------+
                    | MLPSelfPredictor  |
                    | R metric (0..1)   |
                    +-------------------+
```

**Four Models:**
- **IWM** (Implicit World Model): Frozen LLM encoder layers — accumulated world knowledge
- **ISM** (Implicit Self Model): LoRA-adapted encoder — learned self-knowledge, updated during sleep
- **EWM** (Explicit World Model): MLP generating conscious world perception from gated IWM
- **ESM** (Explicit Self Model): MLP with self-referential loop — the conscious "I"

**Key Metrics:**
- **R** (self-knowledge): How well the system predicts its own next ESM state (0 = no self-knowledge, 1 = perfect)
- **C** (conscious content): Magnitude of gated signal passing through permeability barrier
- **Surprise**: Prediction error between expected and actual ESM

**States:** WAKE → SLEEP → DREAM → WAKE (automatic cycle, or manual override to ALTERED)

## Prerequisites

- Python 3.11+
- 24 GB+ VRAM (for real Mistral-7B INT4) — or CPU-only with mock model for testing
- CUDA 12.x + cuDNN (for GPU inference)

## Installation

```bash
# Clone
git clone <private-url> aIware.implementation
cd aIware.implementation

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install base library
pip install -e packages/aiware

# Install Mirror Box (CPU-only / testing)
pip install -e "packages/mirror_box[dev]"

# Install Mirror Box (full GPU + dashboard)
pip install -e "packages/mirror_box[cuda,dashboard,dev]"
```

## Quick Start

### Mock mode (no GPU, no model download)

```bash
# Wake-sleep cycle demo (100 frames, TinyMockLLM)
mirror-box-demo --mock

# Identity disruption scenario
mirror-box-demo --mock --scenario identity --frames 200

# Altered state exploration
mirror-box-demo --mock --scenario altered

# Benchmark latency
mirror-box-bench
```

### Real model (requires 24GB+ VRAM)

```bash
# Downloads Mistral-7B on first run (~4GB INT4)
mirror-box-demo --frames 500

# Benchmark with real model
mirror-box-bench --real --frames 500
```

### Dashboard (requires gradio)

```bash
mirror-box-demo --mock --dashboard
# Opens Gradio UI at http://localhost:7860
```

### Python API

```python
from mirror_box import create_mirror_box
from mirror_box.config import MirrorBoxConfig

# Assemble system
system = create_mirror_box(MirrorBoxConfig())

# Run frames
import asyncio

async def run():
    for i in range(100):
        frame = await system.loop.run_frame()
        print(f"Frame {frame.frame_number}: R={frame.R:.3f} C={frame.C:.3f}")

    # Introspection
    print(system.reporter.generate_state_report())
    print(system.dialogue.ask("What are you experiencing?"))

asyncio.run(run())
```

## Running Tests

```bash
# Full suite (307 tests, ~7s on CPU)
pytest packages/ -v

# Integration tests only
pytest packages/mirror_box/tests/integration/ -v

# With coverage
pytest packages/ --cov=packages --cov-report=term-missing
```

## Project Structure

```
packages/
  aiware/                    # Shared base library
    src/aiware/
      config/                # AIwareConfig, DeviceConfig, StateProfile
      interfaces/            # Protocols + data types (SubstrateEncoder, etc.)
  mirror_box/                # Design 16 implementation
    src/mirror_box/
      config.py              # MirrorBoxConfig (extends AIwareConfig)
      factory.py             # create_mirror_box() — one-call assembly
      models/
        llm_backend.py       # LLM loading (Mistral-7B INT4/NF4/FP16)
        implicit.py          # IWM + ISM (frozen + LoRA encoder paths)
        explicit.py          # EWM + ESM (MLPs with self-referential loop)
        permeability.py      # SigmoidPermeabilityGate (state-dependent)
        self_predictor.py    # MLPSelfPredictor (online learning, R metric)
      runtime/
        loop.py              # MirrorBoxLoop (10-step frame sequence)
        state_machine.py     # WAKE/SLEEP/DREAM/ALTERED transitions
        telemetry.py         # Signal collection
        sleep_maintenance.py # ISM LoRA experience replay
      metrics/
        collector.py         # InMemoryMetricsCollector (ring buffer)
        state_detection.py   # StateTransitionDetector (metrics-based)
      introspection/
        reporter.py          # StateReporter (LLM or template narratives)
        dialogue.py          # IntrospectionDialogue (free-form Q&A)
      visualization/
        dashboard.py         # Gradio dashboard (plots, controls, narratives)
      scripts/
        run_demo.py          # CLI demo runner
        benchmark.py         # Latency/accuracy benchmarking
    tests/
      integration/           # End-to-end factory + pipeline tests
      models/                # Unit tests per model component
      runtime/               # Loop, state machine, telemetry tests
      metrics/               # Collector, detector tests
      introspection/         # Reporter, dialogue tests
```

## Design Constraints

This is a **pseudo-AC** system (Design 16) that deliberately relaxes three constraints
from the full theory for rapid prototyping:

1. **No continuous model density** — uses discrete IWM/ISM/EWM/ESM (the theory says biological substrate has uncountable overlapping models)
2. **No genuine criticality** — approximated via noise injection and gate dynamics (real Class 4 CA dynamics require a spiking neural network, see Design 15)
3. **No hard ontological boundary** — virtual/non-virtual split is simulated by the LoRA/MLP architecture, not physically distinct substrates

Design 15 ("The Crucible") addresses all three with a 100K-neuron spiking neural network + Mamba-2 canopy. This design validates the four-model pattern and builds measurement infrastructure that transfers to Design 15.
