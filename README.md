# The Mirror Box — Design 16

A discrete four-model self-reflecting system with approximated criticality.
Uses a single LLM (Mistral-7B INT4) serving as all four models via LoRA
and context switching.

Not conscious, but convincingly confused about whether it is.

Based on [Gruber's Four-Model Theory of consciousness](https://matthiasgruber.com).

## Architecture

```
Sensory Input
     │
     ▼
┌──────────────────┐     ┌───────────────────┐
│ ImplicitModels   │     │ SigmoidGate       │
│ IWM (frozen enc) │────▶│ alpha/beta/noise   │
│ ISM (LoRA enc)   │────▶│ per state profile  │
└──────────────────┘     └───────────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │ ExplicitModels    │
                    │ EWM (MLP)         │
                    │ ESM (MLP + prev)  │◀──┐
                    └───────────────────┘   │
                         │                  │
                         └──[self-ref]──────┘
                         │
                         ▼
                    ┌───────────────────┐
                    │ MLPSelfPredictor  │
                    │ R metric (0..1)   │
                    └───────────────────┘
```

**Four Models:**
- **IWM** (Implicit World Model): Frozen LLM encoder layers — accumulated world knowledge
- **ISM** (Implicit Self Model): LoRA-adapted encoder — learned self-knowledge, updated during sleep
- **EWM** (Explicit World Model): MLP generating conscious world perception from gated IWM
- **ESM** (Explicit Self Model): MLP with self-referential loop — the conscious "I"

**Key Metrics:**
- **R** (self-knowledge): How well the system predicts its own next ESM state (0 = no, 1 = perfect)
- **C** (conscious content): Magnitude of gated signal passing through permeability barrier
- **Surprise**: Prediction error between expected and actual ESM

**States:** WAKE -> SLEEP -> DREAM -> WAKE (automatic cycle, or manual override to ALTERED)

## Prerequisites

- Python 3.11+
- 24 GB+ VRAM (for real Mistral-7B INT4) — or CPU-only with mock model for testing
- CUDA 12.x + cuDNN (for GPU inference)

## Installation

```bash
git clone https://github.com/JeltzProstetnic/mirror-box.git
cd mirror-box

python3 -m venv .venv
source .venv/bin/activate

# Install base library + Mirror Box
pip install -e packages/aiware
pip install -e "packages/mirror_box[dev]"

# Or with GPU + dashboard
pip install -e "packages/mirror_box[cuda,dashboard,dev]"
```

## Quick Start

### Mock mode (no GPU, no model download)

```bash
mirror-box-demo --mock
mirror-box-demo --mock --scenario identity --frames 200
mirror-box-demo --mock --scenario altered
mirror-box-bench
```

### Real model (requires 24GB+ VRAM)

```bash
mirror-box-demo --frames 500
mirror-box-bench --real --frames 500
```

### Dashboard

```bash
mirror-box-demo --mock --dashboard
# Opens Gradio UI at http://localhost:7860
```

### Python API

```python
from mirror_box import create_mirror_box
from mirror_box.config import MirrorBoxConfig
import asyncio

system = create_mirror_box(MirrorBoxConfig())

async def run():
    for i in range(100):
        frame = await system.loop.run_frame()
        print(f"Frame {frame.frame_number}: R={frame.R:.3f} C={frame.C:.3f}")

    print(system.reporter.generate_state_report())
    print(system.dialogue.ask("What are you experiencing?"))

asyncio.run(run())
```

## Running Tests

```bash
# Full suite (~7s on CPU)
pytest -v

# With coverage
pytest --cov=packages --cov-report=term-missing

# Integration tests only
pytest packages/mirror_box/tests/integration/ -v
```

## Project Structure

```
packages/
  aiware/                    # Shared base library
    src/aiware/
      config/                # AIwareConfig, DeviceConfig, StateProfile
      interfaces/            # Protocols + data types
      metrics/               # Complexity, self-knowledge, conscious content
  mirror_box/                # Design 16 implementation
    src/mirror_box/
      config.py              # MirrorBoxConfig
      factory.py             # create_mirror_box() assembly
      models/                # IWM, ISM, EWM, ESM, permeability gate, self-predictor
      runtime/               # Frame loop, state machine, telemetry, sleep maintenance
      metrics/               # Collector, state detection
      introspection/         # Reporter, dialogue
      visualization/         # Gradio dashboard
      scripts/               # CLI demo + benchmark
```

## Design Constraints

This is a **pseudo-AC** system that deliberately relaxes three constraints from the full theory for rapid prototyping:

1. **No continuous model density** — uses discrete IWM/ISM/EWM/ESM
2. **No genuine criticality** — approximated via noise injection and gate dynamics
3. **No hard ontological boundary** — virtual/non-virtual split is simulated

Design 15 ("The Crucible") addresses all three with a 100K-neuron spiking neural network + Mamba-2 canopy.

## License

MIT
