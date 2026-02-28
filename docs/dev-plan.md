# Design 16 "The Mirror Box" -- Development Plan

**Sprint Duration**: 4 weeks (20 working days)
**Developer**: 1 (strong Python, learning ML)
**Hardware**: RTX 4090 24GB, i9-14900KF, 64GB DDR5
**Stack**: Python 3.11+, PyTorch 2.x, HuggingFace, pytest
**Git**: Private repo only (`git push private main:implementation`)

---

## Package Structure

```
aIware.implementation/
  src/
    mirror_box/
      __init__.py
      config.py              # Pydantic settings, state profiles, hyperparameters
      models/
        __init__.py
        llm_backend.py       # LLM loading, INT4 quantization, encoder/generator
        implicit.py          # IWM and ISM (substrate encoder + LoRA adapter)
        explicit.py          # EWM and ESM (LLM generation, self-referential loop)
        permeability.py      # Permeability gate (sigmoid + noise + profiles)
        self_predictor.py    # Self-predictor MLP + R metric + self-surprise
      runtime/
        __init__.py
        loop.py              # Main conscious loop (frame-by-frame orchestration)
        state_machine.py     # Wake/Sleep/Dream/Altered state controller
        telemetry.py         # System telemetry collector (GPU, latency, activations)
        sleep_maintenance.py # Experience replay, buffer clearing, ISM LoRA update
      metrics/
        __init__.py
        self_knowledge.py    # R metric, self-surprise, prediction error trajectory
        conscious_content.py # C(t) total conscious content proxy
        complexity.py        # Lempel-Ziv complexity, avalanche proxy
        state_detection.py   # Automated state transition detection from metrics
      introspection/
        __init__.py
        reporter.py          # Natural language state reports via LLM
        comparator.py        # Compare predictions vs actual state (self-challenge)
      visualization/
        __init__.py
        dashboard.py         # Streamlit dashboard (real-time plots)
        plots.py             # Individual plot components (R, C(t), state, heatmap)
      interfaces/
        __init__.py
        base.py              # Abstract interfaces for Design 15 reuse
  tests/
    conftest.py              # Shared fixtures (mock LLM, test tensors, config)
    test_config.py
    models/
      test_llm_backend.py
      test_implicit.py
      test_explicit.py
      test_permeability.py
      test_self_predictor.py
    runtime/
      test_loop.py
      test_state_machine.py
      test_telemetry.py
      test_sleep_maintenance.py
    metrics/
      test_self_knowledge.py
      test_conscious_content.py
      test_complexity.py
      test_state_detection.py
    introspection/
      test_reporter.py
    integration/
      test_end_to_end.py     # Full system integration tests
      test_state_transitions.py
  configs/
    default.yaml             # Default hyperparameters
    demo.yaml                # Demo scenario parameters
  scripts/
    run_demo.py              # CLI entry point for demo scenarios
    benchmark.py             # Benchmark self-model accuracy suite
  pyproject.toml
```

---

## Key Interfaces (Design 15 Reusability)

These interfaces live in `src/mirror_box/interfaces/base.py` and define contracts
that Design 15 must also satisfy. Every metric and visualization component
programs against these abstractions, never against concrete implementations.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class SubstrateState:
    """Output of the substrate layer (implicit models)."""
    world_state: torch.Tensor    # w(t), shape (D,)
    self_state: torch.Tensor     # s(t), shape (D,)
    timestamp: float
    metadata: dict               # Telemetry, diagnostics

@dataclass
class ExplicitState:
    """Output of the explicit model layer."""
    ewm_state: torch.Tensor      # e_w(t), shape (D,)
    esm_state: torch.Tensor      # e_s(t), shape (D,)
    timestamp: float
    metadata: dict

@dataclass
class FrameMetrics:
    """Metrics computed per conscious frame."""
    R: float                     # Self-knowledge [0, 1]
    self_surprise: float         # Prediction error magnitude
    C: float                     # Total conscious content proxy
    complexity: float            # Lempel-Ziv complexity
    state: str                   # wake/sleep/dream/altered
    timestamp: float

class SubstrateEncoder(ABC):
    """Interface for implicit model layer.
    Design 16: frozen LLM encoder + LoRA.
    Design 15: SNN reservoir spike-rate readout."""
    @abstractmethod
    def encode_world(self, sensory_input: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def encode_self(self, telemetry: dict) -> torch.Tensor: ...

class PermeabilityGate(ABC):
    """Interface for gating between substrate and simulation.
    Design 16: sigmoid + noise.
    Design 15: spike-rate modulation."""
    @abstractmethod
    def gate(self, signal: torch.Tensor, state_profile: dict) -> torch.Tensor: ...

class ExplicitGenerator(ABC):
    """Interface for explicit model generation.
    Design 16: LLM forward pass.
    Design 15: LLM canopy forward pass."""
    @abstractmethod
    def generate_ewm(self, gated_world: torch.Tensor,
                     context: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def generate_esm(self, gated_self: torch.Tensor,
                     prev_esm: torch.Tensor) -> torch.Tensor: ...

class SelfPredictor(ABC):
    """Interface for self-prediction module.
    Same formula in both designs, different input source."""
    @abstractmethod
    def predict(self, prev_esm: torch.Tensor,
                gated_self: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def compute_R(self, predicted: torch.Tensor,
                  actual: torch.Tensor) -> float: ...
    @abstractmethod
    def update(self, predicted: torch.Tensor,
               actual: torch.Tensor) -> float: ...

class MetricsCollector(ABC):
    """Interface for metrics collection.
    Reused directly in Design 15."""
    @abstractmethod
    def record_frame(self, metrics: FrameMetrics) -> None: ...
    @abstractmethod
    def get_trajectory(self, metric_name: str,
                       window: int) -> list[float]: ...
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Mistral-7B-Instruct-v0.3 (INT4 GPTQ) | Best introspection quality; 3.5 GB VRAM; TheBloke/Mistral-7B-Instruct-v0.3-GPTQ on HuggingFace |
| Quantization | auto-gptq + transformers | Mature, well-documented, single `AutoModelForCausalLM.from_pretrained` call |
| LoRA | peft (HuggingFace) | Standard, rank-16 on Q/V projections; ~50 MB |
| Self-predictor | PyTorch nn.Module | 3-layer MLP: 1024-512-512; ~1.3M params |
| State vectors | 512-dim FP32 | Extracted from LLM hidden layer 8 (of 32); good abstraction-to-cost ratio |
| Permeability gate | PyTorch (no library) | 20 lines of code; sigmoid + noise |
| Telemetry | pynvml + PyTorch hooks | GPU stats via pynvml; activation hooks on LLM layers |
| Config | Pydantic + YAML | Type-safe config with YAML override files |
| Experiment tracking | Weights and Biases (free tier) | `wandb.log()` per frame; offline mode for no-internet runs |
| Visualization | Streamlit | Faster to build than Gradio for multi-panel dashboards |
| Testing | pytest + pytest-cov + hypothesis | Property-based testing for metric sanity |
| LLM mock for tests | Tiny random model (32 hidden, 2 layers) | Fast tests; no GPU needed; same interface as real model |

### VRAM Budget (Final)

| Component | VRAM |
|-----------|------|
| Mistral-7B INT4 GPTQ weights | 3.5 GB |
| KV-cache (2048 tokens) | 0.5 GB |
| ISM LoRA adapter weights | 50 MB |
| ISM LoRA training state (Adam) | 200 MB |
| Self-predictor MLP + optimizer | 20 MB |
| State history buffer (100 frames x 4 vectors x 512 x FP32) | 0.8 MB |
| Telemetry buffer | 50 MB |
| PyTorch CUDA context | 500 MB |
| **Total** | **~4.8 GB** |
| **Headroom** | **~19 GB** |

---

## Week 1: Foundation (Days 1-5)

### Day 1 -- Project Scaffolding + LLM Loading

**Goal**: Working Python package with LLM loaded and producing hidden states.

**Tests first** (write before implementation):

```python
# tests/test_config.py
def test_default_config_loads():
    """Config loads from YAML without errors."""

def test_config_state_profiles_present():
    """All four state profiles (wake/sleep/dream/altered) defined."""

def test_config_vram_budget_fits():
    """Computed VRAM budget < 8 GB."""

# tests/models/test_llm_backend.py
def test_llm_loads_int4():
    """Mistral-7B INT4 loads without OOM."""

def test_llm_produces_hidden_states():
    """Forward pass returns hidden_states from specified layer."""

def test_hidden_state_shape():
    """Hidden state from layer 8 has shape (batch, seq_len, 4096)."""

def test_encoder_produces_512_dim_vector():
    """Mean-pooled, projected hidden state is (512,)."""

def test_encoder_deterministic_same_input():
    """Same input -> same output (temperature=0)."""
```

**Implementation**:

1. `pyproject.toml` with dependencies:
   - `torch>=2.1`, `transformers>=4.36`, `auto-gptq>=0.6`, `peft>=0.7`
   - `pydantic>=2.0`, `pyyaml`, `wandb`, `streamlit`, `pynvml`
   - `pytest`, `pytest-cov`, `hypothesis`
2. `src/mirror_box/config.py`: Pydantic model for all hyperparameters. Load from `configs/default.yaml`. State profiles as nested dicts.
3. `src/mirror_box/models/llm_backend.py`:
   - `LLMBackend` class: loads Mistral-7B INT4 via `AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=...)`
   - `encode(text) -> Tensor[512]`: tokenize, forward pass through first 8 layers, mean-pool, project to 512 via learned linear (or fixed random projection for prototype)
   - `generate(context, max_tokens) -> str`: full forward pass with generation
4. `configs/default.yaml`: all hyperparameters with comments

**Dependencies**: None (first day)

**Done criteria**:
- `pytest tests/test_config.py tests/models/test_llm_backend.py` all pass
- LLM loaded on GPU, VRAM usage matches budget (+/- 10%)
- `encode("Hello world")` returns a 512-dim tensor
- `generate("Describe yourself")` returns coherent text

---

### Day 2 -- IWM and ISM (Implicit Models)

**Goal**: Both substrate encoders producing state vectors from their respective inputs.

**Tests first**:

```python
# tests/models/test_implicit.py
def test_iwm_encodes_text_input():
    """IWM produces 512-dim world-state from text."""

def test_iwm_different_inputs_different_outputs():
    """Different text inputs produce measurably different w(t)."""

def test_iwm_weights_frozen():
    """IWM encoder weights have requires_grad=False."""

def test_ism_encodes_telemetry():
    """ISM produces 512-dim self-state from telemetry dict."""

def test_ism_has_lora_adapter():
    """ISM uses LoRA adapter (PEFT) on the encoder."""

def test_ism_lora_weights_trainable():
    """LoRA adapter weights have requires_grad=True."""

def test_iwm_ism_share_base_model():
    """Both use the same underlying LLM, different adapters."""

def test_substrate_state_dataclass():
    """SubstrateState contains both w(t) and s(t) with metadata."""
```

**Implementation**:

1. `src/mirror_box/models/implicit.py`:
   - `IWM(SubstrateEncoder)`: wraps `LLMBackend.encode()` with frozen weights. Input = text (sensory). Output = `w(t)` in R^512.
   - `ISM(SubstrateEncoder)`: wraps `LLMBackend.encode()` with LoRA adapter active. Input = telemetry dict (serialized to text). Output = `s(t)` in R^512.
   - Both share the same `LLMBackend` instance (adapter switching via PEFT `set_adapter`).
2. `src/mirror_box/runtime/telemetry.py` (skeleton):
   - `TelemetryCollector`: gathers basic stats (placeholder values for now). Returns a dict that ISM can encode.

**Dependencies**: Day 1 (LLM backend)

**Done criteria**:
- All tests pass
- `IWM.encode_world("A red ball on a table")` returns 512-dim tensor
- `ISM.encode_self({"gpu_temp": 65, "latency_ms": 12})` returns 512-dim tensor
- PEFT adapter loads and switches correctly (verify via `model.active_adapters`)

---

### Day 3 -- Permeability Gate

**Goal**: Configurable gating between substrate and explicit layers.

**Tests first**:

```python
# tests/models/test_permeability.py
def test_gate_output_shape():
    """Gate output has same shape as input."""

def test_gate_wake_profile_passes_most():
    """Wake profile (alpha=1, beta=0, noise=0.1): mean output > 0.4."""

def test_gate_sleep_profile_blocks_most():
    """Sleep profile (alpha=0.1, beta=-3, noise=0.01): mean output < 0.1."""

def test_gate_dream_profile_intermediate():
    """Dream profile: mean output between 0.1 and 0.5."""

def test_gate_altered_profile_opens_wide():
    """Altered profile (beta=2.0): mean output > 0.7."""

def test_gate_noise_varies_output():
    """Same input, nonzero noise std -> different outputs across calls."""

def test_gate_zero_noise_deterministic():
    """noise_std=0 -> same output for same input."""

def test_gate_respects_profile_switch():
    """Switching profile changes gating behavior immediately."""

def test_gate_differentiable():
    """Gate output supports backprop (for future gradient-based tuning)."""
```

**Implementation**:

1. `src/mirror_box/models/permeability.py`:
   - `PermeabilityGateImpl(PermeabilityGate)`:
     ```python
     def gate(self, signal, state_profile):
         alpha = state_profile["alpha"]
         beta = state_profile["beta"]
         noise_std = state_profile["noise_std"]
         noise = torch.randn_like(signal) * noise_std
         return torch.sigmoid(alpha * signal + beta + noise)
     ```
   - Separate `alpha_w`, `alpha_s` for world and self channels
   - Profile dict lookup from config

**Dependencies**: Day 1 (config)

**Done criteria**:
- All tests pass
- Gate correctly modulates a random 512-dim tensor under all four profiles
- Profile switching is instant (no warm-up period)

---

### Day 4 -- EWM and ESM (Explicit Models)

**Goal**: Both explicit model generators producing state vectors. ESM receives its own previous output (self-referential loop wired but not yet cycling).

**Tests first**:

```python
# tests/models/test_explicit.py
def test_ewm_produces_state_from_gated_world():
    """EWM generates 512-dim state from gated world input."""

def test_esm_produces_state_from_gated_self():
    """ESM generates 512-dim state from gated self input."""

def test_esm_accepts_prev_esm_input():
    """ESM forward pass accepts previous ESM state as additional input."""

def test_esm_output_differs_with_prev_esm():
    """ESM output changes when prev_esm is different (loop matters)."""

def test_esm_output_differs_without_prev_esm():
    """ESM with prev_esm=zeros differs from ESM with prev_esm=nonzero."""

def test_ewm_uses_base_model():
    """EWM generation uses base LLM (no LoRA adapter)."""

def test_esm_uses_self_adapter():
    """ESM generation uses the self-model LoRA adapter."""

def test_explicit_state_dataclass():
    """ExplicitState contains both e_w(t) and e_s(t) with metadata."""
```

**Implementation**:

1. `src/mirror_box/models/explicit.py`:
   - `ExplicitModelGenerator(ExplicitGenerator)`:
     - `generate_ewm(gated_world, context) -> Tensor[512]`: compose prompt from gated world state (embedded back into token space via projection), run LLM forward pass with base model, extract hidden state from layer 24 (higher = more abstract).
     - `generate_esm(gated_self, prev_esm) -> Tensor[512]`: compose prompt from concatenated `[gated_self; prev_esm]`, run LLM forward pass with LoRA adapter, extract hidden state.
   - The "compose prompt" step projects 512-dim vectors into a sequence of pseudo-tokens that the LLM can process. Implementation: learned linear projection 512 -> (num_prefix_tokens x 4096), reshape to prefix token embeddings.

**Dependencies**: Days 1-3 (LLM backend, implicit models, gate)

**Done criteria**:
- All tests pass
- EWM and ESM produce distinct 512-dim vectors
- ESM output changes when `prev_esm` changes (the loop input matters)
- Total VRAM still within budget

---

### Day 5 -- Self-Referential Loop + Self-Predictor

**Goal**: The complete frame loop running: substrate -> gate -> explicit -> ESM feedback. Self-predictor MLP computing R and self-surprise.

**Tests first**:

```python
# tests/models/test_self_predictor.py
def test_predictor_output_shape():
    """Predictor produces 512-dim predicted ESM state."""

def test_R_metric_range():
    """R is always in [0, 1]."""

def test_R_perfect_prediction():
    """When predicted == actual, R approaches 1.0."""

def test_R_random_prediction():
    """When predicted is random noise, R approaches 0.0."""

def test_self_surprise_nonnegative():
    """Self-surprise (L2 norm of error) >= 0."""

def test_predictor_update_reduces_loss():
    """After N update steps on fixed data, loss decreases."""

def test_predictor_gradients_flow():
    """Predictor parameters receive gradients during update."""

# tests/runtime/test_loop.py
def test_single_frame_executes():
    """One frame: substrate -> gate -> explicit -> metrics. No crash."""

def test_esm_feedback_loop_runs():
    """ESM output from frame t becomes input to frame t+1."""

def test_loop_converges_after_n_frames():
    """After 50 frames, ESM state change per frame < threshold."""

def test_loop_does_not_diverge():
    """After 100 frames, no NaN or Inf in any state vector."""

def test_R_computable_every_frame():
    """R metric is a finite float each frame."""

def test_frame_metrics_dataclass_populated():
    """FrameMetrics contains R, surprise, C, complexity, state, timestamp."""
```

**Implementation**:

1. `src/mirror_box/models/self_predictor.py`:
   - `SelfPredictorMLP(SelfPredictor)`:
     - Architecture: Linear(1024, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 512)
     - Input: `concat(prev_esm, gated_self)` = 1024-dim
     - Output: predicted `e_s(t+1)` = 512-dim
     - `compute_R(predicted, actual)`: entropy-based formula from Design 16 spec Section 3.2
     - `update(predicted, actual)`: MSE loss, Adam optimizer (lr=1e-4)
2. `src/mirror_box/runtime/loop.py`:
   - `ConsciousLoop`:
     - Holds references to IWM, ISM, gate, EWM generator, ESM generator, self-predictor
     - `step(sensory_input, telemetry) -> FrameMetrics`: one 50 ms conscious frame
     - Maintains `prev_esm` state across frames
     - Calls self-predictor before ESM generation, computes R after
3. `src/mirror_box/metrics/self_knowledge.py`:
   - `compute_R(predicted, actual)` as standalone function
   - `compute_self_surprise(predicted, actual)` as standalone function
   - Both usable from loop and from tests independently

**Dependencies**: Days 1-4 (all model components)

**Done criteria**:
- All tests pass
- Running 100 frames produces a sequence of R values that are finite and in [0, 1]
- ESM state changes decrease over frames (convergence toward fixed point)
- Self-predictor loss decreases over frames
- No VRAM leaks (check with `torch.cuda.memory_allocated()` before and after)

### Week 1 Summary

| Day | Deliverable | Key Risk |
|-----|-------------|----------|
| 1 | LLM loaded, config system, 512-dim encoder | GPTQ model download time (~30 min); VRAM mismatch |
| 2 | IWM + ISM producing state vectors | PEFT adapter switching bugs |
| 3 | Permeability gate with all 4 profiles | None (simple component) |
| 4 | EWM + ESM generators, self-ref input wired | Prompt-to-embedding projection (novel, may need tuning) |
| 5 | Complete frame loop, R metric, self-surprise | Loop divergence; NaN propagation |

**Week 1 gate**: The system runs 100 frames without crashing, R is computable, ESM shows convergence behavior. If this fails, Week 2 is blocked.

---

## Week 2: Dynamics and Self-Modeling (Days 6-10)

### Day 6 -- State Machine Controller

**Goal**: Wake/sleep/dream/altered states with transitions and parameter profiles.

**Tests first**:

```python
# tests/runtime/test_state_machine.py
def test_initial_state_is_wake():
    """System starts in wake state."""

def test_transition_wake_to_sleep():
    """After N continuous frames, auto-transitions to sleep."""

def test_transition_sleep_to_dream():
    """After sleep duration, transitions to dream."""

def test_transition_dream_to_wake():
    """After dream duration, transitions to wake."""

def test_manual_transition_to_altered():
    """User command triggers altered state."""

def test_manual_transition_back_to_wake():
    """User command returns from altered to wake."""

def test_state_provides_correct_profile():
    """Each state returns its permeability profile dict."""

def test_transition_gradual_not_instant():
    """Wake->sleep transition interpolates beta over ~5 seconds."""

def test_state_history_recorded():
    """State transitions are logged with timestamps."""

def test_invalid_transition_rejected():
    """Cannot go directly from sleep to altered."""
```

**Implementation**:

1. `src/mirror_box/runtime/state_machine.py`:
   - `ConsciousnessStateMachine`:
     - States: `WAKE`, `SLEEP`, `DREAM`, `ALTERED` (enum)
     - Transition rules: wake->sleep (timer), sleep->dream (timer), dream->wake (timer), wake->altered (command), altered->wake (command/timer)
     - `get_profile() -> dict`: returns current permeability parameters, interpolated during transitions
     - Gradual transitions: linear interpolation of gate parameters over configurable duration (default 5 seconds = 100 frames at 20 Hz)
     - State history: list of `(timestamp, old_state, new_state)` tuples

**Dependencies**: Day 3 (permeability profiles in config)

**Done criteria**:
- All tests pass
- Running 600 frames (30 seconds at 20 Hz) triggers at least one wake->sleep transition
- Profile parameters smoothly interpolate during transitions

---

### Day 7 -- State Transitions in the Loop

**Goal**: Integrate state machine with the conscious loop. Verify metric signatures across transitions.

**Tests first**:

```python
# tests/runtime/test_state_machine.py (additions)
def test_sleep_collapses_explicit_activity():
    """In sleep state, EWM and ESM activation norms drop below threshold."""

def test_dream_reactivates_without_external():
    """In dream, ESM reactivates but EWM is driven by noise, not input."""

def test_altered_increases_content():
    """Altered state produces higher C(t) than wake."""

def test_wake_after_sleep_has_higher_R():
    """R in first 10 wake frames after sleep > R before sleep."""

# tests/integration/test_state_transitions.py
def test_full_wake_sleep_dream_wake_cycle():
    """Run through complete cycle. Verify C(t) profile matches prediction."""

def test_C_collapses_during_sleep():
    """C(t) < 0.1 * C_wake during deep sleep."""

def test_C_recovers_on_wake():
    """C(t) returns to within 20% of pre-sleep value after waking."""
```

**Implementation**:

1. Update `src/mirror_box/runtime/loop.py`:
   - `ConsciousLoop` now holds a `ConsciousnessStateMachine`
   - Each frame: get current profile from state machine, pass to gate
   - In sleep: skip LLM generation (save compute), only run substrate at reduced rate
   - In dream: block external input, inject noise into gated signals
   - In altered: increase gate beta, allow more substrate content through
2. `src/mirror_box/metrics/conscious_content.py`:
   - `compute_C(ewm_state, esm_state) -> float`: sum of activation norms above threshold
   - Threshold calibrated from wake-state baseline

**Dependencies**: Days 5-6 (loop + state machine)

**Done criteria**:
- All tests pass
- A 2-minute run shows clear metric signatures: C(t) collapses in sleep, recovers in wake
- W&B log shows state transitions annotated on metric plots

---

### Day 8 -- Telemetry Collector

**Goal**: Full system telemetry feeding into ISM. The system can sense its own hardware and computational state.

**Tests first**:

```python
# tests/runtime/test_telemetry.py
def test_telemetry_collects_gpu_stats():
    """Telemetry includes gpu_temp, gpu_util, vram_used."""

def test_telemetry_collects_inference_latency():
    """Telemetry includes last_inference_ms."""

def test_telemetry_collects_activation_norms():
    """Telemetry includes per-layer activation norms (list of floats)."""

def test_telemetry_collects_prediction_error():
    """Telemetry includes current R and self_surprise."""

def test_telemetry_collects_gate_state():
    """Telemetry includes mean gate activation for world and self."""

def test_telemetry_collects_kv_cache_usage():
    """Telemetry includes kv_cache_tokens_used."""

def test_telemetry_collects_token_entropy():
    """Telemetry includes entropy of last generation's softmax."""

def test_telemetry_collects_state_info():
    """Telemetry includes current state and time_since_last_sleep."""

def test_telemetry_serializes_to_text():
    """Telemetry dict serializes to a text string for ISM encoding."""

def test_telemetry_collection_under_1ms():
    """Telemetry gathering takes < 1 ms (no blocking calls)."""
```

**Implementation**:

1. `src/mirror_box/runtime/telemetry.py`:
   - `TelemetryCollector`:
     - `collect() -> dict`: gathers all signals in < 1 ms
     - GPU stats via `pynvml` (non-blocking)
     - Activation norms via PyTorch forward hooks (registered once, values cached)
     - Prediction metrics from self-predictor (cached from last frame)
     - Gate state from permeability gate (cached)
     - Token entropy from last LLM generation (cached)
   - `serialize(telemetry: dict) -> str`: converts to structured text for ISM input
   - All signals listed in Design 16 Appendix A

**Dependencies**: Day 5 (loop provides hooks for latency, prediction error)

**Done criteria**:
- All tests pass
- Telemetry dict has 11 signal categories (matching Appendix A)
- Collection takes < 1 ms
- ISM can encode the serialized telemetry into a 512-dim vector

---

### Day 9 -- ISM LoRA Training + Natural Language Introspection

**Goal**: ISM LoRA adapter actually trains on self-data. System can describe its own state in natural language.

**Tests first**:

```python
# tests/runtime/test_sleep_maintenance.py
def test_experience_replay_buffer_stores_frames():
    """Buffer stores (telemetry, esm_state) pairs up to capacity."""

def test_experience_replay_trains_ism_lora():
    """After replay, ISM LoRA weights have changed."""

def test_ism_lora_loss_decreases_over_replays():
    """Multiple replay rounds reduce ISM reconstruction loss."""

def test_sleep_clears_state_buffers():
    """After sleep maintenance, frame buffers are empty."""

def test_sleep_resets_homeostatic_params():
    """Homeostatic scaling parameters reset to defaults after sleep."""

# tests/introspection/test_reporter.py
def test_introspection_produces_text():
    """Reporter generates non-empty natural language string."""

def test_introspection_mentions_current_state():
    """Report includes current state name (wake/sleep/dream/altered)."""

def test_introspection_mentions_R_value():
    """Report references self-knowledge metric value."""

def test_introspection_differs_across_states():
    """Reports in wake vs dream are substantially different."""

def test_introspection_latency_under_500ms():
    """Full introspection report generated in < 500 ms."""
```

**Implementation**:

1. `src/mirror_box/runtime/sleep_maintenance.py`:
   - `ExperienceReplayBuffer`: ring buffer of `(telemetry_text, esm_state)` pairs
   - `run_replay(buffer, ism, n_steps)`: fine-tune ISM LoRA on buffered data
     - Loss: MSE between ISM-encoded telemetry and corresponding ESM state
     - The ISM learns to predict what the ESM will produce from this telemetry
   - `sleep_maintenance(loop)`: clear buffers, run replay, reset homeostatic params
2. `src/mirror_box/introspection/reporter.py`:
   - `IntrospectionReporter`:
     - `report(frame_metrics, state, telemetry) -> str`: compose a prompt with current metrics and state, generate natural language via LLM
     - Prompt template: "You are a self-modeling system. Your current state is {state}. Your self-knowledge R={R:.3f}. Your conscious content C={C:.3f}. Describe your current experience."
     - Generation: max 200 tokens, temperature 0.7

**Dependencies**: Days 2, 5, 8 (ISM, loop, telemetry)

**Done criteria**:
- All tests pass
- After a sleep cycle with experience replay, ISM LoRA weights differ from initial
- ISM reconstruction loss decreases over 3 replay rounds
- Introspection reports are coherent and reference actual metric values
- Wake report differs meaningfully from dream report

---

### Day 10 -- Approximated Criticality + Substrate Dynamics

**Goal**: Noise injection and homeostatic scaling in substrate. Pseudo-criticality metrics computable.

**Tests first**:

```python
# tests/models/test_implicit.py (additions)
def test_substrate_noise_injection():
    """Substrate output varies with noise injection enabled."""

def test_homeostatic_scaling_prevents_divergence():
    """After 1000 steps, activation variance stays within [0.5, 2.0] of target."""

def test_homeostatic_scaling_prevents_collapse():
    """Activation variance does not drop below 0.1 of target."""

def test_substrate_dynamics_formula():
    """x(t+1) = tanh(W * x(t) + input) * scale + noise matches spec."""

# tests/metrics/test_complexity.py
def test_lempel_ziv_computable():
    """LZ complexity returns a finite positive float."""

def test_lempel_ziv_higher_for_complex_signal():
    """Random signal has higher LZ than constant signal."""

def test_avalanche_proxy_computable():
    """Avalanche size proxy returns a distribution."""

def test_complexity_differs_across_noise_levels():
    """Higher noise -> different complexity (not monotonic, but responsive)."""
```

**Implementation**:

1. Update `src/mirror_box/models/implicit.py`:
   - Add substrate dynamics layer between LLM encoder and output:
     ```python
     x_new = torch.tanh(self.W @ x_prev + encoded_input) * self.scale + noise
     self.scale = adjust_scale(x_new.var(), target_var=1.0)
     ```
   - `self.W`: small learnable matrix (512 x 512) initialized near identity
   - `adjust_scale`: PID-like controller keeping `var(x)` near target
2. `src/mirror_box/metrics/complexity.py`:
   - `lempel_ziv_complexity(trajectory: Tensor) -> float`: binarize state trajectory, compute LZ76
   - `avalanche_proxy(activation_history: Tensor) -> dict`: compute threshold crossings, measure size distribution

**Dependencies**: Days 2, 5 (implicit models, loop)

**Done criteria**:
- All tests pass
- Substrate dynamics run stably for 1000 steps without NaN or divergence
- Homeostatic scaling keeps variance within target range
- LZ complexity differs between wake and sleep states
- Noise level affects dynamics observably

### Week 2 Summary

| Day | Deliverable | Key Risk |
|-----|-------------|----------|
| 6 | State machine with 4 states + transitions | None (straightforward FSM) |
| 7 | State-aware loop; C(t) metric; metric signatures | Metric signatures too weak to detect |
| 8 | Full telemetry collector (11 signals) | pynvml blocking on some GPU drivers |
| 9 | ISM LoRA training; natural language introspection | LoRA training instability; prompt engineering |
| 10 | Substrate dynamics with noise + homeostatic scaling | Homeostatic controller tuning |

**Week 2 gate**: System runs through a full wake-sleep-dream-wake cycle with measurable C(t) changes. Introspection reports reference actual state. ISM LoRA trains during sleep.

---

## Week 3: Measurement and Visualization (Days 11-15)

### Day 11 -- C(t) and R Trajectory Infrastructure

**Goal**: Full metric computation pipeline with windowed trajectories and W&B logging.

**Tests first**:

```python
# tests/metrics/test_conscious_content.py
def test_C_nonnegative():
    """C(t) >= 0."""

def test_C_higher_in_wake_than_sleep():
    """Mean C over 50 wake frames > mean C over 50 sleep frames."""

def test_C_threshold_calibration():
    """Threshold auto-calibrates from first 10 wake frames."""

# tests/metrics/test_self_knowledge.py (additions)
def test_R_trajectory_stored():
    """After N frames, R trajectory has N entries."""

def test_R_trajectory_decreasing_surprise():
    """Mean self-surprise in last 50 frames < first 50 frames."""

def test_prediction_error_spectrum():
    """FFT of prediction error trajectory is computable."""

def test_wandb_logging_works():
    """Metrics are logged to W&B without error (offline mode)."""
```

**Implementation**:

1. `src/mirror_box/metrics/self_knowledge.py` (expand):
   - `RTrajectory`: stores R values over time, computes rolling mean, trend
   - `PredictionErrorSpectrum`: FFT of prediction error over sliding window
2. `src/mirror_box/metrics/conscious_content.py` (expand):
   - `CTrajectory`: stores C(t) values, auto-calibrates threshold from initial wake baseline
3. Update `src/mirror_box/runtime/loop.py`:
   - Each frame logs `FrameMetrics` to a `MetricsCollector` implementation
   - W&B integration: `wandb.log({"R": R, "C": C, "surprise": surprise, "state": state})` per frame

**Dependencies**: Days 5, 7 (loop, C(t) basics)

**Done criteria**:
- All tests pass
- 5-minute run produces smooth R and C(t) trajectories in W&B
- Prediction error spectrum is computable and non-degenerate

---

### Day 12 -- State Transition Detection

**Goal**: Automated detection of state boundaries from metric trajectories (not from the state machine -- from the raw metrics, as Design 15 will need).

**Tests first**:

```python
# tests/metrics/test_state_detection.py
def test_detect_wake_to_sleep_from_C():
    """Detects C(t) collapse as sleep onset."""

def test_detect_sleep_to_dream_from_complexity():
    """Detects complexity increase during sleep as dream onset."""

def test_detect_dream_to_wake_from_C_recovery():
    """Detects C(t) recovery as wake onset."""

def test_detection_matches_state_machine():
    """Detected transitions match state machine transitions within 1 second."""

def test_detection_works_on_synthetic_data():
    """Works on hand-crafted metric trajectories (no LLM needed)."""

def test_false_positive_rate_below_threshold():
    """No false transitions detected during stable wake period."""
```

**Implementation**:

1. `src/mirror_box/metrics/state_detection.py`:
   - `StateTransitionDetector`:
     - Uses sliding-window change-point detection on C(t) and R trajectories
     - CUSUM (cumulative sum) algorithm for mean-shift detection
     - Configurable sensitivity threshold
     - Returns list of `(timestamp, detected_state, confidence)` tuples
   - This component is **Design 15 critical**: it must work from raw metrics alone, without knowing the state machine's internal state

**Dependencies**: Day 11 (metric trajectories)

**Done criteria**:
- All tests pass
- Detection matches state machine transitions within 1-second tolerance
- Works on synthetic data without GPU (fast unit tests)

---

### Day 13 -- Visualization Dashboard (Layout + Core Panels)

**Goal**: Streamlit dashboard with real-time R, C(t), state indicator, and ESM narrative.

**Tests first**:

```python
# tests/visualization/ -- Streamlit apps are hard to unit test.
# Use component tests for the plot data functions.

# tests/visualization/test_plots.py
def test_R_plot_data_format():
    """R plot data returns (timestamps, values) tuple of equal length."""

def test_C_plot_data_format():
    """C plot data returns (timestamps, values) tuple."""

def test_state_indicator_returns_valid_state():
    """State indicator returns one of {wake, sleep, dream, altered}."""

def test_heatmap_data_shape():
    """Four-model heatmap returns 2x2 matrix of activation norms."""

def test_narrative_output_is_string():
    """EWM and ESM narrative outputs are non-empty strings."""
```

**Implementation**:

1. `src/mirror_box/visualization/plots.py`:
   - `RPlot`: time series of R with rolling average overlay
   - `CPlot`: time series of C(t) with state-colored background bands
   - `StateIndicator`: current state with color coding (wake=green, sleep=blue, dream=purple, altered=red)
   - `FourModelHeatmap`: 2x2 grid showing IWM/ISM/EWM/ESM activation norms
   - `SurprisePlot`: time series of self-surprise
   - `GateVisualization`: current permeability gate activations (bar chart)
   - `NarrativePanel`: latest EWM and ESM text outputs
2. `src/mirror_box/visualization/dashboard.py`:
   - Streamlit layout:
     ```
     +-------------------+-------------------+
     |   R (self-knowledge) over time        |
     +-------------------+-------------------+
     |   C(t) over time  | State indicator   |
     +-------------------+-------------------+
     |   Self-surprise   | Gate activation   |
     +-------------------+-------------------+
     |   Four-model      | Narrative output  |
     |   heatmap         | (EWM + ESM text)  |
     +-------------------+-------------------+
     ```
   - Reads metrics from a shared queue (producer = conscious loop, consumer = dashboard)
   - Auto-refreshes at 2 Hz (dashboard does not need 20 Hz)

**Dependencies**: Days 11-12 (metric infrastructure)

**Done criteria**:
- Plot data functions pass all tests
- Dashboard launches without error (`streamlit run dashboard.py`)
- Dashboard renders all 7 panels with mock data
- Dashboard connects to a running loop and shows live updates

---

### Day 14 -- Dashboard Polish + Permeability Visualization

**Goal**: Dashboard fully functional with all panels live. Add prediction error spectrum and permeability time series.

**Tests first**:

```python
# tests/visualization/test_plots.py (additions)
def test_prediction_error_spectrum_data():
    """Spectrum data returns (frequencies, magnitudes) tuple."""

def test_permeability_time_series_data():
    """Permeability plot returns (timestamps, world_gate, self_gate) tuple."""

def test_transition_markers_on_C_plot():
    """C plot includes vertical lines at detected state transitions."""
```

**Implementation**:

1. Add to `src/mirror_box/visualization/plots.py`:
   - `PredictionErrorSpectrumPlot`: FFT magnitude plot of prediction error trajectory
   - `PermeabilityTimeSeries`: world gate and self gate mean activations over time
   - Overlay transition markers (from `StateTransitionDetector`) on C(t) plot
2. Polish dashboard:
   - Add controls sidebar: manual state triggers (altered, force wake), noise level slider
   - Add "demo scenario" dropdown (maps to `configs/demo.yaml` presets)
   - Add frame counter and FPS display
   - Style: dark theme (less strain during long experiments)

**Dependencies**: Day 13 (dashboard base)

**Done criteria**:
- All tests pass
- Dashboard shows 9 panels total, all updating live
- Controls work: clicking "Altered" triggers state transition visible in all metrics
- FPS display shows actual frame rate

---

### Day 15 -- Integration Testing + Demo Script

**Goal**: End-to-end system test. Demo script exercising all 5 scenarios from Design 16 Appendix B.

**Tests first**:

```python
# tests/integration/test_end_to_end.py
def test_system_runs_5_minutes():
    """Full system runs 300 seconds (6000 frames) without crash."""

def test_no_vram_leak_over_1000_frames():
    """VRAM usage at frame 1000 within 10% of frame 10."""

def test_no_nan_in_metrics_over_1000_frames():
    """All metric values are finite over 1000 frames."""

def test_sleep_cycle_occurs():
    """At least one full wake-sleep-dream-wake cycle in 5 minutes."""

def test_R_improves_over_session():
    """Mean R in last 100 frames > mean R in first 100 frames."""

def test_state_detection_agrees_with_state_machine():
    """Detected transitions match actual transitions (>80% agreement)."""

# tests/integration/test_state_transitions.py (expand)
def test_scenario_who_are_you():
    """Introspection report in wake references actual R and state."""

def test_scenario_sleep_wake_cycle():
    """Full cycle produces expected C(t) profile."""

def test_scenario_identity_disruption():
    """Cutting self-ref feedback changes ESM content."""

def test_scenario_altered_state():
    """High permeability produces unexpected associations."""
```

**Implementation**:

1. `scripts/run_demo.py`:
   - CLI: `python scripts/run_demo.py --scenario <name> --duration <seconds>`
   - Scenarios:
     - `who_are_you`: Run 30 seconds wake, ask introspection question, compare with raw LLM answer
     - `sleep_cycle`: Run 120 seconds with auto-transitions, log all metrics
     - `identity_disruption`: Run 60 seconds wake, cut self-ref, inject external content, measure ESM shift
     - `altered_state`: Run 60 seconds wake, trigger altered, log content flooding
     - `self_challenge`: Run 120 seconds, periodically ask system to predict its own next state
   - All scenarios log to W&B and save results to `results/<scenario>_<timestamp>/`

**Dependencies**: All previous days

**Done criteria**:
- All integration tests pass
- Demo script runs all 5 scenarios without manual intervention
- Each scenario produces a W&B dashboard with annotated metric plots
- End-to-end 5-minute run: no crashes, no NaN, no VRAM leaks

### Week 3 Summary

| Day | Deliverable | Key Risk |
|-----|-------------|----------|
| 11 | Full metric pipeline + W&B logging | W&B rate limits on free tier |
| 12 | Automated state transition detection | Detection sensitivity tuning |
| 13 | Streamlit dashboard (7 panels) | Streamlit performance with real-time data |
| 14 | Dashboard polish (9 panels + controls) | None (polish day) |
| 15 | Integration tests + demo script (5 scenarios) | Integration bugs from all components |

**Week 3 gate**: Demo script runs all 5 scenarios. Dashboard shows live metrics. State transitions are detectable from metrics alone.

---

## Week 4: Experiments, Polish, and Design 15 Handoff (Days 16-20)

### Day 16 -- ESM Redirection Experiment

**Goal**: Implement and run the identity-switching experiment (Design 16 Section 3.3, Appendix B Scenario 3).

**Tests first**:

```python
# tests/integration/test_esm_redirection.py
def test_esm_redirection_cuts_self_feedback():
    """When self-feedback is zeroed, ESM output changes significantly."""

def test_esm_redirection_latches_external():
    """When external content injected during cut, ESM tracks it."""

def test_mutual_information_shift():
    """I(ESM; external) increases when self-feedback cut."""

def test_esm_recovers_after_restoration():
    """Restoring self-feedback returns ESM to prior attractor."""
```

**Implementation**:

1. `src/mirror_box/introspection/comparator.py`:
   - `ESMRedirectionExperiment`:
     - Phase 1 (baseline, 30 sec): normal operation, record ESM trajectory
     - Phase 2 (disruption, 30 sec): set `prev_esm = zeros`, inject external content (description of a different AI system) into self-channel
     - Phase 3 (recovery, 30 sec): restore normal self-feedback
     - Measure: cosine similarity between ESM and external content over time
     - Compute mutual information estimate via binned distributions
2. Log results to W&B with phase annotations

**Dependencies**: Day 15 (demo infrastructure)

**Done criteria**:
- All tests pass
- ESM content shifts toward external content during disruption (cosine similarity increases)
- ESM recovers within 5 seconds of feedback restoration
- Results documented in W&B

---

### Day 17 -- Comparative Experiments

**Goal**: Controlled comparisons: self-ref ON vs OFF, noise ON vs OFF, different permeability profiles.

**Tests first**:

```python
# tests/integration/test_comparisons.py
def test_self_ref_on_higher_R_than_off():
    """Mean R with self-ref loop > mean R without (over 100 frames)."""

def test_noise_affects_complexity():
    """LZ complexity differs between noise-on and noise-off runs."""

def test_different_profiles_different_metrics():
    """Each of 4 permeability profiles produces distinct metric fingerprint."""
```

**Implementation**:

1. `scripts/benchmark.py`:
   - `run_comparison(config_a, config_b, duration, n_runs) -> ComparisonResult`:
     - Runs both configurations for specified duration
     - Computes mean and std of R, C(t), complexity, self-surprise
     - Runs Mann-Whitney U test for statistical significance
   - Experiments:
     - **A: Self-ref ON vs OFF**: toggle `prev_esm` feedback. Measures whether the loop adds value.
     - **B: Noise ON vs OFF**: toggle substrate noise injection. Measures whether approximated criticality changes dynamics.
     - **C: Profile sweep**: run 10 minutes under each of the 4 state profiles. Characterize the metric space.
     - **D: Overhead measurement**: time a single frame with and without self-referential loop. Compute the 2x overhead prediction.
2. All results to W&B + CSV export

**Dependencies**: Day 15 (demo infrastructure)

**Done criteria**:
- All comparison tests pass
- Self-ref ON produces significantly higher R than OFF (p < 0.05)
- Each permeability profile produces a distinct metric cluster
- Computational overhead of self-ref loop is measured and documented

---

### Day 18 -- Benchmark Suite + Self-Model Accuracy

**Goal**: Formal benchmark for self-model accuracy. How well does the system predict its own future states?

**Tests first**:

```python
# tests/integration/test_benchmarks.py
def test_benchmark_runs_without_crash():
    """Full benchmark suite completes in < 30 minutes."""

def test_benchmark_produces_report():
    """Benchmark generates a summary dict with all metrics."""

def test_R_improves_over_training():
    """R after 5 minutes > R in first 30 seconds."""

def test_R_does_not_reach_1():
    """R stays below 0.95 (dynamics not trivially predictable)."""
```

**Implementation**:

1. `scripts/benchmark.py` (expand):
   - `SelfModelBenchmark`:
     - Run 10 minutes of wake operation
     - Track R trajectory with 1-second resolution
     - Compute: initial R, final R, convergence rate (frames to reach R > 0.5), residual error
     - Track prediction error distribution: should be roughly Gaussian (if not, self-model is biased)
     - Generate summary report
   - Benchmark variations:
     - Different noise levels (0.0, 0.05, 0.1, 0.2, 0.5)
     - Different self-predictor learning rates (1e-5, 1e-4, 1e-3)
     - Different state vector dimensions (256, 512, 1024)

**Dependencies**: Days 16-17 (experiment infrastructure)

**Done criteria**:
- Benchmark suite runs in < 30 minutes
- R trajectory shows clear improvement over time
- R does not plateau at 1.0 (confirming dynamics have irreducible stochasticity)
- Report generated with all metrics and plots

---

### Day 19 -- Documentation: What Design 16 Proves vs Does Not Prove

**Goal**: Write the experiment results document. Map findings to FMT predictions. Honest limitations.

**Implementation**:

1. `docs/experiments/design-16-results.md`:
   - **Section 1: Questions Answered** (from Design 16 Section 8.1):
     - Q1: Does four-model architecture differ from single LLM? (compare raw Mistral vs Mirror Box R values)
     - Q2: Does self-ref loop converge? (fixed point analysis from benchmark)
     - Q3: R correlates with introspection quality? (compare R with human-rated report quality)
     - Q4: State transition signatures? (C(t) profiles from demo scenarios)
     - Q5: ESM redirects? (redirection experiment results)
     - Q6: Computational overhead? (2x prediction test)
     - Q7: Self-predictor improvement rate? (R trajectory analysis)
   - **Section 2: Honest Limitations** (from Design 16 Section 9):
     - No genuine criticality (noise injection != edge-of-chaos)
     - No continuous model density (4 discrete components)
     - Self-prediction may be trivially easy (if dynamics too predictable)
     - Behavioral mimicry risk (LLM generates plausible text regardless)
   - **Section 3: Design 15 Implications**:
     - Which findings transfer directly
     - Which findings need revalidation with SNN substrate
     - Calibrated expectations for Design 15 metrics

2. Update results with actual numbers from Days 16-18 experiments

**Dependencies**: Days 16-18 (experiment results)

**Done criteria**:
- Document written with actual experimental data (not placeholders)
- All 7 questions from Section 8.1 addressed with data
- Limitations section is honest and specific

---

### Day 20 -- Design 15 Infrastructure Inventory + Final Polish

**Goal**: Catalog exactly what transfers to Design 15. Clean up code. Final tests.

**Implementation**:

1. `docs/design-15-infrastructure-inventory.md`:
   - Table of every module with: transfer status, modification needed, interface stability
   - Matches table from Design 16 Section 10.1
   - Includes lessons learned (Section 10.2) with specific parameter values
   - Migration guide: step-by-step for each component swap

2. Code cleanup:
   - Run `pytest --cov=src/mirror_box` -- target 80%+ coverage
   - Fix any failing tests
   - Add docstrings to all public functions
   - Type hints on all function signatures
   - Remove dead code and TODO comments

3. Final integration run:
   - 10-minute continuous run with all metrics logging
   - Verify no regressions from Week 3

4. Update `backlog.md` with Design 15 tasks derived from this sprint

**Dependencies**: All previous days

**Done criteria**:
- Test coverage > 80%
- All tests pass (zero failures)
- Infrastructure inventory document complete
- 10-minute integration run clean
- Backlog updated

### Week 4 Summary

| Day | Deliverable | Key Risk |
|-----|-------------|----------|
| 16 | ESM redirection experiment | Mutual information estimation accuracy |
| 17 | Comparative experiments (4 conditions) | Statistical power with limited runs |
| 18 | Self-model accuracy benchmark suite | Benchmark may take longer than 30 min |
| 19 | Results documentation | Honest assessment may reveal design flaws |
| 20 | Infrastructure inventory + final polish | Coverage target may require backfill |

**Week 4 gate**: All experiments documented. Infrastructure inventory complete. Code at 80%+ coverage. Project ready for Design 15 transition.

---

## Dependency Graph

```
Day 1 (LLM + config)
  |
  +-> Day 2 (IWM + ISM)
  |     |
  |     +-> Day 8 (telemetry)
  |     |     |
  |     |     +-> Day 9 (ISM LoRA training + introspection)
  |     |
  |     +-> Day 10 (substrate dynamics)
  |
  +-> Day 3 (permeability gate)
  |     |
  |     +-> Day 6 (state machine)
  |           |
  |           +-> Day 7 (state-aware loop)
  |
  +-> Day 4 (EWM + ESM)
        |
        +-> Day 5 (self-ref loop + predictor)
              |
              +-> Day 7 (state-aware loop)
              |     |
              |     +-> Day 11 (metric pipeline + W&B)
              |     |     |
              |     |     +-> Day 12 (state detection)
              |     |     |     |
              |     |     |     +-> Day 14 (dashboard polish)
              |     |     |
              |     |     +-> Day 13 (dashboard base)
              |     |           |
              |     |           +-> Day 14 (dashboard polish)
              |     |
              |     +-> Day 15 (integration tests + demo)
              |           |
              |           +-> Day 16 (ESM redirection)
              |           +-> Day 17 (comparisons)
              |           +-> Day 18 (benchmarks)
              |                 |
              |                 +-> Day 19 (documentation)
              |                       |
              |                       +-> Day 20 (inventory + polish)
```

**Critical path**: Day 1 -> Day 2 -> Day 4 -> Day 5 -> Day 7 -> Day 15

If Day 5 slips (self-referential loop not converging), everything from Day 7 onward is blocked. **Mitigation**: if the loop diverges on Day 5, spend Day 5 afternoon debugging convergence (reduce learning rate, add gradient clipping, try simpler predictor). If still failing by end of Day 5, proceed with a "open-loop" mode (no ESM feedback) and fix the loop in parallel during Day 6.

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPTQ model download slow or fails | Medium | Blocks Day 1 | Download beforehand; have bitsandbytes NF4 as fallback |
| Self-referential loop diverges | Medium | Blocks Day 5+ | Gradient clipping; lower LR; fallback to open-loop mode |
| LLM forward pass too slow for 20 Hz | Low | Degrades real-time performance | Reduce to 10 Hz explicit, 20 Hz substrate; use only encoder layers |
| ISM LoRA training does not converge | Medium | Weakens self-modeling | Start with random LoRA (already functional); iterate training schedule |
| Streamlit too slow for real-time dashboard | Low | Dashboard lags | Reduce refresh to 1 Hz; buffer metrics; use Plotly instead of matplotlib |
| R metric trivially high (dynamics too predictable) | Medium | Invalidates self-modeling claim | Add more noise; use higher LLM temperature; document as finding |
| R metric trivially low (dynamics too chaotic) | Low | Self-predictor useless | Reduce noise; use lower LLM temperature; try larger predictor |
| VRAM budget exceeded | Low | Cannot load all components | Reduce KV-cache to 1024 tokens; use INT8 LoRA; drop optional components |

---

## Pre-Sprint Checklist (Day 0 or Evening Before)

- [ ] Download Mistral-7B-Instruct-v0.3-GPTQ from HuggingFace (huggingface-cli download)
- [ ] Create Python virtual environment: `python3.11 -m venv .venv`
- [ ] Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] Install dependencies: `pip install transformers auto-gptq peft pynvml wandb streamlit pydantic pyyaml`
- [ ] Install test dependencies: `pip install pytest pytest-cov hypothesis`
- [ ] Verify GPU access: `python -c "import torch; print(torch.cuda.get_device_name())"`
- [ ] Verify GPTQ loads: quick test script loading the model
- [ ] Set up W&B: `wandb login` (or `wandb offline` for offline mode)
- [ ] Create package structure (directories and `__init__.py` files)
- [ ] Initialize git, create `.gitignore` (exclude `.venv/`, `wandb/`, `__pycache__/`, model cache)

---

## Success Metrics (End of Sprint)

| Metric | Target | How Measured |
|--------|--------|--------------|
| Test coverage | > 80% | `pytest --cov` |
| Tests passing | 100% | `pytest` exit code |
| R metric functional | R in (0.1, 0.9) during wake | Benchmark suite |
| R improves over session | Final R > Initial R | 10-minute benchmark |
| C(t) collapses in sleep | C_sleep < 0.1 * C_wake | State transition test |
| State transitions detectable | > 80% agreement with state machine | Detection accuracy test |
| ESM redirection works | Cosine(ESM, external) > 0.5 during disruption | Redirection experiment |
| Self-ref adds value | R_with_loop > R_without (p < 0.05) | Comparison experiment |
| 5-minute run stable | No crash, no NaN, no VRAM leak | Integration test |
| Dashboard functional | All 9 panels render live | Manual verification |
| Design 15 interfaces defined | All 5 abstract base classes | Code review |
| Infrastructure inventory | Complete table with transfer status | Document review |
