"""Abstract interfaces that AC implementations must fulfill.

All metrics, telemetry, dashboard, and runtime components program against
these abstractions. Implementations (mirror_box, spiking_canopy) provide
concrete classes.
"""

from aiware.interfaces.data_types import (
    ConsciousState,
    ExplicitState,
    FrameMetrics,
    GateOutput,
    SubstrateState,
)
from aiware.interfaces.protocols import (
    ConsciousLoop,
    ExplicitGenerator,
    MetricsCollector,
    PermeabilityGate,
    SelfPredictor,
    StateController,
    SubstrateEncoder,
    TelemetrySource,
)

__all__ = [
    # Data types
    "ConsciousState",
    "ExplicitState",
    "FrameMetrics",
    "GateOutput",
    "SubstrateState",
    # Protocols
    "ConsciousLoop",
    "ExplicitGenerator",
    "MetricsCollector",
    "PermeabilityGate",
    "SelfPredictor",
    "StateController",
    "SubstrateEncoder",
    "TelemetrySource",
]
