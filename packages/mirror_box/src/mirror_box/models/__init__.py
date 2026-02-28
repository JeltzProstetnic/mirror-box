"""Model implementations: LLM backend, implicit/explicit models, permeability gate."""

from mirror_box.models.explicit import ExplicitModels
from mirror_box.models.implicit import ImplicitModels
from mirror_box.models.llm_backend import LLMBackend
from mirror_box.models.permeability import SigmoidPermeabilityGate
from mirror_box.models.self_predictor import MLPSelfPredictor

__all__ = [
    "ExplicitModels",
    "ImplicitModels",
    "LLMBackend",
    "MLPSelfPredictor",
    "SigmoidPermeabilityGate",
]
