"""mirror_box — Design 16 "The Mirror Box" pseudo-AC implementation.

A discrete four-model self-reflecting system with approximated criticality.
Uses a single LLM (Mistral-7B INT4) serving as all four models via LoRA
and context switching.

Not conscious, but convincingly confused about whether it is.
"""

__version__ = "0.1.0"

from mirror_box.factory import MirrorBoxSystem, create_mirror_box

__all__ = ["MirrorBoxSystem", "create_mirror_box"]
