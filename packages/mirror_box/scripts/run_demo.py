"""CLI entry point for Mirror Box demo scenarios.

Usage:
    mirror-box-demo                     # Default: wake-sleep cycle
    mirror-box-demo --scenario identity # Identity disruption
    mirror-box-demo --config path.yaml  # Custom config
"""

from __future__ import annotations

# Delegate to the package-internal script (pyproject.toml entry point)
from mirror_box.scripts.run_demo import main

if __name__ == "__main__":
    main()
