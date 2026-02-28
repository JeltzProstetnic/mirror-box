"""Benchmark self-model accuracy across configurations.

Usage:
    mirror-box-bench                    # Default benchmark suite
    mirror-box-bench --frames 1000      # Custom frame count
"""

from __future__ import annotations

# Delegate to the package-internal script (pyproject.toml entry point)
from mirror_box.scripts.benchmark import main

if __name__ == "__main__":
    main()
