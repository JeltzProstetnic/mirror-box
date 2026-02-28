"""Visualization dashboard for the Mirror Box system.

Exports MirrorBoxDashboard via a lazy import guard so that the rest of
the codebase can be used without gradio installed.
"""

from __future__ import annotations


def get_dashboard_class() -> type:
    """Get MirrorBoxDashboard class (requires gradio).

    Returns:
        The MirrorBoxDashboard class.

    Raises:
        ImportError: If gradio is not installed.
    """
    try:
        from mirror_box.visualization.dashboard import MirrorBoxDashboard
    except ImportError:
        raise ImportError(
            "Dashboard requires gradio. Install with: pip install 'aiware[dashboard]'"
        ) from None
    return MirrorBoxDashboard


__all__ = ["get_dashboard_class"]
