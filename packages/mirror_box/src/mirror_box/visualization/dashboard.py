"""Real-time Gradio dashboard for the Mirror Box system.

Provides a browser-based monitoring UI that displays:
- Metric trajectories (R, C, self-surprise, complexity) as line plots
- Current conscious state with color coding
- Permeability gate activations as a bar chart
- EWM/ESM narrative text from the introspection reporter
- Controls for state transitions and sensory input

All external dependencies (gradio, pandas) are imported lazily so the
module can be imported without them installed. The dashboard degrades
gracefully when the system is not running (shows cached/empty data).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gradio as gr

    from mirror_box.introspection.reporter import StateReporter

from aiware.interfaces.data_types import ConsciousState
from aiware.interfaces.protocols import ConsciousLoop, MetricsCollector, StateController

logger = logging.getLogger(__name__)

# Color scheme for conscious states
_STATE_COLORS: dict[str, str] = {
    "wake": "#22c55e",    # green
    "sleep": "#6366f1",   # indigo
    "dream": "#a855f7",   # purple
    "altered": "#ef4444",  # red
}

_STATE_LABELS: dict[str, str] = {
    "wake": "WAKE -- fully conscious, external input active",
    "sleep": "SLEEP -- subcritical, simulation collapsed",
    "dream": "DREAM -- near-critical, internally generated content",
    "altered": "ALTERED -- increased permeability, content flooding",
}


class MirrorBoxDashboard:
    """Real-time visualization dashboard for the Mirror Box system.

    Shows metric trajectories, state indicators, gate visualization,
    introspection narratives, and control buttons. Built on Gradio Blocks.

    The dashboard works even when the system is not running -- it displays
    whatever data the MetricsCollector has cached (or empty plots if none).

    Args:
        loop: The conscious processing loop (for is_running / frame_count).
        metrics: Metrics collector providing time-series data.
        state_ctrl: State controller for current state and transitions.
        reporter: Optional introspection reporter for EWM/ESM narratives.
    """

    def __init__(
        self,
        loop: ConsciousLoop,
        metrics: MetricsCollector,
        state_ctrl: StateController,
        reporter: StateReporter | None = None,
    ) -> None:
        self._loop = loop
        self._metrics = metrics
        self._state_ctrl = state_ctrl
        self._reporter = reporter

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_metric_plot(self, metric_name: str, window: int = 200) -> Any:
        """Generate a pandas DataFrame for a metric trajectory.

        Args:
            metric_name: One of "R", "C", "self_surprise", "complexity",
                or any key in FrameMetrics.extras.
            window: Number of most recent frames to include.

        Returns:
            pandas DataFrame with columns "frame" and the metric name,
            or None if no data is available.
        """
        import pandas as pd

        values = self._metrics.get_trajectory(metric_name, window=window)
        if not values:
            return pd.DataFrame({"frame": [], metric_name: []})

        # Frame indices: use frame_count to align to actual frame numbers
        total = self._metrics.frame_count()
        start = max(0, total - len(values))
        frames = list(range(start, start + len(values)))

        return pd.DataFrame({"frame": frames, metric_name: values})

    def _get_gate_data(self) -> Any:
        """Generate a DataFrame for the gate activation bar chart.

        Returns:
            pandas DataFrame with columns "channel" and "activation".
        """
        import pandas as pd

        latest = self._metrics.get_latest()
        if latest is None:
            return pd.DataFrame({
                "channel": ["world", "self"],
                "activation": [0.0, 0.0],
            })

        world_act = latest.extras.get("gate_world_activation", 0.0)
        self_act = latest.extras.get("gate_self_activation", 0.0)

        return pd.DataFrame({
            "channel": ["world", "self"],
            "activation": [world_act, self_act],
        })

    def _get_state_display(self) -> str:
        """Generate state indicator text with formatting.

        Returns:
            Markdown-formatted string showing current state, time in state,
            and frame count.
        """
        state = self._state_ctrl.current_state()
        color = _STATE_COLORS.get(state.value, "#888888")
        label = _STATE_LABELS.get(state.value, state.value.upper())
        time_s = self._state_ctrl.time_in_state()
        running = self._loop.is_running()
        frames = self._loop.frame_count()

        lines = [
            f"### <span style='color:{color}'>{state.value.upper()}</span>",
            "",
            f"**Description:** {label}",
            f"**Time in state:** {time_s:.1f}s",
            f"**Frames processed:** {frames}",
            f"**Loop running:** {'Yes' if running else 'No'}",
        ]

        return "\n".join(lines)

    def _get_ewm_narrative(self) -> str:
        """Get EWM narrative text from the reporter.

        Returns:
            EWM narrative string, or a placeholder if no reporter.
        """
        if self._reporter is None:
            return "(No introspection reporter configured)"
        try:
            return self._reporter.generate_ewm_narrative()
        except Exception as exc:
            logger.warning("EWM narrative generation failed: %s", exc)
            return f"(EWM narrative error: {exc})"

    def _get_esm_narrative(self) -> str:
        """Get ESM narrative text from the reporter.

        Returns:
            ESM narrative string, or a placeholder if no reporter.
        """
        if self._reporter is None:
            return "(No introspection reporter configured)"
        try:
            return self._reporter.generate_esm_narrative()
        except Exception as exc:
            logger.warning("ESM narrative generation failed: %s", exc)
            return f"(ESM narrative error: {exc})"

    def _trigger_state_transition(self, target_state: str) -> str:
        """Handle state transition button click.

        Args:
            target_state: Target state name (wake/sleep/dream/altered).

        Returns:
            Status message describing the result.
        """
        try:
            target = ConsciousState(target_state)
        except ValueError:
            return f"Invalid state: {target_state}"

        current = self._state_ctrl.current_state()
        if current == target:
            return f"Already in {target.value.upper()}"

        accepted = self._state_ctrl.transition_to(target)
        if accepted:
            return f"Transitioned: {current.value.upper()} -> {target.value.upper()}"
        return f"Transition to {target.value.upper()} rejected"

    # ------------------------------------------------------------------
    # Refresh callback (called by timer or button)
    # ------------------------------------------------------------------

    def _refresh_all(self) -> tuple[Any, ...]:
        """Refresh all dashboard components.

        Returns:
            Tuple of updated values for all output components in order:
            (R_plot, C_plot, surprise_plot, complexity_plot,
             state_display, gate_plot, ewm_text, esm_text)
        """
        return (
            self._get_metric_plot("R"),
            self._get_metric_plot("C"),
            self._get_metric_plot("self_surprise"),
            self._get_metric_plot("complexity"),
            self._get_state_display(),
            self._get_gate_data(),
            self._get_ewm_narrative(),
            self._get_esm_narrative(),
        )

    # ------------------------------------------------------------------
    # Build the Gradio interface
    # ------------------------------------------------------------------

    def build(self) -> gr.Blocks:
        """Build the Gradio Blocks interface.

        Layout:
        - Top row: State indicator + controls (state transition buttons)
        - Middle: 2x2 grid of metric line plots (R, C, surprise, complexity)
        - Bottom left: Gate activation bar chart
        - Bottom right: Introspection text (EWM/ESM narratives)
        - Footer: Sensory input text box + refresh button

        Returns:
            A gr.Blocks instance ready to launch.
        """
        import gradio as gr

        # Store theme for launch(); Gradio 6 moved theme from Blocks to launch()
        self._theme = gr.themes.Soft()

        with gr.Blocks(
            title="Mirror Box -- Consciousness Dashboard",
        ) as demo:
            gr.Markdown("# Mirror Box -- Consciousness Dashboard")
            gr.Markdown(
                "Real-time monitoring of the Four-Model pseudo-AC system. "
                "Displays self-knowledge (R), conscious content (C), "
                "self-surprise, complexity, and introspection narratives."
            )

            # Auto-refresh timer (every 500ms when active)
            timer = gr.Timer(value=0.5, active=False)

            # ---- Top row: state indicator + controls ----
            with gr.Row():
                with gr.Column(scale=2):
                    state_display = gr.Markdown(
                        value=self._get_state_display(),
                        label="Conscious State",
                    )
                with gr.Column(scale=3):
                    gr.Markdown("### Controls")
                    with gr.Row():
                        btn_wake = gr.Button("Wake", variant="primary", size="sm")
                        btn_sleep = gr.Button("Sleep", variant="secondary", size="sm")
                        btn_dream = gr.Button("Dream", variant="secondary", size="sm")
                        btn_altered = gr.Button("Altered", variant="stop", size="sm")
                    transition_status = gr.Textbox(
                        label="Transition Status",
                        interactive=False,
                        max_lines=1,
                    )
                    with gr.Row():
                        btn_refresh = gr.Button("Refresh Now", variant="secondary")
                        btn_auto = gr.Button("Start Auto-Refresh", variant="primary")
                        btn_stop_auto = gr.Button("Stop Auto-Refresh", variant="stop")

            # ---- Middle: 2x2 metric plots ----
            gr.Markdown("### Metric Trajectories")
            with gr.Row():
                with gr.Column():
                    r_plot = gr.LinePlot(
                        value=self._get_metric_plot("R"),
                        x="frame",
                        y="R",
                        title="Self-Knowledge R(t)",
                        y_title="R [0,1]",
                        x_title="Frame",
                    )
                with gr.Column():
                    c_plot = gr.LinePlot(
                        value=self._get_metric_plot("C"),
                        x="frame",
                        y="C",
                        title="Conscious Content C(t)",
                        y_title="C",
                        x_title="Frame",
                    )
            with gr.Row():
                with gr.Column():
                    surprise_plot = gr.LinePlot(
                        value=self._get_metric_plot("self_surprise"),
                        x="frame",
                        y="self_surprise",
                        title="Self-Surprise",
                        y_title="Surprise",
                        x_title="Frame",
                    )
                with gr.Column():
                    complexity_plot = gr.LinePlot(
                        value=self._get_metric_plot("complexity"),
                        x="frame",
                        y="complexity",
                        title="Trajectory Complexity",
                        y_title="LZ Complexity",
                        x_title="Frame",
                    )

            # ---- Bottom: gate visualization + narratives ----
            gr.Markdown("### Gate & Introspection")
            with gr.Row():
                with gr.Column(scale=1):
                    gate_plot = gr.BarPlot(
                        value=self._get_gate_data(),
                        x="channel",
                        y="activation",
                        title="Permeability Gate Activations",
                        y_title="Activation [0,1]",
                        x_title="Channel",
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            ewm_text = gr.Textbox(
                                value=self._get_ewm_narrative(),
                                label="EWM Narrative (World Perception)",
                                lines=5,
                                interactive=False,
                            )
                        with gr.Column():
                            esm_text = gr.Textbox(
                                value=self._get_esm_narrative(),
                                label="ESM Narrative (Self-Model)",
                                lines=5,
                                interactive=False,
                            )

            # ---- Footer: sensory input ----
            gr.Markdown("### Sensory Input")
            with gr.Row():
                sensory_input = gr.Textbox(
                    label="Text Input (sets sensory tokens for next frame)",
                    placeholder="Type text to feed as sensory input...",
                    lines=1,
                )
                btn_send_input = gr.Button("Send Input", variant="primary", size="sm")

            input_status = gr.Textbox(
                label="Input Status",
                interactive=False,
                max_lines=1,
            )

            # ---- All output components for refresh ----
            all_outputs = [
                r_plot, c_plot, surprise_plot, complexity_plot,
                state_display, gate_plot, ewm_text, esm_text,
            ]

            # ---- Wire up events ----

            # State transition buttons
            btn_wake.click(
                fn=lambda: self._trigger_state_transition("wake"),
                outputs=transition_status,
            )
            btn_sleep.click(
                fn=lambda: self._trigger_state_transition("sleep"),
                outputs=transition_status,
            )
            btn_dream.click(
                fn=lambda: self._trigger_state_transition("dream"),
                outputs=transition_status,
            )
            btn_altered.click(
                fn=lambda: self._trigger_state_transition("altered"),
                outputs=transition_status,
            )

            # Manual refresh
            btn_refresh.click(
                fn=self._refresh_all,
                outputs=all_outputs,
            )

            # Auto-refresh timer controls
            btn_auto.click(
                fn=lambda: gr.Timer(active=True),
                outputs=timer,
            )
            btn_stop_auto.click(
                fn=lambda: gr.Timer(active=False),
                outputs=timer,
            )

            # Timer tick triggers refresh
            timer.tick(
                fn=self._refresh_all,
                outputs=all_outputs,
            )

            # Sensory input handler
            def _handle_sensory_input(text: str) -> str:
                """Convert text to token IDs and set on the loop."""
                if not text.strip():
                    return "No input provided"
                try:
                    import torch

                    # Simple character-level encoding (real system uses tokenizer)
                    token_ids = torch.tensor(
                        [ord(c) % 256 for c in text[:128]],
                        dtype=torch.long,
                    )
                    # Check if the loop has set_sensory_input
                    if hasattr(self._loop, "set_sensory_input"):
                        self._loop.set_sensory_input(token_ids)
                        return f"Sent {len(token_ids)} tokens from '{text[:32]}...'"
                    return "Loop does not support set_sensory_input"
                except Exception as exc:
                    return f"Error: {exc}"

            btn_send_input.click(
                fn=_handle_sensory_input,
                inputs=sensory_input,
                outputs=input_status,
            )

        return demo

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self, **kwargs: Any) -> None:
        """Build and launch the dashboard.

        Args:
            **kwargs: Passed through to gr.Blocks.launch()
                (e.g., server_name, server_port, share, etc.).
        """
        demo = self.build()
        # Pass theme at launch time (Gradio 6 API)
        if "theme" not in kwargs and hasattr(self, "_theme"):
            kwargs["theme"] = self._theme
        demo.launch(**kwargs)
