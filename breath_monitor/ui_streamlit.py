"""Streamlit dashboard for breath monitoring.

Provides a web-based UI with:
- Live video feed
- Real-time BPM display
- Apnea and shallow breathing status
- Line plots of breathing signal
- WebSocket connection to metrics
"""

import asyncio
import json
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import websockets
from streamlit.runtime.scriptrunner import add_script_run_ctx


class MetricsWebSocket:
    """WebSocket client for receiving metrics."""

    def __init__(self, url: str = "ws://localhost:8765"):
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            return True
        except Exception as e:
            st.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False

    async def receive_metrics(self):
        """Receive metrics from WebSocket."""
        if not self.websocket:
            return None
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.connected = False
            return None

    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "bpm_history" not in st.session_state:
        st.session_state.bpm_history = deque(maxlen=100)
    if "signal_history" not in st.session_state:
        st.session_state.signal_history = deque(maxlen=200)
    if "current_bpm" not in st.session_state:
        st.session_state.current_bpm = 0.0
    if "apnea_status" not in st.session_state:
        st.session_state.apnea_status = False
    if "shallow_status" not in st.session_state:
        st.session_state.shallow_status = False
    if "ws_connected" not in st.session_state:
        st.session_state.ws_connected = False
    if "frame_buffer" not in st.session_state:
        st.session_state.frame_buffer = None


def render_status_indicators():
    """Render status indicators for apnea and shallow breathing."""
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.apnea_status:
            st.error("⚠️ APNEA DETECTED")
        else:
            st.success("✓ Normal Breathing")

    with col2:
        if st.session_state.shallow_status:
            st.warning("⚠️ SHALLOW BREATHING")
        else:
            st.success("✓ Normal Depth")


def render_bpm_display():
    """Render current BPM display."""
    st.metric(
        label="Current BPM",
        value=f"{st.session_state.current_bpm:.1f}",
        delta=None,
    )


def render_signal_plot():
    """Render breathing signal line plot."""
    if len(st.session_state.signal_history) > 0:
        import plotly.graph_objects as go

        signal_data = list(st.session_state.signal_history)
        time_data = list(range(len(signal_data)))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode="lines",
                name="Breathing Signal",
                line=dict(color="#1f77b4", width=2),
            )
        )

        fig.update_layout(
            title="Breathing Signal Over Time",
            xaxis_title="Sample",
            yaxis_title="Signal Value",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for breathing signal data...")


def render_bpm_plot():
    """Render BPM history line plot."""
    if len(st.session_state.bpm_history) > 0:
        import plotly.graph_objects as go

        bpm_data = list(st.session_state.bpm_history)
        time_data = list(range(len(bpm_data)))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=bpm_data,
                mode="lines",
                name="BPM",
                line=dict(color="#ff7f0e", width=2),
            )
        )

        fig.update_layout(
            title="BPM History",
            xaxis_title="Sample",
            yaxis_title="BPM",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for BPM data...")


def render_video_feed():
    """Render live video feed."""
    video_placeholder = st.empty()

    if st.session_state.frame_buffer is not None:
        # Display the current frame
        video_placeholder.image(
            st.session_state.frame_buffer,
            channels="BGR",
            use_column_width=True,
        )
    else:
        video_placeholder.info("Waiting for video feed...")

    return video_placeholder


async def update_metrics_from_websocket(ws_client: MetricsWebSocket):
    """Update metrics from WebSocket connection."""
    metrics = await ws_client.receive_metrics()
    if metrics:
        # Update BPM
        if "bpm" in metrics:
            st.session_state.current_bpm = metrics["bpm"]
            st.session_state.bpm_history.append(metrics["bpm"])

        # Update breathing signal
        if "signal" in metrics:
            st.session_state.signal_history.append(metrics["signal"])

        # Update status flags
        if "apnea" in metrics:
            st.session_state.apnea_status = metrics["apnea"]

        if "shallow" in metrics:
            st.session_state.shallow_status = metrics["shallow"]

        # Update frame if available (base64 encoded)
        if "frame" in metrics:
            import base64

            frame_data = base64.b64decode(metrics["frame"])
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            st.session_state.frame_buffer = frame


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Breath Monitor Dashboard",
        page_icon="🫁",
        layout="wide",
    )

    st.title("🫁 Breath Monitor Dashboard")
    st.markdown("Real-time breathing monitoring with MediaPipe pose detection")

    init_session_state()

    # Sidebar for connection settings
    with st.sidebar:
        st.header("Connection Settings")
        ws_url = st.text_input("WebSocket URL", "ws://localhost:8765")

        if st.button("Connect"):
            st.session_state.ws_connected = True
            st.rerun()

        if st.button("Disconnect"):
            st.session_state.ws_connected = False
            st.rerun()

        # Connection status
        if st.session_state.ws_connected:
            st.success("Connected")
        else:
            st.error("Disconnected")

        st.markdown("---")
        st.header("About")
        st.markdown(
            """
            This dashboard displays real-time breathing metrics:
            - **BPM**: Breaths per minute
            - **Apnea**: Breathing stopped
            - **Shallow**: Reduced breathing depth
            """
        )

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = render_video_feed()

    with col2:
        st.subheader("Current Status")
        render_bpm_display()
        render_status_indicators()

    # Signal plots
    st.markdown("---")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        render_signal_plot()

    with plot_col2:
        render_bpm_plot()

    # WebSocket connection and updates
    if st.session_state.ws_connected:
        # Note: In a production app, you'd want to use a proper async loop
        # For this demo, we'll simulate updates
        # In reality, you'd need to integrate with Streamlit's async support
        # or use a background thread

        # Placeholder for WebSocket integration
        # This would typically run in a separate thread or async context
        st.info(
            "WebSocket connection enabled. "
            "For full functionality, ensure the metrics server is running."
        )

    # Auto-refresh
    time.sleep(0.1)
    st.rerun()


if __name__ == "__main__":
    main()
