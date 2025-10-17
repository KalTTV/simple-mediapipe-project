"""
Streamlit UI for breathing monitor.
Provides live video, BPM display, status badges, trend chart, and confidence meter.
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

from .capture import CameraCapture
from .pose_backend import PoseBackend
from .signal import BreathingAnalyzer
from .draw import BreathingVisualizer


# Page config
st.set_page_config(
    page_title="Breathing Monitor",
    page_icon="🫁",
    layout="wide"
)

# Title and warning
st.title("🫁 Infant Breathing Rate Monitor")
st.warning("⚠️ FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE")

# Sidebar controls
st.sidebar.header("Settings")

camera_id = st.sidebar.number_input("Camera ID", value=0, min_value=0, max_value=10)
min_sec = st.sidebar.slider("Buffer Window (sec)", 10.0, 30.0, 15.0)
bpf_low = st.sidebar.slider("Bandpass Low (Hz)", 0.2, 1.0, 0.5, 0.1)
bpf_high = st.sidebar.slider("Bandpass High (Hz)", 0.8, 2.0, 1.2, 0.1)
apnea_sec = st.sidebar.slider("Apnea Threshold (sec)", 10.0, 40.0, 20.0)
tachy = st.sidebar.slider("Tachypnea Threshold (BPM)", 50.0, 80.0, 60.0)
brady = st.sidebar.slider("Bradypnea Threshold (BPM)", 20.0, 40.0, 30.0)

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.capture = None
    st.session_state.pose = None
    st.session_state.analyzer = None
    st.session_state.visualizer = None
    st.session_state.bpm_history = deque(maxlen=300)  # 30 seconds at ~10 Hz
    st.session_state.time_history = deque(maxlen=300)

# Start/Stop button
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", disabled=st.session_state.running):
        st.session_state.capture = CameraCapture(camera_id, 640, 480, 30)
        st.session_state.pose = PoseBackend()
        st.session_state.analyzer = BreathingAnalyzer(
            window_sec=min_sec,
            bpf_low=bpf_low,
            bpf_high=bpf_high,
            apnea_sec=apnea_sec,
            tachy_threshold=tachy,
            brady_threshold=brady
        )
        st.session_state.visualizer = BreathingVisualizer()
        
        if st.session_state.capture.start():
            st.session_state.running = True
            st.session_state.start_time = time.time()
            st.rerun()

with col2:
    if st.button("⏹️ Stop", disabled=not st.session_state.running):
        if st.session_state.capture:
            st.session_state.capture.stop()
        if st.session_state.pose:
            st.session_state.pose.close()
        st.session_state.running = False
        st.rerun()

# Main UI
if not st.session_state.running:
    st.info("Click 'Start' to begin monitoring")
else:
    # Create layout
    video_col, stats_col = st.columns([2, 1])
    
    with video_col:
        st.subheader("Live Video")
        video_placeholder = st.empty()
    
    with stats_col:
        st.subheader("Status")
        bpm_placeholder = st.empty()
        badges_placeholder = st.empty()
        confidence_placeholder = st.empty()
    
    chart_placeholder = st.empty()
    
    # Process frames
    frame_count = 0
    max_frames = 1000  # Process up to 1000 frames
    
    for frame, timestamp in st.session_state.capture.frames():
        frame_count += 1
        
        # Process frame
        pose_result = st.session_state.pose.infer(frame)
        
        # Add signal sample
        if pose_result["detected"] and pose_result["confidence"] > 0.5:
            if pose_result["mid_shoulder_xy"]:
                _, y = pose_result["mid_shoulder_xy"]
                st.session_state.analyzer.add_sample(timestamp, y)
        
        # Analyze
        analysis = st.session_state.analyzer.analyze()
        
        # Draw visualization
        display_frame = st.session_state.visualizer.draw_all(frame, pose_result, analysis)
        
        # Update history
        bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
        elapsed = timestamp - st.session_state.start_time
        st.session_state.time_history.append(elapsed)
        st.session_state.bpm_history.append(bpm if bpm else 0)
        
        # Update UI every 3 frames
        if frame_count % 3 == 0:
            # Convert BGR to RGB for display
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
            
            # BPM display
            if bpm is not None:
                bpm_color = "green"
                if analysis.get("tachypnea"):
                    bpm_color = "orange"
                elif analysis.get("bradypnea"):
                    bpm_color = "red"
                
                bpm_placeholder.markdown(
                    f"<h1 style='text-align: center; color: {bpm_color};'>{bpm:.1f} BPM</h1>",
                    unsafe_allow_html=True
                )
            else:
                bpm_placeholder.markdown(
                    "<h1 style='text-align: center; color: gray;'>-- BPM</h1>",
                    unsafe_allow_html=True
                )
            
            # Status badges
            badges = []
            if analysis.get("apnea"):
                badges.append("🔴 APNEA")
            if analysis.get("shallow"):
                badges.append("🟠 SHALLOW")
            if analysis.get("tachypnea"):
                badges.append("🟠 TACHYPNEA")
            if analysis.get("bradypnea"):
                badges.append("🟡 BRADYPNEA")
            
            if badges:
                badges_placeholder.markdown(" | ".join(badges))
            else:
                badges_placeholder.markdown("🟢 Normal")
            
            # Confidence meter
            confidence = pose_result.get("confidence", 0.0)
            confidence_placeholder.progress(confidence, text=f"Confidence: {confidence:.0%}")
            
            # Trend chart
            if len(st.session_state.bpm_history) > 10:
                fig, ax = plt.subplots(figsize=(10, 3))
                times = list(st.session_state.time_history)
                bpms = list(st.session_state.bpm_history)
                
                # Filter out zeros
                times_filtered = [t for t, b in zip(times, bpms) if b > 0]
                bpms_filtered = [b for b in bpms if b > 0]
                
                if times_filtered:
                    ax.plot(times_filtered, bpms_filtered, 'g-', linewidth=2)
                    ax.axhline(y=tachy, color='orange', linestyle='--', label='Tachy')
                    ax.axhline(y=brady, color='red', linestyle='--', label='Brady')
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("BPM")
                    ax.set_title("30-Second Breathing Rate Trend")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Show only last 30 seconds
                    if len(times_filtered) > 0:
                        ax.set_xlim(max(0, times_filtered[-1] - 30), times_filtered[-1])
                    
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
        
        # Stop after max frames or if user clicked stop
        if frame_count >= max_frames or not st.session_state.running:
            break
        
        # Small delay to prevent overwhelming the UI
        time.sleep(0.01)


def main():
    """Main entry point for Streamlit UI."""
    pass


if __name__ == "__main__":
    main()

