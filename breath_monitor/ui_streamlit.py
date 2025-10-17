"""
Streamlit UI for breathing monitor.
Chest expansion detection with multi-point tracking controls.
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

from breath_monitor.capture import CameraCapture
from breath_monitor.pose_backend import PoseBackend
from breath_monitor.signal import BreathingAnalyzer
from breath_monitor.draw import BreathingVisualizer


# Page config
st.set_page_config(
    page_title="Breathing Monitor",
    page_icon="🫁",
    layout="wide"
)

# Title and warning
st.title("🫁 Breathing Rate Monitor - Chest Expansion Detection")
st.error("⚠️ NOT A MEDICAL DEVICE - FOR RESEARCH/DEMO ONLY")

# Sidebar controls
st.sidebar.header("Settings")

camera_id = st.sidebar.number_input("Camera ID", value=0, min_value=0, max_value=10)
min_sec = st.sidebar.slider("Buffer Window (sec)", 10.0, 60.0, 30.0)

# Respiratory band
st.sidebar.subheader("Respiratory Band")
resp_low = st.sidebar.slider("Low (Hz)", 0.05, 1.0, 0.08, 0.01)
resp_high = st.sidebar.slider("High (Hz)", 0.5, 2.0, 1.2, 0.1)

# Tracking controls
st.sidebar.subheader("Chest Tracking")
trackers = st.sidebar.slider("Feature Points", 8, 48, 32)
min_ibi_sec = st.sidebar.slider("Min IBI (debounce) sec", 0.5, 5.0, 2.0, 0.1)

# Apnea threshold
apnea_sec = st.sidebar.slider("Apnea Threshold (sec)", 10.0, 40.0, 20.0)

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.capture = None
    st.session_state.pose = None
    st.session_state.analyzer = None
    st.session_state.bpm_history = deque(maxlen=600)
    st.session_state.time_history = deque(maxlen=600)

# Start/Stop button
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", disabled=st.session_state.running):
        st.session_state.capture = CameraCapture(camera_id, 640, 480, 30)
        st.session_state.pose = PoseBackend(num_trackers=trackers)
        st.session_state.analyzer = BreathingAnalyzer(
            window_sec=min_sec,
            resp_low=resp_low,
            resp_high=resp_high,
            min_ibi_sec=min_ibi_sec,
            apnea_sec=apnea_sec
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
        tracks_placeholder = st.empty()
    
    chart_placeholder = st.empty()
    
    # Process frames
    frame_count = 0
    max_frames = 3000  # ~100 seconds
    
    for frame, timestamp in st.session_state.capture.frames():
        frame_count += 1
        
        # Process frame
        pose_result = st.session_state.pose.infer(frame, timestamp)
        
        # Add signal sample
        visibility = pose_result.get("confidence", 0.0)
        tracks_alive = pose_result.get("num_tracked_points", 0)
        
        if pose_result["detected"] and visibility > 0.3:
            tracks_y = pose_result.get("tracks_y", np.array([]))
            
            value = 0.5
            if pose_result.get("mid_shoulder_xy"):
                _, y = pose_result["mid_shoulder_xy"]
                value = y
            
            st.session_state.analyzer.add_sample(timestamp, value, tracks_y)
        
        # Analyze
        analysis = st.session_state.analyzer.analyze(
            visibility=visibility,
            tracks_alive=tracks_alive,
            total_tracks=trackers
        )
        
        # Draw visualization with tracking markers
        if st.session_state.visualizer:
            frame = st.session_state.visualizer.draw_all(frame, pose_result, analysis)
        
        # Update history
        bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
        elapsed = timestamp - st.session_state.start_time
        st.session_state.time_history.append(elapsed)
        st.session_state.bpm_history.append(bpm if bpm else 0)
        
        # Update UI every 3 frames
        if frame_count % 3 == 0:
            # Convert BGR to RGB for display (frame already has visualizations)
            display_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
            
            # BPM display
            conf = analysis.get("conf", 0.0)
            
            if bpm is not None and np.isfinite(bpm):
                # Color based on confidence
                if conf >= 0.7:
                    bpm_color = "green"
                elif conf >= 0.4:
                    bpm_color = "orange"
                else:
                    bpm_color = "yellow"
                
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
                badges.append("🟡 SHALLOW")
            
            if badges:
                badges_placeholder.markdown(" | ".join(badges))
            else:
                badges_placeholder.markdown("🟢 Normal")
            
            # Confidence and tracking info
            confidence_placeholder.progress(conf, text=f"Confidence: {conf:.0%}")
            tracks_placeholder.metric("Active Trackers", f"{tracks_alive}/{trackers}")
            
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
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("BPM")
                    ax.set_title("Breathing Rate Trend (Last 60s)")
                    ax.grid(True, alpha=0.3)
                    
                    # Show last 60 seconds
                    if len(times_filtered) > 0:
                        ax.set_xlim(max(0, times_filtered[-1] - 60), times_filtered[-1])
                    
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
        
        # Stop after max frames or if user clicked stop
        if frame_count >= max_frames or not st.session_state.running:
            break
        
        time.sleep(0.01)


if __name__ == "__main__":
    pass
