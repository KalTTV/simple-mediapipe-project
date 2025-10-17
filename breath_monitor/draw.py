"""
Drawing utilities for pose overlay and HUD display.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class BreathingVisualizer:
    """Visualizer for breathing monitor with pose overlay and HUD."""
    
    def __init__(self, trace_length: int = 100):
        """
        Initialize visualizer.
        
        Args:
            trace_length: Number of points in mid-shoulder trace
        """
        self.trace_points = deque(maxlen=trace_length)
        self.mp_drawing = None
        self.mp_pose = None
        
        try:
            import mediapipe as mp
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
        except ImportError:
            print("[WARN] MediaPipe not available for drawing")
    
    def add_trace_point(self, x: float, y: float, frame_shape: Tuple[int, int]):
        """
        Add point to mid-shoulder trace.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            frame_shape: (height, width) of frame
        """
        h, w = frame_shape[:2]
        px = int(x * w)
        py = int(y * h)
        self.trace_points.append((px, py))
    
    def draw_pose(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Frame with pose overlay
        """
        if landmarks and self.mp_drawing and self.mp_pose:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        return frame
    
    def draw_trace(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw mid-shoulder movement trace.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with trace overlay
        """
        if len(self.trace_points) < 2:
            return frame
        
        # Draw trace line
        points = list(self.trace_points)
        for i in range(len(points) - 1):
            # Fade color from old to new
            alpha = i / len(points)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, points[i], points[i + 1], color, 2)
        
        return frame
    
    def draw_hud(self, frame: np.ndarray, bpm: Optional[float], apnea: bool, 
                shallow: bool, confidence: float, tachypnea: bool = False, 
                bradypnea: bool = False) -> np.ndarray:
        """
        Draw HUD with BPM and status badges.
        
        Args:
            frame: Input frame
            bpm: Current BPM
            apnea: Apnea status
            shallow: Shallow breathing status
            confidence: Confidence level
            tachypnea: Tachypnea status
            bradypnea: Bradypnea status
            
        Returns:
            Frame with HUD overlay
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw BPM
        if bpm is not None and confidence > 0.5:
            bpm_text = f"BPM: {bpm:.1f}"
            color = (0, 255, 0)  # Green
            
            # Change color based on thresholds
            if tachypnea:
                color = (0, 165, 255)  # Orange
            elif bradypnea:
                color = (0, 255, 255)  # Yellow
        else:
            bpm_text = "BPM: --"
            color = (128, 128, 128)  # Gray
        
        cv2.putText(frame, bpm_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, color, 3)
        
        # Draw confidence bar
        conf_width = int(200 * confidence)
        cv2.rectangle(frame, (20, 70), (220, 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 70), (20 + conf_width, 90), (0, 255, 0), -1)
        cv2.putText(frame, f"Conf: {confidence:.0%}", (230, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw status badges
        y_offset = 110
        badge_spacing = 10
        
        badges = []
        if apnea:
            badges.append(("APNEA", (0, 0, 255)))  # Red
        if shallow:
            badges.append(("SHALLOW", (0, 165, 255)))  # Orange
        if tachypnea:
            badges.append(("TACHY", (0, 165, 255)))  # Orange
        if bradypnea:
            badges.append(("BRADY", (0, 255, 255)))  # Yellow
        
        x_offset = 20
        for text, color in badges:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            badge_width = text_size[0] + 20
            
            # Draw badge background
            cv2.rectangle(frame, (x_offset, y_offset), 
                         (x_offset + badge_width, y_offset + 30), color, -1)
            
            # Draw badge text
            cv2.putText(frame, text, (x_offset + 10, y_offset + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            x_offset += badge_width + badge_spacing
        
        return frame
    
    def draw_chest_roi(self, frame: np.ndarray, roi: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw chest ROI polygon.
        
        Args:
            frame: Input frame
            roi: 4-point polygon or None
            
        Returns:
            Frame with ROI overlay
        """
        if roi is not None and len(roi) == 4:
            roi_int = roi.astype(np.int32)
            cv2.polylines(frame, [roi_int], isClosed=True, color=(255, 255, 0), thickness=2)
        return frame
    
    def draw_tracked_points(self, frame: np.ndarray, tracked_points: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw tracked chest points.
        
        Args:
            frame: Input frame
            tracked_points: Array of shape (N, 1, 2) or None
            
        Returns:
            Frame with tracked points overlay
        """
        if tracked_points is not None and len(tracked_points) > 0:
            for point in tracked_points:
                x, y = point[0]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)  # Yellow dots
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), 1)   # Yellow rings
        return frame
    
    def draw_all(self, frame: np.ndarray, pose_result: dict, analysis_result: dict) -> np.ndarray:
        """
        Draw complete visualization with tracking markers.
        
        Args:
            frame: Input frame
            pose_result: Result from PoseBackend.infer()
            analysis_result: Result from BreathingAnalyzer.analyze()
            
        Returns:
            Frame with all overlays
        """
        # Draw pose skeleton
        if pose_result.get("detected") and pose_result.get("landmarks"):
            frame = self.draw_pose(frame, pose_result["landmarks"])
            
            # Draw chest ROI
            chest_roi = pose_result.get("chest_roi")
            if chest_roi is not None:
                frame = self.draw_chest_roi(frame, chest_roi)
            
            # Draw tracked points
            tracked_points = pose_result.get("tracked_points")
            if tracked_points is not None:
                frame = self.draw_tracked_points(frame, tracked_points)
            
            # Add trace point for mid-shoulder
            if pose_result.get("mid_shoulder_xy"):
                x, y = pose_result["mid_shoulder_xy"]
                self.add_trace_point(x, y, frame.shape)
        
        # Draw movement trace
        frame = self.draw_trace(frame)
        
        # Draw HUD
        bpm = analysis_result.get("bpm_smooth") or analysis_result.get("bpm")
        frame = self.draw_hud(
            frame,
            bpm=bpm,
            apnea=analysis_result.get("apnea", False),
            shallow=analysis_result.get("shallow", False),
            confidence=pose_result.get("confidence", 0.0),
            tachypnea=False,  # Not using these in new version
            bradypnea=False
        )
        
        return frame


def draw_badge(frame: np.ndarray, text: str, position: Tuple[int, int], 
               color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Draw a status badge on frame.
    
    Args:
        frame: Input frame
        text: Badge text
        position: (x, y) position
        color: BGR color
        
    Returns:
        Frame with badge
    """
    x, y = position
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    # Background
    cv2.rectangle(frame, (x, y), (x + text_size[0] + 20, y + 35), color, -1)
    
    # Text
    cv2.putText(frame, text, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
    return frame


def draw_waveform(frame: np.ndarray, signal_data: np.ndarray, 
                 position: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Draw signal waveform on frame.
    
    Args:
        frame: Input frame
        signal_data: Signal values
        position: (x, y, width, height) of plot area
        
    Returns:
        Frame with waveform
    """
    if len(signal_data) < 2:
        return frame
    
    x, y, width, height = position
    
    # Normalize signal to fit in plot area
    signal_norm = (signal_data - np.min(signal_data)) / (np.ptp(signal_data) + 1e-6)
    signal_scaled = height - (signal_norm * height).astype(int)
    
    # Sample points to fit width
    indices = np.linspace(0, len(signal_data) - 1, width).astype(int)
    
    # Draw waveform
    points = [(x + i, y + signal_scaled[indices[i]]) for i in range(len(indices))]
    
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
    
    return frame

