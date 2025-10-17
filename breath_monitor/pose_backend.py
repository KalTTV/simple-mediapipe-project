"""
MediaPipe Pose Landmarker backend.
Extracts chest proxy signal from shoulder landmarks.
"""

import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple
import cv2


class PoseBackend:
    """MediaPipe Pose Landmarker for breathing detection."""
    
    def __init__(self):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for shoulders
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        
    def infer(self, frame: np.ndarray) -> Dict:
        """
        Process frame and extract breathing proxy signal.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary containing:
                - landmarks: Full pose landmarks
                - mid_shoulder_xy: (x, y) mid-shoulder position in image coordinates
                - mid_shoulder_world: (x, y, z) mid-shoulder position in world coordinates (if available)
                - confidence: Average visibility of shoulders
                - detected: Whether pose was detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                "landmarks": None,
                "mid_shoulder_xy": None,
                "mid_shoulder_world": None,
                "confidence": 0.0,
                "detected": False
            }
        
        # Extract shoulder landmarks
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        
        # Calculate mid-shoulder position in image coordinates
        mid_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Calculate confidence from visibility
        confidence = (left_shoulder.visibility + right_shoulder.visibility) / 2
        
        # World coordinates (if available)
        mid_world = None
        if results.pose_world_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            left_world = world_landmarks[self.LEFT_SHOULDER]
            right_world = world_landmarks[self.RIGHT_SHOULDER]
            mid_world = (
                (left_world.x + right_world.x) / 2,
                (left_world.y + right_world.y) / 2,
                (left_world.z + right_world.z) / 2
            )
        
        return {
            "landmarks": results.pose_landmarks,
            "mid_shoulder_xy": (mid_x, mid_y),
            "mid_shoulder_world": mid_world,
            "confidence": confidence,
            "detected": True
        }
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def main():
    """Test pose detection with webcam."""
    import sys
    sys.path.append("..")
    from breath_monitor.capture import CameraCapture
    
    capture = CameraCapture()
    backend = PoseBackend()
    
    if not capture.start():
        return
    
    print("Press 'q' to quit")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    try:
        for frame, timestamp in capture.frames():
            result = backend.infer(frame)
            
            # Draw pose landmarks
            if result["detected"]:
                mp_drawing.draw_landmarks(
                    frame,
                    result["landmarks"],
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Draw mid-shoulder point
                if result["mid_shoulder_xy"]:
                    h, w = frame.shape[:2]
                    x, y = result["mid_shoulder_xy"]
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    
                    # Display confidence
                    cv2.putText(frame, f"Conf: {result['confidence']:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Pose Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        capture.stop()
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

