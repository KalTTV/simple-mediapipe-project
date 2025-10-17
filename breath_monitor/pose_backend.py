"""MediaPipe Pose backend for extracting keypoints from video frames."""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple


class PoseBackend:
    """Wrapper around MediaPipe Pose for extracting body keypoints from frames."""
    
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose backend.
        
        Args:
            static_image_mode: If True, treats each frame independently.
            model_complexity: Complexity of pose model (0, 1, or 2).
            smooth_landmarks: If True, filters landmarks to reduce jitter.
            enable_segmentation: If True, generates segmentation mask.
            smooth_segmentation: If True, filters segmentation mask.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single video frame and extract pose keypoints.
        
        Args:
            frame: Input frame in BGR format (OpenCV format).
            
        Returns:
            Dictionary containing keypoints and metadata, or None if no pose detected.
            Format:
            {
                'landmarks': list of {x, y, z, visibility} dicts for each keypoint,
                'world_landmarks': list of {x, y, z, visibility} dicts in world coordinates,
                'frame_shape': (height, width, channels)
            }
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks is None:
            return None
            
        # Extract landmarks as list of dicts
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # Extract world landmarks if available
        world_landmarks = []
        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                world_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        
        return {
            'landmarks': landmarks,
            'world_landmarks': world_landmarks,
            'frame_shape': frame.shape
        }
    
    def get_keypoint_indices(self) -> Dict[str, int]:
        """
        Get mapping of keypoint names to their indices.
        
        Returns:
            Dictionary mapping keypoint names to indices.
        """
        return {
            'NOSE': 0,
            'LEFT_EYE_INNER': 1,
            'LEFT_EYE': 2,
            'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4,
            'RIGHT_EYE': 5,
            'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7,
            'RIGHT_EAR': 8,
            'MOUTH_LEFT': 9,
            'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_PINKY': 17,
            'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19,
            'RIGHT_INDEX': 20,
            'LEFT_THUMB': 21,
            'RIGHT_THUMB': 22,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29,
            'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31,
            'RIGHT_FOOT_INDEX': 32
        }
    
    def get_keypoint(self, landmarks: List[Dict], keypoint_name: str) -> Optional[Dict]:
        """
        Get specific keypoint by name.
        
        Args:
            landmarks: List of landmark dictionaries from process_frame.
            keypoint_name: Name of keypoint (e.g., 'LEFT_SHOULDER').
            
        Returns:
            Dictionary with {x, y, z, visibility} or None if not found.
        """
        indices = self.get_keypoint_indices()
        if keypoint_name not in indices:
            return None
        
        idx = indices[keypoint_name]
        if idx >= len(landmarks):
            return None
            
        return landmarks[idx]
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """
        Draw pose landmarks on frame.
        
        Args:
            frame: Input frame to draw on.
            landmarks: List of landmark dictionaries.
            
        Returns:
            Frame with landmarks drawn.
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Draw connections
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Convert landmarks back to MediaPipe format for drawing
        from mediapipe.framework.formats import landmark_pb2
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        for lm in landmarks:
            landmark = landmark_list.landmark.add()
            landmark.x = lm['x']
            landmark.y = lm['y']
            landmark.z = lm['z']
            landmark.visibility = lm['visibility']
        
        # Create a dummy results object for drawing
        class DummyResults:
            def __init__(self, landmarks):
                self.pose_landmarks = landmarks
        
        results = DummyResults(landmark_list)
        
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        return annotated_frame
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
