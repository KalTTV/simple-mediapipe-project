"""
MediaPipe Pose Landmarker backend.
Enhanced with multi-point chest tracking for accurate breathing detection.
Integrated with pose classification for safe/unsafe position detection.
"""

import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple, List
import cv2
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


class PoseBackend:
    """MediaPipe Pose Landmarker with robust chest expansion tracking and position classification."""
    
    def __init__(self, num_trackers: int = 32, tracker_refresh: float = 0.5,
                 roi_shrink: float = 0.05, feature_quality: float = 0.001,
                 model_path: Optional[str] = None):
        """Initialize MediaPipe Pose and KLT tracking.
        
        Args:
            num_trackers: Number of chest feature points to track (default: 32)
            tracker_refresh: Time between forced refreshes or survival ratio (default: 0.5)
            roi_shrink: ROI shrink factor (default: 0.05)
            feature_quality: Quality level for goodFeaturesToTrack (default: 0.001)
            model_path: Path to trained position classifier (default: auto-detect)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        
        # KLT tracking state
        self.num_trackers = num_trackers
        self.tracker_refresh = tracker_refresh
        self.roi_shrink = roi_shrink
        self.feature_quality = feature_quality
        self.prev_gray = None
        self.tracked_points = None
        self.last_refresh_time = 0.0
        self.frame_count = 0
        
        # Load position classifier
        self.position_classifier = None
        self.position_scaler = None
        self._load_position_classifier(model_path)
        
    def _load_position_classifier(self, model_path: Optional[str] = None):
        """Load the trained position classifier model.
        
        Args:
            model_path: Path to the model file (default: auto-detect)
        """
        if model_path is None:
            # Auto-detect model path
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(current_dir, '..', 'models', 'position_model.joblib')
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.position_classifier = model_data['model']
                self.position_scaler = model_data['scaler']
                print(f"✅ Position classifier loaded from: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load position classifier: {e}")
                self.position_classifier = None
                self.position_scaler = None
        else:
            print(f"ℹ️  Position classifier not found at: {model_path}")
            print(f"   Run 'python -m breath_monitor.train_pose_classifier' to train the model.")
    
    def _extract_pose_features(self, landmarks) -> Optional[np.ndarray]:
        """
        Extract feature vector from pose landmarks for classification.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Feature vector matching the training format or None
        """
        if not landmarks:
            return None
        
        # Extract raw landmarks: [x, y, z, visibility] * 33
        lm_array = []
        for lm in landmarks.landmark:
            lm_array.extend([lm.x, lm.y, lm.z, lm.visibility])
        lm_array = np.array(lm_array)
        
        # Extract geometric features (same as training)
        lm = lm_array.reshape(33, 4)
        features = []
        
        # Key landmark positions
        nose = lm[0][:3]
        left_shoulder = lm[11][:3]
        right_shoulder = lm[12][:3]
        left_hip = lm[23][:3]
        right_hip = lm[24][:3]
        left_knee = lm[25][:3]
        right_knee = lm[26][:3]
        left_ankle = lm[27][:3]
        right_ankle = lm[28][:3]
        
        # 1. Torso orientation
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        spine_vector = mid_hip - mid_shoulder
        spine_angle = np.arctan2(spine_vector[1], spine_vector[0])
        features.extend([spine_angle, np.linalg.norm(spine_vector)])
        
        # 2. Head-torso alignment
        head_torso_vector = nose - mid_shoulder
        features.extend(head_torso_vector)
        
        # 3. Shoulder and hip width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_width = np.linalg.norm(right_hip - left_hip)
        features.extend([shoulder_width, hip_width])
        
        # 4. Body symmetry
        left_arm_length = np.linalg.norm(lm[15][:3] - left_shoulder)
        right_arm_length = np.linalg.norm(lm[16][:3] - right_shoulder)
        left_leg_length = np.linalg.norm(left_ankle - left_hip)
        right_leg_length = np.linalg.norm(right_ankle - right_hip)
        features.extend([
            left_arm_length - right_arm_length,
            left_leg_length - right_leg_length
        ])
        
        # 5. Key angles
        left_hip_angle = self._compute_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._compute_angle(right_shoulder, right_hip, right_knee)
        features.extend([left_hip_angle, right_hip_angle])
        
        # 6. Elevation ratios
        features.extend([nose[1], mid_shoulder[1], mid_hip[1]])
        
        # 7. Z-depth variation
        z_std = np.std([lm[i][2] for i in [0, 11, 12, 23, 24]])
        features.append(z_std)
        
        # 8. Visibility statistics
        face_visibility = np.mean([lm[i][3] for i in range(0, 10)])
        body_visibility = np.mean([lm[i][3] for i in [11, 12, 23, 24]])
        features.extend([face_visibility, body_visibility])
        
        # Combine raw landmarks + geometric features
        full_features = np.concatenate([lm_array, features])
        
        return full_features
    
    @staticmethod
    def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute angle at point b formed by points a-b-c."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return angle
    
    def classify_position(self, landmarks) -> Dict:
        """
        Classify sleeping position as safe or unsafe.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary containing:
                - position_safe: Boolean indicating if position is safe
                - position_label: String label ('safe' or 'danger')
                - position_confidence: Prediction probability (0-1)
                - classifier_available: Whether classifier is loaded
        """
        if self.position_classifier is None or landmarks is None:
            return {
                'position_safe': None,
                'position_label': 'unknown',
                'position_confidence': 0.0,
                'classifier_available': False
            }
        
        try:
            # Extract features
            features = self._extract_pose_features(landmarks)
            if features is None:
                return {
                    'position_safe': None,
                    'position_label': 'unknown',
                    'position_confidence': 0.0,
                    'classifier_available': True
                }
            
            # Normalize and predict
            features_scaled = self.position_scaler.transform(features.reshape(1, -1))
            prediction = self.position_classifier.predict(features_scaled)[0]
            probabilities = self.position_classifier.predict_proba(features_scaled)[0]
            
            # 0 = safe, 1 = danger
            is_safe = (prediction == 0)
            label = 'safe' if is_safe else 'danger'
            confidence = probabilities[prediction]
            
            return {
                'position_safe': is_safe,
                'position_label': label,
                'position_confidence': float(confidence),
                'classifier_available': True
            }
            
        except Exception as e:
            print(f"⚠️  Position classification error: {e}")
            return {
                'position_safe': None,
                'position_label': 'error',
                'position_confidence': 0.0,
                'classifier_available': True
            }
    
    def get_chest_roi(self, landmarks, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Build chest ROI from shoulders and hips.
        
        Args:
            landmarks: MediaPipe pose landmarks
            img_shape: (height, width) of the image
            
        Returns:
            4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None
        """
        if not landmarks:
            return None
        
        h, w = img_shape
        lm = landmarks.landmark
        
        # Get shoulder coordinates
        l_shoulder = np.array([lm[self.LEFT_SHOULDER].x * w, lm[self.LEFT_SHOULDER].y * h])
        r_shoulder = np.array([lm[self.RIGHT_SHOULDER].x * w, lm[self.RIGHT_SHOULDER].y * h])
        
        # Try to get hips
        try:
            l_hip = np.array([lm[self.LEFT_HIP].x * w, lm[self.LEFT_HIP].y * h])
            r_hip = np.array([lm[self.RIGHT_HIP].x * w, lm[self.RIGHT_HIP].y * h])
            
            # Check if hips are reliable (visibility > 0.5)
            if lm[self.LEFT_HIP].visibility > 0.5 and lm[self.RIGHT_HIP].visibility > 0.5:
                # Use shoulder-hip quad
                center = (l_shoulder + r_shoulder + l_hip + r_hip) / 4.0
                shrink_factor = 1.0 - self.roi_shrink
                
                l_shoulder = center + (l_shoulder - center) * shrink_factor
                r_shoulder = center + (r_shoulder - center) * shrink_factor
                l_hip = center + (l_hip - center) * shrink_factor
                r_hip = center + (r_hip - center) * shrink_factor
                
                roi = np.array([l_shoulder, r_shoulder, r_hip, l_hip], dtype=np.float32)
                return roi
        except:
            pass
        
        # Fallback: extend downward from shoulders
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
        
        # Estimate torso length (roughly shoulder width * 1.5)
        torso_length = shoulder_width * 1.5
        
        # Build ROI extending down
        top_left = l_shoulder
        top_right = r_shoulder
        bottom_left = l_shoulder + np.array([0, torso_length])
        bottom_right = r_shoulder + np.array([0, torso_length])
        
        roi = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        return roi
    
    def seed_chest_points(self, gray_img: np.ndarray, roi: np.ndarray, 
                         k: int = 32) -> Optional[np.ndarray]:
        """Seed feature points in chest ROI using Shi-Tomasi.
        
        Args:
            gray_img: Grayscale image
            roi: 4-point polygon defining chest region
            k: Number of points to track
            
        Returns:
            Points array of shape (k, 1, 2) or None if failed
        """
        # Create mask from ROI
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi.astype(np.int32)], 255)
        
        # Detect good features to track
        points = cv2.goodFeaturesToTrack(
            gray_img,
            maxCorners=k,
            qualityLevel=self.feature_quality,
            minDistance=5,
            mask=mask,
            blockSize=7
        )
        
        return points
    
    def track_points(self, prev_gray: np.ndarray, gray: np.ndarray,
                    prev_pts: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Track points using LK optical flow with forward-backward check.
        
        Args:
            prev_gray: Previous grayscale frame
            gray: Current grayscale frame
            prev_pts: Previous point positions (N, 1, 2)
            
        Returns:
            (new_pts, status) or (None, None) if tracking failed
        """
        if prev_pts is None or len(prev_pts) == 0:
            return None, None
        
        # Lucas-Kanade parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        try:
            # Forward flow
            new_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )
            
            if new_pts is None or status_fwd is None:
                return None, None
            
            # Backward flow for validation
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, new_pts, None, **lk_params
            )
            
            if back_pts is None or status_bwd is None:
                return new_pts, status_fwd
            
            # Forward-backward consistency check
            fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).flatten()
            fb_threshold = 1.0  # pixels
            fb_status = (fb_error < fb_threshold).astype(np.uint8)
            
            # Combine statuses
            combined_status = status_fwd.flatten() & status_bwd.flatten() & fb_status
            
            # MAD outlier rejection on vertical motion
            if np.sum(combined_status) > 3:
                good_pts = new_pts[combined_status > 0]
                good_prev = prev_pts[combined_status > 0]
                vertical_motion = good_pts[:, 0, 1] - good_prev[:, 0, 1]
                
                median_motion = np.median(vertical_motion)
                mad = np.median(np.abs(vertical_motion - median_motion))
                
                if mad > 0:
                    outliers = np.abs(vertical_motion - median_motion) > 3 * mad
                    combined_status[combined_status > 0] &= ~outliers
            
            return new_pts, combined_status.reshape(-1, 1)
            
        except Exception:
            return None, None
    
    def infer(self, frame: np.ndarray, timestamp: float = 0.0) -> Dict:
        """
        Process frame and extract chest expansion signal.
        
        Args:
            frame: BGR image from camera
            timestamp: Frame timestamp for refresh timing
            
        Returns:
            Dictionary containing:
                - landmarks: Full pose landmarks
                - confidence: Average shoulder visibility
                - detected: Whether pose was detected
                - tracks_y: Array of vertical displacements (px)
                - num_tracked_points: Number of active tracks
                - fps: Estimated FPS
                - mid_shoulder_xy: Fallback for legacy code
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            # Reset tracking on pose loss
            self.tracked_points = None
            self.prev_gray = gray
            return {
                "landmarks": None,
                "confidence": 0.0,
                "detected": False,
                "tracks_y": np.array([]),
                "num_tracked_points": 0,
                "fps": 30.0,
                "mid_shoulder_xy": None,
                "position_safe": None,
                "position_label": 'unknown',
                "position_confidence": 0.0,
                "classifier_available": self.position_classifier is not None
            }
        
        # Calculate shoulder visibility/confidence
        landmarks = results.pose_landmarks.landmark
        l_shoulder_vis = landmarks[self.LEFT_SHOULDER].visibility
        r_shoulder_vis = landmarks[self.RIGHT_SHOULDER].visibility
        confidence = (l_shoulder_vis + r_shoulder_vis) / 2.0
        
        # Get chest ROI
        roi = self.get_chest_roi(results.pose_landmarks, (h, w))
        
        # Track or seed points
        tracks_y = np.array([])
        
        if roi is not None:
            # Check if we need to refresh
            refresh_needed = (
                self.tracked_points is None or
                self.prev_gray is None or
                (timestamp - self.last_refresh_time) > self.tracker_refresh
            )
            
            if not refresh_needed and self.tracked_points is not None:
                # Try to track existing points
                new_pts, status = self.track_points(self.prev_gray, gray, self.tracked_points)
                
                if new_pts is not None and status is not None:
                    # Keep only successfully tracked points
                    good_new = new_pts[status.flatten() == 1]
                    good_old = self.tracked_points[status.flatten() == 1]
                    
                    # Check survival ratio
                    survival_ratio = len(good_new) / max(1, len(self.tracked_points))
                    
                    if survival_ratio < 0.7:
                        refresh_needed = True
                    else:
                        # Extract vertical displacements
                        displacements = good_new[:, 0, 1] - good_old[:, 0, 1]
                        tracks_y = displacements
                        self.tracked_points = good_new
                else:
                    refresh_needed = True
            
            # Seed new points if needed
            if refresh_needed:
                points = self.seed_chest_points(gray, roi, self.num_trackers)
                if points is not None and len(points) > 0:
                    self.tracked_points = points
                    self.last_refresh_time = timestamp
                    # First frame after seeding has zero displacement
                    tracks_y = np.zeros(len(points))
        
        # Fallback: if <3 valid tracks, use mid-shoulder proxy
        mid_shoulder_xy = None
        if len(tracks_y) < 3:
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            mid_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_y = (left_shoulder.y + right_shoulder.y) / 2
            mid_shoulder_xy = (mid_x, mid_y)
            
            # Compute displacement for fallback
            if hasattr(self, 'prev_mid_y') and self.prev_mid_y is not None:
                fallback_disp = (mid_y - self.prev_mid_y) * h  # Convert to pixels
                tracks_y = np.array([fallback_disp])
            self.prev_mid_y = mid_y
        
        self.prev_gray = gray
        
        # Classify sleeping position
        position_info = self.classify_position(results.pose_landmarks)
        
        return {
            "landmarks": results.pose_landmarks,
            "confidence": confidence,
            "detected": True,
            "tracks_y": tracks_y,
            "num_tracked_points": len(tracks_y),
            "fps": 30.0,
            "mid_shoulder_xy": mid_shoulder_xy,
            "chest_roi": roi,  # For visualization
            "tracked_points": self.tracked_points,  # For visualization
            # Position classification results
            "position_safe": position_info['position_safe'],
            "position_label": position_info['position_label'],
            "position_confidence": position_info['position_confidence'],
            "classifier_available": position_info['classifier_available']
        }
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def main():
    """Test pose detection with webcam and breathing monitoring."""
    import sys
    sys.path.append("..")
    from breath_monitor.capture import CameraCapture
    from breath_monitor.signal import BreathingAnalyzer
    
    capture = CameraCapture()
    backend = PoseBackend()
    analyzer = BreathingAnalyzer(window_sec=30.0)
    
    if not capture.start():
        return
    
    print("Press 'q' to quit")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    try:
        for frame, timestamp in capture.frames():
            result = backend.infer(frame, timestamp)
            
            # Update breathing analyzer with tracking data
            if result["detected"] and len(result["tracks_y"]) > 0:
                # Use median of tracks as single value for buffer
                median_y = np.median(result["tracks_y"])
                analyzer.add_sample(timestamp, median_y, result["tracks_y"])
                
                # Analyze breathing
                breath_result = analyzer.analyze(
                    visibility=result["confidence"],
                    tracks_alive=result["num_tracked_points"],
                    total_tracks=backend.num_trackers
                )
            else:
                breath_result = {"bpm_smooth": None, "conf": 0.0}
            
            # Draw pose landmarks
            if result["detected"]:
                mp_drawing.draw_landmarks(
                    frame,
                    result["landmarks"],
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Display tracking info
                cv2.putText(frame, f"Tracks: {result['num_tracked_points']}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {result['confidence']:.2f}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display BPM (breaths per minute)
                bpm = breath_result.get("bpm_smooth")
                if bpm is not None:
                    bpm_text = f"BPM: {bpm:.1f}"
                    bpm_color = (0, 255, 255)  # Cyan for BPM
                    
                    # Add visual indicator for abnormal rates
                    if bpm < 8 or bpm > 60:
                        bpm_color = (0, 165, 255)  # Orange for abnormal
                    
                    cv2.putText(frame, bpm_text, 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bpm_color, 2)
                else:
                    cv2.putText(frame, "BPM: --", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # Display position classification
                if result.get('classifier_available'):
                    label = result.get('position_label', 'unknown')
                    conf = result.get('position_confidence', 0.0)
                    
                    # Color coding: green for safe, red for danger
                    if label == 'safe':
                        color = (0, 255, 0)
                        status_text = f"Position: SAFE ({conf:.2f})"
                    elif label == 'danger':
                        color = (0, 0, 255)
                        status_text = f"Position: DANGER ({conf:.2f})"
                    else:
                        color = (128, 128, 128)
                        status_text = f"Position: {label}"
                    
                    cv2.putText(frame, status_text, 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No pose detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
