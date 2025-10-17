"""
Train a pose classifier using MediaPipe landmarks.
Extracts pose features from images and trains a scikit-learn classifier.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PoseFeatureExtractor:
    """Extract pose landmark features using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Flattened landmark array of shape (33 * 4,) or None if pose not detected
            Features: [x, y, z, visibility] for each of 33 landmarks
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  Failed to read: {image_path}")
            return None
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks: [x, y, z, visibility] * 33 landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        return np.array(landmarks)
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract additional geometric features from landmarks.
        
        Args:
            landmarks: Raw landmark array (33 * 4,)
            
        Returns:
            Extended feature vector with geometric relationships
        """
        # Reshape to (33, 4) for easier indexing
        lm = landmarks.reshape(33, 4)
        
        features = []
        
        # Key landmark indices
        nose = lm[0][:3]
        left_shoulder = lm[11][:3]
        right_shoulder = lm[12][:3]
        left_hip = lm[23][:3]
        right_hip = lm[24][:3]
        left_knee = lm[25][:3]
        right_knee = lm[26][:3]
        left_ankle = lm[27][:3]
        right_ankle = lm[28][:3]
        
        # 1. Torso orientation (angle of spine)
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        spine_vector = mid_hip - mid_shoulder
        spine_angle = np.arctan2(spine_vector[1], spine_vector[0])
        features.extend([spine_angle, np.linalg.norm(spine_vector)])
        
        # 2. Head-torso alignment
        head_torso_vector = nose - mid_shoulder
        features.extend(head_torso_vector)
        
        # 3. Shoulder width and hip width
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_width = np.linalg.norm(right_hip - left_hip)
        features.extend([shoulder_width, hip_width])
        
        # 4. Body symmetry (left-right differences)
        left_arm_length = np.linalg.norm(lm[15][:3] - left_shoulder)  # wrist to shoulder
        right_arm_length = np.linalg.norm(lm[16][:3] - right_shoulder)
        left_leg_length = np.linalg.norm(left_ankle - left_hip)
        right_leg_length = np.linalg.norm(right_ankle - right_hip)
        features.extend([
            left_arm_length - right_arm_length,
            left_leg_length - right_leg_length
        ])
        
        # 5. Key angles
        # Shoulder-hip-knee angles
        left_hip_angle = self._compute_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._compute_angle(right_shoulder, right_hip, right_knee)
        features.extend([left_hip_angle, right_hip_angle])
        
        # 6. Elevation ratios (useful for detecting face-down position)
        head_elevation = nose[1]  # y-coordinate (lower is higher in image)
        shoulder_elevation = mid_shoulder[1]
        hip_elevation = mid_hip[1]
        features.extend([head_elevation, shoulder_elevation, hip_elevation])
        
        # 7. Z-depth variation (face orientation)
        z_std = np.std([lm[i][2] for i in [0, 11, 12, 23, 24]])  # nose, shoulders, hips
        features.append(z_std)
        
        # 8. Visibility statistics (face-down detection)
        face_visibility = np.mean([lm[i][3] for i in range(0, 10)])  # face landmarks
        body_visibility = np.mean([lm[i][3] for i in [11, 12, 23, 24]])
        features.extend([face_visibility, body_visibility])
        
        return np.array(features)
    
    @staticmethod
    def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute angle at point b formed by points a-b-c."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return angle
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def load_dataset(data_dir: str, extractor: PoseFeatureExtractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and process images from danger/ and safe/ directories.
    
    Args:
        data_dir: Path to data directory containing danger/ and safe/ subdirectories
        extractor: PoseFeatureExtractor instance
        
    Returns:
        (X, y, image_paths) where:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (0=safe, 1=danger)
            image_paths: List of successfully processed image paths
    """
    X = []
    y = []
    image_paths = []
    
    classes = {
        'safe': 0,
        'danger': 1
    }
    
    print("📂 Loading dataset...")
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Directory not found: {class_dir}")
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        print(f"   Processing {len(image_files)} images from '{class_name}'...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            # Extract raw landmarks
            landmarks = extractor.extract_landmarks(img_path)
            if landmarks is None:
                continue
            
            # Extract geometric features
            geometric_features = extractor.extract_geometric_features(landmarks)
            
            # Combine raw landmarks + geometric features
            features = np.concatenate([landmarks, geometric_features])
            
            X.append(features)
            y.append(label)
            image_paths.append(img_path)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Loaded {len(X)} samples:")
    print(f"   Safe: {np.sum(y == 0)}")
    print(f"   Danger: {np.sum(y == 1)}")
    print(f"   Feature dimension: {X.shape[1]}")
    
    return X, y, image_paths


def train_classifier(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest classifier on pose features.
    
    Args:
        X: Feature matrix
        y: Labels
        
    Returns:
        (trained_model, scaler)
    """
    print("\n🎯 Training pose classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with class weighting for imbalanced data
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n📊 Training accuracy: {train_score:.3f}")
    print(f"📊 Test accuracy: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"📊 Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Danger']))
    
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    return model, scaler


def save_model(model: RandomForestClassifier, scaler: StandardScaler, output_path: str):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        output_path: Path to save the model
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save both model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'version': '1.0'
    }
    
    joblib.dump(model_data, output_path)
    print(f"\n💾 Model saved to: {output_path}")


def main():
    """Main training pipeline."""
    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'position_model.joblib')
    
    print("=" * 60)
    print("🏋️  Pose Classifier Training")
    print("=" * 60)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        response = input(f"\n⚠️  Model already exists at {MODEL_PATH}\nRetrain? (y/N): ")
        if response.lower() != 'y':
            print("ℹ️  Skipping training.")
            return
    
    # Initialize extractor
    extractor = PoseFeatureExtractor()
    
    try:
        # Load dataset
        X, y, image_paths = load_dataset(DATA_DIR, extractor)
        
        if len(X) == 0:
            print("❌ No valid samples found. Check your data directory.")
            return
        
        # Train classifier
        model, scaler = train_classifier(X, y)
        
        # Save model
        save_model(model, scaler, MODEL_PATH)
        
        print("\n✅ Training complete!")
        
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
