#!/usr/bin/env python3
"""
Test script to verify position classifier integration.
Tests both training and inference pipelines.

Usage:
    python test_position_classifier.py
    or
    py test_position_classifier.py  (Windows)
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_training_module():
    """Test that training module can be imported and has all required components."""
    print("=" * 60)
    print("Test 1: Training Module Import")
    print("=" * 60)
    
    try:
        from breath_monitor.train_pose_classifier import (
            PoseFeatureExtractor,
            load_dataset,
            train_classifier,
            save_model
        )
        print("✅ All training components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return False


def test_pose_backend_integration():
    """Test that PoseBackend has classifier integration."""
    print("\n" + "=" * 60)
    print("Test 2: PoseBackend Integration")
    print("=" * 60)
    
    try:
        from breath_monitor.pose_backend import PoseBackend
        
        # Create backend instance
        backend = PoseBackend()
        
        # Check if classifier methods exist
        assert hasattr(backend, 'position_classifier'), "Missing position_classifier attribute"
        assert hasattr(backend, 'position_scaler'), "Missing position_scaler attribute"
        assert hasattr(backend, 'classify_position'), "Missing classify_position method"
        assert hasattr(backend, '_extract_pose_features'), "Missing _extract_pose_features method"
        
        print("✅ PoseBackend has all required classifier components")
        
        # Test classify_position with None input
        result = backend.classify_position(None)
        assert 'position_safe' in result, "Missing position_safe in result"
        assert 'position_label' in result, "Missing position_label in result"
        assert 'position_confidence' in result, "Missing position_confidence in result"
        assert 'classifier_available' in result, "Missing classifier_available in result"
        
        print("✅ classify_position returns correct structure")
        
        backend.close()
        return True
        
    except Exception as e:
        print(f"❌ PoseBackend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_directory():
    """Test that data directory structure is correct."""
    print("\n" + "=" * 60)
    print("Test 3: Data Directory Structure")
    print("=" * 60)
    
    data_dir = Path(__file__).parent / "breath_monitor" / "data"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    safe_dir = data_dir / "safe"
    danger_dir = data_dir / "danger"
    
    if not safe_dir.exists():
        print(f"⚠️  Safe directory not found: {safe_dir}")
        safe_count = 0
    else:
        safe_files = list(safe_dir.glob("*.png")) + list(safe_dir.glob("*.jpg")) + \
                     list(safe_dir.glob("*.jpeg")) + list(safe_dir.glob("*.webp"))
        safe_count = len(safe_files)
        print(f"✅ Found {safe_count} images in safe/")
    
    if not danger_dir.exists():
        print(f"⚠️  Danger directory not found: {danger_dir}")
        danger_count = 0
    else:
        danger_files = list(danger_dir.glob("*.png")) + list(danger_dir.glob("*.jpg")) + \
                       list(danger_dir.glob("*.jpeg")) + list(danger_dir.glob("*.webp"))
        danger_count = len(danger_files)
        print(f"✅ Found {danger_count} images in danger/")
    
    if safe_count + danger_count > 0:
        print(f"✅ Total training images: {safe_count + danger_count}")
        return True
    else:
        print("⚠️  No training images found")
        return False


def test_model_output():
    """Test that infer() returns position classification fields."""
    print("\n" + "=" * 60)
    print("Test 4: Inference Output Format")
    print("=" * 60)
    
    try:
        from breath_monitor.pose_backend import PoseBackend
        import cv2
        
        backend = PoseBackend()
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run inference
        result = backend.infer(dummy_frame, 0.0)
        
        # Check required fields
        required_fields = [
            'landmarks', 'confidence', 'detected', 'tracks_y', 
            'num_tracked_points', 'fps', 'mid_shoulder_xy',
            'position_safe', 'position_label', 'position_confidence', 
            'classifier_available'
        ]
        
        missing_fields = [f for f in required_fields if f not in result]
        
        if missing_fields:
            print(f"❌ Missing fields in infer() output: {missing_fields}")
            return False
        
        print("✅ infer() returns all required fields:")
        print(f"   - position_safe: {result['position_safe']}")
        print(f"   - position_label: {result['position_label']}")
        print(f"   - position_confidence: {result['position_confidence']}")
        print(f"   - classifier_available: {result['classifier_available']}")
        
        backend.close()
        return True
        
    except Exception as e:
        print(f"❌ Inference output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Test 5: Dependencies Check")
    print("=" * 60)
    
    dependencies = {
        'mediapipe': 'MediaPipe',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'joblib': 'Joblib'
    }
    
    all_installed = True
    
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✅ {display_name} is installed")
        except ImportError:
            print(f"❌ {display_name} is NOT installed")
            if module_name == 'cv2':
                print(f"   Install with: pip install opencv-python")
            else:
                print(f"   Install with: pip install {module_name}")
            all_installed = False
    
    return all_installed


def main():
    """Run all tests."""
    print("\n🧪 Position Classifier Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Training Module", test_training_module),
        ("PoseBackend Integration", test_pose_backend_integration),
        ("Data Directory", test_data_directory),
        ("Inference Output", test_model_output)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Integration is successful.")
        print("\nNext steps:")
        print("1. Train the model: python -m breath_monitor.train_pose_classifier")
        print("2. Test with camera: python -m breath_monitor.pose_backend")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        if any(name == "Dependencies" and not passed for name, passed in results):
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

