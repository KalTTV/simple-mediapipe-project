"""
MediaPipe Demo Application
Supports multiple modes:
- face: Tongue Detection Meme Display
- breath: Infant Breathing Rate Monitor

See TUTORIAL.md for detailed explanations
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import sys

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Window settings - approximately half monitor size (1920x1080 / 2)
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

# Tongue detection threshold - adjust this value to change sensitivity
# Higher value = less sensitive (requires wider mouth opening)
# Lower value = more sensitive (detects smaller mouth openings)
# Recommended range: 0.02 - 0.05
TONGUE_OUT_THRESHOLD = 0.03

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

# Initialize MediaPipe Face Mesh
# This creates a face detection model that tracks 468 facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,  # Confidence threshold for initial detection
    min_tracking_confidence=0.5,   # Confidence threshold for tracking
    max_num_faces=1                # We only need to track one face
)

def is_tongue_out(face_landmarks):
    """
    Detect if tongue is out by analyzing mouth landmarks.
    
    This function uses MediaPipe Face Mesh landmarks to determine if the
    mouth is open wide enough to indicate the tongue is sticking out.
    
    Key landmarks used:
    - Landmark #13: Upper lip center
    - Landmark #14: Lower lip center
    - Landmark #0: Nose tip (reference point)
    - Landmark #17: Chin bottom
    
    MediaPipe provides 468 total landmarks. See the landmark map:
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    
    Args:
        face_landmarks: MediaPipe face landmarks object containing 468 3D points
        
    Returns:
        bool: True if tongue appears to be out, False otherwise
    """
    
    # Get mouth landmarks (normalized coordinates 0.0 to 1.0)
    upper_lip = face_landmarks.landmark[13]  # Upper lip center point
    lower_lip = face_landmarks.landmark[14]  # Lower lip center point
    
    # Additional landmarks for reference (not currently used, but available)
    mouth_top = face_landmarks.landmark[0]     # Nose tip
    mouth_bottom = face_landmarks.landmark[17] # Chin bottom
    
    # Calculate vertical distance between lips
    # Since coordinates are normalized (0.0-1.0), the result is a percentage
    # of the total frame height
    mouth_opening = abs(upper_lip.y - lower_lip.y)
    
    # Optional: Print for debugging/calibration
    # Uncomment the line below to see mouth opening values in real-time
    # print(f"Mouth opening: {mouth_opening:.4f}")
    
    # Compare to threshold and return result
    # If mouth opening exceeds threshold, tongue is considered "out"
    return mouth_opening > TONGUE_OUT_THRESHOLD

def main():
    """
    Main application loop.
    
    This function:
    1. Loads the meme images
    2. Initializes the webcam
    3. Creates display windows
    4. Runs the main detection loop
    5. Handles cleanup on exit
    """
    
    # ========================================================================
    # STEP 1: Load and prepare meme images
    # ========================================================================
    
    print("=" * 60)
    print("Tongue Detection Meme Display")
    print("=" * 60)
    
    # Check if required image files exist
    if not os.path.exists('apple.png'):
        print("\n[ERROR] apple.png not found!")
        print("Please add this image to the project directory.")
        print("This image is displayed when tongue is NOT out.")
        return
    
    if not os.path.exists('appletongue.png'):
        print("\n[ERROR] appletongue.png not found!")
        print("Please add this image to the project directory.")
        print("This image is displayed when tongue IS out.")
        return
    
    # Load images using OpenCV (images are loaded in BGR format)
    apple_img = cv2.imread('apple.png')
    appletongue_img = cv2.imread('appletongue.png')
    
    # Verify images loaded successfully
    if apple_img is None or appletongue_img is None:
        print("\n[ERROR] Could not load meme images.")
        print("Please check that the files are valid PNG images.")
        return
    
    print("[OK] Meme images loaded successfully!")
    
    # Resize images to fit the output window
    # This ensures consistent display regardless of original image size
    apple_img = cv2.resize(apple_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    appletongue_img = cv2.resize(appletongue_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # ========================================================================
    # STEP 2: Initialize webcam
    # ========================================================================
    
    # Open the default webcam (index 0)
    # If you have multiple cameras, try changing 0 to 1, 2, etc.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        print("Please check:")
        print("  - Webcam is connected")
        print("  - No other application is using the webcam")
        print("  - Webcam permissions are enabled")
        return
    
    # Set webcam resolution (may not match exactly depending on hardware)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    
    print("[OK] Webcam initialized successfully!")
    
    # ========================================================================
    # STEP 3: Create display windows
    # ========================================================================
    
    # Create two windows: one for camera input, one for meme output
    cv2.namedWindow('Camera Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Meme Output', cv2.WINDOW_NORMAL)
    
    # Set window sizes
    cv2.resizeWindow('Camera Input', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Meme Output', WINDOW_WIDTH, WINDOW_HEIGHT)
    
    print("\n" + "=" * 60)
    print("[OK] Application started successfully!")
    print("=" * 60)
    print("\n[CAMERA] Windows opened")
    print("[TONGUE] Stick your tongue out to change the meme!")
    print("[QUIT] Press 'q' to quit\n")
    
    # Default state - start with normal apple image
    current_meme = apple_img.copy()
    
    # ========================================================================
    # STEP 4: Main detection loop
    # ========================================================================
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("\n[ERROR] Could not read frame from webcam.")
            break
        
        # Flip frame horizontally for mirror effect (makes it easier to use)
        # Without this, moving left would make the image move right
        frame = cv2.flip(frame, 1)
        
        # Ensure frame matches our target window size
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        # OpenCV uses BGR color order, but MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        # This detects faces and returns 468 facial landmarks per face
        results = face_mesh.process(rgb_frame)
        
        # ====================================================================
        # Detect tongue and select appropriate meme
        # ====================================================================
        
        if results.multi_face_landmarks:
            # Face detected! Process landmarks
            for face_landmarks in results.multi_face_landmarks:
                # Check if tongue is out using our detection function
                if is_tongue_out(face_landmarks):
                    # Tongue detected - show tongue meme
                    current_meme = appletongue_img.copy()
                    
                    # Add visual indicator on camera feed (green text)
                    cv2.putText(frame, "TONGUE OUT!", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                else:
                    # No tongue - show normal meme
                    current_meme = apple_img.copy()
                    
                    # Add status text (yellow text)
                    cv2.putText(frame, "No tongue detected", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # No face detected in frame
            current_meme = apple_img.copy()
            
            # Add warning text (red text)
            cv2.putText(frame, "No face detected", (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ====================================================================
        # Display windows
        # ====================================================================
        
        # Show camera feed with detection status
        cv2.imshow('Camera Input', frame)
        
        # Show current meme image
        cv2.imshow('Meme Output', current_meme)
        
        # ====================================================================
        # Handle keyboard input
        # ====================================================================
        
        # Wait 1ms for key press, check if 'q' was pressed
        # The & 0xFF is needed for compatibility with some systems
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[QUIT] Quitting application...")
            break
    
    # ========================================================================
    # STEP 5: Cleanup and exit
    # ========================================================================
    
    # Release webcam
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Close MediaPipe Face Mesh
    face_mesh.close()
    
    print("[OK] Application closed successfully.")
    print("Thanks for using Tongue Detection Meme Display!\n")

def run_breath_monitor():
    """Run breathing monitor mode."""
    from breath_monitor.cli import main as breath_main
    breath_main()


if __name__ == "__main__":
    # Parse mode argument
    parser = argparse.ArgumentParser(
        description="MediaPipe Demo Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=["face", "breath"], default="breath",
                       help="Application mode: face (tongue detection) or breath (breathing monitor)")
    
    # Parse known args to allow pass-through of other arguments
    args, remaining = parser.parse_known_args()
    
    if args.mode == "face":
        print("\n=== Running Face/Tongue Detection Mode ===\n")
        main()
    elif args.mode == "breath":
        print("\n=== Running Breathing Monitor Mode ===\n")
        # Pass remaining arguments to breath monitor
        sys.argv = [sys.argv[0]] + remaining
        run_breath_monitor()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

