"""
OpenCV webcam capture with configurable parameters.
Provides frames and monotonic timestamps with clean shutdown.
"""

import cv2
import time
import argparse
from typing import Generator, Tuple
import numpy as np


class CameraCapture:
    """Webcam capture with configuration and clean shutdown."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device index
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        print(f"[OK] Camera started: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator yielding frames with monotonic timestamps.
        
        Yields:
            Tuple of (frame, timestamp)
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
                
            timestamp = time.monotonic()
            yield frame, timestamp
    
    def stop(self):
        """Stop camera capture and release resources."""
        self.running = False
        if self.cap:
            self.cap.release()
        print("[OK] Camera stopped")


def main():
    """Test camera capture with display."""
    parser = argparse.ArgumentParser(description="Test camera capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()
    
    capture = CameraCapture(args.camera, args.width, args.height, args.fps)
    
    if not capture.start():
        return
    
    print("Press 'q' to quit")
    
    try:
        for frame, timestamp in capture.frames():
            cv2.imshow("Camera Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

