"""Video capture module for breathing monitoring."""

import cv2
from typing import Generator, Optional
import numpy as np


class VideoCapture:
    """Handles video capture from webcam or file."""
    
    def __init__(self, source: int | str = 0):
        """Initialize video capture.
        
        Args:
            source: Camera index (int) or video file path (str)
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        
    def __enter__(self):
        """Context manager entry."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cap:
            self.cap.release()
            
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames from video source.
        
        Yields:
            numpy array of frame (BGR format)
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
    def get_fps(self) -> float:
        """Get video FPS."""
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0
