"""
Main CLI entrypoint for breath monitoring.
Chest expansion detection with multi-point tracking.
"""

import argparse
import time
import cv2
import sys
import numpy as np
from typing import Optional

from .capture import CameraCapture
from .pose_backend import PoseBackend
from .signal import BreathingAnalyzer
from .draw import BreathingVisualizer


class BreathMonitor:
    """Main breath monitoring application."""
    
    def __init__(self, args):
        """Initialize monitor with arguments."""
        self.args = args
        
        # Initialize components
        self.capture = CameraCapture(args.camera, args.width, args.height, args.fps)
        self.pose = PoseBackend(
            num_trackers=args.trackers,
            tracker_refresh=args.tracker_refresh,
            roi_shrink=args.roi_shrink,
            feature_quality=args.feature_quality
        )
        self.analyzer = BreathingAnalyzer(
            window_sec=args.min_sec,
            resp_low=args.resp_low,
            resp_high=args.resp_high,
            min_ibi_sec=args.min_ibi_sec,
            apnea_sec=args.apnea_sec
        )
        
        # Visualization
        self.draw_enabled = args.draw == "on"
        self.visualizer = BreathingVisualizer() if self.draw_enabled else None
        
        # Statistics
        self.frame_count = 0
        self.last_log_time = time.time()
        
    def start(self) -> bool:
        """Start all components."""
        print("=" * 60)
        print("Breathing Rate Monitor - Chest Expansion Detection")
        print("=" * 60)
        print("\n[WARNING] NOT A MEDICAL DEVICE - FOR RESEARCH/DEMO ONLY\n")
        
        if not self.capture.start():
            return False
        
        print(f"[CONFIG] Window: {self.args.min_sec}s, "
              f"Band: {self.args.resp_low:.2f}-{self.args.resp_high:.2f} Hz "
              f"({self.args.resp_low*60:.0f}-{self.args.resp_high*60:.0f} BPM)")
        print(f"[CONFIG] Trackers: {self.args.trackers}, "
              f"Min IBI: {self.args.min_ibi_sec}s (debounce)")
        print(f"[CONFIG] Draw: {self.args.draw}")
        print("\nPress 'q' to quit\n")
        
        return True
    
    def process_frame(self, frame, timestamp):
        """Process a single frame."""
        # Pose detection with multi-point tracking
        pose_result = self.pose.infer(frame, timestamp)
        
        # Add signal sample with tracks
        visibility = pose_result.get("confidence", 0.0)
        tracks_alive = pose_result.get("num_tracked_points", 0)
        
        if pose_result["detected"] and visibility > 0.3:
            tracks_y = pose_result.get("tracks_y", np.array([]))
            
            # Use fallback value if no tracks
            value = 0.5  # Default
            if pose_result.get("mid_shoulder_xy"):
                _, y = pose_result["mid_shoulder_xy"]
                value = y
            
            self.analyzer.add_sample(timestamp, value, tracks_y)
        
        # Analyze breathing
        analysis = self.analyzer.analyze(
            visibility=visibility,
            tracks_alive=tracks_alive,
            total_tracks=self.args.trackers
        )
        
        # Draw visualization if enabled
        if self.visualizer:
            frame = self.visualizer.draw_all(frame, pose_result, analysis)
        
        return frame, analysis
    
    def log_status(self, analysis):
        """Log status once per second."""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
            conf = analysis.get("conf", 0.0)
            snr = analysis.get("snr_db", 0.0)
            tracks = analysis.get("tracks_alive", 0)
            ibi_median = analysis.get("ibi_median", 0.0)
            breath_count = analysis.get("breath_count", 0)
            
            # Format BPM
            bpm_str = f"{bpm:.1f}" if (bpm is not None and np.isfinite(bpm)) else "--"
            
            # Status flags
            flags = []
            if analysis.get("apnea"):
                flags.append("APNEA")
            if analysis.get("shallow"):
                flags.append("SHALLOW")
            
            status = " | ".join(flags) if flags else "OK"
            
            print(f"[{self.frame_count:06d}] BPM: {bpm_str:>6s} | "
                  f"Breaths: {breath_count:2d} | "
                  f"Conf: {conf:.2f} | SNR: {snr:>5.1f} dB | "
                  f"Tracks: {tracks:2d} | IBI: {ibi_median:.2f}s | {status}")
            
            self.last_log_time = current_time
    
    def run(self):
        """Main processing loop."""
        if not self.start():
            return
        
        try:
            for frame, timestamp in self.capture.frames():
                self.frame_count += 1
                
                # Process frame
                display_frame, analysis = self.process_frame(frame, timestamp)
                
                # Log status once per second
                self.log_status(analysis)
                
                # Display (visualization already applied in process_frame)
                if self.draw_enabled:
                    cv2.imshow("Breathing Monitor", display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all components."""
        print("\n[INFO] Shutting down...")
        self.capture.stop()
        self.pose.close()
        
        if self.draw_enabled:
            cv2.destroyAllWindows()
        
        print("[OK] Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Breathing Monitor - Chest Expansion Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Camera settings
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    # Signal processing
    parser.add_argument("--min-sec", type=float, default=30.0, 
                       help="Signal buffer window (seconds)")
    parser.add_argument("--resp-low", type=float, default=0.08, 
                       help="Respiratory band low cutoff (Hz)")
    parser.add_argument("--resp-high", type=float, default=1.2, 
                       help="Respiratory band high cutoff (Hz)")
    parser.add_argument("--min-ibi-sec", type=float, default=2.0,
                       help="Minimum inter-breath interval / debounce (seconds)")
    
    # Multi-point tracking
    parser.add_argument("--trackers", type=int, default=32,
                       help="Number of chest feature points to track")
    parser.add_argument("--tracker-refresh", type=float, default=0.5,
                       help="Tracker refresh period (seconds)")
    parser.add_argument("--roi-shrink", type=float, default=0.05,
                       help="ROI shrink factor (0.05 = 5%)")
    parser.add_argument("--feature-quality", type=float, default=0.001,
                       help="Feature quality threshold for goodFeaturesToTrack")
    
    # Detection thresholds
    parser.add_argument("--apnea-sec", type=float, default=20.0, 
                       help="Apnea detection threshold (seconds)")
    
    # Output settings
    parser.add_argument("--draw", choices=["on", "off"], default="on", 
                       help="Enable visual overlay")
    
    args = parser.parse_args()
    
    monitor = BreathMonitor(args)
    monitor.run()


if __name__ == "__main__":
    main()
