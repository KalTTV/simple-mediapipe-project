"""
Main CLI entrypoint for breath monitoring.
Integrates capture, pose detection, signal processing, and visualization.
"""

import argparse
import time
import cv2
import sys
from typing import Optional

from .capture import CameraCapture
from .pose_backend import PoseBackend
from .signal import BreathingAnalyzer
from .draw import BreathingVisualizer
from .events import WebSocketBroadcaster


class BreathMonitor:
    """Main breath monitoring application."""
    
    def __init__(self, args):
        """Initialize monitor with arguments."""
        self.args = args
        
        # Initialize components
        self.capture = CameraCapture(args.camera, args.width, args.height, args.fps)
        self.pose = PoseBackend()
        self.analyzer = BreathingAnalyzer(
            window_sec=args.min_sec,
            bpf_low=args.bpf_low,
            bpf_high=args.bpf_high,
            apnea_sec=args.apnea_sec,
            tachy_threshold=args.tachy,
            brady_threshold=args.brady
        )
        self.visualizer = BreathingVisualizer() if args.draw == "on" else None
        self.broadcaster = WebSocketBroadcaster() if args.ws == "on" else None
        
        # Statistics
        self.frame_count = 0
        self.last_log_time = time.time()
        
    def start(self) -> bool:
        """Start all components."""
        print("=" * 60)
        print("Infant Breathing Rate Monitor")
        print("=" * 60)
        print("\n[WARNING] FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE\n")
        
        if not self.capture.start():
            return False
        
        if self.broadcaster:
            self.broadcaster.start()
        
        print(f"[CONFIG] Window: {self.args.min_sec}s, "
              f"Bandpass: {self.args.bpf_low}-{self.args.bpf_high} Hz")
        print(f"[CONFIG] Apnea: {self.args.apnea_sec}s, "
              f"Tachy: {self.args.tachy} BPM, Brady: {self.args.brady} BPM")
        print(f"[CONFIG] Draw: {self.args.draw}, WebSocket: {self.args.ws}")
        print("\nPress 'q' to quit\n")
        
        return True
    
    def process_frame(self, frame, timestamp):
        """Process a single frame."""
        # Pose detection
        pose_result = self.pose.infer(frame)
        
        # Add signal sample if pose detected with sufficient confidence
        if pose_result["detected"] and pose_result["confidence"] > 0.5:
            if pose_result["mid_shoulder_xy"]:
                # Use Y coordinate (vertical motion) as breathing proxy
                _, y = pose_result["mid_shoulder_xy"]
                self.analyzer.add_sample(timestamp, y)
        
        # Analyze breathing
        analysis = self.analyzer.analyze()
        
        # Broadcast state if WebSocket enabled
        if self.broadcaster:
            self.broadcaster.broadcast_state(
                bpm=analysis.get("bpm_smooth"),
                apnea=analysis.get("apnea", False),
                shallow=analysis.get("shallow", False),
                confidence=pose_result.get("confidence", 0.0),
                timestamp=timestamp
            )
        
        # Draw visualization if enabled
        if self.visualizer:
            frame = self.visualizer.draw_all(frame, pose_result, analysis)
        
        return frame, analysis
    
    def log_status(self, analysis):
        """Log status every second."""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            bpm = analysis.get("bpm_smooth") or analysis.get("bpm")
            if bpm is not None:
                bpm_str = f"{bpm:.1f}"
            else:
                bpm_str = "--"
            
            status_flags = []
            if analysis.get("apnea"):
                status_flags.append("APNEA")
            if analysis.get("shallow"):
                status_flags.append("SHALLOW")
            if analysis.get("tachypnea"):
                status_flags.append("TACHY")
            if analysis.get("bradypnea"):
                status_flags.append("BRADY")
            
            status = " | ".join(status_flags) if status_flags else "OK"
            
            print(f"[{self.frame_count:06d}] BPM: {bpm_str:>6s} | "
                  f"Peaks: {analysis.get('peak_count', 0):2d} | "
                  f"Conf: {analysis.get('confidence', 0.0):.2f} | "
                  f"Status: {status}")
            
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
                
                # Log status
                self.log_status(analysis)
                
                # Display if drawing enabled
                if self.args.draw == "on":
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
        
        if self.broadcaster:
            self.broadcaster.stop()
        
        if self.args.draw == "on":
            cv2.destroyAllWindows()
        
        print("[OK] Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Infant Breathing Rate Monitor using MediaPipe Pose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Camera settings
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    # Signal processing settings
    parser.add_argument("--min-sec", type=float, default=15.0, 
                       help="Signal buffer window (seconds)")
    parser.add_argument("--bpf-low", type=float, default=0.5, 
                       help="Bandpass filter low cutoff (Hz)")
    parser.add_argument("--bpf-high", type=float, default=1.2, 
                       help="Bandpass filter high cutoff (Hz)")
    
    # Detection thresholds
    parser.add_argument("--apnea-sec", type=float, default=20.0, 
                       help="Apnea detection threshold (seconds)")
    parser.add_argument("--tachy", type=float, default=60.0, 
                       help="Tachypnea threshold (BPM)")
    parser.add_argument("--brady", type=float, default=30.0, 
                       help="Bradypnea threshold (BPM)")
    
    # Output settings
    parser.add_argument("--ws", choices=["on", "off"], default="off", 
                       help="Enable WebSocket broadcasting")
    parser.add_argument("--draw", choices=["on", "off"], default="on", 
                       help="Enable visual overlay")
    
    args = parser.parse_args()
    
    monitor = BreathMonitor(args)
    monitor.run()


if __name__ == "__main__":
    main()

