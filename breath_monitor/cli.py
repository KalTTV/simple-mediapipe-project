#!/usr/bin/env python3
"""CLI entrypoint for breath monitoring application."""

import argparse
import sys
import time
import asyncio
import logging
from typing import Optional

import cv2
import numpy as np

from breath_monitor.capture import CaptureSource
from breath_monitor.pose_backend import PoseBackend
from breath_monitor.signal import SignalProcessor
from breath_monitor.events import EventBus, BreathMetricsEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreathMonitorCLI:
    """CLI application for breath monitoring."""

    def __init__(self, mode: str, websocket: bool = False, port: int = 8765):
        """
        Initialize CLI application.

        Args:
            mode: Operating mode ('face' or 'breath')
            websocket: Whether to serve WebSocket
            port: WebSocket port number
        """
        self.mode = mode
        self.websocket = websocket
        self.port = port
        self.running = False

        # Components
        self.capture: Optional[CaptureSource] = None
        self.pose_backend: Optional[PoseBackend] = None
        self.signal_processor: Optional[SignalProcessor] = None
        self.event_bus: Optional[EventBus] = None

    def initialize_components(self) -> bool:
        """Initialize all required components."""
        try:
            logger.info(f"Initializing components for {self.mode} mode...")

            # Initialize capture
            self.capture = CaptureSource(source=0)
            if not self.capture.is_opened():
                logger.error("Failed to open webcam")
                return False

            # Initialize pose backend (for breath mode)
            if self.mode == 'breath':
                self.pose_backend = PoseBackend()
                self.signal_processor = SignalProcessor(
                    window_size=150,
                    sampling_rate=30.0
                )
                self.event_bus = EventBus()

                # Subscribe to metrics events
                self.event_bus.subscribe(
                    BreathMetricsEvent,
                    self._handle_metrics_event
                )

            logger.info("Components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def _handle_metrics_event(self, event: BreathMetricsEvent):
        """Handle breath metrics events."""
        print(f"\n=== Breath Metrics ===")
        print(f"Rate: {event.rate:.2f} breaths/min")
        print(f"Depth: {event.depth:.2f}")
        print(f"Regularity: {event.regularity:.2f}")
        print(f"Confidence: {event.confidence:.2f}")
        print(f"Timestamp: {event.timestamp:.2f}")

    def run_breath_mode(self):
        """Run breath monitoring mode."""
        logger.info("Starting breath monitoring...")
        self.running = True

        frame_count = 0
        fps_time = time.time()
        fps = 0.0

        try:
            while self.running:
                # Capture frame
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue

                frame_count += 1

                # Calculate FPS
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps = frame_count / (current_time - fps_time)
                    frame_count = 0
                    fps_time = current_time

                # Process with pose backend
                landmarks = self.pose_backend.process_frame(frame)

                if landmarks:
                    # Extract chest keypoint (average of shoulders and hips)
                    # Landmarks: 11=left_shoulder, 12=right_shoulder
                    #            23=left_hip, 24=right_hip
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]

                    # Calculate chest center (midpoint between shoulders and hips center)
                    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                    hip_center_y = (left_hip.y + right_hip.y) / 2
                    chest_y = (shoulder_center_y + hip_center_y) / 2

                    # Send to signal processor
                    self.signal_processor.add_sample(chest_y, current_time)

                    # Process metrics if enough data
                    metrics = self.signal_processor.compute_metrics()
                    if metrics:
                        # Emit metrics event
                        event = BreathMetricsEvent(
                            rate=metrics['rate'],
                            depth=metrics['depth'],
                            regularity=metrics['regularity'],
                            confidence=metrics['confidence'],
                            timestamp=current_time
                        )
                        self.event_bus.emit(event)

                    # Draw visualization
                    h, w = frame.shape[:2]
                    chest_pixel_y = int(chest_y * h)
                    shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)

                    # Draw chest point
                    cv2.circle(frame, (shoulder_center_x, chest_pixel_y),
                              10, (0, 255, 0), -1)

                    # Draw skeleton connections
                    def draw_landmark(lm, color=(255, 0, 0)):
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 5, color, -1)

                    # Draw key points
                    draw_landmark(left_shoulder, (0, 255, 255))
                    draw_landmark(right_shoulder, (0, 255, 255))
                    draw_landmark(left_hip, (255, 255, 0))
                    draw_landmark(right_hip, (255, 255, 0))

                    # Draw connecting lines
                    cv2.line(frame,
                            (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                            (int(right_shoulder.x * w), int(right_shoulder.y * h)),
                            (255, 0, 0), 2)
                    cv2.line(frame,
                            (int(left_hip.x * w), int(left_hip.y * h)),
                            (int(right_hip.x * w), int(right_hip.y * h)),
                            (255, 0, 0), 2)

                # Display FPS and status
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Mode: {self.mode}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show frame
                cv2.imshow('Breath Monitor', frame)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Exit requested")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in breath mode: {e}", exc_info=True)
        finally:
            self.cleanup()

    def run_face_mode(self):
        """Run face detection mode (placeholder)."""
        logger.info("Starting face mode...")
        logger.info("Face mode is not yet implemented")
        logger.info("This is a placeholder for future face detection functionality")

        self.running = True
        try:
            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue

                # Display frame
                cv2.putText(frame, "Face Mode - Not Implemented", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Mode', frame)

                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.running = False

        if self.capture:
            self.capture.release()

        if self.pose_backend:
            self.pose_backend.close()

        cv2.destroyAllWindows()
        logger.info("Cleanup complete")

    def run(self):
        """Run the CLI application."""
        if not self.initialize_components():
            logger.error("Failed to initialize, exiting")
            return 1

        try:
            if self.mode == 'breath':
                self.run_breath_mode()
            elif self.mode == 'face':
                self.run_face_mode()
            else:
                logger.error(f"Unknown mode: {self.mode}")
                return 1

            return 0

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            return 1
        finally:
            self.cleanup()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Breath monitoring CLI application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode breath              # Run breath monitoring
  %(prog)s --mode face                # Run face detection (not implemented)
  %(prog)s --mode breath --websocket  # Run with WebSocket server
  %(prog)s --mode breath --port 9000  # Use custom WebSocket port

Controls:
  Press 'q' or ESC to quit
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['face', 'breath'],
        required=True,
        help='Operating mode: face detection or breath monitoring'
    )

    parser.add_argument(
        '--websocket',
        action='store_true',
        help='Enable WebSocket server for real-time data streaming'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Create and run CLI
    logger.info(f"Starting CLI in {args.mode} mode")
    if args.websocket:
        logger.info(f"WebSocket server will be available on port {args.port}")

    cli = BreathMonitorCLI(
        mode=args.mode,
        websocket=args.websocket,
        port=args.port
    )

    sys.exit(cli.run())


if __name__ == '__main__':
    main()
