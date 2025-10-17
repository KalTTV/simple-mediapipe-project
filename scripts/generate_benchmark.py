"""
Generate synthetic test video with simulated breathing motion.
Creates a video with a figure showing periodic vertical motion suitable for breathing rate testing.
"""

import cv2
import numpy as np
import argparse


def draw_simple_person(frame, center_x, center_y, shoulder_offset=0):
    """
    Draw a simple stick figure with shoulders at specified vertical offset.
    
    Args:
        frame: Image to draw on
        center_x: Horizontal center
        center_y: Vertical center (will be offset by shoulder_offset)
        shoulder_offset: Vertical offset for breathing simulation (pixels)
    """
    # Adjust center for breathing
    y = center_y + int(shoulder_offset)
    
    # Colors
    color = (255, 255, 255)
    thickness = 3
    
    # Head
    cv2.circle(frame, (center_x, y - 60), 25, color, thickness)
    
    # Body (vertical line)
    cv2.line(frame, (center_x, y - 35), (center_x, y + 80), color, thickness)
    
    # Shoulders (horizontal line) - this is what we track
    cv2.line(frame, (center_x - 50, y), (center_x + 50, y), color, thickness + 2)
    cv2.circle(frame, (center_x - 50, y), 8, (0, 255, 0), -1)  # Left shoulder marker
    cv2.circle(frame, (center_x + 50, y), 8, (0, 255, 0), -1)  # Right shoulder marker
    
    # Arms
    cv2.line(frame, (center_x - 50, y), (center_x - 40, y + 60), color, thickness)
    cv2.line(frame, (center_x + 50, y), (center_x + 40, y + 60), color, thickness)
    
    # Legs
    cv2.line(frame, (center_x, y + 80), (center_x - 30, y + 140), color, thickness)
    cv2.line(frame, (center_x, y + 80), (center_x + 30, y + 140), color, thickness)


def generate_video(output_path, duration=30, fps=30, breathing_rate=0.8, amplitude=15):
    """
    Generate a test video with simulated breathing motion.
    
    Args:
        output_path: Output video file path
        duration: Video duration in seconds
        fps: Frames per second
        breathing_rate: Breathing frequency in Hz (breaths per second)
        amplitude: Breathing amplitude in pixels
    """
    # Video settings
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    center_x = width // 2
    center_y = height // 2
    
    print(f"Generating {duration}s video at {fps} FPS...")
    print(f"Breathing rate: {breathing_rate} Hz ({breathing_rate * 60:.1f} BPM)")
    print(f"Amplitude: {amplitude} pixels")
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate breathing offset (sinusoidal motion)
        t = frame_num / fps
        offset = amplitude * np.sin(2 * np.pi * breathing_rate * t)
        
        # Draw person with breathing motion
        draw_simple_person(frame, center_x, center_y, shoulder_offset=offset)
        
        # Add info text
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Target BPM: {breathing_rate * 60:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_num % 30 == 0:
            print(f"Progress: {frame_num}/{total_frames} ({100*frame_num/total_frames:.0f}%)")
    
    out.release()
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Duration: {duration}s")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Expected BPM: {breathing_rate * 60:.1f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic breathing test video")
    parser.add_argument("--output", default="benchmark.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--bpm", type=float, default=48.0, help="Target breathing rate (BPM)")
    parser.add_argument("--amplitude", type=float, default=15.0, help="Motion amplitude (pixels)")
    args = parser.parse_args()
    
    # Convert BPM to Hz
    breathing_rate = args.bpm / 60.0
    
    generate_video(
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        breathing_rate=breathing_rate,
        amplitude=args.amplitude
    )


if __name__ == "__main__":
    main()

