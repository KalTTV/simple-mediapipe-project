# Breathing Monitor MVP

A real-time breathing rate monitoring application using MediaPipe Pose Landmarking and OpenCV. This MVP detects shoulder movement patterns to estimate respiratory rate from webcam video.

**⚠️ SAFETY WARNING: NOT A MEDICAL DEVICE ⚠️**

This software is provided for educational and experimental purposes ONLY. It is NOT a medical device and has NOT been validated for clinical use. DO NOT use this application for medical diagnosis, treatment decisions, or health monitoring. Always consult qualified healthcare professionals for medical advice and monitoring.

## Features

- Real-time breathing rate estimation via shoulder landmark tracking
- Visual feedback with breathing rate display (breaths per minute)
- Status indicators: "Detecting...", "Calculating...", "Monitoring"
- Configurable camera resolution and detection parameters
- CSV data logging for analysis
- Built on MediaPipe Pose Landmarking (33-point model)

## Quickstart Usage

### Basic Usage

Run the breathing monitor with default settings:

```bash
python -m breath_monitor.main
```

Or using the convenience script:

**Windows:**
```bash
run.bat
```

**macOS/Linux:**
```bash
./run.sh
```

### With Custom Settings

```bash
# Set custom camera resolution
python -m breath_monitor.main --width 1280 --height 720

# Enable data logging
python -m breath_monitor.main --log-data --log-file breath_data.csv

# Adjust detection sensitivity
python -m breath_monitor.main --min-detection-confidence 0.7 --min-tracking-confidence 0.7

# Use specific camera (if multiple cameras available)
python -m breath_monitor.main --camera-index 1
```

## Command Flags

### Camera Settings
- `--camera-index INDEX` - Camera device index (default: 0)
- `--width WIDTH` - Camera frame width in pixels (default: 640)
- `--height HEIGHT` - Camera frame height in pixels (default: 480)

### Detection Parameters
- `--min-detection-confidence CONF` - Minimum confidence for pose detection (0.0-1.0, default: 0.5)
- `--min-tracking-confidence CONF` - Minimum confidence for landmark tracking (0.0-1.0, default: 0.5)

### Breathing Analysis
- `--smoothing-window SIZE` - Smoothing window size for signal processing (default: 5)
- `--breath-history SIZE` - Number of breath cycles to average (default: 3)

### Data Logging
- `--log-data` - Enable CSV data logging
- `--log-file FILENAME` - Output CSV file path (default: breath_monitor_log.csv)

### Display Options
- `--no-display` - Run without video display window (headless mode for logging)
- `--fullscreen` - Run in fullscreen mode

### Help
- `--help` or `-h` - Show all available options and exit

## Installation

### Requirements
- Python 3.11+ (tested on 3.11, 3.12)
- Webcam (built-in or USB)
- Operating System: Windows 10/11, macOS 10.14+, or Linux

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KalTTV/simple-mediapipe-project.git
   cd simple-mediapipe-project
   git checkout breathing-mvp
   ```

2. **Install Python dependencies:**
   
   **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Or using the Makefile:**
   ```bash
   make install
   ```

3. **Run tests (optional):**
   ```bash
   make test
   ```

4. **Start monitoring:**
   ```bash
   python -m breath_monitor.main
   ```

## How It Works

The breathing monitor uses MediaPipe's Pose Landmarker to track 33 body landmarks in real-time. It focuses on shoulder landmarks (left and right) to detect the subtle vertical movement patterns associated with breathing:

1. **Pose Detection**: MediaPipe identifies person in frame and tracks 33 landmarks
2. **Shoulder Tracking**: Monitors y-coordinate changes of left/right shoulder landmarks (landmarks 11 & 12)
3. **Signal Processing**: Applies smoothing and filtering to reduce noise
4. **Peak Detection**: Identifies local maxima in smoothed signal as breath cycles
5. **Rate Calculation**: Computes breathing rate (breaths/minute) from inter-peak intervals

See [docs/algorithm.md](docs/algorithm.md) for detailed technical documentation.

## Known Failure Cases

This MVP has limitations and will not work reliably in the following scenarios:

### Environmental Limitations
- **Poor Lighting**: Dim lighting or harsh shadows degrade pose detection accuracy
- **Cluttered Background**: Busy backgrounds can interfere with person segmentation
- **Camera Placement**: Extreme angles or unstable camera mounting affects tracking
- **Distance**: Too far (>3m) or too close (<0.5m) from camera reduces accuracy

### User-Related Limitations
- **Movement**: Walking, fidgeting, or upper body movement creates false positives
- **Clothing**: Bulky clothing or accessories may obscure shoulder landmarks
- **Body Position**: Lying down, turned away, or partial occlusion breaks tracking
- **Multiple People**: Only tracks one person; multiple people in frame cause unreliable results
- **Fast Breathing**: Respiratory rates >30 BPM may be underestimated
- **Shallow Breathing**: Minimal chest/shoulder excursion may not be detected

### Technical Limitations
- **Low Frame Rate**: Cameras <20 FPS may miss breathing cycles
- **CPU Performance**: Slow systems may cause lag and dropped frames
- **Initial Calibration**: First 10-15 seconds of data are unreliable (calibration period)
- **Signal Noise**: Even with smoothing, occasional false peaks occur

### Medical Conditions (⚠️ Do Not Use Clinically)
- **Irregular Breathing**: Conditions like sleep apnea or Cheyne-Stokes respiration not detected
- **Paradoxical Breathing**: Abnormal breathing patterns may give false readings
- **Accessory Muscle Use**: Respiratory distress patterns not accurately captured

**Important**: These limitations mean this tool should NEVER be used for medical monitoring, diagnosis, or treatment decisions.

## Controls

- **ESC or Q**: Quit the application
- **Window**: Click and drag to reposition display window

## Output

The application displays:
- Live webcam feed with pose landmarks overlaid
- Current breathing rate (breaths per minute)
- Status indicator
- Shoulder landmark visualization

If logging is enabled (`--log-data`), a CSV file is created with:
- Timestamp
- Left shoulder Y position
- Right shoulder Y position  
- Calculated breathing rate
- Detection confidence scores

## Troubleshooting

**No video feed / Camera not found:**
- Check camera permissions in system settings
- Try different `--camera-index` values (0, 1, 2...)
- Ensure no other application is using the camera

**Pose not detected:**
- Ensure adequate lighting
- Position yourself fully in frame, facing camera
- Try lowering `--min-detection-confidence` to 0.3
- Check if camera is too close or too far

**Erratic breathing rate readings:**
- Remain still for 15-20 seconds (calibration period)
- Avoid upper body movement
- Increase `--smoothing-window` to 7 or 10
- Ensure proper lighting and camera position

**Low FPS / Laggy performance:**
- Reduce resolution: `--width 320 --height 240`
- Close other applications
- Ensure Python is using system GPU if available

## Project Structure

```
breath_monitor/
├── __init__.py          # Package initialization
├── main.py              # Main application entry point
├── pose_detector.py     # MediaPipe pose detection wrapper
├── breath_analyzer.py   # Breathing rate calculation logic
└── visualizer.py        # Display and UI rendering

tests/
├── test_pose_detector.py
├── test_breath_analyzer.py
└── test_integration.py

docs/
└── algorithm.md         # Detailed algorithm documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MediaPipe](https://google.github.io/mediapipe/) by Google
- Computer vision powered by [OpenCV](https://opencv.org/)
- Signal processing with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)

## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. THE AUTHORS AND COPYRIGHT HOLDERS DISCLAIM ALL LIABILITY FOR ANY DAMAGES OR INJURIES ARISING FROM USE OF THIS SOFTWARE. THIS IS NOT A MEDICAL DEVICE AND MUST NOT BE USED FOR MEDICAL PURPOSES.
