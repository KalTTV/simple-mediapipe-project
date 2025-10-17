# MediaPipe Demo Project

A collection of MediaPipe and OpenCV applications for learning computer vision and real-time video processing.

**GitHub:** [https://github.com/aaronhubhachen/simple-mediapipe-project](https://github.com/aaronhubhachen/simple-mediapipe-project)

## 🆕 New: Infant Breathing Rate Monitor

Real-time breathing rate monitoring using MediaPipe Pose. Track breathing from shoulder movement with anomaly detection (apnea, shallow breathing, tachypnea, bradypnea).

**⚠️ FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE**

Enhanced with **chest expansion detection**: 32-point multi-tracker with hysteresis breath counting (prevents BPM overestimation).

```bash
# Quick start (with chest expansion detection)
pip install -e .
python -m breath_monitor --draw on --trackers 32 --min-ibi-sec 2.0

# Or launch web UI
streamlit run breath_monitor/ui_streamlit.py
```

See [docs/BREATHING_README.md](docs/BREATHING_README.md) for full documentation.

## Original: Tongue Detection Meme Display

A fun application that detects when your tongue is out and displays different meme images in real-time.

Good for learning about:
- MediaPipe Face Mesh and Pose detection
- Real-time video processing with OpenCV
- Facial and body landmark analysis
- Computer vision basics
- Signal processing and filtering

New to this project? Start with the [Quick Start Guide](QUICKSTART.md) to get running in 5 minutes.

Want to understand how it works? Check out the [Tutorial](TUTORIAL.md) for detailed explanations.

macOS user? See the [macOS Setup Guide](SETUP_MACOS.md) for platform-specific instructions.

## Features

### Breathing Monitor
- Real-time breathing rate (BPM) from shoulder movement
- Anomaly detection: Apnea, shallow breathing, tachypnea, bradypnea
- Visual overlay: Pose skeleton, movement trace, HUD badges
- Streamlit web UI with live charts and controls
- WebSocket broadcasting for external integrations
- Fully configurable thresholds and filters
- Comprehensive unit tests

### Tongue Detection
- Real-time webcam face detection using MediaPipe Face Mesh
- Tongue-out detection algorithm
- Dual window display: Camera input and Meme output
- Large window sizes (960x720) for clear visibility
- Real-time switching between normal and tongue-out meme images

## Requirements

- **Python 3.8+** (3.8, 3.9, 3.10, or 3.11 recommended)
- Webcam (built-in or USB)
- For tongue detection: Two meme images (`apple.png` and `appletongue.png`)
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux

## Installation

### Option 1: Quick Install (Recommended for Breathing Monitor)

```bash
# Clone the repository
git clone https://github.com/aaronhubhachen/simple-mediapipe-project.git
cd simple-mediapipe-project

# Install with pip (includes all dependencies)
pip install -e .

# Or use Make
make setup
```

### Option 2: Legacy Install (For Tongue Detection Only)

**Windows:**
```bash
python -m pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m pip install -r requirements.txt
```

2. **Add your meme images:**
   
   You need to provide two PNG images:
   - `apple.png` - Displayed when tongue is NOT out
   - `appletongue.png` - Displayed when tongue IS out
   
   **How to get images:**
   - Find any meme images you like online
   - Create your own custom images
   - Use any PNG images (they will be automatically resized to 960x720)
   - Recommended: Use images with transparent backgrounds or high contrast
   
   **Example ideas:**
   - Apple emoji + Apple with tongue out emoji
   - Happy face + Silly face
   - Normal pet + Derpy pet
   - Any before/after style meme format

## Usage

### Mode Selection

The application now supports two modes:

```bash
# Breathing monitor (default)
python main.py --mode breath [OPTIONS]

# Tongue detection
python main.py --mode face
```

### Breathing Monitor

```bash
# CLI with chest expansion detection
python -m breath_monitor --draw on --trackers 32 --min-ibi-sec 2.0

# Adjust debounce (higher = prevents overcounting, lower = more sensitive)
python -m breath_monitor --draw on --trackers 32 --min-ibi-sec 3.0

# Wide respiratory band (5-72 BPM)
python -m breath_monitor --draw on --resp-low 0.08 --resp-high 1.2

# Web UI
streamlit run breath_monitor/ui_streamlit.py

# Using Make
make run    # CLI
make ui     # Web UI
```

**Key Flags**:
- `--trackers 32`: Number of chest points to track (default: 32)
- `--min-ibi-sec 2.0`: Minimum inter-breath interval / debounce (prevents double counts)
- `--resp-low 0.08 --resp-high 1.2`: Wide respiratory band (5-72 BPM)
- `--tracker-refresh 0.5`: Tracker refresh period (seconds)

**Note**: BPM is displayed as computed (no age-range clamping). Use confidence to judge reliability.

See [docs/BREATHING_README.md](docs/BREATHING_README.md) for detailed usage.

### Tongue Detection (Original)

**Windows:**

**Option 1: Double-click the batch file (easiest):**
- Simply double-click `run.bat` in File Explorer

**Option 2: Run from PowerShell:**
```powershell
.\run.bat
```

**Option 3: Run from Command Prompt (cmd):**
```cmd
run.bat
```

**Option 4: Run directly with Python:**
```bash
python3.11 main.py
```

### macOS/Linux:

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
simple-mediapipe-project/
├── breath_monitor/          # 🆕 Breathing monitor package
│   ├── __init__.py
│   ├── __main__.py         # Module entrypoint
│   ├── capture.py          # Camera capture
│   ├── pose_backend.py     # MediaPipe Pose integration
│   ├── signal.py           # Signal processing
│   ├── events.py           # WebSocket broadcasting
│   ├── draw.py             # Visualization
│   ├── cli.py              # CLI interface
│   └── ui_streamlit.py     # Web UI
├── tests/                   # 🆕 Unit tests
│   └── test_signal.py
├── docs/                    # 🆕 Documentation
│   ├── BREATHING_README.md
│   ├── algorithm.md
│   └── validation.md
├── .github/workflows/       # 🆕 CI/CD
│   └── ci.yml
├── main.py                  # Application entrypoint (with mode selection)
├── pyproject.toml           # 🆕 Modern Python packaging
├── Makefile                 # 🆕 Build automation
├── pytest.ini               # 🆕 Test configuration
├── requirements.txt         # Legacy dependencies
├── run.bat                  # Windows launcher
├── run.sh                   # Unix launcher
├── README.md                # This file
├── QUICKSTART.md
├── TUTORIAL.md
├── IMAGE_GUIDE.md
├── SETUP_MACOS.md
├── CONTRIBUTING.md
└── LICENSE
```

## Technical Details

- **Face Detection**: Uses MediaPipe Face Mesh for real-time facial landmark detection
- **Tongue Detection**: Analyzes mouth landmark distances to determine if tongue is extended
- **Window Size**: 960x720 pixels (approximately half of a 1920x1080 monitor)
- **Frame Processing**: Mirror effect applied for natural interaction

## Dependencies

- `mediapipe==0.10.7` - Face mesh detection and tracking
- `opencv-python==4.8.1.78` - Video capture and display
- `numpy==1.24.3` - Numerical operations

## Customization Ideas

Want to make this project your own? Try these modifications:

1. **Different gestures**: Modify the detection logic to detect smiles, winks, or eyebrow raises
2. **More images**: Add multiple states (happy, sad, surprised) instead of just two
3. **Sound effects**: Play sounds when tongue is detected
4. **Record mode**: Save funny moments to video files
5. **Filters/Effects**: Add Instagram-style filters to the camera feed
6. **Hand gestures**: Combine with MediaPipe Hands for hand gesture detection
7. **Green screen**: Replace the background instead of showing meme images

## Troubleshooting Common Issues

### "No module named 'cv2'"
- Make sure you installed the requirements with Python 3.11
- Run: `python3.11 -m pip install -r requirements.txt`

### Webcam shows black screen
- Check if another application is using the webcam
- Try allowing webcam permissions in Windows Settings

### Detection is laggy
- Close other applications using the webcam
- Lower the camera resolution in `main.py`
- Ensure your computer meets minimum requirements

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
