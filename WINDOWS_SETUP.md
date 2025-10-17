# Windows Setup Guide for Breathing Monitor

## Prerequisites

### 1. Install Python

Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/)

**Important**: During installation, check "Add Python to PATH"

Verify installation:
```powershell
python --version
```

Should show something like `Python 3.11.x`

### 2. Install Git (Optional, for cloning)

Download from [git-scm.com](https://git-scm.com/download/win)

Or download the repository as a ZIP file from GitHub.

## Installation Steps

### Option 1: Quick Install (Recommended)

```powershell
# Navigate to project directory
cd C:\path\to\simple-mediapipe-project

# Install package and dependencies
python -m pip install -e .
```

This will install:
- opencv-python
- mediapipe
- numpy
- scipy
- websockets
- streamlit
- matplotlib

### Option 2: Manual Install

```powershell
# Install individual packages
pip install opencv-python mediapipe numpy scipy websockets streamlit matplotlib
```

## Running the Application

### Method 1: CLI with Visualization (Recommended First Test)

```powershell
python -m breath_monitor --draw on
```

You should see:
- A window showing your webcam feed
- Pose skeleton overlay (green lines)
- BPM display in top-left corner
- Status badges (APNEA, SHALLOW, etc. when detected)

Press 'q' to quit.

### Method 2: Streamlit Web UI

```powershell
streamlit run breath_monitor/ui_streamlit.py
```

Your browser will open automatically with the web interface.

### Method 3: Using Main.py with Mode Selection

```powershell
# Breathing monitor (default)
python main.py --mode breath --draw on

# Original tongue detection
python main.py --mode face
```

## Common Issues

### "Python was not found"

**Solution**: Python is not in your PATH.
1. Reinstall Python and check "Add Python to PATH"
2. Or manually add Python to PATH in System Environment Variables

### "No module named 'cv2'" or similar

**Solution**: Dependencies not installed.
```powershell
python -m pip install -e .
```

### "Could not open camera"

**Solution**:
- Check camera is connected
- Close other apps using the camera (Zoom, Teams, etc.)
- Try a different camera: `--camera 1`
- Check Windows camera privacy settings

### "No pose detected"

**Solution**:
- Ensure good lighting
- Position yourself 1-2 meters from camera
- Make sure your shoulders are visible
- Try adjusting camera angle

## Configuration Options

### Adjust for Different Breathing Rates

```powershell
# For faster breathing (toddlers, exercise)
python -m breath_monitor --bpf-high 1.5 --tachy 70 --draw on

# For slower breathing (sleeping adult)
python -m breath_monitor --bpf-low 0.3 --brady 20 --draw on
```

### Change Camera Settings

```powershell
# Use different camera
python -m breath_monitor --camera 1 --draw on

# Higher resolution
python -m breath_monitor --width 1280 --height 720 --draw on

# Lower resolution for performance
python -m breath_monitor --width 320 --height 240 --draw on
```

### Enable WebSocket Broadcasting

```powershell
python -m breath_monitor --ws on --draw off
```

Then connect from JavaScript, Python, or any WebSocket client to `ws://localhost:8765`

## Testing

### Run Unit Tests

```powershell
# Install pytest first
pip install pytest

# Run tests
pytest -v
```

You should see all tests passing (green).

### Generate Test Video

```powershell
python scripts/generate_benchmark.py --output test.mp4 --duration 30 --bpm 48
```

This creates a synthetic video with known breathing rate for validation.

## Performance Tips

### If Application is Slow

1. **Lower resolution**:
   ```powershell
   python -m breath_monitor --width 320 --height 240 --fps 15 --draw on
   ```

2. **Close other applications**: Browser tabs, video editors, etc.

3. **Check CPU usage**: MediaPipe Pose is CPU-intensive (15-30% typical)

### Expected Performance

- **FPS**: 30 FPS on modern laptops (Core i5/Ryzen 5 or better)
- **CPU**: 15-30% single core
- **RAM**: ~200 MB
- **Latency**: 10-15 seconds for BPM to stabilize

## Next Steps

1. **Read Documentation**:
   - [Algorithm Overview](docs/algorithm.md)
   - [Breathing Monitor README](docs/BREATHING_README.md)
   - [Validation Protocol](docs/validation.md)

2. **Try Different Scenarios**:
   - Normal breathing
   - Slow breathing (hold breath briefly)
   - Deep breathing
   - Movement (see how it affects detection)

3. **Validate Accuracy**:
   - Follow [docs/validation.md](docs/validation.md)
   - Compare with manual counting

## Safety Reminder

⚠️ **FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE**

Never use this application:
- As the sole means of monitoring an infant
- For medical diagnosis or treatment
- In any life-critical situation

Always use proper medical equipment under healthcare professional supervision.

## Support

If you encounter issues:
1. Check this guide and [docs/BREATHING_README.md](docs/BREATHING_README.md)
2. Search for similar issues on GitHub
3. Open a new GitHub issue with:
   - Windows version
   - Python version
   - Camera model
   - Error messages or screenshots

## Windows-Specific Notes

### Make Commands Won't Work

The Makefile uses Unix commands. Use Python commands directly:

```powershell
# Instead of "make run"
python -m breath_monitor --draw on

# Instead of "make ui"
streamlit run breath_monitor/ui_streamlit.py

# Instead of "make test"
pytest -v
```

### PowerShell vs CMD

Both work, but PowerShell is recommended for better output formatting.

### Webcam Privacy Settings

If camera doesn't work:
1. Open Windows Settings
2. Privacy & Security → Camera
3. Ensure "Camera access" is On
4. Ensure "Let desktop apps access your camera" is On

