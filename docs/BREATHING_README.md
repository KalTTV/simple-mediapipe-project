# Infant Breathing Rate Monitor

Real-time breathing rate monitoring using MediaPipe Pose and computer vision.

## ⚠️ IMPORTANT SAFETY NOTICE

**FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY**

This is NOT a medical device and should NEVER be used:
- As the sole means of monitoring an infant
- For clinical diagnosis or treatment decisions  
- As a replacement for medical-grade equipment
- In any life-critical application

Always use proper medical monitoring equipment under healthcare professional supervision.

## Features

- **Real-time BPM Detection**: Measures breathing rate from shoulder movement
- **Visual Overlay**: Pose skeleton, movement trace, and HUD display
- **Anomaly Detection**: Alerts for apnea, shallow breathing, tachypnea, bradypnea
- **Multiple Interfaces**: CLI with OpenCV display and Streamlit web UI
- **WebSocket Broadcasting**: Real-time state updates for external integrations
- **Configurable**: All thresholds adjustable via command-line flags
- **Unit Tested**: Comprehensive test suite with synthetic signals

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or using the Makefile
make setup
```

### Run CLI Monitor

```bash
# Basic run with visualization
python -m breath_monitor --draw on

# Or using Make
make run

# With custom parameters
python -m breath_monitor --camera 0 --width 640 --height 480 --apnea-sec 15 --draw on
```

### Run Streamlit UI

```bash
# Launch web UI
streamlit run breath_monitor/ui_streamlit.py

# Or using Make
make ui
```

### Run Tests

```bash
# Run unit tests
pytest -v

# Or using Make
make test
```

## Usage

### CLI Options

```bash
python -m breath_monitor [OPTIONS]

Camera Settings:
  --camera INT       Camera device ID (default: 0)
  --width INT        Frame width (default: 640)
  --height INT       Frame height (default: 480)
  --fps INT          Target FPS (default: 30)

Signal Processing:
  --min-sec FLOAT    Buffer window in seconds (default: 15.0)
  --bpf-low FLOAT    Bandpass low cutoff Hz (default: 0.5)
  --bpf-high FLOAT   Bandpass high cutoff Hz (default: 1.2)

Detection Thresholds:
  --apnea-sec FLOAT  Apnea threshold seconds (default: 20.0)
  --tachy FLOAT      Tachypnea threshold BPM (default: 60.0)
  --brady FLOAT      Bradypnea threshold BPM (default: 30.0)

Output:
  --ws on|off        WebSocket broadcasting (default: off)
  --draw on|off      Visual overlay (default: on)
```

### Example Commands

```bash
# Monitor with stricter apnea threshold
python -m breath_monitor --apnea-sec 10 --draw on

# Adjust for older child (slower breathing)
python -m breath_monitor --bpf-low 0.3 --bpf-high 1.0 --brady 20

# Enable WebSocket for external app integration
python -m breath_monitor --ws on --draw off

# High resolution capture
python -m breath_monitor --width 1280 --height 720 --fps 60
```

## How It Works

See [docs/algorithm.md](./algorithm.md) for detailed technical documentation.

**Summary**:
1. Capture video from webcam
2. Detect pose using MediaPipe Pose Landmarker
3. Extract mid-shoulder vertical position as breathing proxy
4. Buffer 15 seconds of signal data
5. Detrend to remove baseline drift
6. Bandpass filter (0.5-1.2 Hz for infant breathing range)
7. Detect peaks with refractory period
8. Calculate BPM from median inter-peak interval
9. Smooth for display with exponential moving average
10. Detect anomalies (apnea, shallow, tachy/brady)

## Camera Setup

### Optimal Configuration

- **Distance**: 1-2 meters from subject
- **Angle**: Slightly elevated (30-45° above horizontal)
- **Framing**: Full torso visible, shoulders clearly in frame
- **Lighting**: Good, even lighting (avoid shadows)
- **Background**: Uncluttered, contrasting with subject

### Clothing Recommendations

- Fitted clothing works best
- Avoid thick blankets covering shoulders
- Light colors on dark background (or vice versa)

## Validation

See [docs/validation.md](./validation.md) for validation protocol.

**Expected Accuracy**: ±10% of ground truth (tested with synthetic signals)

**Real-World Performance**: Depends on camera quality, lighting, subject movement, and clothing.

## Troubleshooting

### No Pose Detected

- Check lighting (need good illumination)
- Verify camera is working (`--draw on` to see video)
- Ensure subject is in frame and shoulders are visible
- Try different camera angle

### BPM Shows "--"

- Confidence too low (< 0.5)
- Improve pose visibility
- Wait 10-15 seconds for buffer to fill
- Check that shoulders are not occluded

### Incorrect BPM

- Verify manual count matches (see validation protocol)
- Adjust bandpass filter for subject's breathing rate
- Reduce subject movement
- Ensure tight-fitting clothing
- Check camera angle and distance

### Slow/Laggy

- Reduce resolution: `--width 320 --height 240`
- Lower FPS: `--fps 15`
- Close other applications
- MediaPipe Pose runs best on systems with good CPUs

## Known Limitations

- **Occlusion**: Blankets covering shoulders will lose tracking
- **Movement**: Subject movement can contaminate signal
- **Camera Angle**: Best with slightly elevated camera
- **Clothing**: Loose/thick clothing reduces accuracy
- **Low Light**: Poor lighting reduces landmark confidence
- **Latency**: 10-15 seconds to stabilize after start

## Project Structure

```
breath_monitor/
├── __init__.py          # Package init
├── __main__.py          # Module entrypoint
├── capture.py           # Camera capture
├── pose_backend.py      # MediaPipe Pose integration
├── signal.py            # Signal processing (detrend, filter, peaks, BPM)
├── events.py            # WebSocket broadcasting
├── draw.py              # Visualization utilities
├── cli.py               # CLI entrypoint
└── ui_streamlit.py      # Streamlit web UI

tests/
└── test_signal.py       # Unit tests

docs/
├── algorithm.md         # Technical details
└── validation.md        # Validation protocol
```

## API Usage (Advanced)

```python
from breath_monitor.capture import CameraCapture
from breath_monitor.pose_backend import PoseBackend
from breath_monitor.signal import BreathingAnalyzer

# Initialize components
capture = CameraCapture(camera_id=0, width=640, height=480, fps=30)
pose = PoseBackend()
analyzer = BreathingAnalyzer(window_sec=15.0, bpf_low=0.5, bpf_high=1.2)

capture.start()

# Process frames
for frame, timestamp in capture.frames():
    # Get pose
    result = pose.infer(frame)
    
    # Add signal sample
    if result["detected"] and result["confidence"] > 0.5:
        _, y = result["mid_shoulder_xy"]
        analyzer.add_sample(timestamp, y)
    
    # Analyze
    analysis = analyzer.analyze()
    bpm = analysis.get("bpm_smooth")
    
    print(f"BPM: {bpm}")
    
    # Your custom logic here...
```

## WebSocket API

When `--ws on` is enabled, state is broadcast to `ws://localhost:8765`:

```json
{
  "bpm": 42.5,
  "apnea": false,
  "shallow": false,
  "confidence": 0.85,
  "timestamp": 1634567890.123
}
```

Connect with any WebSocket client:

```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
  const state = JSON.parse(event.data);
  console.log(`BPM: ${state.bpm}`);
};
```

## Performance Notes

- **CPU Usage**: ~15-30% (single core, pose detection)
- **FPS**: 30 FPS achievable on modern laptops
- **RAM**: ~200 MB
- **Latency**: 33ms per frame + 1-2s display smoothing

## Contributing

Contributions welcome! Areas for improvement:
- Multi-landmark fusion (hips, chest)
- Better motion artifact rejection
- Adaptive bandpass tuning
- Mobile/edge deployment optimization

## License

MIT License - see LICENSE file

## Credits

- **MediaPipe**: Google's ML framework for pose detection
- **OpenCV**: Video capture and display
- **SciPy**: Signal processing (filtering, peak detection)
- **Streamlit**: Web UI framework

## Support

For issues, questions, or validation reports:
- Open a GitHub issue
- Include: camera specs, lighting conditions, sample video if possible
- Check docs/validation.md for troubleshooting guide

