# Tongue Detection Meme Display

A fun Python application using MediaPipe and OpenCV that detects when your tongue is out and displays different meme images in real-time.

**GitHub:** [https://github.com/aaronhubhachen/simple-mediapipe-project](https://github.com/aaronhubhachen/simple-mediapipe-project)

Good for learning about:
- MediaPipe Face Mesh detection
- Real-time video processing with OpenCV
- Facial landmark analysis
- Computer vision basics

New to this project? Start with the [Quick Start Guide](QUICKSTART.md) to get running in 5 minutes.

Want to understand how it works? Check out the [Tutorial](TUTORIAL.md) for detailed explanations.

macOS user? See the [macOS Setup Guide](SETUP_MACOS.md) for platform-specific instructions.

## Features

- Real-time webcam face detection using MediaPipe Face Mesh
- Tongue-out detection algorithm
- Dual window display: Camera input and Meme output
- Large window sizes (960x720) for clear visibility
- Real-time switching between normal and tongue-out meme images

## Requirements

- **Python 3.11** (specifically 3.11, not 3.13 or other versions)
- Webcam (built-in or USB)
- Two meme images: `apple.png` and `appletongue.png`
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux

## Installation

### For All Operating Systems:

1. **Install Python dependencies (using Python 3.11):**
   
   **Windows:**
   ```bash
   python3.11 -m pip install -r requirements.txt
   ```
   
   **macOS/Linux:**
   ```bash
   python3.11 -m pip install -r requirements.txt
   # or if python3.11 command doesn't exist:
   pip3 install -r requirements.txt
   ```

2. **Add your meme images:**
   - Place `apple.png` and `appletongue.png` in the project directory
   - Or use your own images (update the paths in `main.py`)

## Usage

### Quick Start (Recommended):

**Windows:**
```bash
run.bat
```

**macOS/Linux:**
```bash
./run.sh
# or
chmod +x run.sh && ./run.sh
```

### Manual Start:

```bash
python main.py
```

or

```bash
python3.11 main.py
```

## Controls

- **q** or **ESC**: Quit the application
- Stick out your tongue to see the meme change!

## How It Works

1. **Face Detection**: Uses MediaPipe Face Mesh to detect 478 facial landmarks in real-time
2. **Tongue Detection**: Analyzes the vertical distance between specific mouth landmarks
3. **Image Switching**: When tongue is detected (distance > threshold), switches to the "tongue out" meme
4. **Dual Display**: Shows both the webcam feed and the corresponding meme image

For a detailed technical explanation, see the [Tutorial](TUTORIAL.md).

## Project Structure

```
simple-mediapipe-project/
├── main.py                 # Main application code
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows launcher
├── run.sh                  # macOS/Linux launcher
├── apple.png              # Normal state meme
├── appletongue.png        # Tongue out meme
├── QUICKSTART.md          # Quick start guide
├── TUTORIAL.md            # Detailed tutorial
├── SETUP_MACOS.md         # macOS-specific setup
├── CONTRIBUTING.md        # Contribution guidelines
├── IMAGE_GUIDE.md         # Image requirements guide
└── LICENSE                # MIT License
```

## Customization Ideas

1. **Sound effects**: Add audio when tongue is detected
2. **Different gestures**: Detect winks, smiles, or head tilts
3. **Multiple memes**: Cycle through different images
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

Contributions are welcome! Check out the [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

Ideas for contributions:
- Improve tongue detection algorithm
- Add support for multiple gestures (smile, wink, etc.)
- Create a GUI for adjusting sensitivity
- Add gesture recording/playback
- Optimize performance
- Improve cross-platform compatibility

## Platform Support

This project is cross-platform:

| Platform | Status | Instructions |
|----------|--------|-------------|
| Windows 10/11 | Fully Supported | Use `run.bat` or see [Quick Start](QUICKSTART.md) |
| macOS 10.14+ | Fully Supported | See [macOS Setup Guide](SETUP_MACOS.md) |
| Linux | Supported | Use `run.sh` or see [Quick Start](QUICKSTART.md) |

## Credits

Built with:
- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- Python 3.11

## License

MIT License - Feel free to use and modify as needed. See LICENSE file for details.

## Breathing Monitor - MVP

This branch contains the initial MVP implementation of a real-time infant breathing monitor using MediaPipe Pose detection.

### Project Structure
- `breath_monitor/` - Core monitoring modules
  - `capture.py` - Video capture handling
  - `pose_backend.py` - MediaPipe Pose integration
  - `signal.py` - Signal processing for breathing detection
  - `events.py` - Event detection and alerting
  - `cli.py` - Command-line interface
  - `ui_streamlit.py` - Streamlit web interface
- `tests/` - Unit tests
- `docs/algorithm.md` - Algorithm documentation
- `scripts/benchmark.mp4` - Sample video for testing

### Features
- Real-time pose landmark detection
- Breathing pattern analysis
- Alert system for irregular breathing
- Web-based monitoring interface
- CLI tool for quick testing
