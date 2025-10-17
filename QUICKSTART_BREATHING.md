# Quick Start: Breathing Monitor (Windows)

## ⚠️ SAFETY FIRST
**FOR RESEARCH/DEMO ONLY - NOT A MEDICAL DEVICE**

## Prerequisites (Install First)

1. **Python 3.8 or newer** from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
   
2. **Verify Python is installed**:
   ```powershell
   python --version
   ```

## Step 1: Install Dependencies

```powershell
# Navigate to project folder
cd C:\Users\KK\simple-mediapipe-project

# Install everything
python -m pip install -e .
```

This installs: opencv-python, mediapipe, numpy, scipy, websockets, streamlit, matplotlib

## Step 2: Run the Monitor

### Option A: CLI with Video (Recommended First)

```powershell
python -m breath_monitor --draw on
```

**What you'll see**:
- Webcam window opens
- Green skeleton overlays your body
- BPM counter in top-left
- Press **'q'** to quit

### Option B: Web Interface

```powershell
streamlit run breath_monitor/ui_streamlit.py
```

Browser opens automatically with interactive dashboard.

## Step 3: Position Yourself

**For best results**:
- Sit or lie 1-2 meters from camera
- Keep shoulders visible and unobstructed
- Good lighting (avoid shadows)
- Fitted clothing preferred

## Expected Behavior

1. **First 15 seconds**: BPM shows "--" (buffer filling)
2. **After 15-20 seconds**: BPM appears and stabilizes
3. **Confidence meter**: Should be > 0.7 (green)
4. **Status badges**: Show APNEA, SHALLOW, TACHY, BRADY when detected

## Common Commands

```powershell
# Basic run
python -m breath_monitor --draw on

# Web UI
streamlit run breath_monitor/ui_streamlit.py

# Custom camera
python -m breath_monitor --camera 1 --draw on

# Stricter apnea detection
python -m breath_monitor --apnea-sec 10 --draw on

# Enable WebSocket broadcasting
python -m breath_monitor --ws on --draw off

# Run tests
pytest -v

# Generate test video
python scripts/generate_benchmark.py --output test.mp4 --bpm 48
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python was not found" | Reinstall Python, check "Add to PATH" |
| "No module named 'cv2'" | Run: `python -m pip install -e .` |
| "Could not open camera" | Close other camera apps, try `--camera 1` |
| BPM shows "--" forever | Wait 15s, check lighting, ensure shoulders visible |
| No pose detected | Better lighting, adjust angle, move 1-2m from camera |

## Performance Tips

**For slower computers**:
```powershell
python -m breath_monitor --width 320 --height 240 --fps 15 --draw on
```

**For faster computers**:
```powershell
python -m breath_monitor --width 1280 --height 720 --fps 60 --draw on
```

## Expected Performance

- **FPS**: 30 (Core i5 or better)
- **CPU**: 15-30% single core
- **RAM**: ~200 MB
- **Stabilization**: 10-15 seconds

## Next Steps

1. ✅ Get it running with `python -m breath_monitor --draw on`
2. 📖 Read [docs/BREATHING_README.md](docs/BREATHING_README.md) for detailed docs
3. 🧪 Follow [docs/validation.md](docs/validation.md) to validate accuracy
4. ⚙️ Experiment with different settings

## Documentation

- **Full Guide**: [docs/BREATHING_README.md](docs/BREATHING_README.md)
- **Windows Setup**: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **Algorithm Details**: [docs/algorithm.md](docs/algorithm.md)
- **Validation**: [docs/validation.md](docs/validation.md)
- **Project Summary**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## Original Tongue Detection Mode

Still available:
```powershell
python main.py --mode face
```

## Support

Issues? Check:
1. This guide
2. [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
3. [docs/BREATHING_README.md](docs/BREATHING_README.md)

## Remember

🚫 **Never use as sole infant monitor**  
🚫 **Not for medical decisions**  
✅ **Research and demo only**  
✅ **Always use proper medical equipment**

---

**Enjoy exploring computer vision and signal processing!** 🎥📊

