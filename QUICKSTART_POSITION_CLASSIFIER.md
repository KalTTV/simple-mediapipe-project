# Quick Start: Position Classifier

Train and use the sleeping position classifier in 3 easy steps.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- scikit-learn (machine learning)
- joblib (model saving)
- Other existing dependencies

## Step 2: Train the Model

```bash
python -m breath_monitor.train_pose_classifier
```

Or on Windows:
```bash
py -m breath_monitor.train_pose_classifier
```

**What happens:**
- Reads images from `breath_monitor/data/safe/` and `breath_monitor/data/danger/`
- Extracts pose landmarks using MediaPipe
- Trains a Random Forest classifier
- Saves model to `models/position_model.joblib`

**Expected time:** 1-3 minutes (depending on dataset size)

## Step 3: Use It

The classifier is automatically loaded when you use `PoseBackend`:

```python
from breath_monitor.pose_backend import PoseBackend

backend = PoseBackend()
result = backend.infer(frame, timestamp)

# Check position safety
if result['position_safe'] == False:
    print("⚠️  WARNING: Unsafe sleeping position detected!")
    print(f"   Confidence: {result['position_confidence']:.2f}")
```

## Test with Webcam

See the classifier in action:

```bash
python -m breath_monitor.pose_backend
```

You'll see:
- ✅ "Position: SAFE" in green for safe positions
- ⚠️  "Position: DANGER" in red for unsafe positions

Press 'q' to quit.

## That's It!

The position classifier is now integrated and running automatically. No code changes needed.

---

## Need Help?

### "Model not found" error
Run step 2 to train the model first.

### "No images found" error
Make sure images are in:
- `breath_monitor/data/safe/`
- `breath_monitor/data/danger/`

### Retrain the model
Delete `models/position_model.joblib` and run step 2 again.

### More info
See `POSITION_CLASSIFIER_INTEGRATION.md` for detailed documentation.

