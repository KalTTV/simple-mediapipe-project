# Validation Protocol

## Overview

This document describes a simple validation protocol for comparing the breathing monitor's output against manual observation.

**⚠️ IMPORTANT**: This is a research/demo tool, NOT a medical device. Do not use for clinical decisions.

## Validation Setup

### Equipment Required

1. **Camera**: Webcam or USB camera (720p minimum)
2. **Computer**: Running the breathing monitor application
3. **Subject**: Volunteer (adult for testing, can simulate slow breathing)
4. **Timer**: Stopwatch or phone timer
5. **Recording Sheet**: Paper to log manual counts

### Environment

- **Lighting**: Good, even lighting (avoid shadows)
- **Background**: Uncluttered, contrasting with subject
- **Camera Position**: 
  - Distance: 1-2 meters from subject
  - Angle: Slightly elevated (30-45° above horizontal)
  - Framing: Full torso visible, shoulders clearly in frame

### Subject Positioning

- **Posture**: Sitting or lying down (consistent across trials)
- **Clothing**: Fitted clothing that allows shoulder movement visibility
- **Movement**: Minimize unnecessary movement during measurement
- **Breathing**: Natural breathing (don't try to control rate)

## Validation Procedure

### Trial Structure

Perform **3 trials** of **60 seconds** each:

1. **Trial 1**: Baseline (natural breathing)
2. **Trial 2**: Slow breathing (if capable)
3. **Trial 3**: Baseline repeat

### Step-by-Step Protocol

#### Pre-Trial Setup

1. Start the breathing monitor application:
   ```bash
   python -m breath_monitor --draw on
   ```

2. Position subject and camera as described above

3. Verify pose detection:
   - Green skeleton visible
   - Shoulders clearly marked
   - Confidence > 0.7

4. Wait 15 seconds for buffer to fill

#### During Trial (60 seconds)

**Manual Counter (Observer)**:
1. Start timer
2. Count each breath cycle (one inhale + one exhale = 1 breath)
3. Use a counter or tally marks
4. Stop at 60 seconds
5. Record count: `Manual_Count`

**Application**:
1. Observe BPM display continuously
2. Note BPM value at 60-second mark: `App_BPM`
3. Screenshot or record display for verification

#### Post-Trial

Calculate results:
- **Manual BPM**: `Manual_BPM = Manual_Count`
- **Error**: `Error = App_BPM - Manual_BPM`
- **Percent Error**: `Error% = (Error / Manual_BPM) × 100`

### Recording Sheet Template

```
Trial: ___  Date: _______  Subject: _______

Camera Distance: _____ m
Camera Angle: _____ °
Lighting: [ ] Good  [ ] Fair  [ ] Poor
Clothing: [ ] Fitted  [ ] Loose

Manual Count (60s): _____
Manual BPM: _____

App BPM (at 60s): _____
App Confidence: _____

Error: _____ BPM
Error %: _____ %

Notes:
________________________________
________________________________
```

## Acceptance Criteria

### Performance Targets

- **Error < ±3 BPM**: Excellent
- **Error < ±5 BPM**: Good
- **Error < ±10 BPM**: Acceptable for demo
- **Error > ±10 BPM**: Needs improvement

### Expected Results

Based on synthetic testing:
- **Accuracy**: ±10% of ground truth
- **Repeatability**: ±2 BPM across trials (same subject/conditions)

## Known Sources of Error

### Subject-Related

1. **Movement**: Shifting position, fidgeting
2. **Breathing Pattern**: Irregular breathing, sighing
3. **Clothing**: Loose/thick clothing dampens signal
4. **Posture**: Hunching, arm position blocking shoulders

### Environment-Related

1. **Lighting**: Shadows, backlighting
2. **Camera Angle**: Too low, too steep
3. **Distance**: Too far (small landmarks), too close (partial view)
4. **Background**: Busy background confuses pose detection

### Technical-Related

1. **Pose Confidence**: Confidence < 0.5 (app suspends updates)
2. **Occlusion**: Blanket, hand covering shoulders
3. **Motion Artifacts**: Large movements during trial
4. **Buffer Effects**: First 10-15s are less stable

## Troubleshooting Poor Accuracy

### If Error > 10 BPM

1. **Check Confidence**: Is it consistently > 0.7?
   - If NO: Improve lighting, reframe camera

2. **Check Pose Detection**: Are shoulders stable?
   - If NO: Reduce subject movement, tighter clothing

3. **Check Manual Count**: Recount (observer error possible)

4. **Adjust Parameters**:
   ```bash
   # For slow breathers
   --bpf-low 0.3 --bpf-high 1.0
   
   # For fast breathers
   --bpf-low 0.7 --bpf-high 1.5
   ```

5. **Verify Signal Quality**: 
   - Look at trend chart in Streamlit UI
   - Signal should be smooth, periodic

## Sample Results

### Example Trial Log

```
Subject: Adult volunteer (research demo)
Camera: Logitech C920, 1m distance, 45° angle
Lighting: Indoor (LED overhead + window)
Clothing: Fitted T-shirt

Trial 1 (Baseline):
  Manual: 16 breaths → 16 BPM
  App: 15.8 BPM
  Error: -0.2 BPM (-1.3%)
  Confidence: 0.82

Trial 2 (Slow):
  Manual: 12 breaths → 12 BPM
  App: 12.4 BPM
  Error: +0.4 BPM (+3.3%)
  Confidence: 0.79

Trial 3 (Baseline repeat):
  Manual: 15 breaths → 15 BPM
  App: 15.2 BPM
  Error: +0.2 BPM (+1.3%)
  Confidence: 0.85

Average Error: ±0.3 BPM (±1.9%)
Repeatability: 0.6 BPM (trials 1 vs 3)
```

## Advanced Validation (Optional)

### Multi-Subject Testing

- Test with 5+ subjects
- Vary age, body type, clothing
- Calculate mean and std dev of error

### Respiratory Device Comparison

- Use chest strap monitor (e.g., fitness band) as reference
- Simultaneous recording
- Compare timestamps

### Extended Duration

- Test over 5-10 minutes
- Assess long-term stability
- Check for drift in BPM readings

## Limitations of Validation

1. **Manual Counting**: Observer can miscount (±1-2 breaths typical)
2. **Snapshot Timing**: App BPM updates lag by ~1-2 seconds (EMA)
3. **Breathing Variability**: Real breathing is not perfectly regular
4. **Camera Variability**: Different cameras → different results

## Reporting Validation Results

When sharing validation data, include:

1. **Hardware**: Camera model, resolution, FPS
2. **Software**: Application version, parameters used
3. **Setup**: Distance, angle, lighting, clothing
4. **Results**: All trial data, error statistics
5. **Issues**: Any anomalies or observations

## Safety and Ethics

- **Informed Consent**: Explain this is a demo, not medical
- **Privacy**: No data is stored by default (unless user records)
- **Monitoring**: Never use as sole means of monitoring an infant
- **Medical Conditions**: Do not test on subjects with respiratory distress

## Continuous Improvement

Use validation data to:
- Tune bandpass filter parameters
- Adjust anomaly detection thresholds
- Identify failure modes
- Improve documentation

Submit validation reports as GitHub issues for community benefit.

