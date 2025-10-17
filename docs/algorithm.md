# Breathing Signal Processing Pipeline

This document explains the complete algorithm used to detect and monitor breathing patterns in real-time using MediaPipe pose landmarks.

## Overview

The breathing detection system processes video frames to extract chest movement, filters the signal, detects breathing cycles, and identifies abnormal breathing events such as apnea (breathing stopped) and shallow breathing.

## Pipeline Stages

### 1. Chest Keypoint Extraction

The first stage extracts relevant pose landmarks from each video frame using MediaPipe Pose.

**Process:**
- MediaPipe detects 33 pose landmarks on the body
- We focus on chest-related landmarks:
  - **Left Shoulder** (landmark 11)
  - **Right Shoulder** (landmark 12)
  - **Left Hip** (landmark 23)
  - **Right Hip** (landmark 24)

**Chest Distance Calculation:**
```python
# Calculate midpoints
left_mid = (left_shoulder + left_hip) / 2
right_mid = (right_shoulder + right_hip) / 2

# Compute Euclidean distance
chest_distance = ||left_mid - right_mid||
```

This distance metric captures chest expansion and contraction during breathing. As the chest expands during inhalation, the distance increases; during exhalation, it decreases.

**Normalization:**
The raw distance is normalized to account for varying distances from the camera:
```python
normalized_distance = chest_distance / torso_height
```
where `torso_height` is the distance between shoulder midpoint and hip midpoint.

### 2. Signal Buffering

Raw chest distance measurements are stored in a circular buffer for temporal processing.

**Buffer Properties:**
- **Size**: Typically 300 frames (10 seconds at 30 FPS)
- **Purpose**: Provides historical context for filtering and pattern detection
- **Implementation**: Fixed-size deque or numpy array with rolling window

**Benefits:**
- Enables frequency-domain analysis
- Smooths out single-frame anomalies
- Provides sufficient data for breathing cycle detection (normal breathing: 12-20 breaths/minute)

### 3. Butterworth Filtering

A Butterworth bandpass filter removes noise and isolates the breathing frequency range.

**Filter Specifications:**
- **Type**: Bandpass Butterworth filter
- **Order**: 4th order (steeper rolloff, better frequency isolation)
- **Frequency Range**: 0.1 - 0.5 Hz (6-30 breaths per minute)
  - Lower bound (0.1 Hz): Filters out very slow drift and body movements
  - Upper bound (0.5 Hz): Removes high-frequency noise (camera jitter, small movements)

**Why Butterworth?**
- Maximally flat frequency response in passband
- No ripples in the breathing frequency range
- Smooth phase response reduces signal distortion

**Implementation:**
```python
from scipy.signal import butter, filtfilt

# Design filter
nyquist = sampling_rate / 2
low = 0.1 / nyquist  # 6 BPM
high = 0.5 / nyquist  # 30 BPM
b, a = butter(4, [low, high], btype='band')

# Apply zero-phase filtering
filtered_signal = filtfilt(b, a, signal_buffer)
```

The `filtfilt` function applies the filter twice (forward and backward) for zero phase distortion.

### 4. Peak Detection for BPM

Breathing cycles are identified by detecting peaks in the filtered signal.

**Peak Detection Algorithm:**
```python
from scipy.signal import find_peaks

# Detect peaks with constraints
peaks, properties = find_peaks(
    filtered_signal,
    distance=fps * 1.5,      # Minimum 1.5 seconds between breaths
    prominence=threshold,     # Minimum peak prominence
    height=mean + 0.5 * std  # Adaptive threshold
)
```

**Parameters:**
- **Distance**: Minimum frames between peaks (prevents double-counting)
- **Prominence**: Peak must stand out from surrounding signal
- **Height**: Adaptive threshold based on signal statistics

**BPM Calculation:**
```python
if len(peaks) >= 2:
    # Calculate time between consecutive peaks
    intervals = np.diff(peaks) / fps  # Convert to seconds
    
    # Average interval over recent breaths
    avg_interval = np.mean(intervals[-5:])  # Last 5 breaths
    
    # Convert to breaths per minute
    bpm = 60.0 / avg_interval
```

**Smoothing:**
BPM values are smoothed using an exponential moving average to reduce jitter:
```python
smoothed_bpm = alpha * new_bpm + (1 - alpha) * previous_bpm
```
where `alpha = 0.3` provides good responsiveness while filtering noise.

### 5. Event Logic for Apnea and Shallow Breathing

The system continuously monitors the filtered signal to detect abnormal breathing patterns.

#### 5.1 Apnea Detection (Breathing Stopped)

**Definition:** No breathing detected for a prolonged period.

**Detection Logic:**
```python
APNEA_THRESHOLD_SECONDS = 10  # No breath for 10 seconds

# Check time since last detected peak
if current_time - last_peak_time > APNEA_THRESHOLD_SECONDS:
    trigger_apnea_alert()
```

**Additional Validation:**
- Signal variance must be low (confirms lack of movement, not just missed detection)
- No significant peaks below detection threshold

**Alert Behavior:**
- Visual warning on screen
- Optional audio alert
- Log timestamp and duration

#### 5.2 Shallow Breathing Detection

**Definition:** Breathing amplitude is significantly reduced compared to normal baseline.

**Detection Logic:**
```python
SHALLOW_THRESHOLD = 0.6  # 60% of normal amplitude

# Calculate baseline from recent history
baseline_amplitude = np.percentile(peak_heights[-20:], 75)

# Check current peak amplitudes
if current_peak_height < SHALLOW_THRESHOLD * baseline_amplitude:
    shallow_breath_count += 1
    
    if shallow_breath_count >= 3:  # 3 consecutive shallow breaths
        trigger_shallow_breathing_alert()
```

**Adaptive Baseline:**
The baseline adjusts over time to account for:
- Different body types
- Camera positioning
- Clothing effects

**Reset Conditions:**
- Normal breathing detected: Reset counter
- Deep breath detected: Update baseline upward

#### 5.3 Event State Machine

```
[NORMAL] ──(no peaks for 10s)──> [APNEA]
    │                                │
    │                                │
    │                          (peak detected)
    │                                │
    │                                ↓
    ├───(3 shallow breaths)──> [SHALLOW] ──(normal breath)──> [NORMAL]
    │                                │
    │                                │
    └────────(normal breath)─────────┘
```

**State Transitions:**
- Normal → Apnea: No breath for threshold duration
- Normal → Shallow: Multiple consecutive shallow breaths
- Apnea → Normal: Any breath detected
- Shallow → Normal: Normal amplitude breath detected

## Performance Considerations

### Real-time Processing

- **Frame Rate**: Target 30 FPS
- **Latency**: < 100ms per frame
- **Buffer Update**: O(1) with circular buffer
- **Filtering**: Applied every N frames (e.g., every 10 frames) to balance responsiveness and performance

### Accuracy Improvements

1. **Multi-point averaging**: Use multiple landmark pairs for redundancy
2. **Outlier rejection**: Remove statistical outliers before filtering
3. **Confidence thresholding**: Only process frames with high landmark confidence
4. **Calibration period**: Initial 10-second calibration to establish personal baseline

## Configuration Parameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `buffer_size` | 300 frames | Signal history length |
| `filter_order` | 4 | Butterworth filter order |
| `low_freq` | 0.1 Hz | Minimum breathing frequency |
| `high_freq` | 0.5 Hz | Maximum breathing frequency |
| `peak_distance` | 1.5 sec | Minimum time between breaths |
| `apnea_threshold` | 10 sec | Time before apnea alert |
| `shallow_threshold` | 0.6 | Amplitude ratio for shallow breathing |
| `bpm_smoothing` | 0.3 | EMA smoothing factor |

## Edge Cases and Limitations

### Handled Cases
- **Camera movement**: Normalization by torso height
- **Distance variation**: Relative distance measurements
- **Clothing**: Loose vs. tight clothing affects amplitude but not frequency

### Known Limitations
1. **Lateral orientation**: Algorithm assumes frontal or back view
2. **Occlusion**: Requires clear view of shoulder and hip landmarks
3. **Very shallow breathing**: May not detect breaths below noise floor
4. **Non-breathing movement**: Large body movements can create false peaks

## Testing and Validation

See `tests/test_signal.py` for unit tests covering:
- Keypoint extraction accuracy
- Filter frequency response
- Peak detection precision
- Event trigger conditions
- Edge case handling

## References

- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
- Butterworth Filter Design: scipy.signal documentation
- Respiratory Rate Monitoring: Clinical standards (12-20 breaths/min for adults)

---

*Last updated: 2025-10-16*
