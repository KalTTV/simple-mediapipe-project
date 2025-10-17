# Algorithm Overview

## Pipeline Diagram

```
┌──────────────┐
│   Camera     │
│   Capture    │
└──────┬───────┘
       │ RGB Frame + Timestamp
       ▼
┌──────────────┐
│  MediaPipe   │
│  Pose Model  │
└──────┬───────┘
       │ 33 Landmarks (image + world coords)
       ▼
┌──────────────┐
│  Extract     │
│ Mid-Shoulder │ ← Average of L/R shoulder landmarks
│  (Y coord)   │
└──────┬───────┘
       │ Raw signal (normalized Y position)
       ▼
┌──────────────┐
│ Time-based   │
│ Ring Buffer  │ ← 15-second sliding window
└──────┬───────┘
       │ Buffered signal + timestamps
       ▼
┌──────────────┐
│   Detrend    │
│ (Polynomial  │ ← Remove baseline drift
│  or HP)      │
└──────┬───────┘
       │ Detrended signal
       ▼
┌──────────────┐
│  Butterworth │
│  Bandpass    │ ← 0.5-1.2 Hz (30-72 BPM)
│  Filter      │
└──────┬───────┘
       │ Filtered signal
       ▼
┌──────────────┐
│    Peak      │
│  Detection   │ ← scipy.signal.find_peaks
└──────┬───────┘
       │ Peak indices + timestamps
       ▼
┌──────────────┐
│  Calculate   │
│     BPM      │ ← 60 / median(inter-peak intervals)
└──────┬───────┘
       │ Raw BPM
       ▼
┌──────────────┐
│     EMA      │
│  Smoothing   │ ← Exponential moving average (α=0.3)
└──────┬───────┘
       │ Smoothed BPM
       ▼
┌──────────────┐
│   Anomaly    │
│  Detection   │ ← Apnea, Shallow, Tachy/Brady
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│  Display + WebSocket Output  │
└──────────────────────────────┘
```

## Rationale for Shoulder Proxy

### Why Shoulders?

1. **Visibility**: Shoulders are consistently visible from most camera angles, unlike chest directly
2. **Motion**: Chest expansion/contraction causes vertical shoulder displacement
3. **Robustness**: Less affected by clothing than direct chest tracking
4. **Landmark Quality**: MediaPipe Pose provides high-confidence shoulder landmarks

### Signal Extraction

- **Primary**: Mid-shoulder Y coordinate (image space, normalized 0-1)
  - Upward chest expansion → shoulders rise (Y decreases in image coords)
  - Chest contraction → shoulders lower (Y increases)
  
- **Fallback**: World Z coordinate (depth)
  - For frontal views, Z motion may also capture breathing
  - Less reliable due to depth estimation noise

### Limitations

- **Occlusion**: Blankets covering shoulders will lose tracking
- **Motion**: Subject movement can contaminate signal (mitigated by detrending)
- **Camera Angle**: Best with slightly elevated camera viewing infant from above
- **Clothing**: Tight-fitting clothing provides better landmark stability

## Frequency Band Selection

### Infant Breathing Rate Range

- **Normal**: 30-60 breaths/minute (0.5-1.0 Hz)
- **Extended**: 20-70 breaths/minute (0.33-1.17 Hz)

### Chosen Bandpass: 0.5-1.2 Hz (30-72 BPM)

**Rationale**:
- Covers normal + tachypnea range
- Excludes cardiac signal (~1.5-2.5 Hz for infants)
- Removes low-frequency drift (<0.5 Hz)
- Configurable via CLI flags for different age groups

### Filter Design

- **Type**: Butterworth (4th order)
- **Implementation**: Zero-phase filtfilt (no delay)
- **Trade-off**: Narrow band for specificity vs. false negatives

## Peak Detection Strategy

### scipy.signal.find_peaks

**Parameters**:
- `distance`: Refractory period ≈ 0.6s (prevents double-counting)
- `height`: None (adaptive to signal amplitude)
- `prominence`: Implicit (handled by bandpass)

### BPM Calculation

- **Method**: Median of inter-peak intervals (robust to outliers)
- **Formula**: BPM = 60 / median(diff(peak_times))
- **Minimum Peaks**: 2 (returns None if insufficient)

### Display Smoothing

- **EMA**: α = 0.3 (balances responsiveness and stability)
- **Purpose**: Reduces display jitter while tracking changes

## Anomaly Detection

### Apnea

**Definition**: No peaks detected for ≥ apnea_sec (default 20s)

**Logic**:
```python
if len(peaks) == 0:
    time_since_start >= apnea_sec
elif len(peaks) > 0:
    time_since_last_peak >= apnea_sec
```

### Shallow Breathing

**Definition**: Peak-to-peak amplitude < adaptive threshold

**Logic**:
- Maintain rolling history of recent amplitudes (100 samples)
- Threshold = median(history) × 0.5
- Flag if current amplitude < threshold

### Tachypnea / Bradypnea

- **Tachypnea**: BPM > tachy_threshold (default 60)
- **Bradypnea**: BPM < brady_threshold (default 30)

## Confidence Estimation

**Factors**:
1. **Shoulder Visibility**: Average of L/R shoulder visibility scores
2. **Peak Regularity**: 1 - CV(inter-peak intervals)
3. **Peak Count**: Normalized by target (10 peaks)

**Formula**:
```python
regularity = 1 - std(intervals) / mean(intervals)
confidence = min(1.0, regularity × (peak_count / 10.0))
```

**Behavior**:
- Confidence < 0.5: Suspend BPM updates, display "--"
- High confidence: Green display
- Low confidence: Yellow/gray display

## Performance Characteristics

### Computational Cost

- **Pose Detection**: ~30-50ms per frame (CPU)
- **Signal Processing**: <1ms per frame
- **Total Pipeline**: ~33ms (30 FPS achievable)

### Accuracy

- **Synthetic Signals**: ±10% of ground truth (tested)
- **Real-World**: Depends on:
  - Camera quality
  - Lighting conditions
  - Subject movement
  - Clothing fit

### Latency

- **Detection**: 10-15 seconds for stable BPM (buffer fill time)
- **Updates**: ~1 second (EMA smoothing lag)
- **Apnea Alert**: apnea_sec + ~1 second

## Configurability

All thresholds exposed via CLI:

```bash
--min-sec 15          # Buffer window
--bpf-low 0.5         # Bandpass low cutoff
--bpf-high 1.2        # Bandpass high cutoff
--apnea-sec 20        # Apnea threshold
--tachy 60            # Tachypnea threshold
--brady 30            # Bradypnea threshold
```

Adapt for different age groups:
- **Newborns**: `--bpf-high 1.5 --tachy 70`
- **Toddlers**: `--bpf-low 0.4 --brady 25`

## Known Failure Modes

1. **False Positives**: Hand movements near face
2. **False Negatives**: Very shallow breathing with thick clothing
3. **Noise**: Subject rolling/moving significantly
4. **Occlusion**: Blanket covering shoulders
5. **Lighting**: Very low light reduces landmark confidence

## Future Improvements

- Multi-landmark fusion (hips, chest, shoulders)
- Adaptive bandpass based on detected frequency
- Motion artifact rejection via accelerometer-like filtering
- Thermal camera integration for ground truth

