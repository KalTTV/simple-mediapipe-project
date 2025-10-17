"""
Signal processing for breathing rate detection.
Chest expansion detection with hysteresis breath counting (no double counts).
"""

import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from collections import deque
from typing import List, Tuple, Optional, Dict


class RingBuffer:
    """Time-based ring buffer for signal and tracking data."""
    
    def __init__(self, window_seconds: float = 30.0, estimated_fps: float = 30.0):
        """
        Initialize ring buffer.
        
        Args:
            window_seconds: Time window to keep in seconds
            estimated_fps: Estimated sampling rate for buffer sizing
        """
        self.window_seconds = window_seconds
        max_size = int(window_seconds * estimated_fps * 1.5)  # 50% margin
        self.timestamps = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.tracks_history = deque(maxlen=max_size)  # Multi-track data
        
    def add(self, timestamp: float, value: float, tracks_y: Optional[np.ndarray] = None):
        """Add a sample to the buffer."""
        self.timestamps.append(timestamp)
        self.values.append(value)
        self.tracks_history.append(tracks_y if tracks_y is not None else np.array([]))
        self._trim_old_samples()
    
    def _trim_old_samples(self):
        """Remove samples older than window."""
        if len(self.timestamps) < 2:
            return
            
        cutoff_time = self.timestamps[-1] - self.window_seconds
        while len(self.timestamps) > 0 and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            self.values.popleft()
            self.tracks_history.popleft()
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, List]:
        """Get current buffer data."""
        return np.array(self.timestamps), np.array(self.values), list(self.tracks_history)
    
    def __len__(self) -> int:
        return len(self.timestamps)


def estimate_fs(timestamps):
    """Estimate sampling frequency from timestamps."""
    if len(timestamps) < 3:
        return 0.0
    dt = np.median(np.diff(np.asarray(timestamps, float)))
    return float(1.0 / dt) if dt > 0 else 0.0


def build_expansion_trace(tracks_y_list: List[np.ndarray]) -> np.ndarray:
    """
    Build chest expansion trace from multi-track vertical displacements.
    
    Args:
        tracks_y_list: List of per-frame vertical displacement arrays
        
    Returns:
        Expansion signal (positive = inhalation)
    """
    if len(tracks_y_list) == 0:
        return np.array([])
    
    # Filter out empty tracks
    valid_tracks = [t for t in tracks_y_list if len(t) > 0]
    
    if len(valid_tracks) == 0:
        return np.array([])
    
    # Find minimum track count
    min_k = min(len(t) for t in valid_tracks)
    
    if min_k == 0:
        return np.array([])
    
    # Build matrix: (N_frames, k)
    tracks_matrix = np.array([t[:min_k] for t in valid_tracks])
    
    # Remove outliers per frame using MAD
    expansion = np.zeros(tracks_matrix.shape[0])
    for i in range(tracks_matrix.shape[0]):
        frame_tracks = tracks_matrix[i, :]
        median_val = np.median(frame_tracks)
        mad = np.median(np.abs(frame_tracks - median_val))
        
        if mad > 1e-6:
            # Keep inliers
            inliers = np.abs(frame_tracks - median_val) < 3 * mad
            if np.sum(inliers) > 0:
                expansion[i] = np.median(frame_tracks[inliers])
            else:
                expansion[i] = median_val
        else:
            expansion[i] = median_val
    
    # Negate so inhalation (upward motion) = positive expansion
    return -expansion


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove trend using Savitzky-Golay filter."""
    if len(x) < 11:
        return x
    
    try:
        window_length = min(51, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if window_length < 5:
            return signal.detrend(x)
        trend = savgol_filter(x, window_length=window_length, polyorder=2)
        return x - trend
    except:
        return signal.detrend(x)


def bandpass(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    if len(x) < order * 3:
        return x
    
    nyquist = fs / 2
    low = max(0.01, min(0.99, fmin / nyquist))
    high = max(0.01, min(0.99, fmax / nyquist))
    
    if low >= high:
        return x
    
    try:
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, x)
    except:
        return x


def detect_breaths_hysteresis(expansion: np.ndarray, timestamps: np.ndarray, 
                              min_ibi_sec: float = 2.0) -> Tuple[np.ndarray, List[float]]:
    """
    Detect breath events using hysteresis (Schmitt trigger) to prevent double counts.
    
    Args:
        expansion: Chest expansion signal (positive = inhalation)
        timestamps: Corresponding timestamps
        min_ibi_sec: Minimum inter-breath interval (refractory/debounce period)
        
    Returns:
        (breath_indices, ibi_list)
    """
    if len(expansion) < 10:
        return np.array([]), []
    
    # Compute rolling IQR for adaptive threshold
    window_samples = min(len(expansion), 200)  # ~10-20s at 20 Hz
    rolling_iqr = []
    
    for i in range(len(expansion)):
        start = max(0, i - window_samples)
        window = expansion[start:i+1]
        if len(window) > 5:
            iqr = np.percentile(window, 75) - np.percentile(window, 25)
            rolling_iqr.append(iqr)
        else:
            rolling_iqr.append(0.1)
    
    rolling_iqr = np.array(rolling_iqr)
    
    # Hysteresis thresholds
    theta = 0.25 * rolling_iqr  # Adaptive threshold
    
    # Schmitt trigger state machine
    breath_indices = []
    last_breath_time = -999.0  # Initialize far in past
    state = "IDLE"  # States: IDLE, RISING, FALLING
    
    for i in range(1, len(expansion)):
        t = timestamps[i]
        
        # Debounce: skip if too soon after last breath
        if (t - last_breath_time) < min_ibi_sec:
            continue
        
        e = expansion[i]
        thresh = theta[i]
        
        if state == "IDLE":
            if e > thresh:
                state = "RISING"
        elif state == "RISING":
            # Peak detected, now wait for fall
            if e < -thresh:
                # Breath event: crossed from +theta to -theta
                breath_indices.append(i)
                last_breath_time = t
                state = "IDLE"
    
    breath_indices = np.array(breath_indices)
    
    # Calculate IBIs
    ibi_list = []
    if len(breath_indices) > 1:
        breath_times = timestamps[breath_indices]
        ibi_list = list(np.diff(breath_times))
    
    return breath_indices, ibi_list


def calculate_bpm_from_ibi(ibi_list: List[float], smoothing_alpha: float = 0.3) -> Optional[float]:
    """Calculate BPM from inter-breath intervals."""
    if len(ibi_list) < 2:
        return None
    
    # Use median for robustness
    median_ibi = np.median(ibi_list)
    
    if median_ibi <= 0:
        return None
    
    bpm = 60.0 / median_ibi
    return bpm


def estimate_confidence(expansion: np.ndarray, fs: float, visibility: float,
                       tracks_alive: int, total_tracks: int) -> Tuple[float, float]:
    """
    Estimate confidence and SNR.
    
    Returns:
        (confidence, snr_db)
    """
    if len(expansion) < 50:
        return 0.0, 0.0
    
    # Tracker survival
    tracker_score = tracks_alive / max(1, total_tracks) if total_tracks > 0 else 0.0
    
    # Spectral SNR using Welch
    try:
        nperseg = min(256, max(64, len(expansion) // 4))
        freqs, psd = signal.welch(expansion, fs=fs, nperseg=nperseg)
        
        # Find peak in respiratory band (0.08-1.2 Hz)
        mask = (freqs >= 0.08) & (freqs <= 1.2)
        if np.any(mask):
            psd_band = psd[mask]
            peak_power = np.max(psd_band)
            noise_floor = np.median(psd_band)
            snr_db = 10 * np.log10(peak_power / (noise_floor + 1e-9))
        else:
            snr_db = 0.0
    except:
        snr_db = 0.0
    
    # Combine factors
    snr_score = min(1.0, max(0.0, snr_db / 20.0))
    conf = min(visibility, (tracker_score + snr_score) / 2.0)
    
    return conf, snr_db


class BpmSmoother:
    """Exponential moving average smoother for BPM display."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: Optional[float]) -> Optional[float]:
        if new_value is None:
            return self.value
        
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def reset(self):
        self.value = None


class BreathingAnalyzer:
    """Breathing analysis with chest expansion detection and debouncing."""
    
    def __init__(self, window_sec: float = 30.0, resp_low: float = 0.08, resp_high: float = 1.2,
                 min_ibi_sec: float = 2.0, apnea_sec: float = 20.0):
        """
        Initialize breathing analyzer.
        
        Args:
            window_sec: Signal buffer window in seconds
            resp_low: Low cutoff frequency (Hz)
            resp_high: High cutoff frequency (Hz)
            min_ibi_sec: Minimum inter-breath interval (debounce)
            apnea_sec: Apnea detection threshold (seconds)
        """
        self.buffer = RingBuffer(window_sec)
        self.resp_low = resp_low
        self.resp_high = resp_high
        self.min_ibi_sec = min_ibi_sec
        self.apnea_sec = apnea_sec
        self.smoother = BpmSmoother(alpha=0.3)
        
        # Track last breath time for apnea detection
        self.last_breath_time = None
        
    def add_sample(self, timestamp: float, value: float, tracks_y: Optional[np.ndarray] = None):
        """Add a new sample to the buffer."""
        self.buffer.add(timestamp, value, tracks_y)
    
    def analyze(self, visibility: float = 1.0, tracks_alive: int = 0, 
                total_tracks: int = 32) -> Dict:
        """
        Perform breathing analysis with chest expansion detection.
        
        Args:
            visibility: Current pose visibility score
            tracks_alive: Number of active tracking points
            total_tracks: Total number of trackers
        
        Returns:
            Dictionary with analysis results
        """
        timestamps, values, tracks_y_list = self.buffer.get_data()
        
        if len(timestamps) < 20:
            return {
                "bpm": None,
                "bpm_smooth": None,
                "conf": 0.0,
                "confidence": 0.0,
                "snr_db": 0.0,
                "tracks_alive": tracks_alive,
                "ibi_median": 0.0,
                "apnea": False,
                "shallow": False,
                "breath_count": 0
            }
        
        # Estimate FPS
        fs = estimate_fs(timestamps)
        if fs < 5.0:
            return {
                "bpm": None,
                "bpm_smooth": None,
                "conf": 0.0,
                "confidence": 0.0,
                "snr_db": 0.0,
                "tracks_alive": tracks_alive,
                "ibi_median": 0.0,
                "apnea": False,
                "shallow": False,
                "breath_count": 0
            }
        
        # Build expansion trace from multi-track data
        if len(tracks_y_list) > 0 and any(len(t) > 0 for t in tracks_y_list):
            expansion = build_expansion_trace(tracks_y_list)
        else:
            # Fallback to single-point signal
            expansion = -np.array(values)  # Negate for inhalation = positive
        
        if len(expansion) < 20:
            return {
                "bpm": None,
                "bpm_smooth": None,
                "conf": 0.0,
                "confidence": 0.0,
                "snr_db": 0.0,
                "tracks_alive": tracks_alive,
                "ibi_median": 0.0,
                "apnea": False,
                "shallow": False,
                "breath_count": 0
            }
        
        # Detrend
        detrended = detrend(expansion)
        
        # Bandpass filter (wide band - no clamping)
        filtered = bandpass(detrended, fs, self.resp_low, self.resp_high)
        
        # Detect breaths with hysteresis (prevents double counts)
        breath_indices, ibi_list = detect_breaths_hysteresis(
            filtered, 
            timestamps[:len(filtered)], 
            self.min_ibi_sec
        )
        
        # Calculate BPM from IBIs
        bpm = calculate_bpm_from_ibi(ibi_list)
        bpm_smooth = self.smoother.update(bpm)
        
        # Update last breath time
        if len(breath_indices) > 0:
            self.last_breath_time = timestamps[breath_indices[-1]]
        
        # Apnea detection
        apnea = False
        if self.last_breath_time is not None:
            time_since_last = timestamps[-1] - self.last_breath_time
            if time_since_last >= self.apnea_sec:
                apnea = True
        elif len(timestamps) > 0:
            time_span = timestamps[-1] - timestamps[0]
            if time_span >= self.apnea_sec and len(breath_indices) == 0:
                apnea = True
        
        # Shallow breathing detection
        shallow = False
        if len(filtered) > 10:
            amplitude = np.std(filtered)
            if amplitude < 0.02:  # Very low amplitude
                shallow = True
        
        # Confidence and SNR
        conf, snr_db = estimate_confidence(filtered, fs, visibility, tracks_alive, total_tracks)
        
        # Median IBI
        ibi_median = np.median(ibi_list) if len(ibi_list) > 0 else 0.0
        
        return {
            "bpm": bpm,
            "bpm_smooth": bpm_smooth,
            "conf": conf,
            "confidence": conf,  # Alias for compatibility
            "snr_db": snr_db,
            "tracks_alive": tracks_alive,
            "ibi_median": ibi_median,
            "apnea": apnea,
            "shallow": shallow,
            "breath_count": len(breath_indices)
        }


# Export key functions
__all__ = [
    "RingBuffer",
    "estimate_fs",
    "build_expansion_trace",
    "detrend",
    "bandpass",
    "detect_breaths_hysteresis",
    "calculate_bpm_from_ibi",
    "estimate_confidence",
    "BpmSmoother",
    "BreathingAnalyzer"
]
