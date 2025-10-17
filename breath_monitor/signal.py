"""
Signal processing for breathing rate detection.
Includes detrending, bandpass filtering, peak detection, and anomaly detection.
"""

import numpy as np
from scipy import signal
from collections import deque
from typing import List, Tuple, Optional, Dict


class RingBuffer:
    """Time-based ring buffer for signal data."""
    
    def __init__(self, window_seconds: float = 15.0, estimated_fps: float = 30.0):
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
        
    def add(self, timestamp: float, value: float):
        """Add a sample to the buffer."""
        self.timestamps.append(timestamp)
        self.values.append(value)
        self._trim_old_samples()
    
    def _trim_old_samples(self):
        """Remove samples older than window."""
        if len(self.timestamps) < 2:
            return
            
        cutoff_time = self.timestamps[-1] - self.window_seconds
        while len(self.timestamps) > 0 and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            self.values.popleft()
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current buffer data as numpy arrays."""
        return np.array(self.timestamps), np.array(self.values)
    
    def __len__(self) -> int:
        return len(self.timestamps)


def detrend_signal(signal_data: np.ndarray, method: str = "polynomial", order: int = 2) -> np.ndarray:
    """
    Remove trend from signal.
    
    Args:
        signal_data: Input signal
        method: "polynomial" or "highpass"
        order: Polynomial order for polynomial method
        
    Returns:
        Detrended signal
    """
    if len(signal_data) < 10:
        return signal_data
    
    if method == "polynomial":
        # Fit polynomial and subtract
        x = np.arange(len(signal_data))
        coeffs = np.polyfit(x, signal_data, order)
        trend = np.polyval(coeffs, x)
        return signal_data - trend
    elif method == "highpass":
        # Simple high-pass filter
        return signal.detrend(signal_data)
    else:
        return signal_data


def bandpass_filter(signal_data: np.ndarray, fs: float, lowcut: float = 0.5, 
                   highcut: float = 1.2, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if len(signal_data) < order * 3:
        return signal_data
    
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are in valid range
    low = max(0.01, min(0.99, low))
    high = max(0.01, min(0.99, high))
    
    if low >= high:
        return signal_data
    
    try:
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered = signal.sosfiltfilt(sos, signal_data)
        return filtered
    except Exception as e:
        print(f"[WARN] Bandpass filter failed: {e}")
        return signal_data


def detect_peaks(signal_data: np.ndarray, fs: float, refractory_sec: float = 0.6) -> np.ndarray:
    """
    Detect peaks in signal using scipy.signal.find_peaks.
    
    Args:
        signal_data: Input signal
        fs: Sampling frequency
        refractory_sec: Minimum time between peaks (seconds)
        
    Returns:
        Array of peak indices
    """
    if len(signal_data) < 10:
        return np.array([])
    
    distance = max(1, int(refractory_sec * fs))
    
    try:
        peaks, _ = signal.find_peaks(signal_data, distance=distance)
        return peaks
    except Exception as e:
        print(f"[WARN] Peak detection failed: {e}")
        return np.array([])


def calculate_bpm(peak_indices: np.ndarray, timestamps: np.ndarray) -> Optional[float]:
    """
    Calculate BPM from peak timestamps.
    
    Args:
        peak_indices: Indices of peaks
        timestamps: Full timestamp array
        
    Returns:
        BPM value or None if insufficient peaks
    """
    if len(peak_indices) < 2:
        return None
    
    # Get peak timestamps
    peak_times = timestamps[peak_indices]
    
    # Calculate inter-peak intervals
    intervals = np.diff(peak_times)
    
    if len(intervals) == 0:
        return None
    
    # Use median interval for robustness
    median_interval = np.median(intervals)
    
    if median_interval <= 0:
        return None
    
    # Convert to BPM (breaths per minute)
    bpm = 60.0 / median_interval
    return bpm


class BpmSmoother:
    """Exponential moving average smoother for BPM display."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive.
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: Optional[float]) -> Optional[float]:
        """
        Update and return smoothed value.
        
        Args:
            new_value: New BPM measurement
            
        Returns:
            Smoothed BPM value
        """
        if new_value is None:
            return self.value
        
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def reset(self):
        """Reset smoother."""
        self.value = None


class BreathingAnalyzer:
    """Complete breathing analysis pipeline."""
    
    def __init__(self, window_sec: float = 15.0, bpf_low: float = 0.5, bpf_high: float = 1.2,
                 apnea_sec: float = 20.0, tachy_threshold: float = 60.0, brady_threshold: float = 30.0):
        """
        Initialize breathing analyzer.
        
        Args:
            window_sec: Signal buffer window in seconds
            bpf_low: Bandpass filter low cutoff (Hz)
            bpf_high: Bandpass filter high cutoff (Hz)
            apnea_sec: Apnea detection threshold (seconds without peaks)
            tachy_threshold: Tachypnea threshold (BPM)
            brady_threshold: Bradypnea threshold (BPM)
        """
        self.buffer = RingBuffer(window_sec)
        self.bpf_low = bpf_low
        self.bpf_high = bpf_high
        self.apnea_sec = apnea_sec
        self.tachy_threshold = tachy_threshold
        self.brady_threshold = brady_threshold
        self.smoother = BpmSmoother(alpha=0.3)
        
        # Adaptive threshold for shallow breathing
        self.amplitude_history = deque(maxlen=100)
        
    def add_sample(self, timestamp: float, value: float):
        """Add a new sample to the buffer."""
        self.buffer.add(timestamp, value)
    
    def analyze(self) -> Dict:
        """
        Perform full breathing analysis.
        
        Returns:
            Dictionary with analysis results:
                - bpm: Current breathing rate
                - bpm_smooth: Smoothed BPM for display
                - apnea: Apnea detected
                - shallow: Shallow breathing detected
                - tachypnea: Tachypnea detected
                - bradypnea: Bradypnea detected
                - confidence: Analysis confidence
                - peak_count: Number of peaks detected
        """
        timestamps, values = self.buffer.get_data()
        
        if len(values) < 20:
            return {
                "bpm": None,
                "bpm_smooth": None,
                "apnea": False,
                "shallow": False,
                "tachypnea": False,
                "bradypnea": False,
                "confidence": 0.0,
                "peak_count": 0
            }
        
        # Calculate sampling frequency
        if len(timestamps) > 1:
            fs = 1.0 / np.median(np.diff(timestamps))
        else:
            fs = 30.0
        
        # Detrend
        detrended = detrend_signal(values, method="polynomial", order=2)
        
        # Bandpass filter
        filtered = bandpass_filter(detrended, fs, self.bpf_low, self.bpf_high)
        
        # Detect peaks
        peaks = detect_peaks(filtered, fs, refractory_sec=0.6)
        
        # Calculate BPM
        bpm = calculate_bpm(peaks, timestamps)
        bpm_smooth = self.smoother.update(bpm)
        
        # Check for apnea (no peaks in recent window)
        apnea = False
        if len(peaks) == 0:
            time_since_start = timestamps[-1] - timestamps[0]
            if time_since_start >= self.apnea_sec:
                apnea = True
        elif len(peaks) > 0:
            time_since_last_peak = timestamps[-1] - timestamps[peaks[-1]]
            if time_since_last_peak >= self.apnea_sec:
                apnea = True
        
        # Check for shallow breathing
        shallow = False
        if len(peaks) > 2:
            peak_values = filtered[peaks]
            amplitude = np.ptp(peak_values)  # Peak-to-peak
            self.amplitude_history.append(amplitude)
            
            if len(self.amplitude_history) >= 10:
                threshold = np.median(self.amplitude_history) * 0.5
                if amplitude < threshold:
                    shallow = True
        
        # Check for tachypnea/bradypnea
        tachypnea = bpm is not None and bpm > self.tachy_threshold
        bradypnea = bpm is not None and bpm < self.brady_threshold
        
        # Calculate confidence (based on signal quality and peak regularity)
        confidence = 0.0
        if len(peaks) >= 3:
            peak_times = timestamps[peaks]
            intervals = np.diff(peak_times)
            regularity = 1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-6))
            confidence = min(1.0, regularity * (len(peaks) / 10.0))
        
        return {
            "bpm": bpm,
            "bpm_smooth": bpm_smooth,
            "apnea": apnea,
            "shallow": shallow,
            "tachypnea": tachypnea,
            "bradypnea": bradypnea,
            "confidence": confidence,
            "peak_count": len(peaks)
        }


# Export key functions for unit testing
__all__ = [
    "RingBuffer",
    "detrend_signal",
    "bandpass_filter",
    "detect_peaks",
    "calculate_bpm",
    "BpmSmoother",
    "BreathingAnalyzer"
]

