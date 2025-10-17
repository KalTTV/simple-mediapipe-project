"""Signal processing module for breathing analysis.

Provides:
- Ring buffer for signal buffering
- Butterworth filtering for noise reduction
- Peak detection for breathing cycles
- BPM (breaths per minute) calculation
- Apnea and shallow breathing detection
"""

import numpy as np
from scipy import signal
from collections import deque
from typing import Optional, Tuple, List
import time


class RingBuffer:
    """Circular buffer for efficient signal storage."""
    
    def __init__(self, maxlen: int):
        """Initialize ring buffer.
        
        Args:
            maxlen: Maximum buffer size
        """
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.timestamps = deque(maxlen=maxlen)
    
    def append(self, value: float, timestamp: Optional[float] = None):
        """Add value to buffer.
        
        Args:
            value: Signal value to append
            timestamp: Optional timestamp (defaults to current time)
        """
        self.buffer.append(value)
        if timestamp is None:
            timestamp = time.time()
        self.timestamps.append(timestamp)
    
    def get_array(self) -> np.ndarray:
        """Get buffer contents as numpy array.
        
        Returns:
            Numpy array of buffered values
        """
        return np.array(self.buffer)
    
    def get_timestamps(self) -> np.ndarray:
        """Get timestamps as numpy array.
        
        Returns:
            Numpy array of timestamps
        """
        return np.array(self.timestamps)
    
    def __len__(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.maxlen
    
    def clear(self):
        """Clear buffer contents."""
        self.buffer.clear()
        self.timestamps.clear()


class BreathingSignalProcessor:
    """Process breathing signals to extract respiratory metrics."""
    
    def __init__(
        self,
        buffer_size: int = 300,
        sample_rate: float = 30.0,
        lowcut: float = 0.1,
        highcut: float = 0.5,
        filter_order: int = 4,
        apnea_threshold: float = 10.0,
        shallow_threshold: float = 0.3
    ):
        """Initialize breathing signal processor.
        
        Args:
            buffer_size: Size of ring buffer (number of samples)
            sample_rate: Expected sample rate in Hz
            lowcut: Low cutoff frequency for bandpass filter (Hz)
            highcut: High cutoff frequency for bandpass filter (Hz)
            filter_order: Order of Butterworth filter
            apnea_threshold: Time threshold for apnea detection (seconds)
            shallow_threshold: Amplitude threshold for shallow breathing (relative)
        """
        self.buffer = RingBuffer(buffer_size)
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.apnea_threshold = apnea_threshold
        self.shallow_threshold = shallow_threshold
        
        # Design Butterworth filter
        self._design_filter()
        
        # Peak detection state
        self.last_peak_time = None
        self.peak_times = deque(maxlen=20)  # Store recent peak times for BPM
        self.peak_values = deque(maxlen=20)
        
        # Apnea detection state
        self.last_breath_time = time.time()
        self.apnea_active = False
        self.shallow_breathing = False
    
    def _design_filter(self):
        """Design Butterworth bandpass filter."""
        nyquist = 0.5 * self.sample_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Ensure frequencies are in valid range (0, 1)
        low = max(0.001, min(0.999, low))
        high = max(0.001, min(0.999, high))
        
        if low >= high:
            # Fallback to lowpass if range is invalid
            self.sos = signal.butter(
                self.filter_order,
                high,
                btype='low',
                output='sos'
            )
        else:
            self.sos = signal.butter(
                self.filter_order,
                [low, high],
                btype='band',
                output='sos'
            )
    
    def add_sample(self, value: float, timestamp: Optional[float] = None):
        """Add a new sample to the buffer.
        
        Args:
            value: Chest keypoint position (e.g., y-coordinate)
            timestamp: Optional timestamp (defaults to current time)
        """
        self.buffer.append(value, timestamp)
    
    def filter_signal(self, signal_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Butterworth filter to signal.
        
        Args:
            signal_data: Optional signal array (defaults to buffer contents)
        
        Returns:
            Filtered signal as numpy array
        """
        if signal_data is None:
            signal_data = self.buffer.get_array()
        
        if len(signal_data) < self.filter_order * 3:
            # Not enough data for filtering, return original
            return signal_data
        
        # Apply zero-phase filtering (forward-backward)
        try:
            filtered = signal.sosfiltfilt(self.sos, signal_data)
            return filtered
        except Exception:
            # Return original if filtering fails
            return signal_data
    
    def detect_peaks(
        self,
        signal_data: Optional[np.ndarray] = None,
        min_distance: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks in breathing signal.
        
        Args:
            signal_data: Optional signal array (defaults to filtered buffer)
            min_distance: Minimum distance between peaks (samples)
        
        Returns:
            Tuple of (peak_indices, peak_values)
        """
        if signal_data is None:
            signal_data = self.filter_signal()
        
        if len(signal_data) < min_distance * 2:
            return np.array([]), np.array([])
        
        # Find peaks (inhalations - maximum chest expansion)
        peaks, properties = signal.find_peaks(
            signal_data,
            distance=min_distance,
            prominence=np.std(signal_data) * 0.3  # Adaptive threshold
        )
        
        if len(peaks) > 0:
            peak_values = signal_data[peaks]
            return peaks, peak_values
        
        return np.array([]), np.array([])
    
    def calculate_bpm(self, window_seconds: float = 30.0) -> Optional[float]:
        """Calculate breathing rate in breaths per minute.
        
        Args:
            window_seconds: Time window for BPM calculation (seconds)
        
        Returns:
            BPM value or None if insufficient data
        """
        if len(self.buffer) < 2:
            return None
        
        # Get filtered signal and detect peaks
        filtered = self.filter_signal()
        peaks, _ = self.detect_peaks(filtered)
        
        if len(peaks) < 2:
            return None
        
        # Get timestamps for peaks
        timestamps = self.buffer.get_timestamps()
        peak_timestamps = timestamps[peaks]
        
        # Filter peaks within time window
        current_time = time.time()
        recent_peaks = peak_timestamps[peak_timestamps >= (current_time - window_seconds)]
        
        if len(recent_peaks) < 2:
            # Use all available peaks if not enough in window
            recent_peaks = peak_timestamps
        
        if len(recent_peaks) < 2:
            return None
        
        # Calculate average time between breaths
        breath_intervals = np.diff(recent_peaks)
        avg_interval = np.mean(breath_intervals)
        
        if avg_interval <= 0:
            return None
        
        # Convert to BPM
        bpm = 60.0 / avg_interval
        
        # Store peak time for apnea detection
        if len(peaks) > 0:
            self.last_breath_time = peak_timestamps[-1]
        
        # Clamp to reasonable range (2-60 BPM)
        bpm = max(2.0, min(60.0, bpm))
        
        return bpm
    
    def detect_apnea(self) -> bool:
        """Detect if apnea (absence of breathing) is occurring.
        
        Returns:
            True if apnea detected, False otherwise
        """
        current_time = time.time()
        time_since_breath = current_time - self.last_breath_time
        
        # Apnea detected if no breath for threshold duration
        self.apnea_active = time_since_breath > self.apnea_threshold
        
        return self.apnea_active
    
    def detect_shallow_breathing(self) -> bool:
        """Detect shallow breathing based on signal amplitude.
        
        Returns:
            True if shallow breathing detected, False otherwise
        """
        if len(self.buffer) < 30:
            return False
        
        filtered = self.filter_signal()
        
        # Calculate signal amplitude (peak-to-peak)
        signal_range = np.ptp(filtered[-30:])  # Last 30 samples (~1 second)
        
        # Compare to overall signal range
        overall_range = np.ptp(filtered) if len(filtered) > 60 else signal_range
        
        if overall_range == 0:
            return False
        
        relative_amplitude = signal_range / overall_range
        
        self.shallow_breathing = relative_amplitude < self.shallow_threshold
        
        return self.shallow_breathing
    
    def get_metrics(self) -> dict:
        """Get all breathing metrics.
        
        Returns:
            Dictionary containing:
                - bpm: Breaths per minute
                - apnea: Boolean indicating apnea
                - shallow: Boolean indicating shallow breathing
                - signal_quality: Quality metric (0-1)
        """
        bpm = self.calculate_bpm()
        apnea = self.detect_apnea()
        shallow = self.detect_shallow_breathing()
        
        # Calculate signal quality based on buffer fullness and variance
        signal_quality = 0.0
        if len(self.buffer) > 0:
            fullness = len(self.buffer) / self.buffer.maxlen
            signal_data = self.buffer.get_array()
            variance = np.var(signal_data) if len(signal_data) > 1 else 0
            # Normalize variance (assume reasonable range)
            variance_score = min(1.0, variance * 100)
            signal_quality = (fullness + variance_score) / 2
        
        return {
            'bpm': bpm,
            'apnea': apnea,
            'shallow': shallow,
            'signal_quality': signal_quality,
            'buffer_size': len(self.buffer),
            'buffer_full': self.buffer.is_full()
        }
    
    def reset(self):
        """Reset processor state."""
        self.buffer.clear()
        self.last_breath_time = time.time()
        self.apnea_active = False
        self.shallow_breathing = False
        self.peak_times.clear()
        self.peak_values.clear()


if __name__ == '__main__':
    # Test the signal processor
    print("Testing BreathingSignalProcessor...")
    
    processor = BreathingSignalProcessor(
        buffer_size=300,
        sample_rate=30.0,
        apnea_threshold=5.0
    )
    
    # Simulate breathing signal (sine wave)
    t = np.linspace(0, 10, 300)  # 10 seconds at 30 Hz
    breathing_rate = 0.25  # 15 BPM (0.25 Hz)
    signal_sim = np.sin(2 * np.pi * breathing_rate * t) + np.random.normal(0, 0.1, 300)
    
    # Add samples
    for i, value in enumerate(signal_sim):
        timestamp = time.time() + i / 30.0  # Simulate 30 Hz
        processor.add_sample(value, timestamp)
    
    # Get metrics
    metrics = processor.get_metrics()
    print(f"\nMetrics:")
    print(f"  BPM: {metrics['bpm']:.1f}" if metrics['bpm'] else "  BPM: N/A")
    print(f"  Apnea: {metrics['apnea']}")
    print(f"  Shallow: {metrics['shallow']}")
    print(f"  Signal Quality: {metrics['signal_quality']:.2f}")
    print(f"  Buffer: {metrics['buffer_size']}/{processor.buffer.maxlen}")
    
    # Test filtering
    filtered = processor.filter_signal()
    print(f"\nFiltered signal shape: {filtered.shape}")
    
    # Test peak detection
    peaks, peak_vals = processor.detect_peaks()
    print(f"Detected {len(peaks)} peaks")
    
    print("\n✓ Signal processor test complete!")
