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
        if timestamp is None:
            timestamp = time.time()
        self.buffer.append(value)
        self.timestamps.append(timestamp)

    def get_array(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        return np.array(self.buffer)

    def get_timestamps(self) -> np.ndarray:
        """Get timestamps as numpy array."""
        return np.array(self.timestamps)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.maxlen

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.timestamps.clear()

    def __len__(self):
        return len(self.buffer)


class BreathingSignalProcessor:
    """Process breathing signals with filtering, peak detection, and analysis."""

    def __init__(
        self,
        buffer_size: int = 300,
        sample_rate: float = 30.0,
        lowcut: float = 0.1,
        highcut: float = 0.5,
        filter_order: int = 4,
        apnea_threshold: float = 10.0,
        shallow_threshold: float = 0.3,
        window_seconds: float = 30.0,
    ):
        """Initialize breathing signal processor.

        Args:
            buffer_size: Maximum number of samples to store
            sample_rate: Sampling frequency in Hz
            lowcut: Low cutoff frequency for bandpass filter (Hz)
            highcut: High cutoff frequency for bandpass filter (Hz)
            filter_order: Order of Butterworth filter
            apnea_threshold: Seconds without breathing to detect apnea
            shallow_threshold: Minimum amplitude ratio for shallow breathing
            window_seconds: Time window for BPM calculation (seconds)
        """
        self.buffer = RingBuffer(buffer_size)
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.apnea_threshold = apnea_threshold
        self.shallow_threshold = shallow_threshold
        self.window_seconds = window_seconds

        # State tracking
        self.last_breath_time = time.time()
        self.apnea_active = False
        self.shallow_breathing = False

        # Peak detection history
        self.peak_times = deque(maxlen=50)
        self.peak_values = deque(maxlen=50)

        # Design Butterworth bandpass filter
        nyquist = self.sample_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.filter_b, self.filter_a = signal.butter(
            self.filter_order, [low, high], btype="band"
        )

    def add_sample(self, value: float, timestamp: Optional[float] = None):
        """Add a new sample to the buffer.

        Args:
            value: Breathing signal value
            timestamp: Optional timestamp (defaults to current time)
        """
        self.buffer.append(value, timestamp)

    def filter_signal(self) -> np.ndarray:
        """Apply Butterworth bandpass filter to buffer.

        Returns:
            Filtered signal as numpy array
        """
        if len(self.buffer) < 10:
            return self.buffer.get_array()

        raw_signal = self.buffer.get_array()
        try:
            filtered = signal.filtfilt(self.filter_b, self.filter_a, raw_signal)
            return filtered
        except Exception:
            return raw_signal

    def detect_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks (breaths) in the filtered signal.

        Returns:
            Tuple of (peak_indices, peak_values)
        """
        if len(self.buffer) < 10:
            return np.array([]), np.array([])

        filtered = self.filter_signal()

        # Dynamic threshold based on signal statistics
        mean_val = np.mean(filtered)
        std_val = np.std(filtered)
        threshold = mean_val + 0.5 * std_val

        # Find peaks
        peaks, properties = signal.find_peaks(
            filtered,
            height=threshold,
            distance=int(self.sample_rate * 1.5),  # Min 1.5s between breaths
        )

        if len(peaks) > 0:
            peak_values = filtered[peaks]
            timestamps = self.buffer.get_timestamps()
            if len(timestamps) > 0:
                peak_times = timestamps[peaks]
                self.peak_times.extend(peak_times)
                self.peak_values.extend(peak_values)

        return peaks, filtered[peaks] if len(peaks) > 0 else np.array([])

    def calculate_bpm(self, window_seconds: float = None) -> Optional[float]:
        """Calculate breaths per minute from recent peaks.

        Args:
            window_seconds: Time window to consider (defaults to self.window_seconds)

        Returns:
            BPM value or None if insufficient data
        """
        if window_seconds is None:
            window_seconds = self.window_seconds

        if len(self.peak_times) < 2:
            return None

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Get peaks within time window
        recent_peaks = [t for t in self.peak_times if t >= cutoff_time]

        if len(recent_peaks) < 2:
            return None

        # Calculate average time between peaks
        intervals = np.diff(recent_peaks)
        mean_interval = np.mean(intervals)

        if mean_interval <= 0:
            return None

        # Convert to breaths per minute
        bpm = 60.0 / mean_interval

        # Sanity check
        if bpm < 5 or bpm > 60:
            return None

        return bpm

    def check_apnea(self) -> bool:
        """Check if apnea (no breathing) is detected.

        Returns:
            True if apnea detected
        """
        if len(self.peak_times) == 0:
            return False

        current_time = time.time()
        time_since_last_peak = current_time - self.peak_times[-1]

        self.apnea_active = time_since_last_peak > self.apnea_threshold
        return self.apnea_active

    def check_shallow_breathing(self) -> bool:
        """Check if breathing is shallow.

        Returns:
            True if shallow breathing detected
        """
        if len(self.peak_values) < 3:
            return False

        recent_peaks = list(self.peak_values)[-5:]
        mean_amplitude = np.mean(np.abs(recent_peaks))
        max_amplitude = np.max(np.abs(list(self.peak_values)))

        if max_amplitude == 0:
            return False

        amplitude_ratio = mean_amplitude / max_amplitude
        self.shallow_breathing = amplitude_ratio < self.shallow_threshold
        return self.shallow_breathing

    def get_metrics(self) -> dict:
        """Get current breathing metrics.

        Returns:
            Dictionary with BPM, apnea status, and signal quality
        """
        self.detect_peaks()
        bpm = self.calculate_bpm()
        apnea = self.check_apnea()
        shallow = self.check_shallow_breathing()

        # Calculate signal quality
        if len(self.buffer) > 10:
            filtered = self.filter_signal()
            snr = np.std(filtered) / (np.mean(np.abs(np.diff(filtered))) + 1e-6)
            signal_quality = min(1.0, snr / 10.0)
        else:
            signal_quality = 0.0

        return {
            "bpm": bpm,
            "apnea": apnea,
            "shallow": shallow,
            "signal_quality": signal_quality,
            "buffer_size": len(self.buffer),
            "buffer_full": self.buffer.is_full(),
        }

    def process(self) -> dict:
        """Process signal and return breathing metrics (test-compatible interface).

        Returns:
            Dictionary with 'breathing_rate', 'amplitude', and 'quality'
        """
        metrics = self.get_metrics()
        peaks, peak_vals = self.detect_peaks()

        # Calculate mean amplitude from recent peaks
        if len(peak_vals) > 0:
            amplitude = float(np.mean(np.abs(peak_vals)))
        else:
            amplitude = 0.0

        return {
            "breathing_rate": metrics["bpm"],
            "amplitude": amplitude,
            "quality": metrics["signal_quality"],
        }

    def reset(self):
        """Reset processor state."""
        self.buffer.clear()
        self.last_breath_time = time.time()
        self.apnea_active = False
        self.shallow_breathing = False
        self.peak_times.clear()
        self.peak_values.clear()


if __name__ == "__main__":
    # Test the signal processor
    print("Testing BreathingSignalProcessor...")

    processor = BreathingSignalProcessor(
        buffer_size=300, sample_rate=30.0, apnea_threshold=5.0
    )

    # Simulate breathing signal (sine wave)
    t = np.linspace(0, 10, 300)  # 10 seconds at 30 Hz
    breathing_rate = 0.25  # 15 BPM (0.25 Hz)
    signal_sim = np.sin(2 * np.pi * breathing_rate * t) + np.random.normal(
        0, 0.1, 300
    )

    # Add samples
    for i, value in enumerate(signal_sim):
        timestamp = time.time() + i / 30.0  # Simulate 30 Hz
        processor.add_sample(value, timestamp)

    # Get metrics
    metrics = processor.get_metrics()
    print(f"\nMetrics:")
    print(
        f"  BPM: {metrics['bpm']:.1f}" if metrics["bpm"] else "  BPM: N/A"
    )
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
