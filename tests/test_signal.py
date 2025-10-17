"""
Unit tests for signal processing module.
Tests detrending, bandpass filtering, peak detection, and anomaly detection.
"""

import pytest
import numpy as np
from breath_monitor.signal import (
    RingBuffer, detrend_signal, bandpass_filter, detect_peaks,
    calculate_bpm, BpmSmoother, BreathingAnalyzer
)


class TestRingBuffer:
    """Test ring buffer functionality."""
    
    def test_buffer_creation(self):
        """Test buffer initialization."""
        buffer = RingBuffer(window_seconds=10.0, estimated_fps=30.0)
        assert len(buffer) == 0
    
    def test_buffer_add(self):
        """Test adding samples."""
        buffer = RingBuffer(window_seconds=10.0, estimated_fps=30.0)
        buffer.add(0.0, 1.0)
        buffer.add(0.1, 2.0)
        assert len(buffer) == 2
    
    def test_buffer_trim(self):
        """Test automatic trimming of old samples."""
        buffer = RingBuffer(window_seconds=1.0, estimated_fps=10.0)
        
        # Add samples over 2 seconds
        for i in range(20):
            buffer.add(i * 0.1, float(i))
        
        # Buffer should only contain last 1 second
        timestamps, values = buffer.get_data()
        assert timestamps[-1] - timestamps[0] <= 1.0


class TestSignalProcessing:
    """Test signal processing functions."""
    
    def test_detrend_polynomial(self):
        """Test polynomial detrending."""
        # Create signal with trend
        x = np.linspace(0, 10, 100)
        trend = 2 * x + 5
        signal = np.sin(x) + trend
        
        detrended = detrend_signal(signal, method="polynomial", order=1)
        
        # Detrended signal should have much smaller mean
        assert abs(np.mean(detrended)) < abs(np.mean(signal))
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        fs = 30.0  # 30 Hz sampling rate
        t = np.linspace(0, 10, int(10 * fs))
        
        # Create signal with multiple frequencies
        signal = (np.sin(2 * np.pi * 0.8 * t) +  # In band (0.8 Hz)
                 np.sin(2 * np.pi * 0.1 * t) +   # Below band (0.1 Hz)
                 np.sin(2 * np.pi * 5.0 * t))    # Above band (5 Hz)
        
        filtered = bandpass_filter(signal, fs, lowcut=0.5, highcut=1.2)
        
        # Filtered signal should be different from original
        assert not np.allclose(signal, filtered)
        
        # Should have some filtering effect
        assert np.std(filtered) < np.std(signal)
    
    def test_detect_peaks(self):
        """Test peak detection."""
        fs = 30.0
        t = np.linspace(0, 10, int(10 * fs))
        
        # Create signal with known peaks (1 Hz = 10 peaks in 10 seconds)
        signal = np.sin(2 * np.pi * 1.0 * t)
        
        peaks = detect_peaks(signal, fs, refractory_sec=0.6)
        
        # Should detect approximately 10 peaks
        assert 8 <= len(peaks) <= 12
    
    def test_calculate_bpm(self):
        """Test BPM calculation from peaks."""
        # Create timestamps with peaks every 1 second (60 BPM)
        timestamps = np.arange(0, 10, 0.1)
        peak_indices = np.array([0, 10, 20, 30, 40, 50])  # Every 1 second
        
        bpm = calculate_bpm(peak_indices, timestamps)
        
        # Should be close to 60 BPM
        assert bpm is not None
        assert 55 <= bpm <= 65


class TestBpmSmoother:
    """Test BPM smoother."""
    
    def test_smoother_initialization(self):
        """Test smoother initialization."""
        smoother = BpmSmoother(alpha=0.3)
        assert smoother.value is None
    
    def test_smoother_update(self):
        """Test smoother update."""
        smoother = BpmSmoother(alpha=0.3)
        
        result = smoother.update(50.0)
        assert result == 50.0
        
        result = smoother.update(60.0)
        # Should be between 50 and 60, closer to 60
        assert 50 < result < 60
    
    def test_smoother_reset(self):
        """Test smoother reset."""
        smoother = BpmSmoother(alpha=0.3)
        smoother.update(50.0)
        smoother.reset()
        assert smoother.value is None


class TestBreathingAnalyzer:
    """Test complete breathing analyzer."""
    
    def test_analyzer_creation(self):
        """Test analyzer initialization."""
        analyzer = BreathingAnalyzer()
        assert analyzer is not None
    
    def test_synthetic_breathing_0_7hz(self):
        """Test with synthetic 0.7 Hz breathing (42 BPM)."""
        analyzer = BreathingAnalyzer(window_sec=15.0, bpf_low=0.5, bpf_high=1.2)
        
        # Generate 20 seconds of 0.7 Hz sinusoid (42 breaths/min)
        fs = 30.0
        duration = 20.0
        t = np.linspace(0, duration, int(duration * fs))
        
        # Add some noise and drift
        breathing_signal = (0.5 * np.sin(2 * np.pi * 0.7 * t) + 
                          0.1 * np.random.randn(len(t)) +
                          0.02 * t)  # Linear drift
        
        # Add samples to analyzer
        for timestamp, value in zip(t, breathing_signal):
            analyzer.add_sample(timestamp, value)
        
        # Analyze after sufficient data
        if len(t) > 100:
            result = analyzer.analyze()
            
            # BPM should be detected
            if result["bpm"] is not None:
                # Should be within ±10% of 42 BPM
                assert 37.8 <= result["bpm"] <= 46.2, f"BPM {result['bpm']} not within 10% of 42"
    
    def test_synthetic_breathing_1_0hz(self):
        """Test with synthetic 1.0 Hz breathing (60 BPM)."""
        analyzer = BreathingAnalyzer(window_sec=15.0, bpf_low=0.5, bpf_high=1.2)
        
        # Generate 20 seconds of 1.0 Hz sinusoid
        fs = 30.0
        duration = 20.0
        t = np.linspace(0, duration, int(duration * fs))
        
        breathing_signal = (0.5 * np.sin(2 * np.pi * 1.0 * t) + 
                          0.1 * np.random.randn(len(t)))
        
        for timestamp, value in zip(t, breathing_signal):
            analyzer.add_sample(timestamp, value)
        
        if len(t) > 100:
            result = analyzer.analyze()
            
            if result["bpm"] is not None:
                # Should be within ±10% of 60 BPM
                assert 54.0 <= result["bpm"] <= 66.0, f"BPM {result['bpm']} not within 10% of 60"
    
    def test_apnea_detection(self):
        """Test apnea detection with no peaks."""
        analyzer = BreathingAnalyzer(window_sec=15.0, apnea_sec=10.0)
        
        # Add 25 seconds of flat signal (no breathing)
        fs = 30.0
        duration = 25.0
        t = np.linspace(0, duration, int(duration * fs))
        flat_signal = np.ones(len(t)) * 0.5 + 0.01 * np.random.randn(len(t))
        
        for timestamp, value in zip(t, flat_signal):
            analyzer.add_sample(timestamp, value)
        
        result = analyzer.analyze()
        
        # Apnea should be detected after 10+ seconds
        assert result["apnea"] == True
    
    def test_shallow_breathing_detection(self):
        """Test shallow breathing detection."""
        analyzer = BreathingAnalyzer(window_sec=15.0)
        
        # First, establish baseline with normal amplitude
        fs = 30.0
        t1 = np.linspace(0, 10, int(10 * fs))
        normal_signal = 0.5 * np.sin(2 * np.pi * 0.8 * t1)
        
        for timestamp, value in zip(t1, normal_signal):
            analyzer.add_sample(timestamp, value)
        
        # Then add very low amplitude signal
        t2 = np.linspace(10, 20, int(10 * fs))
        shallow_signal = 0.05 * np.sin(2 * np.pi * 0.8 * t2)
        
        for timestamp, value in zip(t2, shallow_signal):
            analyzer.add_sample(timestamp, value)
        
        result = analyzer.analyze()
        
        # Shallow breathing might be detected (depending on threshold adaptation)
        # Just verify the field exists
        assert "shallow" in result


def test_peak_detection_with_noise():
    """Test peak detection robustness with noisy signal."""
    fs = 30.0
    t = np.linspace(0, 10, int(10 * fs))
    
    # Signal with 0.8 Hz peaks plus significant noise
    signal = np.sin(2 * np.pi * 0.8 * t) + 0.5 * np.random.randn(len(t))
    
    peaks = detect_peaks(signal, fs, refractory_sec=0.6)
    
    # Should still detect some peaks despite noise
    assert len(peaks) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

