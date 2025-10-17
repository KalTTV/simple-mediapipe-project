import numpy as np
import pytest
from breath_monitor.signal import BreathingSignalProcessor


class TestBreathingSignalProcessor:
    """Unit tests for BreathingSignalProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        return BreathingSignalProcessor(
            sample_rate=30.0,
            window_seconds=5.0,
            lowcut=0.1,
            highcut=0.5
        )

    def test_initialization(self, processor):
        """Test that processor initializes with correct parameters."""
        assert processor.sample_rate == 30.0
        assert processor.window_seconds == 5.0
        assert processor.lowcut == 0.1
        assert processor.highcut == 0.5
        assert len(processor.buffer) == 0

    def test_normal_breathing_sinusoid(self, processor):
        """Test with synthetic sinusoid representing normal breathing."""
        # Generate 10 seconds of breathing at 0.25 Hz (15 breaths/min)
        t = np.linspace(0, 10, 300)
        signal = np.sin(2 * np.pi * 0.25 * t)
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        assert 'breathing_rate' in result
        assert 'amplitude' in result
        assert 'quality' in result
        
        # Check breathing rate is reasonable (around 15 breaths/min)
        assert 10 <= result['breathing_rate'] <= 20
        
        # Check amplitude is significant
        assert result['amplitude'] > 0.1
        
        # Check quality is good
        assert result['quality'] > 0.5

    def test_apnea_flatline(self, processor):
        """Test with flatline signal representing apnea."""
        # Generate 10 seconds of flatline (no breathing)
        signal = np.zeros(300)
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        
        # Amplitude should be very low or zero
        assert result['amplitude'] < 0.01
        
        # Quality should be low due to lack of signal
        assert result['quality'] < 0.3

    def test_shallow_breathing_low_amplitude(self, processor):
        """Test with low-amplitude signal representing shallow breathing."""
        # Generate 10 seconds of shallow breathing at 0.25 Hz
        t = np.linspace(0, 10, 300)
        signal = 0.05 * np.sin(2 * np.pi * 0.25 * t)  # Very small amplitude
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        
        # Amplitude should be low
        assert result['amplitude'] < 0.1
        
        # Breathing rate should still be detectable
        assert 10 <= result['breathing_rate'] <= 20

    def test_noisy_signal(self, processor):
        """Test with noisy signal to verify filtering works."""
        # Generate breathing signal with noise
        t = np.linspace(0, 10, 300)
        clean_signal = np.sin(2 * np.pi * 0.25 * t)
        noise = 0.1 * np.random.randn(len(t))
        signal = clean_signal + noise
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        # Should still detect breathing despite noise
        assert 10 <= result['breathing_rate'] <= 20

    def test_buffer_management(self, processor):
        """Test that buffer maintains correct size."""
        max_buffer_size = int(processor.sample_rate * processor.window_seconds)
        
        # Add more samples than buffer size
        for i in range(max_buffer_size + 50):
            processor.add_sample(float(i))
        
        # Buffer should not exceed max size
        assert len(processor.buffer) <= max_buffer_size

    def test_insufficient_data(self, processor):
        """Test behavior with insufficient data."""
        # Add only a few samples
        for i in range(10):
            processor.add_sample(float(i))
        
        result = processor.process()
        
        # Should handle gracefully (return None or minimal result)
        # Exact behavior depends on implementation
        assert result is None or isinstance(result, dict)

    def test_rapid_breathing(self, processor):
        """Test with rapid breathing (high frequency)."""
        # Generate 10 seconds of rapid breathing at 0.4 Hz (24 breaths/min)
        t = np.linspace(0, 10, 300)
        signal = np.sin(2 * np.pi * 0.4 * t)
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        # Should detect higher breathing rate
        assert result['breathing_rate'] >= 20

    def test_slow_breathing(self, processor):
        """Test with slow breathing (low frequency)."""
        # Generate 10 seconds of slow breathing at 0.15 Hz (9 breaths/min)
        t = np.linspace(0, 10, 300)
        signal = np.sin(2 * np.pi * 0.15 * t)
        
        # Add all samples to buffer
        for sample in signal:
            processor.add_sample(sample)
        
        # Process the signal
        result = processor.process()
        
        assert result is not None
        # Should detect lower breathing rate
        assert result['breathing_rate'] <= 12

    def test_reset(self, processor):
        """Test that reset clears the buffer."""
        # Add samples
        for i in range(50):
            processor.add_sample(float(i))
        
        assert len(processor.buffer) > 0
        
        # Reset
        processor.reset()
        
        # Buffer should be empty
        assert len(processor.buffer) == 0
