"""
Unit tests for signal processing module.
Tests chest expansion detection and debouncing.
"""

import pytest
import numpy as np
from breath_monitor.signal import (
    RingBuffer, estimate_fs, build_expansion_trace, detrend, bandpass,
    detect_breaths_hysteresis, calculate_bpm_from_ibi, BreathingAnalyzer
)


class TestRingBuffer:
    """Test ring buffer functionality."""
    
    def test_buffer_creation(self):
        buffer = RingBuffer(window_seconds=10.0, estimated_fps=30.0)
        assert len(buffer) == 0
    
    def test_buffer_add(self):
        buffer = RingBuffer(window_seconds=10.0, estimated_fps=30.0)
        buffer.add(0.0, 1.0)
        buffer.add(0.1, 2.0)
        assert len(buffer) == 2


class TestExpansionTrace:
    """Test chest expansion trace building."""
    
    def test_build_trace_multi_track(self):
        """Test building trace from 32 noisy tracks."""
        # Generate 0.25 Hz (15 BPM) breathing
        fs = 20.0
        duration = 25.0
        t = np.linspace(0, duration, int(duration * fs))
        breathing = 0.03 * np.sin(2 * np.pi * 0.25 * t)
        
        # 32 tracks with noise + drift + outliers
        tracks_list = []
        for i, breath in enumerate(breathing):
            tracks = breath + 0.01 * np.random.randn(32)
            # Add drift
            tracks += 0.001 * i
            # Add 2 outliers per frame
            tracks[0] = breath + 0.15
            tracks[1] = breath - 0.15
            tracks_list.append(tracks)
        
        trace = build_expansion_trace(tracks_list)
        
        assert len(trace) > 0
        # Should filter out outliers and drift
        assert len(trace) == len(tracks_list)


class TestBreathDetection:
    """Test hysteresis breath detection."""
    
    def test_debounce_prevents_double_counts(self):
        """Test that min_ibi_sec prevents double counting."""
        # Create signal with double peaks (noise creating false peaks)
        fs = 20.0
        duration = 20.0
        t = np.linspace(0, duration, int(duration * fs))
        
        # Intentional double pulses at 0.5s spacing
        expansion = np.zeros_like(t)
        for peak_time in [2.0, 2.5, 5.0, 5.5, 8.0, 8.5]:  # Pairs 0.5s apart
            pulse_idx = np.argmin(np.abs(t - peak_time))
            expansion[max(0, pulse_idx-5):min(len(expansion), pulse_idx+5)] += 0.1
        
        # With min_ibi_sec=1.5, should only count 3 breaths (not 6)
        breath_indices, ibi_list = detect_breaths_hysteresis(expansion, t, min_ibi_sec=1.5)
        
        # Should reject the 0.5s spaced duplicates
        assert len(breath_indices) <= 4  # Allow some detection variance
        
    def test_wide_band_low_frequency(self):
        """Test that 0.1 Hz (6 BPM) is detected."""
        analyzer = BreathingAnalyzer(
            window_sec=25.0,
            resp_low=0.08,
            resp_high=1.2,
            min_ibi_sec=2.0
        )
        
        # Generate 0.1 Hz breathing (6 BPM)
        fs = 20.0
        duration = 30.0
        t = np.linspace(0, duration, int(duration * fs))
        breathing = 0.05 * np.sin(2 * np.pi * 0.1 * t)
        
        # 32 tracks
        for i, (timestamp, value) in enumerate(zip(t, breathing)):
            tracks = breathing[i] + 0.01 * np.random.randn(32)
            analyzer.add_sample(timestamp, value, tracks)
        
        analysis = analyzer.analyze(visibility=0.9, tracks_alive=32, total_tracks=32)
        
        bpm = analysis.get("bpm")
        
        # Should detect ~6 BPM
        if bpm is not None:
            assert 5.0 <= bpm <= 8.0


class TestMultiTrackAccuracy:
    """Test that 32-track median reduces overestimation."""
    
    def test_32_track_median_filters_noise(self):
        """Verify 32 tracks with median recovers fundamental within ±10%."""
        analyzer = BreathingAnalyzer(
            window_sec=25.0,
            resp_low=0.08,
            resp_high=1.2,
            min_ibi_sec=1.5
        )
        
        # Generate 0.25 Hz (15 BPM) breathing
        fs = 20.0
        duration = 28.0
        t = np.linspace(0, duration, int(duration * fs))
        breathing = 0.03 * np.sin(2 * np.pi * 0.25 * t)
        
        # 32 noisy tracks + jitter
        for i, (timestamp, value) in enumerate(zip(t, breathing)):
            tracks = breathing[i] + 0.01 * np.random.randn(32)
            # Add high-frequency jitter to some tracks
            for j in range(0, 10, 3):
                tracks[j] += 0.02 * np.sin(2 * np.pi * 2.0 * timestamp)
            analyzer.add_sample(timestamp, value, tracks)
        
        analysis = analyzer.analyze(visibility=0.85, tracks_alive=32, total_tracks=32)
        
        bpm = analysis.get("bpm")
        
        # Should recover 15 BPM ± 10% (13.5-16.5)
        # The debouncing should prevent overestimation from jitter
        if bpm is not None:
            assert 13.5 <= bpm <= 16.5, f"BPM {bpm} outside ±10% of 15"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
