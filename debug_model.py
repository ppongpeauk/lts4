#!/usr/bin/env python3
"""
Debug script to test Concert2Studio model with synthetic data
This helps isolate whether the issue is in data loading or model processing
"""

import torch
import numpy as np
import yaml
from model import Concert2StudioModel


def create_synthetic_audio(batch_size=2, channels=2, duration=4.0, sample_rate=44100):
    """Create synthetic test audio with known properties"""
    time_samples = int(duration * sample_rate)

    # Create simple sine waves at different frequencies for each channel
    t = torch.linspace(0, duration, time_samples)

    if channels == 2:
        # Stereo: different frequency for each channel
        left_freq = 440.0  # A4
        right_freq = 880.0  # A5

        left_channel = 0.1 * torch.sin(2 * np.pi * left_freq * t)
        right_channel = 0.1 * torch.sin(2 * np.pi * right_freq * t)

        # Stack channels: (batch, channels, time)
        waveform = (
            torch.stack([left_channel, right_channel], dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
    else:
        # Mono
        freq = 440.0
        waveform = 0.1 * torch.sin(2 * np.pi * freq * t).unsqueeze(0).repeat(
            batch_size, 1
        )

    return waveform


def debug_model_inference():
    """Debug the model step by step"""

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model = Concert2StudioModel(config)
    model.eval()

    # Create test data
    test_audio = create_synthetic_audio(batch_size=1, channels=2, duration=1.0)
    print(
        f"Input audio: shape={test_audio.shape}, range=[{test_audio.min():.4f}, {test_audio.max():.4f}]"
    )

    with torch.no_grad():
        # Extract one channel for debugging
        channel_waveform = test_audio[:, 0, :]  # (1, T)
        print(
            f"Single channel: shape={channel_waveform.shape}, range=[{channel_waveform.min():.4f}, {channel_waveform.max():.4f}]"
        )

        # STFT
        spec = model.stft(channel_waveform)
        magnitude = torch.abs(spec)
        original_phase = torch.angle(spec)
        print(
            f"Original magnitude: shape={magnitude.shape}, range=[{magnitude.min():.4f}, {magnitude.max():.4f}]"
        )
        print(
            f"Original phase: shape={original_phase.shape}, range=[{original_phase.min():.4f}, {original_phase.max():.4f}]"
        )

        # U-Net processing
        magnitude_4d = magnitude.unsqueeze(1)
        print(f"Magnitude 4D input: shape={magnitude_4d.shape}")

        enhanced_magnitude_4d = model.unet(magnitude_4d)
        enhanced_magnitude = enhanced_magnitude_4d.squeeze(1)
        print(
            f"Enhanced magnitude: shape={enhanced_magnitude.shape}, range=[{enhanced_magnitude.min():.4f}, {enhanced_magnitude.max():.4f}]"
        )

        # Check for problematic values
        if torch.isnan(enhanced_magnitude).any():
            print("❌ Enhanced magnitude contains NaN!")
        if torch.isinf(enhanced_magnitude).any():
            print("❌ Enhanced magnitude contains Inf!")
        if torch.any(enhanced_magnitude < 0):
            print("❌ Enhanced magnitude contains negative values!")
        if torch.any(enhanced_magnitude > 1000):
            print("❌ Enhanced magnitude contains very large values!")

        # Clamp and reconstruct
        enhanced_magnitude = torch.clamp(
            enhanced_magnitude, min=1e-8, max=10.0
        )  # More aggressive clamping
        print(
            f"Clamped magnitude: range=[{enhanced_magnitude.min():.4f}, {enhanced_magnitude.max():.4f}]"
        )

        # Create complex spectrogram
        enhanced_complex_spec = enhanced_magnitude * torch.exp(1j * original_phase)
        print(f"Enhanced complex spec: shape={enhanced_complex_spec.shape}")

        # Check complex values
        if torch.isnan(enhanced_complex_spec).any():
            print("❌ Enhanced complex spec contains NaN!")
        if torch.isinf(enhanced_complex_spec).any():
            print("❌ Enhanced complex spec contains Inf!")

        # Try ISTFT
        try:
            reconstructed = torch.istft(
                enhanced_complex_spec,
                n_fft=model.n_fft,
                hop_length=model.hop_length,
                win_length=model.win_length,
                window=model.window,
                normalized=True,
                onesided=True,
                return_complex=False,
                center=True,
            )
            print(
                f"✅ ISTFT successful: shape={reconstructed.shape}, range=[{reconstructed.min():.4f}, {reconstructed.max():.4f}]"
            )

            if torch.isnan(reconstructed).any():
                print("❌ Reconstructed audio contains NaN!")
            if torch.isinf(reconstructed).any():
                print("❌ Reconstructed audio contains Inf!")

        except Exception as e:
            print(f"❌ ISTFT failed: {e}")


if __name__ == "__main__":
    print("🔍 Debugging model inference step by step...")
    debug_model_inference()
