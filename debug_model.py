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

        # Shape: (batch_size, 2, time_samples)
        audio = (
            torch.stack([left_channel, right_channel], dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
    else:
        # Mono
        freq = 440.0
        audio = 0.1 * torch.sin(2 * np.pi * freq * t).unsqueeze(0).repeat(batch_size, 1)

    return audio


def test_model():
    """Test the model with synthetic data"""
    print("🔧 Testing Concert2Studio model with synthetic data...")

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create model
    print("Creating model...")
    model = Concert2StudioModel(config)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create synthetic input
    print("\nCreating synthetic test audio...")
    batch_size = 1
    channels = 2 if config["model"]["unet"]["use_stereo"] else 1
    duration = config["audio"]["segment_length"]
    sample_rate = config["data"]["sample_rate"]

    if channels == 2:
        input_audio = create_synthetic_audio(
            batch_size, channels, duration, sample_rate
        )
        target_audio = input_audio * 1.1  # Slightly louder target
    else:
        input_audio = create_synthetic_audio(batch_size, 1, duration, sample_rate)
        target_audio = input_audio * 1.1

    print(f"Input shape: {input_audio.shape}")
    print(f"Input range: [{input_audio.min():.4f}, {input_audio.max():.4f}]")
    print(f"Target shape: {target_audio.shape}")
    print(f"Target range: [{target_audio.min():.4f}, {target_audio.max():.4f}]")

    # Test forward pass
    print("\n🚀 Testing forward pass...")
    with torch.no_grad():
        try:
            output = model(input_audio)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

            # Check if output is reasonable
            if torch.allclose(output, torch.zeros_like(output), atol=1e-6):
                print("❌ WARNING: Output is silent/zero!")
            elif torch.std(output) < 1e-6:
                print("❌ WARNING: Output has no variation (constant value)!")
            elif torch.abs(output).max() > 1.0:
                print("❌ WARNING: Output exceeds audio range [-1, 1]!")
            else:
                print("✅ Output appears reasonable")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return

    # Test training mode
    print("\n🎯 Testing training mode (with losses)...")
    model.train()
    try:
        output, losses = model(input_audio, target_audio)
        print(f"✅ Training pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        print("Loss breakdown:")
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                print(f"  {loss_name}: {loss_value.item():.6f}")
            else:
                print(f"  {loss_name}: {loss_value}")

        # Check if losses are reasonable
        total_loss = losses["total"].item()
        if total_loss > 100:
            print(f"❌ WARNING: Total loss is very high: {total_loss:.6f}")
        elif total_loss < 1e-8:
            print(f"❌ WARNING: Total loss is suspiciously low: {total_loss:.6f}")
        else:
            print(f"✅ Total loss appears reasonable: {total_loss:.6f}")

    except Exception as e:
        print(f"❌ Training pass failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model()
