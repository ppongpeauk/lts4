#!/usr/bin/env python3
"""
Quick test script to verify forward pass works
"""

import torch
import yaml
from model import Concert2StudioModel

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create model
print("Creating model...")
model = Concert2StudioModel(config)

# Test forward pass
print("Testing forward pass...")
batch_size = 1
seq_length = 48000  # 1 second at 48kHz

test_input = torch.randn(batch_size, seq_length)
test_target = torch.randn(batch_size, seq_length)

try:
    with torch.no_grad():
        output, losses = model(test_input, test_target)

    print(f"✅ Forward pass successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss keys: {list(losses.keys())}")
    print(f"Total loss: {losses['total'].item():.6f}")
    print("✅ All tensor size fixes working correctly!")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
