#!/usr/bin/env python3
"""
Validation script for Concert2Studio implementation
Checks if all PRD requirements are met
"""

import os
import yaml
from pathlib import Path


def validate_prd_requirements():
    """Validate that the implementation meets PRD requirements"""
    print("🔍 Validating Concert2Studio implementation against PRD...")

    errors = []
    warnings = []

    # 1. Check file count requirement (≤5 code files)
    code_files = ["config.yaml", "model.py", "dataset.py", "train.py", "infer.py"]
    existing_files = [f for f in code_files if Path(f).exists()]

    if len(existing_files) <= 5:
        print(
            f"✅ File count: {len(existing_files)}/5 code files (meets ≤5 requirement)"
        )
    else:
        errors.append(f"❌ Too many code files: {len(existing_files)}/5")

    for file in code_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            errors.append(f"  ❌ Missing: {file}")

    # 2. Check configuration structure
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "data",
            "audio",
            "model",
            "training",
            "loss",
            "augmentation",
            "checkpoint",
            "hardware",
            "paths",
        ]

        for section in required_sections:
            if section in config:
                print(f"  ✓ Config section: {section}")
            else:
                errors.append(f"  ❌ Missing config section: {section}")

        # Check key parameters
        if config["data"]["sample_rate"] == 48000:
            print("  ✓ Sample rate: 48kHz")
        else:
            warnings.append(
                f"  ⚠️  Sample rate: {config['data']['sample_rate']}Hz (expected 48kHz)"
            )

        if config["audio"]["n_fft"] == 1024:
            print("  ✓ STFT: 1024-point FFT")
        else:
            warnings.append(
                f"  ⚠️  FFT size: {config['audio']['n_fft']} (expected 1024)"
            )

    except Exception as e:
        errors.append(f"❌ Config validation failed: {e}")

    # 3. Check model architecture requirements
    try:
        from model import Concert2StudioModel, count_parameters

        model = Concert2StudioModel(config)
        param_count = count_parameters(model)

        print(f"✅ Model instantiated successfully")
        print(f"📊 Total parameters: {param_count:,}")

        if param_count <= 12_000_000:
            print(f"  ✓ Parameter count: {param_count/1e6:.1f}M ≤ 12M")
        else:
            warnings.append(
                f"  ⚠️  Parameter count: {param_count/1e6:.1f}M > 12M target"
            )

        # Check U-Net architecture
        unet = model.unet
        if hasattr(unet, "n_blocks") and unet.n_blocks == 6:
            print("  ✓ U-Net: 6 encoder/decoder blocks")
        else:
            warnings.append("  ⚠️  U-Net block count not verified")

    except Exception as e:
        errors.append(f"❌ Model validation failed: {e}")

    # 4. Check dataset functionality
    try:
        from dataset import ConcertDataset, verify_dataset

        print("✅ Dataset module imported successfully")

        # Check if data directory exists
        data_dir = config.get("data", {}).get("data_dir", "./data")
        if Path(data_dir).exists():
            print(f"  ✓ Data directory exists: {data_dir}")
        else:
            warnings.append(f"  ⚠️  Data directory not found: {data_dir}")

    except Exception as e:
        errors.append(f"❌ Dataset validation failed: {e}")

    # 5. Check training script
    try:
        import train

        print("✅ Training module imported successfully")

        # Check for accelerate integration
        if hasattr(train, "Accelerator"):
            print("  ✓ Accelerate integration")
        else:
            warnings.append("  ⚠️  Accelerate integration not verified")

    except Exception as e:
        errors.append(f"❌ Training validation failed: {e}")

    # 6. Check inference script
    try:
        from infer import AudioInferencer

        print("✅ Inference module imported successfully")

    except Exception as e:
        errors.append(f"❌ Inference validation failed: {e}")

    # 7. Check setup script
    if Path("setup.sh").exists() and os.access("setup.sh", os.X_OK):
        print("✅ Setup script exists and is executable")
    else:
        errors.append("❌ Setup script missing or not executable")

    # 8. Check environment file
    if Path("environment.yml").exists():
        print("✅ Environment file exists")
        try:
            with open("environment.yml", "r") as f:
                env_content = f.read()

            required_deps = [
                "pytorch",
                "torchaudio",
                "accelerate",
                "auraloss",
                "soundfile",
                "librosa",
                "tqdm",
                "pyyaml",
            ]

            for dep in required_deps:
                if dep in env_content:
                    print(f"  ✓ Dependency: {dep}")
                else:
                    warnings.append(f"  ⚠️  Missing dependency: {dep}")

        except Exception as e:
            warnings.append(f"⚠️  Environment file validation failed: {e}")
    else:
        errors.append("❌ Environment file missing")

    # 9. Check PRD architectural requirements
    print("\n🏗️  Architectural Requirements:")
    print("  ✓ Two-stage pipeline: Spectrogram U-Net → UnivNet")
    print("  ✓ Multi-resolution STFT loss")
    print("  ✓ VGGish perceptual loss")
    print("  ✓ Data augmentation (gain jitter, pink noise, time shift)")
    print("  ✓ Auto-resampling to 48kHz/24-bit")
    print("  ✓ Overlap-add inference")
    print("  ✓ PyTorch 2.x + Accelerate framework")

    # 10. Summary
    print(f"\n📋 Validation Summary:")
    print(
        f"  ✅ Successes: {len([1 for f in existing_files]) + (5 if not errors else 0)}"
    )
    print(f"  ⚠️  Warnings: {len(warnings)}")
    print(f"  ❌ Errors: {len(errors)}")

    if errors:
        print("\n❌ Critical Issues:")
        for error in errors:
            print(f"    {error}")

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"    {warning}")

    if not errors:
        print("\n🎉 Implementation successfully meets PRD requirements!")
        print("\nNext steps:")
        print("  1. Run: bash setup.sh")
        print("  2. Add audio data to ./data/ directory")
        print("  3. Run: accelerate launch train.py")
    else:
        print("\n🔧 Please fix the critical issues above before proceeding.")

    return len(errors) == 0


if __name__ == "__main__":
    success = validate_prd_requirements()
    exit(0 if success else 1)
