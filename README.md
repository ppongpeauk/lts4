# 🎵 Concert2Studio: AI-Powered Audio Enhancement

Transform amateur concert recordings into studio-master-quality audio using a two-stage neural network pipeline.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd concert2studio

# 2. Run setup script (handles everything automatically)
bash setup.sh

# 3. Activate environment and start training
conda activate concert2studio
accelerate launch train.py --config=config.yaml
```

## 🔧 System Overview

Concert2Studio uses a two-stage architecture:

1. **Spectrogram U-Net**: Removes reverberation, crowd noise, and PA coloration
2. **UnivNet Vocoder**: Reconstructs phase-accurate waveforms with high-frequency detail

**Key Features:**
- 🎯 **Target Performance**: MOS ≈ 4.1, RTF ≤ 0.1 on RTX A6000
- 📦 **Lightweight**: <12M parameters, trains in <24 GPU-hours
- ⚡ **Fast Setup**: One script handles all dependencies
- 🔄 **Auto-Resample**: Handles any input format, outputs 48kHz/24-bit

## 📁 Project Structure

```
concert2studio/
├── setup.sh           # One-click setup script
├── environment.yml     # Conda environment specification
├── config.yaml        # All hyperparameters and settings
├── model.py           # Neural network architectures
├── dataset.py         # Data loading and augmentation
├── train.py           # Training script with Accelerate
├── infer.py           # Inference CLI tool
├── data/              # Audio files (created by setup)
├── checkpoints/       # Model checkpoints (created during training)
├── outputs/           # Training outputs (created during training)
└── logs/              # Training logs (created during training)
```

## 📊 Data Format

Place your audio files in the `data/` directory following this naming convention:

```
data/
├── song1_0.wav        # Studio master recording
├── song1_1.wav        # Concert recording #1
├── song1_2.wav        # Concert recording #2
├── song2_0.wav        # Another studio master
├── song2_1.wav        # Another concert recording
└── ...
```

**Naming Convention:**
- `{songID}_0.wav` → Studio master (target)
- `{songID}_{≠0}.wav` → Concert recordings (input)

**Supported Formats:**
- Any sample rate (auto-resampled to 48kHz)
- Any bit depth (converted to 24-bit)
- Mono or stereo (converted to mono)

## 🎮 Usage

### Training

```bash
# Basic training
accelerate launch train.py

# With custom config
accelerate launch train.py --config=my_config.yaml

# Resume from checkpoint
accelerate launch train.py --resume=checkpoints/epoch_010.pt

# Override data directory
accelerate launch train.py --data-dir=/path/to/my/data
```

### Inference

```bash
# Enhance a single file
python infer.py --input concert_recording.wav --output enhanced.wav

# Use specific checkpoint
python infer.py -i input.wav -o output.wav -c checkpoints/best.pt

# Benchmark performance
python infer.py --benchmark

# CPU-only inference
python infer.py -i input.wav -o output.wav --device cpu
```

### Testing Components

```bash
# Test model architecture
python model.py

# Test dataset loading
python dataset.py

# Verify data structure
python -c "from dataset import verify_dataset; verify_dataset('data')"
```

## 🛠️ Configuration

Edit `config.yaml` to customize training:

```yaml
# Key settings to adjust:
training:
  batch_size: 8          # Reduce if out of memory
  learning_rate: 2e-4    # Learning rate
  num_epochs: 100        # Training duration

data:
  data_dir: "./data"     # Path to audio files
  segment_length: 3.0    # Audio segment length (seconds)

model:
  unet:
    base_channels: 48    # Model capacity
    max_channels: 768    # Maximum channels
```

## 📈 Monitoring Training

Training logs are saved to `logs/` and can be viewed with TensorBoard:

```bash
tensorboard --logdir=logs
```

**Key Metrics:**
- **Total Loss**: Combined loss (lower is better)
- **Multi-Res STFT**: Spectral reconstruction quality
- **VGGish Loss**: Perceptual audio quality
- **Validation Loss**: Generalization performance

## 🎯 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **MOS** | ≥ 4.0 | TBD |
| **PESQ** | ≥ 3.5 | TBD |
| **RTF (A6000)** | ≤ 0.10 | TBD |
| **Parameters** | ≤ 12M | ~8M |

## 🔧 Hardware Requirements

**Minimum:**
- 8GB GPU memory (RTX 3070 or better)
- 16GB system RAM
- 50GB storage

**Recommended:**
- 24GB GPU memory (RTX 4090, A6000)
- 32GB system RAM
- 100GB SSD storage

**CPU-Only:** Supported but 50-100x slower

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 4  # or 2, or 1
```

**"No audio files found"**
```bash
# Check data directory structure
python dataset.py
```

**"Checkpoint not found"**
```bash
# List available checkpoints
ls checkpoints/
# Use specific checkpoint
python infer.py -c checkpoints/latest.pt
```

**"Import errors"**
```bash
# Reinstall environment
conda env remove -n concert2studio
bash setup.sh
```

### Performance Optimization

**For faster training:**
```yaml
hardware:
  compile_model: true      # Enable torch.compile
  channels_last: true      # Memory layout optimization
  num_workers: 8          # Increase if you have more CPU cores
```

**For lower memory usage:**
```yaml
training:
  batch_size: 4           # Reduce batch size
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

## 📚 Model Architecture Details

### Spectrogram U-Net
- **Input**: 513-bin magnitude spectrogram (1024-pt FFT)
- **Architecture**: 6 encoder + 6 decoder blocks with skip connections
- **Features**: Dilated convolutions (1,2,4,8), self-attention bottleneck
- **Parameters**: ~6M

### UnivNet Vocoder
- **Input**: Enhanced magnitude + original phase
- **Architecture**: Lightweight GAN generator
- **Features**: Multi-resolution discriminator, log-Mel conditioning
- **Parameters**: ~4M

### Loss Functions
- **L1 Loss**: Direct reconstruction error
- **Multi-Resolution STFT**: Spectral fidelity across scales
- **VGGish Perceptual**: Perceptual audio quality
- **Adversarial**: GAN training for vocoder

## 🔬 Advanced Usage

### Custom Data Augmentation

```python
# In dataset.py, modify augmentation parameters:
augmentation:
  gain_jitter_db: 6.0      # Increase for more robust training
  pink_noise_prob: 0.3     # Add more noise augmentation
  time_shift_ms: 100       # Larger time shifts
```

### Synthetic Pre-training

Enable synthetic data generation using room impulse responses:

```yaml
data:
  use_synthetic: true      # Enable RIR-based pre-training
```

### Multi-GPU Training

```bash
# Automatic multi-GPU detection
accelerate config
accelerate launch train.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{concert2studio2024,
  title={Concert2Studio: Neural Audio Enhancement for Live Recordings},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See `/docs` for detailed guides

---

**Ready to enhance your concert recordings? Start with `bash setup.sh`!** 🎶
