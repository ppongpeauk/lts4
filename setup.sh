#!/usr/bin/env bash
set -e

echo "[*] Setting up Concert2Studio environment..."

# 1) Install Miniconda if missing
if ! command -v conda &>/dev/null; then
    echo "[*] Installing Miniconda..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        curl -sSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    else
        echo "[!] Unsupported OS: $OSTYPE"
        exit 1
    fi
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    rm miniconda.sh
fi

# 2) Create environment if absent
if ! conda info --envs | grep -q concert2studio; then
    echo "[*] Creating concert2studio environment..."
    conda env create -f environment.yml
fi

echo "[*] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate concert2studio

# 3) Sanity GPU check
echo "[*] Checking GPU availability..."
python - <<'PY'
import torch
import sys

if torch.cuda.is_available():
    print(f"[✓] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[✓] PyTorch version: {torch.__version__}")
    print(f"[✓] CUDA version: {torch.version.cuda}")
else:
    print("[!] CUDA not detected! Training will be CPU-only (very slow)")

# Check if we can import all required modules
try:
    import torchaudio
    import soundfile
    import librosa
    import accelerate
    import auraloss
    import yaml
    print("[✓] All required packages imported successfully")
except ImportError as e:
    print(f"[!] Import error: {e}")
    sys.exit(1)
PY

# 4) Create data directory if it doesn't exist
mkdir -p data

echo ""
echo "🎶 Setup complete! 🎶"
echo ""
echo "Ready to use! Run the following commands:"
echo "  conda activate concert2studio"
echo "  accelerate launch train.py --config=config.yaml"
echo ""
echo "Don't forget to put your audio files in ./data/ following the naming convention:"
echo "  ./data/{songID}_0.wav → studio master"
echo "  ./data/{songID}_{≠1}.wav → concert recordings"
