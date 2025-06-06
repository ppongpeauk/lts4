"""
Dataset handling for Concert2Studio
Implements ConcertDataset with auto-resample and data augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import soundfile as sf
import librosa
import numpy as np
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
import pyloudnorm as pyln


class ConcertDataset(Dataset):
    """
    Dataset for concert-to-studio audio pairs
    Handles auto-resampling, data augmentation, and efficient loading
    """

    def __init__(self, data_dir: str, config: dict, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_training = is_training

        self.sample_rate = config["data"]["sample_rate"]
        self.segment_length = config["audio"]["segment_length"]
        self.segment_samples = int(self.sample_rate * self.segment_length)

        # Audio processing
        self.bit_depth = config["data"]["bit_depth"]

        # Data augmentation parameters
        self.gain_jitter_db = config["augmentation"]["gain_jitter_db"]
        self.gain_jitter_prob = config["augmentation"]["gain_jitter_prob"]
        self.pink_noise_prob = config["augmentation"]["pink_noise_prob"]
        self.pink_noise_level = config["augmentation"]["pink_noise_level"]
        self.time_shift_ms = config["augmentation"]["time_shift_ms"]
        self.time_shift_prob = config["augmentation"]["time_shift_prob"]

        # Advanced augmentation parameters (research-based)
        self.mixup_prob = config["augmentation"].get("mixup_prob", 0.0)
        self.mixup_alpha = config["augmentation"].get("mixup_alpha", 0.4)
        self.spec_augment_prob = config["augmentation"].get("spec_augment_prob", 0.0)
        self.freq_mask_prob = config["augmentation"].get("freq_mask_prob", 0.0)
        self.time_mask_prob = config["augmentation"].get("time_mask_prob", 0.0)

        # Stereo support
        self.use_stereo = config["model"]["unet"].get("use_stereo", False)

        # LUFS normalization settings
        self.target_lufs = config["data"].get("target_lufs", -23.0)  # EBU R128 standard
        self.lufs_tolerance = config["data"].get(
            "lufs_tolerance", 2.0
        )  # ±2 LUFS tolerance

        # Initialize loudness meter
        self.meter = pyln.Meter(self.sample_rate)

        # Audio quality settings
        self.enable_lufs_normalization = config["data"].get(
            "enable_lufs_normalization", True
        )
        self.enable_dc_removal = config["data"].get("enable_dc_removal", True)
        self.enable_peak_limiting = config["data"].get("enable_peak_limiting", True)
        self.peak_threshold = config["data"].get("peak_threshold", 0.95)

        # Find all audio file pairs
        self.audio_pairs = self._find_audio_pairs()

        print(f"Found {len(self.audio_pairs)} audio pairs in {data_dir}")
        if self.enable_lufs_normalization:
            print(
                f"📊 LUFS normalization enabled: target {self.target_lufs} LUFS ±{self.lufs_tolerance}"
            )

    def _find_audio_pairs(self) -> List[Tuple[str, str]]:
        """
        Find pairs of studio master and concert recordings
        Convention: {songID}_0.wav → studio master, {songID}_{≠1}.wav → concert
        """
        pairs = []
        wav_files = list(self.data_dir.glob("*.wav"))

        # Group files by song ID
        song_groups = {}
        for wav_file in wav_files:
            name = wav_file.stem
            if "_" in name:
                song_id, version = name.rsplit("_", 1)
                if song_id not in song_groups:
                    song_groups[song_id] = {}
                song_groups[song_id][version] = wav_file

        # Create pairs
        for song_id, versions in song_groups.items():
            if "0" in versions:  # Studio master exists
                studio_file = versions["0"]
                for version, concert_file in versions.items():
                    if version != "0":  # Concert recording
                        pairs.append((str(concert_file), str(studio_file)))

        return pairs

    def _normalize_lufs(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to target LUFS level"""
        if not self.enable_lufs_normalization:
            return waveform

        try:
            # Convert to numpy for pyloudnorm
            if self.use_stereo and waveform.dim() == 2:
                # Stereo: (2, T) -> (T, 2) for pyloudnorm
                audio_np = waveform.transpose(0, 1).numpy().astype(np.float64)
            else:
                # Mono: (T,) -> (T,) for pyloudnorm
                audio_np = waveform.numpy().astype(np.float64)

            # Measure current loudness
            try:
                current_loudness = self.meter.integrated_loudness(audio_np)

                # Skip normalization if the measurement failed or audio is too quiet
                if current_loudness == float("-inf") or current_loudness < -70.0:
                    # Very quiet audio - apply gentle gain instead of LUFS normalization
                    gain_db = self.target_lufs - (
                        -40.0
                    )  # Assume -40 LUFS for quiet audio
                    gain_linear = 10 ** (gain_db / 20.0)
                    normalized_audio = audio_np * gain_linear
                else:
                    # Calculate required gain
                    gain_db = self.target_lufs - current_loudness

                    # Limit gain to prevent extreme adjustments
                    gain_db = np.clip(gain_db, -20.0, 20.0)  # ±20dB max adjustment

                    # Apply gain only if needed (outside tolerance)
                    if abs(gain_db) > self.lufs_tolerance:
                        gain_linear = 10 ** (gain_db / 20.0)
                        normalized_audio = audio_np * gain_linear
                    else:
                        normalized_audio = audio_np

            except Exception as e:
                # Fallback: simple RMS normalization
                rms = np.sqrt(np.mean(audio_np**2))
                if rms > 1e-8:  # Avoid division by zero
                    target_rms = 10 ** (
                        self.target_lufs / 20.0
                    )  # Rough LUFS to RMS conversion
                    gain = target_rms / rms
                    gain = np.clip(gain, 0.1, 10.0)  # Limit gain to reasonable range
                    normalized_audio = audio_np * gain
                else:
                    normalized_audio = audio_np

            # Convert back to torch tensor
            normalized_audio = normalized_audio.astype(np.float32)

            if self.use_stereo and waveform.dim() == 2:
                # Convert back to (2, T) format
                result = torch.from_numpy(normalized_audio.T)
            else:
                result = torch.from_numpy(normalized_audio)

            return result

        except Exception as e:
            warnings.warn(f"LUFS normalization failed: {e}")
            return waveform

    def _apply_audio_quality_processing(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio quality processing: DC removal, peak limiting, etc."""

        # DC removal (remove DC offset)
        if self.enable_dc_removal:
            if self.use_stereo and waveform.dim() == 2:
                # Remove DC from each channel
                waveform = waveform - torch.mean(waveform, dim=1, keepdim=True)
            else:
                waveform = waveform - torch.mean(waveform)

        # Peak limiting to prevent clipping
        if self.enable_peak_limiting:
            peak = torch.max(torch.abs(waveform))
            if peak > self.peak_threshold:
                # Apply gentle compression instead of hard limiting
                ratio = self.peak_threshold / peak
                # Soft limiting with gentle curve
                waveform = waveform * ratio * 0.95  # 5% safety margin

        # Final safety clamp
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def _load_and_resample_audio(self, file_path: str) -> torch.Tensor:
        """
        Load audio and resample if necessary
        Handles different sample rates and bit depths
        """
        try:
            # Get file info first to avoid loading large files unnecessarily
            info = torchaudio.info(file_path)

            if (
                info.sample_rate == self.sample_rate
                and info.bits_per_sample == self.bit_depth
            ):
                # No resampling needed
                waveform, _ = torchaudio.load(file_path)
            else:
                # Load and resample
                waveform, orig_sr = torchaudio.load(file_path)

                if orig_sr != self.sample_rate:
                    # Resample
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=orig_sr, new_freq=self.sample_rate
                    )
                    waveform = resampler(waveform)

            # Handle stereo/mono based on configuration
            if self.use_stereo:
                # Ensure stereo output
                if waveform.shape[0] == 1:
                    # Duplicate mono to pseudo-stereo
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    # Take first two channels if more than stereo
                    waveform = waveform[:2]
                # Apply audio quality processing and LUFS normalization
                waveform = self._apply_audio_quality_processing(waveform)
                waveform = self._normalize_lufs(waveform)
                return waveform  # Keep (2, T) shape for stereo
            else:
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform = waveform.squeeze(0)  # Remove channel dimension (T,)
                # Apply audio quality processing and LUFS normalization
                waveform = self._apply_audio_quality_processing(waveform)
                waveform = self._normalize_lufs(waveform)
                return waveform

        except Exception as e:
            # Fallback to librosa for problematic files
            warnings.warn(f"torchaudio failed for {file_path}, using librosa: {e}")
            if self.use_stereo:
                # Load as stereo with librosa
                waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)
                waveform = torch.from_numpy(waveform)
                if waveform.dim() == 1:  # Mono file
                    waveform = waveform.unsqueeze(0).repeat(
                        2, 1
                    )  # Convert to pseudo-stereo
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2]  # Take first two channels

                # Apply audio quality processing and LUFS normalization for stereo librosa fallback
                waveform = self._apply_audio_quality_processing(waveform)
                waveform = self._normalize_lufs(waveform)
            else:
                waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
                waveform = torch.from_numpy(waveform)

            # Apply audio quality processing and LUFS normalization for librosa fallback
            waveform = self._apply_audio_quality_processing(waveform)
            waveform = self._normalize_lufs(waveform)

            return waveform

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation during training with research-based techniques"""
        if not self.is_training:
            return waveform

        # Handle stereo vs mono
        is_stereo_input = waveform.dim() == 2 and waveform.shape[0] == 2

        # Apply augmentations to each channel if stereo
        if is_stereo_input:
            left_channel = self._apply_channel_augmentation(waveform[0])
            right_channel = self._apply_channel_augmentation(waveform[1])
            waveform = torch.stack([left_channel, right_channel], dim=0)
        else:
            waveform = self._apply_channel_augmentation(waveform)

        return waveform

    def _apply_channel_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single channel"""
        # Gain jitter
        if random.random() < self.gain_jitter_prob:
            gain_db = random.uniform(-self.gain_jitter_db, self.gain_jitter_db)
            gain_linear = 10 ** (gain_db / 20)
            waveform = waveform * gain_linear

        # Pink noise overlay
        if random.random() < self.pink_noise_prob:
            pink_noise = self._generate_pink_noise(len(waveform))
            noise_level = 10 ** (self.pink_noise_level / 20)
            waveform = waveform + pink_noise * noise_level

        # Time shift (wrap-around)
        if random.random() < self.time_shift_prob:
            shift_samples = random.randint(
                -int(self.time_shift_ms * self.sample_rate / 1000),
                int(self.time_shift_ms * self.sample_rate / 1000),
            )
            waveform = torch.roll(waveform, shift_samples)

        return waveform

    def _generate_pink_noise(self, length: int) -> torch.Tensor:
        """Generate pink noise (1/f noise)"""
        # Simple approximation of pink noise
        white_noise = torch.randn(length)

        # Apply simple 1/f filter approximation
        # This is a simplified version - for production, use proper pink noise generation
        freqs = torch.fft.fftfreq(length)
        magnitude = 1 / torch.sqrt(torch.abs(freqs) + 1e-8)
        magnitude[0] = 0  # DC component

        # Apply filter in frequency domain
        fft = torch.fft.fft(white_noise)
        filtered_fft = fft * magnitude
        pink_noise = torch.fft.ifft(filtered_fft).real

        return pink_noise

    def _get_random_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment of specified length"""
        # Handle stereo vs mono
        if waveform.dim() == 2:  # Stereo (2, T)
            waveform_length = waveform.shape[1]
            if waveform_length <= self.segment_samples:
                # Pad if too short
                padding = self.segment_samples - waveform_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                # Random crop if too long
                if self.is_training:
                    start_idx = random.randint(
                        0, waveform_length - self.segment_samples
                    )
                else:
                    # Use center crop for validation
                    start_idx = (waveform_length - self.segment_samples) // 2
                waveform = waveform[:, start_idx : start_idx + self.segment_samples]
        else:  # Mono (T,)
            if len(waveform) <= self.segment_samples:
                # Pad if too short
                padding = self.segment_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                # Random crop if too long
                if self.is_training:
                    start_idx = random.randint(0, len(waveform) - self.segment_samples)
                else:
                    # Use center crop for validation
                    start_idx = (len(waveform) - self.segment_samples) // 2
                waveform = waveform[start_idx : start_idx + self.segment_samples]

        return waveform

    def _apply_mixup(
        self, concert_segment: torch.Tensor, studio_segment: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup data augmentation"""
        if not self.is_training or random.random() >= self.mixup_prob:
            return concert_segment, studio_segment

        # Get another random sample for mixing
        mix_idx = random.randint(0, len(self.audio_pairs) - 1)
        mix_concert_path, mix_studio_path = self.audio_pairs[mix_idx]

        # Load the mixing samples
        mix_concert = self._load_and_resample_audio(mix_concert_path)
        mix_studio = self._load_and_resample_audio(mix_studio_path)

        # Get segments
        mix_concert_segment = self._get_random_segment(mix_concert)
        mix_studio_segment = self._get_random_segment(mix_studio)

        # Generate mixing coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Mix the samples
        mixed_concert = lam * concert_segment + (1 - lam) * mix_concert_segment
        mixed_studio = lam * studio_segment + (1 - lam) * mix_studio_segment

        return mixed_concert, mixed_studio

    def __len__(self) -> int:
        return len(self.audio_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of concert and studio audio segments
        Returns: (concert_audio, studio_audio)
        """
        concert_path, studio_path = self.audio_pairs[idx]

        # Load audio files
        concert_audio = self._load_and_resample_audio(concert_path)
        studio_audio = self._load_and_resample_audio(studio_path)

        # Get segments
        concert_segment = self._get_random_segment(concert_audio)
        studio_segment = self._get_random_segment(studio_audio)

        # Apply augmentation
        concert_segment = self._apply_augmentation(concert_segment)
        studio_segment = self._apply_augmentation(studio_segment)

        # Apply mixup augmentation
        concert_segment, studio_segment = self._apply_mixup(
            concert_segment, studio_segment
        )

        # Normalize to prevent clipping
        concert_segment = torch.clamp(concert_segment, -1.0, 1.0)
        studio_segment = torch.clamp(studio_segment, -1.0, 1.0)

        return concert_segment, studio_segment


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    # Create full dataset
    full_dataset = ConcertDataset(
        data_dir=config["data"]["data_dir"], config=config, is_training=True
    )

    # Split into train and validation
    train_size = int(config["data"]["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(config["data"]["random_seed"])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Set validation dataset to non-training mode
    val_dataset.dataset.is_training = False

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        persistent_workers=config["hardware"]["persistent_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        persistent_workers=config["hardware"]["persistent_workers"],
        drop_last=False,
    )

    return train_loader, val_loader


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for pre-training using RIR convolution
    Optional dataset for bootstrapping the U-Net
    """

    def __init__(self, studio_files: List[str], rir_files: List[str], config: dict):
        self.studio_files = studio_files
        self.rir_files = rir_files
        self.config = config

        self.sample_rate = config["data"]["sample_rate"]
        self.segment_length = config["audio"]["segment_length"]
        self.segment_samples = int(self.sample_rate * self.segment_length)

    def __len__(self) -> int:
        return (
            len(self.studio_files) * 10
        )  # Generate multiple synthetic pairs per studio file

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select studio file and RIR
        studio_idx = idx % len(self.studio_files)
        rir_idx = random.randint(0, len(self.rir_files) - 1)

        # Load studio audio
        studio_audio, _ = torchaudio.load(self.studio_files[studio_idx])
        if studio_audio.shape[0] > 1:
            studio_audio = torch.mean(studio_audio, dim=0, keepdim=True)
        studio_audio = studio_audio.squeeze(0)

        # Load RIR
        rir, rir_sr = torchaudio.load(self.rir_files[rir_idx])
        if rir.shape[0] > 1:
            rir = torch.mean(rir, dim=0, keepdim=True)
        rir = rir.squeeze(0)

        # Resample RIR if necessary
        if rir_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(rir_sr, self.sample_rate)
            rir = resampler(rir)

        # Convolve to create synthetic concert audio
        concert_audio = torch.nn.functional.conv1d(
            studio_audio.unsqueeze(0).unsqueeze(0),
            rir.unsqueeze(0).unsqueeze(0),
            padding=len(rir) // 2,
        ).squeeze()

        # Get random segments
        studio_segment = self._get_random_segment(studio_audio)
        concert_segment = self._get_random_segment(concert_audio)

        return concert_segment, studio_segment

    def _get_random_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment of specified length"""
        if len(waveform) <= self.segment_samples:
            padding = self.segment_samples - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            start_idx = random.randint(0, len(waveform) - self.segment_samples)
            waveform = waveform[start_idx : start_idx + self.segment_samples]

        return waveform


def verify_dataset(data_dir: str) -> None:
    """Verify dataset structure and print statistics"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ Data directory {data_dir} does not exist!")
        return

    wav_files = list(data_path.glob("*.wav"))
    print(f"📁 Found {len(wav_files)} WAV files in {data_dir}")

    # Check naming convention
    studio_files = [f for f in wav_files if f.stem.endswith("_0")]
    concert_files = [
        f for f in wav_files if not f.stem.endswith("_0") and "_" in f.stem
    ]

    print(f"🎵 Studio masters: {len(studio_files)}")
    print(f"🎤 Concert recordings: {len(concert_files)}")

    # Check file properties
    sample_rates = set()
    bit_depths = set()

    for wav_file in wav_files[:5]:  # Check first 5 files
        try:
            info = torchaudio.info(str(wav_file))
            sample_rates.add(info.sample_rate)
            bit_depths.add(info.bits_per_sample)
        except Exception as e:
            print(f"⚠️  Could not read {wav_file}: {e}")

    print(f"📊 Sample rates found: {sorted(sample_rates)}")
    print(f"📊 Bit depths found: {sorted(bit_depths)}")

    if 48000 not in sample_rates:
        print("⚠️  No 48kHz files found - auto-resampling will be used")

    if 24 not in bit_depths:
        print("⚠️  No 24-bit files found - this may affect quality")


if __name__ == "__main__":
    # Test dataset
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Verify dataset structure
    verify_dataset(config["data"]["data_dir"])

    # Test dataset loading
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(
            f"✅ Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches"
        )

        # Test loading a batch
        concert_batch, studio_batch = next(iter(train_loader))
        print(
            f"✅ Batch shapes: concert {concert_batch.shape}, studio {studio_batch.shape}"
        )

    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        print(
            "Make sure you have audio files in the data directory following the naming convention:"
        )
