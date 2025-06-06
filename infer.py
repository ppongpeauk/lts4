"""
Inference script for Concert2Studio
CLI tool for processing concert recordings to studio quality
"""

import argparse
import yaml
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
import time
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Local imports
from model import Concert2StudioModel, count_parameters


class AudioInferencer:
    """Audio inference with overlap-add processing"""

    def __init__(self, config: dict, checkpoint_path: str, device: str = "auto"):
        self.config = config

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔧 Using device: {self.device}")

        # Audio parameters
        self.sample_rate = config["data"]["sample_rate"]
        self.segment_length = config["audio"]["segment_length"]
        self.segment_samples = int(self.sample_rate * self.segment_length)

        # Overlap parameters for smoother results
        self.overlap_ratio = 0.25  # 25% overlap
        self.overlap_samples = int(self.segment_samples * self.overlap_ratio)
        self.hop_samples = self.segment_samples - self.overlap_samples

        # Load model
        self.model = self._load_model(checkpoint_path)

        print(f"✅ Model loaded successfully")
        param_count = count_parameters(self.model)
        print(f"📊 Model parameters: {param_count:,}")

    def _load_model(self, checkpoint_path: str) -> Concert2StudioModel:
        """Load model from checkpoint"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📂 Loading model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Initialize model
        model = Concert2StudioModel(self.config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Enable optimizations
        if self.config.get("hardware", {}).get("compile_model", False):
            try:
                model = torch.compile(model)
                print("✅ Model compiled with torch.compile")
            except Exception as e:
                print(f"⚠️  torch.compile failed: {e}")

        if self.config.get("hardware", {}).get("channels_last", False):
            model = model.to(memory_format=torch.channels_last)

        return model

    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        print(f"📂 Loading audio from {file_path}")

        try:
            # Get file info
            info = torchaudio.info(file_path)
            print(
                f"📊 Original: {info.sample_rate}Hz, {info.num_channels} channels, {info.num_frames} samples"
            )

            # Load audio
            waveform, orig_sr = torchaudio.load(file_path)

            # Handle stereo/mono based on model configuration
            use_stereo = self.config["model"]["unet"].get("use_stereo", False)

            if use_stereo:
                # Keep stereo or convert mono to stereo
                if waveform.shape[0] == 1:
                    # Duplicate mono to stereo
                    waveform = waveform.repeat(2, 1)
                    print("🔄 Converted mono to stereo")
                elif waveform.shape[0] > 2:
                    # Take first two channels if more than stereo
                    waveform = waveform[:2]
                    print("🔄 Trimmed to stereo")
                # Keep as (2, T) for stereo processing
            else:
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    print("🔄 Converted to mono")
                # Remove channel dimension for mono: (1, T) -> (T,)
                waveform = waveform.squeeze(0)

            # Resample if necessary
            if orig_sr != self.sample_rate:
                print(f"🔄 Resampling from {orig_sr}Hz to {self.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sample_rate
                )
                if use_stereo:
                    # Resample each channel separately
                    resampled_channels = []
                    for ch in range(waveform.shape[0]):
                        resampled_channels.append(resampler(waveform[ch : ch + 1]))
                    waveform = torch.cat(resampled_channels, dim=0)
                else:
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

            # Normalize
            if torch.max(torch.abs(waveform)) > 1.0:
                waveform = waveform / torch.max(torch.abs(waveform))
                print("🔄 Normalized audio levels")

            print(
                f"✅ Audio loaded: {len(waveform)} samples ({len(waveform)/self.sample_rate:.2f}s)"
            )
            return waveform

        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def _apply_window(self, segment: torch.Tensor) -> torch.Tensor:
        """Apply Hann window to segment edges for smooth overlap-add"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo: segment shape is (2, T)
            if segment.shape[-1] != self.segment_samples:
                return segment

            # Create Hann window for overlap regions
            window = torch.ones_like(segment)

            # Fade in
            fade_in = torch.hann_window(2 * self.overlap_samples)[
                : self.overlap_samples
            ]
            window[:, : self.overlap_samples] = fade_in.unsqueeze(0)

            # Fade out
            fade_out = torch.hann_window(2 * self.overlap_samples)[
                self.overlap_samples :
            ]
            window[:, -self.overlap_samples :] = fade_out.unsqueeze(0)
        else:
            # Mono: segment shape is (T,)
            if len(segment) != self.segment_samples:
                return segment

            # Create Hann window for overlap regions
            window = torch.ones_like(segment)

            # Fade in
            fade_in = torch.hann_window(2 * self.overlap_samples)[
                : self.overlap_samples
            ]
            window[: self.overlap_samples] = fade_in

            # Fade out
            fade_out = torch.hann_window(2 * self.overlap_samples)[
                self.overlap_samples :
            ]
            window[-self.overlap_samples :] = fade_out

        return segment * window

    def _process_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Process a single audio segment"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo processing: segment shape is (2, T)
            original_length = segment.shape[-1]
            if original_length < self.segment_samples:
                padding = self.segment_samples - original_length
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Add batch dimension: (2, T) -> (1, 2, T)
            segment = segment.unsqueeze(0).to(self.device)
        else:
            # Mono processing: segment shape is (T,)
            original_length = len(segment)
            if original_length < self.segment_samples:
                padding = self.segment_samples - original_length
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Add batch dimension: (T,) -> (1, T)
            segment = segment.unsqueeze(0).to(self.device)

        # Set memory format if requested
        if self.config.get("hardware", {}).get("channels_last", False):
            segment = segment.to(memory_format=torch.channels_last)

        # Process with model
        with torch.no_grad():
            enhanced = self.model(segment)

        # Move back to CPU and remove batch dimension
        enhanced = enhanced.cpu().squeeze(0)

        # Trim to original length if padded
        if use_stereo:
            # Stereo: enhanced shape is (2, T)
            if original_length < self.segment_samples:
                enhanced = enhanced[..., :original_length]
        else:
            # Mono: enhanced shape is (T,)
            if original_length < self.segment_samples:
                enhanced = enhanced[:original_length]

        return enhanced

    def process_audio(
        self, input_audio: torch.Tensor, show_progress: bool = True
    ) -> torch.Tensor:
        """Process full audio with overlap-add"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo processing: input_audio shape is (2, T)
            total_samples = input_audio.shape[-1]
        else:
            # Mono processing: input_audio shape is (T,)
            total_samples = len(input_audio)

        if total_samples <= self.segment_samples:
            # Audio is short enough to process in one go
            print("🎵 Processing in single segment")
            return self._process_segment(input_audio)

        # Calculate number of segments
        num_segments = max(
            1, (total_samples - self.segment_samples) // self.hop_samples + 1
        )
        print(f"🎵 Processing {num_segments} overlapping segments")

        # Initialize output
        if use_stereo:
            output_audio = torch.zeros(2, total_samples)
            overlap_counts = torch.zeros(2, total_samples)
        else:
            output_audio = torch.zeros(total_samples)
            overlap_counts = torch.zeros(total_samples)

        # Process segments with overlap-add
        for i in range(num_segments):
            start_idx = i * self.hop_samples
            end_idx = min(start_idx + self.segment_samples, total_samples)

            if show_progress:
                progress = (i + 1) / num_segments * 100
                print(
                    f"\r🔄 Processing segment {i+1}/{num_segments} ({progress:.1f}%)",
                    end="",
                )

            # Extract segment
            if use_stereo:
                segment = input_audio[:, start_idx:end_idx]  # (2, segment_length)
            else:
                segment = input_audio[start_idx:end_idx]  # (segment_length,)

            # Process segment
            enhanced_segment = self._process_segment(segment)

            # Apply windowing for smooth overlap
            if i > 0 or i < num_segments - 1:  # Not first or last segment
                enhanced_segment = self._apply_window(enhanced_segment)

            # Add to output with overlap handling
            if use_stereo:
                actual_length = enhanced_segment.shape[-1]
                output_audio[
                    :, start_idx : start_idx + actual_length
                ] += enhanced_segment
                overlap_counts[:, start_idx : start_idx + actual_length] += 1
            else:
                actual_length = len(enhanced_segment)
                output_audio[start_idx : start_idx + actual_length] += enhanced_segment
                overlap_counts[start_idx : start_idx + actual_length] += 1

        if show_progress:
            print()  # New line after progress

        # Normalize by overlap counts
        output_audio = output_audio / torch.clamp(overlap_counts, min=1)

        return output_audio

    def process_file(
        self, input_path: str, output_path: str, target_lufs: float = -1.0
    ) -> None:
        """Process audio file from input to output"""
        start_time = time.time()

        # Load input audio
        input_audio = self._load_audio(input_path)

        # Process audio
        print("🎵 Starting audio enhancement...")
        enhanced_audio = self.process_audio(input_audio)

        # Post-processing: limiting to target LUFS
        if target_lufs is not None:
            enhanced_audio = self._apply_limiter(enhanced_audio, target_lufs)

        # Save output
        self._save_audio(enhanced_audio, output_path)

        # Performance metrics
        processing_time = time.time() - start_time
        audio_duration = len(input_audio) / self.sample_rate
        rtf = processing_time / audio_duration

        print(f"✅ Processing complete!")
        print(f"📊 Audio duration: {audio_duration:.2f}s")
        print(f"📊 Processing time: {processing_time:.2f}s")
        print(f"📊 Real-time factor: {rtf:.3f}x")
        print(f"💾 Output saved to: {output_path}")

    def _apply_limiter(self, audio: torch.Tensor, target_lufs: float) -> torch.Tensor:
        """Apply simple limiter to achieve target LUFS"""
        # Simple peak limiting - in production you'd use proper LUFS metering
        current_peak = torch.max(torch.abs(audio))
        target_peak = 10 ** (target_lufs / 20)  # Rough LUFS to peak conversion

        if current_peak > target_peak:
            ratio = target_peak / current_peak
            audio = audio * ratio
            print(f"🔧 Applied limiting: {20*torch.log10(ratio):.1f}dB reduction")

        return audio

    def _save_audio(self, audio: torch.Tensor, output_path: str) -> None:
        """Save audio to file"""
        print(f"💾 Saving audio to {output_path}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy and ensure correct format
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo: audio shape is (2, T) -> transpose to (T, 2) for soundfile
            audio_np = audio.transpose(0, 1).numpy().astype(np.float32)
        else:
            # Mono: audio shape is (T,)
            audio_np = audio.numpy().astype(np.float32)

        # Clamp to prevent clipping
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Save with high quality settings
        sf.write(
            output_path, audio_np, self.sample_rate, subtype="PCM_24"  # 24-bit PCM
        )

        print(f"✅ Audio saved successfully")


def benchmark_model(config: dict, checkpoint_path: str, duration: float = 10.0) -> None:
    """Benchmark model performance"""
    print(f"🏃 Benchmarking model performance ({duration}s test audio)")

    # Create inferencer
    inferencer = AudioInferencer(config, checkpoint_path)

    # Generate test audio
    sample_rate = config["data"]["sample_rate"]
    test_samples = int(sample_rate * duration)
    test_audio = torch.randn(test_samples)

    # Warm up
    print("🔥 Warming up...")
    _ = inferencer.process_audio(test_audio, show_progress=False)

    # Benchmark
    print("📊 Running benchmark...")
    start_time = time.time()

    num_runs = 3
    for i in range(num_runs):
        _ = inferencer.process_audio(test_audio, show_progress=False)

    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    rtf = avg_time / duration

    print(f"📊 Average processing time: {avg_time:.3f}s")
    print(f"📊 Real-time factor: {rtf:.3f}x")
    print(f"📊 Throughput: {duration/avg_time:.2f}x real-time")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Concert2Studio Audio Enhancement")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input audio file path"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output audio file path"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="checkpoints/best.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-1.0,
        help="Target LUFS for output limiting",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )

    args = parser.parse_args()

    # Load configuration
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pt"):
                print(f"  - {ckpt}")
        return

    # Run benchmark if requested
    if args.benchmark:
        try:
            benchmark_model(config, args.checkpoint)
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
        return

    # Check input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    # Create inferencer and process file
    try:
        inferencer = AudioInferencer(config, args.checkpoint, args.device)
        inferencer.process_file(args.input, args.output, args.target_lufs)

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
