"""
Inference script for Concert2Studio
CLI tool for processing concert recordings to studio quality
Includes improved stability and overlap-add processing
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
    """Audio inference with enhanced overlap-add processing and stability"""

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

        # Improved overlap parameters for better reconstruction
        self.overlap_ratio = 0.5  # 50% overlap for better stability
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

        return model

    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file with improved error handling"""
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

            # Normalize with improved stability
            max_val = torch.max(torch.abs(waveform))
            if max_val > 1.0:
                waveform = waveform / (max_val * 1.1)  # Add small headroom
                print("🔄 Normalized audio levels with headroom")
            elif max_val < 0.1:
                # Boost very quiet audio
                waveform = waveform / (max_val * 0.5)
                print("🔄 Boosted quiet audio levels")

            print(
                f"✅ Audio loaded: {waveform.shape} samples ({waveform.shape[-1]/self.sample_rate:.2f}s)"
            )
            return waveform

        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def _apply_window(self, segment: torch.Tensor) -> torch.Tensor:
        """Apply improved Hann window to segment edges for smooth overlap-add"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo: segment shape is (2, T)
            if segment.shape[-1] != self.segment_samples:
                return segment

            # Create improved Hann window for overlap regions
            window = torch.ones_like(segment)

            # Smooth fade in/out over overlap regions
            fade_samples = self.overlap_samples
            fade_in = torch.hann_window(2 * fade_samples, device=segment.device)[
                :fade_samples
            ]
            fade_out = torch.hann_window(2 * fade_samples, device=segment.device)[
                fade_samples:
            ]

            window[:, :fade_samples] = fade_in.unsqueeze(0)
            window[:, -fade_samples:] = fade_out.unsqueeze(0)
        else:
            # Mono: segment shape is (T,)
            if len(segment) != self.segment_samples:
                return segment

            # Create improved Hann window for overlap regions
            window = torch.ones_like(segment)

            # Smooth fade in/out over overlap regions
            fade_samples = self.overlap_samples
            fade_in = torch.hann_window(2 * fade_samples, device=segment.device)[
                :fade_samples
            ]
            fade_out = torch.hann_window(2 * fade_samples, device=segment.device)[
                fade_samples:
            ]

            window[:fade_samples] = fade_in
            window[-fade_samples:] = fade_out

        return segment * window

    def _process_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Process a single audio segment with improved error handling"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            # Stereo processing: segment shape is (2, T)
            original_length = segment.shape[-1]
            if original_length < self.segment_samples:
                padding = self.segment_samples - original_length
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Add batch dimension: (2, T) -> (1, 2, T)
            segment_batch = segment.unsqueeze(0).to(self.device)
        else:
            # Mono processing: segment shape is (T,)
            original_length = len(segment)
            if original_length < self.segment_samples:
                padding = self.segment_samples - original_length
                segment = torch.nn.functional.pad(segment, (0, padding))

            # Add batch dimension: (T,) -> (1, T)
            segment_batch = segment.unsqueeze(0).to(self.device)

        # Process with model
        with torch.no_grad():
            try:
                enhanced_segment = self.model(segment_batch)

                # Check for numerical issues
                if (
                    torch.isnan(enhanced_segment).any()
                    or torch.isinf(enhanced_segment).any()
                ):
                    print(
                        "⚠️  Numerical issues detected in model output, using original segment"
                    )
                    enhanced_segment = segment_batch

                # Clamp to reasonable range
                enhanced_segment = torch.clamp(enhanced_segment, -1.0, 1.0)

            except Exception as e:
                print(f"⚠️  Model processing failed: {e}, using original segment")
                enhanced_segment = segment_batch

        # Remove batch dimension and move to CPU
        if use_stereo:
            enhanced_segment = enhanced_segment.squeeze(0).cpu()  # (1, 2, T) -> (2, T)
        else:
            enhanced_segment = enhanced_segment.squeeze(0).cpu()  # (1, T) -> (T,)

        # Trim to original length
        if use_stereo:
            enhanced_segment = enhanced_segment[..., :original_length]
        else:
            enhanced_segment = enhanced_segment[:original_length]

        return enhanced_segment

    def process_audio(
        self, input_audio: torch.Tensor, show_progress: bool = True
    ) -> torch.Tensor:
        """Process audio with improved overlap-add reconstruction"""
        use_stereo = self.config["model"]["unet"].get("use_stereo", False)

        if use_stereo:
            audio_length = input_audio.shape[-1]
            num_channels = input_audio.shape[0]
        else:
            audio_length = len(input_audio)

        # Calculate number of segments with overlap
        if audio_length <= self.segment_samples:
            # Single segment
            windowed_segment = self._apply_window(input_audio)
            return self._process_segment(windowed_segment)

        # Multiple segments with overlap
        num_segments = (audio_length - self.segment_samples) // self.hop_samples + 1
        if (audio_length - self.segment_samples) % self.hop_samples != 0:
            num_segments += 1

        # Initialize output buffer
        if use_stereo:
            output_audio = torch.zeros(
                (num_channels, audio_length), dtype=input_audio.dtype
            )
            weight_buffer = torch.zeros(
                (num_channels, audio_length), dtype=input_audio.dtype
            )
        else:
            output_audio = torch.zeros(audio_length, dtype=input_audio.dtype)
            weight_buffer = torch.zeros(audio_length, dtype=input_audio.dtype)

        # Process segments with progress bar
        iterator = range(num_segments)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Processing segments")

        for i in iterator:
            start_idx = i * self.hop_samples
            end_idx = min(start_idx + self.segment_samples, audio_length)

            # Extract segment
            if use_stereo:
                segment = input_audio[:, start_idx:end_idx].clone()
            else:
                segment = input_audio[start_idx:end_idx].clone()

            # Apply windowing for smooth overlap
            windowed_segment = self._apply_window(segment)

            # Process segment
            enhanced_segment = self._process_segment(windowed_segment)

            # Add to output with proper overlap-add
            actual_length = (
                enhanced_segment.shape[-1] if use_stereo else len(enhanced_segment)
            )
            actual_end = start_idx + actual_length

            if use_stereo:
                # Get the window weights for proper overlap-add
                if actual_length == self.segment_samples:
                    # Full segment
                    window_weights = torch.ones_like(enhanced_segment)
                    fade_samples = self.overlap_samples
                    if fade_samples > 0:
                        fade_in = torch.hann_window(2 * fade_samples)[:fade_samples]
                        fade_out = torch.hann_window(2 * fade_samples)[fade_samples:]
                        window_weights[:, :fade_samples] = fade_in.unsqueeze(0)
                        window_weights[:, -fade_samples:] = fade_out.unsqueeze(0)
                else:
                    # Partial segment
                    window_weights = torch.ones_like(enhanced_segment)

                output_audio[:, start_idx:actual_end] += (
                    enhanced_segment * window_weights
                )
                weight_buffer[:, start_idx:actual_end] += window_weights
            else:
                # Get the window weights for proper overlap-add
                if actual_length == self.segment_samples:
                    # Full segment
                    window_weights = torch.ones_like(enhanced_segment)
                    fade_samples = self.overlap_samples
                    if fade_samples > 0:
                        fade_in = torch.hann_window(2 * fade_samples)[:fade_samples]
                        fade_out = torch.hann_window(2 * fade_samples)[fade_samples:]
                        window_weights[:fade_samples] = fade_in
                        window_weights[-fade_samples:] = fade_out
                else:
                    # Partial segment
                    window_weights = torch.ones_like(enhanced_segment)

                output_audio[start_idx:actual_end] += enhanced_segment * window_weights
                weight_buffer[start_idx:actual_end] += window_weights

        # Normalize by accumulated weights to complete overlap-add
        # Avoid division by zero
        weight_buffer = torch.clamp(weight_buffer, min=1e-8)
        output_audio = output_audio / weight_buffer

        # Final safety check
        if torch.isnan(output_audio).any() or torch.isinf(output_audio).any():
            print("⚠️  Numerical issues in final output, using original audio")
            return input_audio

        return output_audio

    def process_file(
        self, input_path: str, output_path: str, target_lufs: float = -1.0
    ) -> None:
        """Process audio file with improved error handling and LUFS normalization"""
        start_time = time.time()

        try:
            # Load audio
            input_audio = self._load_audio(input_path)

            print(f"🎵 Processing audio...")

            # Process audio
            enhanced_audio = self.process_audio(input_audio, show_progress=True)

            # Apply LUFS normalization if requested
            if target_lufs > -70:  # Valid LUFS range
                enhanced_audio = self._apply_limiter(enhanced_audio, target_lufs)

            # Save output
            self._save_audio(enhanced_audio, output_path)

            processing_time = time.time() - start_time
            audio_duration = (
                input_audio.shape[-1] / self.sample_rate
                if input_audio.dim() > 1
                else len(input_audio) / self.sample_rate
            )
            rtf = processing_time / audio_duration

            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📊 Real-time factor: {rtf:.2f}x")
            print(f"💾 Output saved to: {output_path}")

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

    def _apply_limiter(self, audio: torch.Tensor, target_lufs: float) -> torch.Tensor:
        """Apply basic limiter and LUFS normalization"""
        # Simple peak limiting
        audio = torch.clamp(audio, -0.95, 0.95)

        # Basic RMS-based loudness adjustment (approximation of LUFS)
        if audio.dim() > 1:  # Stereo
            rms = torch.sqrt(torch.mean(audio**2))
        else:  # Mono
            rms = torch.sqrt(torch.mean(audio**2))

        # Convert target LUFS to approximate RMS (rough approximation)
        target_rms = 10 ** (target_lufs / 20.0)

        if rms > 1e-8:  # Avoid division by zero
            gain = target_rms / rms
            gain = min(gain, 10.0)  # Limit maximum gain
            audio = audio * gain

        return audio

    def _save_audio(self, audio: torch.Tensor, output_path: str) -> None:
        """Save audio with improved format handling"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy for saving
        if audio.dim() > 1:  # Stereo
            audio_np = audio.cpu().numpy()
        else:  # Mono
            audio_np = audio.cpu().numpy()

        # Ensure audio is in the right shape for saving
        if audio_np.ndim == 1:
            # Mono: keep as (T,)
            pass
        elif audio_np.ndim == 2:
            # Stereo: transpose to (T, 2) for soundfile
            audio_np = audio_np.T

        # Clamp to valid range
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Save with appropriate format
        try:
            sf.write(str(output_path), audio_np, self.sample_rate, subtype="PCM_24")
            print(f"💾 Saved as 24-bit audio: {output_path}")
        except Exception as e:
            print(f"⚠️  Failed to save as 24-bit, trying 16-bit: {e}")
            sf.write(str(output_path), audio_np, self.sample_rate, subtype="PCM_16")
            print(f"💾 Saved as 16-bit audio: {output_path}")


def benchmark_model(config: dict, checkpoint_path: str, duration: float = 10.0) -> None:
    """Benchmark model performance with various audio lengths"""
    print(f"🏃 Benchmarking model performance...")

    inferencer = AudioInferencer(config, checkpoint_path)

    # Test different audio lengths
    durations = [1.0, 5.0, 10.0, 30.0]
    use_stereo = config["model"]["unet"].get("use_stereo", False)

    for test_duration in durations:
        if test_duration > duration:
            continue

        print(f"\n⏱️  Testing {test_duration}s audio...")

        # Generate test audio
        num_samples = int(test_duration * config["data"]["sample_rate"])
        if use_stereo:
            test_audio = torch.randn(2, num_samples) * 0.1
        else:
            test_audio = torch.randn(num_samples) * 0.1

        # Benchmark processing
        start_time = time.time()
        enhanced_audio = inferencer.process_audio(test_audio, show_progress=False)
        processing_time = time.time() - start_time

        rtf = processing_time / test_duration
        print(f"📊 Processing time: {processing_time:.3f}s, RTF: {rtf:.3f}x")

        # Memory usage (if on CUDA)
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"🧠 Peak GPU memory: {memory_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Concert2Studio Audio Enhancement")
    parser.add_argument("input", type=str, help="Input audio file path")
    parser.add_argument("output", type=str, help="Output audio file path")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-23.0,
        help="Target LUFS level (-70 to disable)",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    try:
        if args.benchmark:
            benchmark_model(config, args.checkpoint)
        else:
            # Initialize inferencer
            inferencer = AudioInferencer(config, args.checkpoint, args.device)

            # Process file
            inferencer.process_file(args.input, args.output, args.target_lufs)

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
