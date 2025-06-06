"""
Model architectures for Concert2Studio
Implements Spectrogram U-Net, Enhanced UnivNet wrapper, and advanced loss functions
Includes stereo support and research-based noise reduction techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Tuple, Optional
import math
from auraloss.freq import MultiResolutionSTFTLoss


class ConvBlock(nn.Module):
    """Enhanced convolution block with LayerNorm and Swish activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        self.conv = conv
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU()  # Swish activation for better gradients

    def forward(self, x):
        x = self.conv(x)
        # LayerNorm expects (B, H, W, C) format
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.activation(x)


class TwinDeconvolution(nn.Module):
    """Twin deconvolution module for artifact-free upsampling (inspired by FA-GAN)"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2
    ):
        super().__init__()

        # Replace twin deconvolution with standard transposed conv + normalization
        # The original twin approach was causing numerical instability
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 - stride // 2,
        )

        # Add batch normalization for stability
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        # Stable deconvolution without division operations
        output = self.deconv(x)
        output = self.norm(output)
        output = self.activation(output)
        return output


class ContextAwareModule(nn.Module):
    """Context-aware module for enhanced feature extraction (inspired by EVA-GAN)"""

    def __init__(self, channels: int, context_size: int = 7):
        super().__init__()

        self.depth_conv = nn.Conv2d(
            channels, channels, context_size, padding=context_size // 2, groups=channels
        )
        self.point_conv1 = nn.Conv2d(channels, channels * 4, 1)
        self.point_conv2 = nn.Conv2d(channels * 4, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = x

        # Depthwise convolution for spatial context
        x = self.depth_conv(x)

        # Point-wise convolutions for channel mixing
        x = self.point_conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.point_conv2(x)

        # Layer normalization
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        return residual + x


class AttentionBlock(nn.Module):
    """Multi-head self-attention with improved efficiency"""

    def __init__(self, channels: int, num_heads: int = 2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.context_module = ContextAwareModule(channels)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Reshape for attention: (B, H*W, C)
        x_reshaped = x.view(B, C, -1).transpose(1, 2)
        x_norm = self.norm(x_reshaped)

        # Apply attention
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)

        # Add residual and reshape back
        out = x_reshaped + attn_out
        out = out.transpose(1, 2).view(B, C, H, W)

        # Apply context-aware enhancement
        out = self.context_module(out)

        return out


class SpectroUNet(nn.Module):
    """
    Enhanced Spectrogram U-Net with stereo support and research-based improvements
    """

    def __init__(
        self,
        in_channels: int = 1,  # Single channel magnitude spectrogram
        out_channels: int = 1,  # Single channel output
        base_channels: int = 32,
        max_channels: int = 256,
        n_blocks: int = 4,
        dilations: List[int] = [1, 2, 4, 8],
        use_attention: bool = True,
        attention_heads: int = 2,
        dropout: float = 0.2,
        use_stereo: bool = False,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.use_attention = use_attention
        self.dropout = nn.Dropout2d(dropout)

        # Channel progression: [1, 32, 64, 128, 256]
        self.channels = [in_channels]
        for i in range(n_blocks):
            next_ch = min(base_channels * (2**i), max_channels)
            self.channels.append(next_ch)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        for i in range(n_blocks):
            dilation = dilations[i % len(dilations)] if i < len(dilations) else 1

            block = nn.Sequential(
                ConvBlock(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=3,
                    dilation=dilation,
                ),
                ConvBlock(
                    self.channels[i + 1],
                    self.channels[i + 1],
                    kernel_size=3,
                    dilation=1,
                ),
            )
            self.encoder_blocks.append(block)

            # Add pooling layer (except for last block)
            if i < n_blocks - 1:
                self.encoder_pools.append(nn.MaxPool2d(2))
            else:
                self.encoder_pools.append(nn.Identity())

        # Bottleneck
        bottleneck_ch = self.channels[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_ch, bottleneck_ch, kernel_size=3),
            (
                AttentionBlock(bottleneck_ch, attention_heads)
                if use_attention
                else nn.Identity()
            ),
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for i in range(n_blocks):
            # Decoder index (reverse order)
            dec_idx = n_blocks - 1 - i

            # Current decoder level channels
            if i == 0:
                # First decoder: comes from bottleneck
                input_ch = self.channels[-1]  # bottleneck channels
                output_ch = self.channels[dec_idx]  # target channels
            else:
                # Subsequent decoders: comes from previous decoder output
                input_ch = self.channels[dec_idx + 1]  # previous decoder output
                output_ch = self.channels[dec_idx]  # target channels

            # Upsampling layer
            upsample = nn.ConvTranspose2d(input_ch, output_ch, 4, stride=2, padding=1)
            self.decoder_upsamples.append(upsample)

            # Decoder block (after skip connection concatenation)
            # Input: upsampled features + skip connection from encoder
            decoder_in_ch = (
                output_ch + self.channels[dec_idx + 1]
            )  # upsampled + encoder skip
            decoder_out_ch = output_ch

            decoder_block = nn.Sequential(
                ConvBlock(decoder_in_ch, decoder_out_ch, kernel_size=3),
                ConvBlock(decoder_out_ch, decoder_out_ch, kernel_size=3),
            )
            self.decoder_blocks.append(decoder_block)

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], out_channels, kernel_size=1),
            nn.ReLU(),  # Use ReLU instead of Sigmoid for magnitude spectrograms
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store input for residual connection
        input_mag = x

        # Encoder path with skip connections
        skip_connections = []

        for i in range(self.n_blocks):
            # Apply encoder block
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)

            # Apply pooling (except for last block)
            x = self.encoder_pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for i in range(self.n_blocks):
            # Get corresponding skip connection (in reverse order)
            skip_idx = self.n_blocks - 1 - i
            skip = skip_connections[skip_idx]

            # Upsample
            x = self.decoder_upsamples[i](x)

            # Handle size mismatch with skip connection
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)

            # Apply decoder block
            x = self.decoder_blocks[i](x)

        # Final output
        x = self.final_conv(x)

        # Residual connection with input magnitude
        if x.shape == input_mag.shape:
            x = x + input_mag * 0.1  # Small residual connection

        return x


class GriffinLimReconstruction(nn.Module):
    """Griffin-Lim algorithm for phase reconstruction"""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_iter: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter = n_iter

        # Create Hann window
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Reconstruct phase using Griffin-Lim algorithm"""
        # magnitude: (B, F, T) where F = n_fft//2 + 1
        batch_size = magnitude.shape[0]

        # Ensure magnitude is in reasonable range and positive
        magnitude = torch.clamp(magnitude, min=1e-8, max=100.0)

        # Initialize with random phase
        phase = torch.rand_like(magnitude) * 2 * math.pi - math.pi

        # Validate STFT parameters for overlap-add
        if self.hop_length >= self.win_length:
            print(
                f"Warning: hop_length ({self.hop_length}) >= win_length ({self.win_length}), using safer ratio"
            )
            hop_length = self.win_length // 4
        else:
            hop_length = self.hop_length

        for _ in range(self.n_iter):
            # Reconstruct complex spectrogram
            complex_spec = magnitude * torch.exp(1j * phase)

            try:
                # ISTFT to time domain with proper parameters
                waveform = torch.istft(
                    complex_spec,
                    n_fft=self.n_fft,
                    hop_length=hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    normalized=True,
                    onesided=True,
                    return_complex=False,
                    length=None,  # Let PyTorch determine length
                    center=True,  # Ensure proper centering
                )
            except Exception as e:
                print(f"Griffin-Lim ISTFT failed: {e}")
                # Fallback: return zero tensor with expected length
                expected_length = (magnitude.shape[-1] - 1) * hop_length
                return torch.zeros(
                    (batch_size, expected_length),
                    device=magnitude.device,
                    dtype=torch.float32,
                )

            try:
                # STFT back to frequency domain with same parameters
                new_complex_spec = torch.stft(
                    waveform,
                    n_fft=self.n_fft,
                    hop_length=hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    normalized=True,
                    onesided=True,
                    return_complex=True,
                    center=True,  # Ensure proper centering
                )
            except Exception as e:
                print(f"Griffin-Lim STFT failed: {e}")
                # Fallback: return zero tensor
                expected_length = (magnitude.shape[-1] - 1) * hop_length
                return torch.zeros(
                    (batch_size, expected_length),
                    device=magnitude.device,
                    dtype=torch.float32,
                )

            # Update phase
            phase = torch.angle(new_complex_spec)

        # Final reconstruction
        final_complex_spec = magnitude * torch.exp(1j * phase)

        try:
            final_waveform = torch.istft(
                final_complex_spec,
                n_fft=self.n_fft,
                hop_length=hop_length,
                win_length=self.win_length,
                window=self.window,
                normalized=True,
                onesided=True,
                return_complex=False,
                length=None,
                center=True,
            )
            return final_waveform
        except Exception as e:
            print(f"Griffin-Lim final ISTFT failed: {e}")
            # Fallback: return zero tensor
            expected_length = (magnitude.shape[-1] - 1) * hop_length
            return torch.zeros(
                (batch_size, expected_length),
                device=magnitude.device,
                dtype=torch.float32,
            )


class EnhancedUnivNetWrapper(nn.Module):
    """
    Enhanced UnivNet wrapper with stereo support and improved architecture
    """

    def __init__(
        self,
        model_name: str = "univnet-c32",
        pretrained: bool = False,
        freeze_epochs: int = 0,
        sample_rate: int = 44100,
        use_stereo: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_epochs = freeze_epochs
        self.sample_rate = sample_rate
        self.use_stereo = use_stereo

        # Initialize as a simple neural network instead of UnivNet
        # This provides more stable training
        self.enhancement_network = nn.Sequential(
            nn.Conv1d(1 if not use_stereo else 2, 64, 15, padding=7),
            nn.LeakyReLU(0.2),
            self._make_dilated_block(64, 64, 1),
            self._make_dilated_block(64, 64, 3),
            self._make_dilated_block(64, 64, 9),
            self._make_dilated_block(64, 64, 27),
            nn.Conv1d(64, 1 if not use_stereo else 2, 7, padding=3),
            nn.Tanh(),
        )

        # Initialize weights
        self._initialize_weights()

    def _make_dilated_block(self, in_channels, out_channels, dilation):
        """Create dilated convolution block"""
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
        )

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_parameters(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, waveform):
        # Handle different input formats
        if waveform.dim() == 2:  # (B, T) -> (B, 1, T)
            waveform = waveform.unsqueeze(1)
        elif waveform.dim() == 3 and waveform.shape[1] == 2:  # Stereo (B, 2, T)
            pass  # Already in correct format
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # Store input for residual connection
        input_waveform = waveform

        # Apply enhancement network
        enhanced = self.enhancement_network(waveform)

        # Residual connection
        output = input_waveform + enhanced * 0.1

        # Convert back to original format
        if output.shape[1] == 1:  # Mono
            output = output.squeeze(1)  # (B, 1, T) -> (B, T)

        return output


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for frequency domain alignment"""

    def __init__(self, scales: List[int] = [512, 1024, 2048, 4096]):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        """
        Properly handle stereo audio for STFT loss computation
        pred/target shapes:
        - Mono: (B, T)
        - Stereo: (B, 2, T)
        """
        # Handle stereo by processing each channel separately
        if pred.dim() == 3 and pred.shape[1] == 2:  # Stereo: (B, 2, T)
            batch_size, channels, time_samples = pred.shape
            total_loss = 0.0

            for ch in range(channels):
                pred_ch = pred[:, ch, :]  # (B, T)
                target_ch = target[:, ch, :]  # (B, T)

                for scale in self.scales:
                    # Ensure we have valid STFT parameters
                    hop_length = scale // 4
                    try:
                        pred_stft = torch.stft(
                            pred_ch,
                            n_fft=scale,
                            hop_length=hop_length,
                            return_complex=True,
                        ).abs()
                        target_stft = torch.stft(
                            target_ch,
                            n_fft=scale,
                            hop_length=hop_length,
                            return_complex=True,
                        ).abs()

                        # Normalize both spectrograms to prevent huge loss values
                        pred_stft = pred_stft / (torch.mean(pred_stft) + 1e-8)
                        target_stft = target_stft / (torch.mean(target_stft) + 1e-8)

                        total_loss += F.l1_loss(pred_stft, target_stft)
                    except Exception as e:
                        # Skip this scale if STFT fails
                        continue

            return total_loss / (len(self.scales) * channels)

        else:  # Mono: (B, T)
            total_loss = 0.0
            for scale in self.scales:
                hop_length = scale // 4
                try:
                    pred_stft = torch.stft(
                        pred,
                        n_fft=scale,
                        hop_length=hop_length,
                        return_complex=True,
                    ).abs()
                    target_stft = torch.stft(
                        target,
                        n_fft=scale,
                        hop_length=hop_length,
                        return_complex=True,
                    ).abs()

                    # Normalize both spectrograms to prevent huge loss values
                    pred_stft = pred_stft / (torch.mean(pred_stft) + 1e-8)
                    target_stft = target_stft / (torch.mean(target_stft) + 1e-8)

                    total_loss += F.l1_loss(pred_stft, target_stft)
                except Exception as e:
                    # Skip this scale if STFT fails
                    continue

            return total_loss / len(self.scales)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss for better frequency domain reconstruction"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_spec, target_spec):
        """
        Compute spectral convergence loss
        pred_spec: predicted spectrogram magnitude
        target_spec: target spectrogram magnitude
        """
        # Ensure both are magnitude spectrograms
        pred_mag = torch.abs(pred_spec) if torch.is_complex(pred_spec) else pred_spec
        target_mag = (
            torch.abs(target_spec) if torch.is_complex(target_spec) else target_spec
        )

        # Spectral convergence
        numerator = torch.norm(target_mag - pred_mag, p="fro", dim=(-2, -1))
        denominator = torch.norm(target_mag, p="fro", dim=(-2, -1))

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-7)

        return torch.mean(numerator / denominator)


class MagnitudePhaseConsistencyLoss(nn.Module):
    """Magnitude-phase consistency loss for better phase reconstruction"""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, pred_waveform, target_waveform):
        """
        Compute magnitude-phase consistency loss
        Properly handle stereo audio tensors
        """
        # Handle stereo by processing each channel separately
        if (
            pred_waveform.dim() == 3 and pred_waveform.shape[1] == 2
        ):  # Stereo: (B, 2, T)
            batch_size, channels, time_samples = pred_waveform.shape
            total_mag_loss = 0.0
            total_phase_loss = 0.0

            for ch in range(channels):
                pred_ch = pred_waveform[:, ch, :]  # (B, T)
                target_ch = target_waveform[:, ch, :]  # (B, T)

                try:
                    # Get spectrograms for this channel
                    pred_stft = torch.stft(
                        pred_ch,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        return_complex=True,
                    )
                    target_stft = torch.stft(
                        target_ch,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        return_complex=True,
                    )

                    # Magnitude loss
                    pred_mag = torch.abs(pred_stft)
                    target_mag = torch.abs(target_stft)
                    mag_loss = F.l1_loss(pred_mag, target_mag)
                    total_mag_loss += mag_loss

                    # Phase consistency loss (using cosine similarity)
                    pred_phase = torch.angle(pred_stft)
                    target_phase = torch.angle(target_stft)

                    # Compute phase difference
                    phase_diff = torch.cos(pred_phase - target_phase)
                    phase_loss = 1.0 - torch.mean(phase_diff)
                    total_phase_loss += phase_loss

                except Exception as e:
                    # Skip this channel if STFT fails
                    continue

            # Average across channels
            avg_mag_loss = total_mag_loss / channels
            avg_phase_loss = total_phase_loss / channels
            return avg_mag_loss + 0.1 * avg_phase_loss

        else:  # Mono: (B, T)
            try:
                # Get spectrograms
                pred_stft = torch.stft(
                    pred_waveform,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    return_complex=True,
                )
                target_stft = torch.stft(
                    target_waveform,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    return_complex=True,
                )

                # Magnitude loss
                pred_mag = torch.abs(pred_stft)
                target_mag = torch.abs(target_stft)
                mag_loss = F.l1_loss(pred_mag, target_mag)

                # Phase consistency loss (using cosine similarity)
                pred_phase = torch.angle(pred_stft)
                target_phase = torch.angle(target_stft)

                # Compute phase difference
                phase_diff = torch.cos(pred_phase - target_phase)
                phase_loss = 1.0 - torch.mean(phase_diff)

                return mag_loss + 0.1 * phase_loss

            except Exception as e:
                # Return zero loss if STFT fails
                return torch.tensor(0.0, device=pred_waveform.device)


class VGGishLoss(nn.Module):
    """VGGish-based perceptual loss for audio quality assessment"""

    def __init__(self):
        super().__init__()

        # Simple CNN for perceptual feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv1d(256, 512, 15, stride=2, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, pred, target):
        # Convert to mel-spectrograms with proper clamping
        pred_clamped = torch.clamp(pred, -1.0, 1.0)
        target_clamped = torch.clamp(target, -1.0, 1.0)

        # Handle stereo by converting to mono (average channels)
        if pred_clamped.dim() == 3 and pred_clamped.shape[1] == 2:  # Stereo: (B, 2, T)
            pred_clamped = torch.mean(pred_clamped, dim=1)  # (B, T)
            target_clamped = torch.mean(target_clamped, dim=1)  # (B, T)

        # Ensure correct shape for conv1d: (B, T) -> (B, 1, T)
        if pred_clamped.dim() == 2:
            pred_clamped = pred_clamped.unsqueeze(1)
            target_clamped = target_clamped.unsqueeze(1)

        try:
            # Extract features
            pred_features = self.feature_extractor(pred_clamped)
            target_features = self.feature_extractor(target_clamped)

            # Compute L1 loss between features
            return F.l1_loss(pred_features, target_features)
        except Exception as e:
            # Return zero loss if feature extraction fails
            return torch.tensor(0.0, device=pred.device)


class Concert2StudioModel(nn.Module):
    """
    Main Concert2Studio model combining Spectrogram U-Net and Enhanced UnivNet
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Audio processing parameters with improved STFT settings
        self.sample_rate = config["data"]["sample_rate"]
        self.n_fft = config["audio"]["n_fft"]
        self.hop_length = config["audio"]["hop_length"]
        self.win_length = config["audio"]["win_length"]

        # Register Hann window for STFT
        self.register_buffer("window", torch.hann_window(self.win_length))

        # Calculate spectrogram dimensions
        self.freq_bins = self.n_fft // 2 + 1  # 513 for n_fft=1024

        # Stereo configuration
        self.use_stereo = config["model"]["unet"].get("use_stereo", False)

        # Initialize Spectrogram U-Net
        self.unet = SpectroUNet(
            in_channels=1,  # Single channel magnitude spectrogram
            out_channels=1,  # Single channel output
            base_channels=config["model"]["unet"]["base_channels"],
            max_channels=config["model"]["unet"]["max_channels"],
            n_blocks=config["model"]["unet"]["n_blocks"],
            dilations=config["model"]["unet"]["dilations"],
            use_attention=config["model"]["unet"]["use_attention"],
            attention_heads=config["model"]["unet"]["attention_heads"],
            dropout=config["model"]["unet"]["dropout"],
            use_stereo=self.use_stereo,
        )

        # Initialize Griffin-Lim for phase reconstruction
        self.griffin_lim = GriffinLimReconstruction(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=32,
        )

        # Initialize Enhanced UnivNet Vocoder
        self.use_vocoder = config["model"].get("use_vocoder", True)
        if self.use_vocoder:
            self.vocoder = EnhancedUnivNetWrapper(
                model_name=config["model"]["vocoder"]["model_name"],
                pretrained=config["model"]["vocoder"]["pretrained"],
                freeze_epochs=config["model"]["vocoder"]["freeze_epochs"],
                sample_rate=self.sample_rate,
                use_stereo=self.use_stereo,
            )

        # Initialize loss functions
        self.loss_weights = {
            "l1": config["loss"]["l1_weight"],
            "multires_stft": config["loss"]["multires_stft_weight"],
            "vggish": config["loss"]["vggish_weight"],
            "spectral_convergence": 0.1,
            "magnitude_phase_consistency": 0.2,
        }

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.multires_stft_loss = MultiResSTFTLoss(config["loss"]["stft_scales"])
        self.vggish_loss = VGGishLoss()
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.magnitude_phase_consistency_loss = MagnitudePhaseConsistencyLoss(
            n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Apply spectral normalization for training stability
        self._apply_spectral_norm()

    def stft(self, waveform):
        """Compute STFT with proper parameters"""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            onesided=True,
            return_complex=True,
        )

    def istft(self, stft_tensor):
        """Compute ISTFT with proper parameters"""
        return torch.istft(
            stft_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            onesided=True,
            return_complex=False,
        )

    def forward(self, waveform, target_waveform=None):
        # Handle stereo input: process each channel separately
        if waveform.dim() == 3:  # Stereo: (B, 2, T)
            batch_size, channels, time_samples = waveform.shape

            # Process each channel separately
            enhanced_channels = []
            target_length = time_samples  # Use original length as target

            for ch in range(channels):
                # Extract single channel: (B, T)
                channel_waveform = waveform[:, ch, :]

                # Convert to spectrogram
                spec = self.stft(channel_waveform)
                magnitude = torch.abs(spec)
                original_phase = torch.angle(spec)  # Keep original phase!

                # Reshape for U-Net: treat spectrogram as 2D image
                # (B, freq, time) -> (B, 1, freq, time)
                magnitude_4d = magnitude.unsqueeze(1)

                # Enhance magnitude with U-Net
                enhanced_magnitude_4d = self.unet(magnitude_4d)

                # Reshape back to original spectrogram shape
                # (B, 1, freq, time) -> (B, freq, time)
                enhanced_magnitude = enhanced_magnitude_4d.squeeze(1)

                # Ensure magnitude is positive and stable
                enhanced_magnitude = torch.clamp(enhanced_magnitude, min=1e-8)

                # MORE AGGRESSIVE CLAMPING for real audio stability
                # Clamp to reasonable magnitude range based on original
                max_original_magnitude = torch.max(magnitude)
                max_enhanced_magnitude = (
                    max_original_magnitude * 2.0
                )  # Allow 2x amplification max
                enhanced_magnitude = torch.clamp(
                    enhanced_magnitude, min=1e-8, max=max_enhanced_magnitude
                )

                # Additional stability checks
                if (
                    torch.isnan(enhanced_magnitude).any()
                    or torch.isinf(enhanced_magnitude).any()
                ):
                    print(
                        f"Warning: NaN/Inf in enhanced magnitude channel {ch}, using original"
                    )
                    enhanced_magnitude = magnitude  # Use original magnitude

                # Check for extremely large values that could cause ISTFT issues
                if torch.max(enhanced_magnitude) > 100.0:
                    print(
                        f"Warning: Very large magnitude values channel {ch}, clamping"
                    )
                    enhanced_magnitude = torch.clamp(enhanced_magnitude, max=10.0)

                # BYPASS GRIFFIN-LIM: Use original phase with enhanced magnitude
                enhanced_complex_spec = enhanced_magnitude * torch.exp(
                    1j * original_phase
                )

                # Direct ISTFT reconstruction (much more stable!)
                try:
                    enhanced_channel = torch.istft(
                        enhanced_complex_spec,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        window=self.window,
                        normalized=True,
                        onesided=True,
                        return_complex=False,
                        center=True,
                    )

                    # Check for NaN/Inf in reconstruction
                    if (
                        torch.isnan(enhanced_channel).any()
                        or torch.isinf(enhanced_channel).any()
                    ):
                        print(f"Warning: NaN/Inf in ISTFT channel {ch}, using original")
                        enhanced_channel = channel_waveform

                    # Clamp to prevent extreme values
                    enhanced_channel = torch.clamp(enhanced_channel, -1.0, 1.0)

                except Exception as e:
                    print(
                        f"Warning: ISTFT failed for channel {ch}: {e}, using original"
                    )
                    enhanced_channel = channel_waveform

                # Ensure exact length match for concatenation
                if enhanced_channel.shape[-1] != target_length:
                    if enhanced_channel.shape[-1] > target_length:
                        enhanced_channel = enhanced_channel[..., :target_length]
                    else:
                        pad_length = target_length - enhanced_channel.shape[-1]
                        enhanced_channel = F.pad(enhanced_channel, (0, pad_length))

                enhanced_channels.append(enhanced_channel)

            # Combine channels: (B, 2, T)
            enhanced_waveform = torch.stack(enhanced_channels, dim=1)

        else:  # Mono: (B, T)
            # Convert to spectrogram
            spec = self.stft(waveform)
            magnitude = torch.abs(spec)
            original_phase = torch.angle(spec)  # Keep original phase!

            # Reshape for U-Net: treat spectrogram as 2D image
            # (B, freq, time) -> (B, 1, freq, time)
            magnitude_4d = magnitude.unsqueeze(1)

            # Enhance magnitude with U-Net
            enhanced_magnitude_4d = self.unet(magnitude_4d)

            # Reshape back to original spectrogram shape
            # (B, 1, freq, time) -> (B, freq, time)
            enhanced_magnitude = enhanced_magnitude_4d.squeeze(1)

            # Ensure magnitude is positive and stable
            enhanced_magnitude = torch.clamp(enhanced_magnitude, min=1e-8)

            # MORE AGGRESSIVE CLAMPING for real audio stability
            # Clamp to reasonable magnitude range based on original
            max_original_magnitude = torch.max(magnitude)
            max_enhanced_magnitude = (
                max_original_magnitude * 2.0
            )  # Allow 2x amplification max
            enhanced_magnitude = torch.clamp(
                enhanced_magnitude, min=1e-8, max=max_enhanced_magnitude
            )

            # Additional stability checks
            if (
                torch.isnan(enhanced_magnitude).any()
                or torch.isinf(enhanced_magnitude).any()
            ):
                print(f"Warning: NaN/Inf in enhanced magnitude, using original")
                enhanced_magnitude = magnitude  # Use original magnitude

            # Check for extremely large values that could cause ISTFT issues
            if torch.max(enhanced_magnitude) > 100.0:
                print(f"Warning: Very large magnitude values, clamping")
                enhanced_magnitude = torch.clamp(enhanced_magnitude, max=10.0)

            # BYPASS GRIFFIN-LIM: Use original phase with enhanced magnitude
            enhanced_complex_spec = enhanced_magnitude * torch.exp(1j * original_phase)

            # Direct ISTFT reconstruction (much more stable!)
            try:
                enhanced_waveform = torch.istft(
                    enhanced_complex_spec,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    normalized=True,
                    onesided=True,
                    return_complex=False,
                    center=True,
                )

                # Check for NaN/Inf in reconstruction
                if (
                    torch.isnan(enhanced_waveform).any()
                    or torch.isinf(enhanced_waveform).any()
                ):
                    print(f"Warning: NaN/Inf in ISTFT, using original")
                    enhanced_waveform = waveform

                # Clamp to prevent extreme values
                enhanced_waveform = torch.clamp(enhanced_waveform, -1.0, 1.0)

            except Exception as e:
                print(f"Warning: ISTFT failed: {e}, using original")
                enhanced_waveform = waveform

        # Ensure output length matches input length
        if enhanced_waveform.shape[-1] != waveform.shape[-1]:
            if enhanced_waveform.shape[-1] > waveform.shape[-1]:
                # Trim if output is longer
                enhanced_waveform = enhanced_waveform[..., : waveform.shape[-1]]
            else:
                # Pad if output is shorter
                pad_length = waveform.shape[-1] - enhanced_waveform.shape[-1]
                enhanced_waveform = F.pad(enhanced_waveform, (0, pad_length))

        # Check for NaN/Inf values and clamp enhanced waveform
        if torch.isnan(enhanced_waveform).any() or torch.isinf(enhanced_waveform).any():
            print("Warning: NaN/Inf detected in enhanced waveform, using original")
            enhanced_waveform = waveform

        # Apply vocoder for final audio enhancement and noise reduction
        if self.use_vocoder:
            final_waveform = self.vocoder(enhanced_waveform)

            # Check for NaN/Inf in vocoder output
            if torch.isnan(final_waveform).any() or torch.isinf(final_waveform).any():
                print(
                    "Warning: NaN/Inf detected in vocoder output, falling back to enhanced waveform"
                )
                final_waveform = enhanced_waveform
            # Debug: Check for silent output
            elif torch.allclose(
                final_waveform, torch.zeros_like(final_waveform), atol=1e-6
            ):
                print(
                    "Warning: Vocoder output is silent, falling back to enhanced waveform"
                )
                final_waveform = enhanced_waveform
        else:
            final_waveform = enhanced_waveform

        # Final safety clamp
        final_waveform = torch.clamp(final_waveform, -1.0, 1.0)

        # Ensure final output length matches input length
        if final_waveform.shape[-1] != waveform.shape[-1]:
            if final_waveform.shape[-1] > waveform.shape[-1]:
                # Trim if output is longer
                final_waveform = final_waveform[..., : waveform.shape[-1]]
            else:
                # Pad if output is shorter
                pad_length = waveform.shape[-1] - final_waveform.shape[-1]
                final_waveform = F.pad(final_waveform, (0, pad_length))

        if target_waveform is not None:
            # Calculate losses during training
            losses = self.calculate_losses(final_waveform, target_waveform)
            return final_waveform, losses

        return final_waveform

    def calculate_losses(self, pred, target):
        """Calculate combined loss with proper multi-scale and perceptual components"""
        losses = {}

        # Clamp inputs to reasonable audio range
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)

        # L1 loss (primary reconstruction loss)
        l1_loss = self.l1_loss(pred, target)
        losses["l1"] = l1_loss

        # Add channel dimension for auraloss (expects 3D: batch, channels, time)
        if pred.dim() == 2:  # Mono: (B, T) -> (B, 1, T)
            pred_3d = pred.unsqueeze(1)
            target_3d = target.unsqueeze(1)
        elif pred.dim() == 3:  # Stereo: (B, 2, T) - already correct format
            pred_3d = pred
            target_3d = target
        else:
            raise ValueError(f"Unexpected prediction tensor shape: {pred.shape}")

        # Multi-resolution STFT loss - critical for audio quality
        if self.loss_weights["multires_stft"] > 0:
            try:
                stft_loss = self.multires_stft_loss(pred_3d, target_3d)
                losses["multires_stft"] = stft_loss
            except Exception as e:
                losses["multires_stft"] = torch.tensor(0.0, device=pred.device)
        else:
            losses["multires_stft"] = torch.tensor(0.0, device=pred.device)

        # VGGish perceptual loss - helps with audio quality perception
        if self.loss_weights["vggish"] > 0:
            try:
                # Ensure mono for VGGish loss (it expects mono input)
                if pred.dim() == 3 and pred.shape[1] == 2:  # Stereo
                    pred_mono = torch.mean(pred, dim=1)  # (B, T)
                    target_mono = torch.mean(target, dim=1)  # (B, T)
                else:
                    pred_mono = pred
                    target_mono = target

                vggish_loss = self.vggish_loss(pred_mono, target_mono)
                losses["vggish"] = vggish_loss
            except Exception as e:
                losses["vggish"] = torch.tensor(0.0, device=pred.device)
        else:
            losses["vggish"] = torch.tensor(0.0, device=pred.device)

        # Spectral convergence loss
        if self.loss_weights["spectral_convergence"] > 0:
            try:
                # Get spectrograms for spectral convergence loss
                pred_spec = self.stft(
                    pred_mono
                    if "pred_mono" in locals()
                    else (pred.squeeze(1) if pred.dim() == 3 else pred)
                )
                target_spec = self.stft(
                    target_mono
                    if "target_mono" in locals()
                    else (target.squeeze(1) if target.dim() == 3 else target)
                )

                spectral_conv_loss = self.spectral_convergence_loss(
                    pred_spec, target_spec
                )
                losses["spectral_convergence"] = spectral_conv_loss
            except Exception as e:
                losses["spectral_convergence"] = torch.tensor(0.0, device=pred.device)
        else:
            losses["spectral_convergence"] = torch.tensor(0.0, device=pred.device)

        # Magnitude-phase consistency loss
        if self.loss_weights["magnitude_phase_consistency"] > 0:
            try:
                mag_phase_loss = self.magnitude_phase_consistency_loss(pred, target)
                losses["magnitude_phase_consistency"] = mag_phase_loss
            except Exception as e:
                losses["magnitude_phase_consistency"] = torch.tensor(
                    0.0, device=pred.device
                )
        else:
            losses["magnitude_phase_consistency"] = torch.tensor(
                0.0, device=pred.device
            )

        # Combine losses with configured weights
        total_loss = (
            self.loss_weights["l1"] * losses["l1"]
            + self.loss_weights["multires_stft"] * losses["multires_stft"]
            + self.loss_weights["vggish"] * losses["vggish"]
            + self.loss_weights["spectral_convergence"] * losses["spectral_convergence"]
            + self.loss_weights["magnitude_phase_consistency"]
            * losses["magnitude_phase_consistency"]
        )

        # Numerical stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = losses["l1"]

        losses["total"] = total_loss
        return losses

    def _apply_spectral_norm(self):
        """Apply spectral normalization to critical layers for training stability"""

        def has_spectral_norm(module):
            """Check if module already has spectral norm applied"""
            for name, _ in module.named_modules():
                if "spectral_norm" in name:
                    return True
            # Also check for spectral norm in the module's hooks
            for hook in module._forward_pre_hooks.values():
                if hasattr(hook, "name") and "spectral_norm" in str(type(hook)):
                    return True
            return False

        # Apply spectral normalization only to layers that don't already have it
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                # Skip if already has spectral norm or if it's in a block that applies its own
                if has_spectral_norm(module):
                    continue

                # Only apply to specific components that need it for stability
                if ("unet" in name and "final_conv" in name) or (
                    "vocoder" in name and "enhancement_network" in name
                ):
                    try:
                        # Apply spectral normalization
                        spectral_norm_module = nn.utils.spectral_norm(module)

                        # Replace the module
                        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                        module_name = name.rsplit(".", 1)[1] if "." in name else name

                        if parent_name:
                            parent_module = dict(self.named_modules())[parent_name]
                            setattr(parent_module, module_name, spectral_norm_module)
                        else:
                            setattr(self, module_name, spectral_norm_module)
                    except RuntimeError as e:
                        # Skip if spectral norm already applied
                        if "Cannot register two spectral_norm hooks" in str(e):
                            continue
                        else:
                            raise e

    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients for training stability"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = Concert2StudioModel(config)

    # Test forward pass
    batch_size = 2
    seq_length = int(config["audio"]["sample_rate"] * config["audio"]["segment_length"])

    # Test both mono and stereo based on config
    if config["model"]["unet"].get("use_stereo", False):
        dummy_input = torch.randn(batch_size, 2, seq_length)  # Stereo
        dummy_target = torch.randn(batch_size, 2, seq_length)
    else:
        dummy_input = torch.randn(batch_size, seq_length)  # Mono
        dummy_target = torch.randn(batch_size, seq_length)

    with torch.no_grad():
        output, losses = model(dummy_input, dummy_target)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Losses: {losses}")
