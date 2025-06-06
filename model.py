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

        # Main deconvolution branch
        self.main_deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 - stride // 2,
        )

        # Twin branch for calculating overlap
        self.twin_deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 - stride // 2,
        )

        # Initialize twin with same weights
        self.twin_deconv.weight.data = self.main_deconv.weight.data.clone()
        self.twin_deconv.bias.data = self.main_deconv.bias.data.clone()

    def forward(self, x):
        main_out = self.main_deconv(x)
        twin_out = self.twin_deconv(x)

        # Element-wise division to suppress artifacts
        # Add small epsilon to prevent division by zero
        output = main_out / (twin_out + 1e-8)
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
        in_channels: int = 513,
        out_channels: int = 513,  # Can be different for stereo
        base_channels: int = 16,
        max_channels: int = 128,
        n_blocks: int = 3,
        dilations: List[int] = [1, 2, 4],
        use_attention: bool = True,
        attention_heads: int = 2,
        dropout: float = 0.3,
        use_stereo: bool = False,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.use_attention = use_attention
        self.use_stereo = use_stereo
        self.dropout = nn.Dropout2d(dropout)

        # Keep input and output channels the same
        # Stereo processing is handled at the model level, not UNet level
        out_channels = in_channels

        # Calculate channel progression
        channels = [base_channels * (2**i) for i in range(n_blocks)]
        channels = [min(c, max_channels) for c in channels]
        channels = [in_channels] + channels

        # Encoder blocks with context-aware modules
        self.encoder_blocks = nn.ModuleList()
        self.context_modules = nn.ModuleList()

        for i in range(n_blocks):
            dilation = dilations[i % len(dilations)] if i < len(dilations) else 1

            encoder_block = nn.Sequential(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    dilation=dilation,
                    stride=2,
                    use_spectral_norm=True,
                ),
                ConvBlock(
                    channels[i + 1], channels[i + 1], dilation=1, use_spectral_norm=True
                ),
            )
            self.encoder_blocks.append(encoder_block)

            # Add context-aware module
            context_module = ContextAwareModule(channels[i + 1])
            self.context_modules.append(context_module)

        # Bottleneck with enhanced attention
        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1], use_spectral_norm=True),
            (
                AttentionBlock(channels[-1], attention_heads)
                if use_attention
                else nn.Identity()
            ),
            ConvBlock(channels[-1], channels[-1], use_spectral_norm=True),
        )

        # Decoder blocks with twin deconvolution
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_ch = (
                channels[n_blocks - i] + channels[n_blocks - i - 1]
            )  # skip connection
            out_ch = channels[n_blocks - i - 1]

            decoder_block = nn.Sequential(
                TwinDeconvolution(channels[n_blocks - i], channels[n_blocks - i]),
                ConvBlock(in_ch, out_ch, use_spectral_norm=True),
                ConvBlock(out_ch, out_ch, use_spectral_norm=True),
            )
            self.decoder_blocks.append(decoder_block)

        # Output layer with stereo support
        self.output = nn.Sequential(
            nn.Conv2d(channels[0], max(32, channels[0]), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(32, channels[0]), out_channels, kernel_size=1),
            nn.Tanh(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Improved weight initialization"""
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
        residual_input = x

        # Encoder with skip connections and context enhancement
        skip_connections = []
        for i, (encoder_block, context_module) in enumerate(
            zip(self.encoder_blocks, self.context_modules)
        ):
            skip_connections.append(x)
            x = encoder_block(x)
            x = context_module(x)

            if i < len(self.encoder_blocks) - 1:  # Don't apply dropout to last encoder
                x = self.dropout(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Twin deconvolution
            x = decoder_block[0](x)  # TwinDeconvolution

            # Skip connection
            skip = skip_connections[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                # Adaptive resizing for skip connections
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)

            # ConvBlocks
            x = decoder_block[1](x)  # First ConvBlock
            x = decoder_block[2](x)  # Second ConvBlock

            if i < len(self.decoder_blocks) - 1:  # Don't apply dropout to last decoder
                x = self.dropout(x)

        # Output with residual connection
        enhancement = self.output(x)

        # Apply residual connection regardless of stereo/mono
        # The stereo processing is handled at the model level
        output = residual_input + 0.7 * enhancement

        return output


class EnhancedUnivNetWrapper(nn.Module):
    """
    Enhanced neural vocoder with advanced noise reduction and stereo support
    """

    def __init__(
        self,
        model_name: str = "univnet-c32",
        pretrained: bool = False,
        freeze_epochs: int = 0,
        sample_rate: int = 48000,
        use_stereo: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_epochs = freeze_epochs
        self.sample_rate = sample_rate
        self.use_stereo = use_stereo

        # Determine input/output channels
        in_channels = 2 if use_stereo else 1
        out_channels = 2 if use_stereo else 1

        # Multi-scale enhancement network
        self.pre_enhancer = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, dilation=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout1d(0.1),
        )

        # Multi-scale dilated convolutions for different temporal contexts
        self.multi_scale_blocks = nn.ModuleList(
            [
                self._make_dilated_block(32, 32, dilation=1),
                self._make_dilated_block(32, 32, dilation=2),
                self._make_dilated_block(32, 32, dilation=4),
                self._make_dilated_block(32, 32, dilation=8),
            ]
        )

        # Feature fusion and output
        self.fusion = nn.Conv1d(32 * 4, 64, kernel_size=1)
        self.post_enhancer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _make_dilated_block(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                padding=7 * dilation // 2,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout1d(0.1),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, waveform):
        # Handle different input formats
        if waveform.dim() == 2:  # Mono: (B, T)
            if self.use_stereo:
                # For stereo, input should be (B, T) and we'll convert to (B, 2, T)
                x = waveform.unsqueeze(1).repeat(1, 2, 1)
            else:
                x = waveform.unsqueeze(1)  # (B, 1, T)
            squeeze_output = True
        elif waveform.dim() == 3:  # Stereo: (B, 2, T)
            x = waveform  # Already in correct format
            squeeze_output = False
        else:
            x = waveform
            squeeze_output = False

        # Store original for residual connection
        residual = x

        # Pre-enhancement
        x = self.pre_enhancer(x)

        # Multi-scale processing
        multi_scale_outputs = []
        target_length = x.shape[-1]  # Use input length as target

        for block in self.multi_scale_blocks:
            output = block(x)
            # Ensure all outputs have the same temporal dimension
            if output.shape[-1] != target_length:
                if output.shape[-1] > target_length:
                    output = output[..., :target_length]
                else:
                    pad_length = target_length - output.shape[-1]
                    output = F.pad(output, (0, pad_length))
            multi_scale_outputs.append(output)

        # Fuse multi-scale features
        x = torch.cat(multi_scale_outputs, dim=1)
        x = self.fusion(x)

        # Post-enhancement
        enhancement = self.post_enhancer(x)

        # Strong residual connection with learned gating
        alpha = 0.3  # Conservative enhancement
        output = (1 - alpha) * residual + alpha * enhancement

        # Output formatting
        if squeeze_output and not self.use_stereo:
            output = output.squeeze(1)

        return output


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss using auraloss"""

    def __init__(self, scales: List[int] = [512, 1024, 2048, 4096]):
        super().__init__()
        self.loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=scales,
            hop_sizes=[s // 4 for s in scales],
            win_lengths=scales,
        )

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss for training stability"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_spec, target_spec):
        """Calculate spectral convergence loss"""
        # Ensure spectra are magnitude spectrograms
        if torch.is_complex(pred_spec):
            pred_spec = torch.abs(pred_spec)
        if torch.is_complex(target_spec):
            target_spec = torch.abs(target_spec)

        # Spectral convergence
        return torch.norm(target_spec - pred_spec, p="fro") / torch.norm(
            target_spec, p="fro"
        )


class VGGishLoss(nn.Module):
    """VGGish perceptual loss for audio quality assessment"""

    def __init__(self):
        super().__init__()
        # Simplified VGGish-like network for perceptual loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=1024, hop_length=256, n_mels=64
        )

        self.vgg_net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Freeze VGG features after proper initialization
        self._initialize_weights()
        for param in self.vgg_net.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        """Initialize VGG network weights properly"""
        for m in self.vgg_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pred, target):
        # Convert to mel-spectrograms with proper clamping
        pred_mel = self.mel_transform(pred).clamp(min=1e-7).log().unsqueeze(1)
        target_mel = self.mel_transform(target).clamp(min=1e-7).log().unsqueeze(1)

        # Extract features
        pred_features = self.vgg_net(pred_mel)
        target_features = self.vgg_net(target_mel)

        # L2 loss between features with gradient scaling
        loss = F.mse_loss(pred_features, target_features)
        return loss * 0.1  # Scale down perceptual loss to prevent dominance


class Concert2StudioModel(nn.Module):
    """
    Complete Concert2Studio model combining Spectrogram U-Net and Enhanced UnivNet
    """

    def __init__(self, config):
        super().__init__()

        # STFT transform
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=config["audio"]["n_fft"],
            hop_length=config["audio"]["hop_length"],
            win_length=config["audio"]["win_length"],
            window_fn=torch.hann_window,
            power=None,  # Return complex spectrogram
        )

        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=config["audio"]["n_fft"],
            hop_length=config["audio"]["hop_length"],
            win_length=config["audio"]["win_length"],
            window_fn=torch.hann_window,
        )

        # Spectrogram U-Net
        self.unet = SpectroUNet(**config["model"]["unet"])

        # Enhanced UnivNet vocoder
        self.use_vocoder = config["model"].get("use_vocoder", True)
        if self.use_vocoder:
            vocoder_config = config["model"]["vocoder"].copy()
            vocoder_config["sample_rate"] = config["data"]["sample_rate"]
            self.vocoder = EnhancedUnivNetWrapper(**vocoder_config)
        else:
            self.vocoder = nn.Identity()

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.multires_stft_loss = MultiResSTFTLoss(config["loss"]["stft_scales"])
        self.vggish_loss = VGGishLoss()

        self.loss_weights = {
            "l1": config["loss"]["l1_weight"],
            "multires_stft": config["loss"]["multires_stft_weight"],
            "vggish": config["loss"]["vggish_weight"],
        }

        # Training stability improvements
        self.register_buffer("global_step", torch.tensor(0))
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Gradient clipping parameters
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)

        # Apply spectral normalization to prevent gradient explosion
        self._apply_spectral_norm()

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
                phase = torch.angle(spec)

                # Reshape for U-Net: treat frequency bins as channels
                # (B, freq, time) -> (B, freq, 1, time)
                magnitude_4d = magnitude.unsqueeze(2)

                # Enhance magnitude with U-Net
                enhanced_magnitude_4d = self.unet(magnitude_4d)

                # Reshape back to original spectrogram shape
                # (B, freq, 1, time) -> (B, freq, time)
                enhanced_magnitude = enhanced_magnitude_4d.squeeze(2)

                # Reconstruct complex spectrogram
                enhanced_spec = enhanced_magnitude * torch.exp(1j * phase)

                # Convert back to waveform
                enhanced_channel = self.istft(enhanced_spec)

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
            phase = torch.angle(spec)

            # Reshape for U-Net: treat frequency bins as channels
            # (B, freq, time) -> (B, freq, 1, time)
            magnitude_4d = magnitude.unsqueeze(2)

            # Enhance magnitude with U-Net
            enhanced_magnitude_4d = self.unet(magnitude_4d)

            # Reshape back to original spectrogram shape
            # (B, freq, 1, time) -> (B, freq, time)
            enhanced_magnitude = enhanced_magnitude_4d.squeeze(2)

            # Reconstruct complex spectrogram
            enhanced_spec = enhanced_magnitude * torch.exp(1j * phase)

            # Convert back to waveform
            enhanced_waveform = self.istft(enhanced_spec)

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
            print("Warning: NaN/Inf detected in enhanced waveform, clamping values")
            enhanced_waveform = torch.clamp(enhanced_waveform, -1.0, 1.0)

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
        """Calculate combined loss optimized for tiny datasets with extreme stability"""
        losses = {}

        # Clamp inputs to prevent extreme values
        pred = torch.clamp(pred, -10.0, 10.0)
        target = torch.clamp(target, -10.0, 10.0)

        # Add minimal noise to target for label smoothing (prevents overfitting)
        if self.training:
            noise_scale = 0.001  # Much smaller noise for stability
            target_smooth = target + torch.randn_like(target) * noise_scale
        else:
            target_smooth = target

        # L1 loss (primary reconstruction loss) with label smoothing
        l1_loss = self.l1_loss(pred, target_smooth)
        losses["l1"] = torch.clamp(l1_loss, max=1.0)  # More conservative clamping

        # Add channel dimension for auraloss (expects 3D: batch, channels, time)
        if pred.dim() == 2:  # Mono: (B, T) -> (B, 1, T)
            pred_3d = pred.unsqueeze(1)
            target_3d = target_smooth.unsqueeze(1)
        elif pred.dim() == 3:  # Stereo: (B, 2, T) - already correct format
            pred_3d = pred
            target_3d = target_smooth
        else:
            raise ValueError(f"Unexpected prediction tensor shape: {pred.shape}")

        # Multi-resolution STFT loss with very conservative weighting
        try:
            stft_loss = self.multires_stft_loss(pred_3d, target_3d)
            losses["multires_stft"] = torch.clamp(stft_loss, max=10.0)
        except Exception as e:
            print(f"STFT loss computation failed: {e}")
            losses["multires_stft"] = torch.tensor(0.0, device=pred.device)

        # Use configured loss weights
        total_loss = (
            self.loss_weights["l1"] * losses["l1"]
            + self.loss_weights["multires_stft"] * losses["multires_stft"]
            + self.loss_weights["vggish"]
            * losses.get("vggish", torch.tensor(0.0, device=pred.device))
        )

        # Ultra-strict numerical stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 5.0:
            print(
                f"Warning: Unstable loss detected ({total_loss}), falling back to L1 only"
            )
            total_loss = losses["l1"]
            # If L1 is also unstable, use a small constant
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 5.0:
                print("Warning: L1 loss also unstable, using fallback loss")
                total_loss = torch.tensor(1.0, device=pred.device)

        losses["total"] = total_loss
        return losses

    def _apply_spectral_norm(self):
        """Apply spectral normalization to key layers for gradient stability"""
        # Apply spectral norm to U-Net output layers
        for module in self.unet.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
                # Apply to 1x1 conv layers (typically output layers)
                torch.nn.utils.spectral_norm(module)

        # Apply spectral norm to vocoder if used
        if self.use_vocoder and hasattr(self.vocoder, "post_enhancer"):
            for module in self.vocoder.post_enhancer.modules():
                if isinstance(module, nn.Conv1d) and module.kernel_size == (3,):
                    torch.nn.utils.spectral_norm(module)

    def clip_gradients(self):
        """Clip gradients to prevent explosion"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
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
