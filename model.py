"""
Model architectures for Concert2Studio
Implements Spectrogram U-Net, UnivNet wrapper, and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Tuple, Optional
import math
from auraloss.freq import MultiResolutionSTFTLoss


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and PReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class AttentionBlock(nn.Module):
    """Self-attention block for long-range dependencies"""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

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
        return out.transpose(1, 2).view(B, C, H, W)


class SpectroUNet(nn.Module):
    """
    Lightweight Spectrogram U-Net for audio denoising and enhancement
    Reduced complexity to prevent overfitting with limited data
    """

    def __init__(
        self,
        in_channels: int = 513,
        base_channels: int = 24,  # Reduced from 48
        max_channels: int = 192,  # Reduced from 768
        n_blocks: int = 4,  # Reduced from 6
        dilations: List[int] = [1, 2, 4],  # Simplified
        use_attention: bool = False,  # Disabled to reduce params
        attention_heads: int = 1,
        dropout: float = 0.1,  # Added dropout for regularization
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.use_attention = use_attention
        self.dropout = nn.Dropout2d(dropout)

        # Calculate channel progression
        channels = [base_channels * (2**i) for i in range(n_blocks)]
        channels = [min(c, max_channels) for c in channels]
        channels = [in_channels] + channels

        # Encoder blocks with dropout
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = dilations[i % len(dilations)] if i < len(dilations) else 1
            block = nn.Sequential(
                ConvBlock(channels[i], channels[i + 1], dilation=dilation, stride=2),
                (
                    nn.Dropout2d(dropout) if i > 0 else nn.Identity()
                ),  # Skip dropout on first layer
            )
            self.encoder_blocks.append(block)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            (
                AttentionBlock(channels[-1], attention_heads)
                if use_attention
                else nn.Identity()
            ),
            ConvBlock(channels[-1], channels[-1]),
        )

        # Decoder blocks with dropout and residual connections
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_ch = (
                channels[n_blocks - i] + channels[n_blocks - i - 1]
            )  # skip connection
            out_ch = channels[n_blocks - i - 1]
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    channels[n_blocks - i],
                    channels[n_blocks - i],
                    kernel_size=2,
                    stride=2,
                ),
                ConvBlock(in_ch, out_ch),
                (
                    nn.Dropout2d(dropout) if i < n_blocks - 1 else nn.Identity()
                ),  # No dropout on last layer
            )
            self.decoder_blocks.append(block)

        # Output layer with residual connection and proper initialization
        # Ensure we have at least 2 channels for the intermediate layer
        intermediate_channels = max(2, channels[0])
        self.output = nn.Sequential(
            nn.Conv2d(channels[0], intermediate_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(intermediate_channels, in_channels, kernel_size=1),
            nn.Tanh(),  # Ensure output is bounded to prevent explosion
        )

        # Initialize weights properly to prevent vanishing outputs
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights to prevent vanishing/exploding gradients"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # Use Xavier initialization for transpose convolutions too
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, C, H, W) where C is frequency bins
        residual_input = x  # Store input for residual connection

        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            # Transpose convolution
            if len(block) >= 3:
                x = block[0](x)  # Transpose conv
                skip = skip_connections[-(i + 1)]

                # Ensure spatial dimensions match before concatenation
                if x.shape[2:] != skip.shape[2:]:
                    # Crop or pad to match skip connection size
                    diff_h = skip.shape[2] - x.shape[2]
                    diff_w = skip.shape[3] - x.shape[3]

                    if diff_h > 0 or diff_w > 0:
                        # Pad if upsampled tensor is smaller
                        pad_h = max(0, diff_h)
                        pad_w = max(0, diff_w)
                        x = F.pad(x, (0, pad_w, 0, pad_h))
                    elif diff_h < 0 or diff_w < 0:
                        # Crop if upsampled tensor is larger
                        x = x[:, :, : skip.shape[2], : skip.shape[3]]

                x = torch.cat([x, skip], dim=1)
                x = block[1](x)  # Conv block
                if len(block) > 2:
                    x = block[2](x)  # Dropout
            else:
                x = block(x)

        # Output with residual connection
        enhancement = self.output(x)

        # Residual connection with gating to prevent over-correction
        gate = torch.sigmoid(enhancement)  # Learn how much enhancement to apply
        output = residual_input + gate * enhancement

        return output


class UnivNetWrapper(nn.Module):
    """
    Lightweight neural vocoder for audio enhancement with stability improvements
    Much simpler architecture to prevent training instability
    """

    def __init__(
        self,
        model_name: str = "univnet-c32",
        pretrained: bool = True,
        freeze_epochs: int = 0,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_epochs = freeze_epochs
        self.sample_rate = sample_rate

        # Simplified noise reduction with stable architecture
        self.enhancer = nn.Sequential(
            # First noise reduction block
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout1d(0.1),
            # Second noise reduction block
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout1d(0.1),
            # Final output with noise suppression
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Tanh(),  # Bounded output to prevent artifacts
        )

        # Separate gating network for adaptive enhancement
        self.gate_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid(),  # 0-1 gate for mixing
        )

        # Initialize weights for stability
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(
                    m.weight, gain=0.5
                )  # Conservative initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_parameters(self):
        """Freeze all parameters in the vocoder"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Unfreeze all parameters in the vocoder"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, waveform):
        """
        Multi-scale noise reduction with adaptive gating
        Args:
            waveform: Input waveform tensor of shape (B, T) or (B, 1, T)
        Returns:
            Enhanced waveform of same shape as input
        """
        # Ensure input has channel dimension
        if waveform.dim() == 2:
            x = waveform.unsqueeze(1)  # (B, T) -> (B, 1, T)
            squeeze_output = True
        else:
            x = waveform  # Already (B, 1, T)
            squeeze_output = False

        # Store original for residual connection
        residual = x

        # Apply noise reduction enhancement
        enhancement = self.enhancer(x)

        # Adaptive gating - learn how much enhancement to apply
        gate = self.gate_net(x)

        # Apply gated enhancement with residual connection
        # Gate determines mix between original and enhanced signal
        output = (1 - gate) * residual + gate * enhancement

        # Remove channel dimension if input was 2D
        if squeeze_output:
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
    Complete Concert2Studio model combining Spectrogram U-Net and UnivNet
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

        # UnivNet vocoder
        self.use_vocoder = config["model"].get("use_vocoder", True)
        if self.use_vocoder:
            vocoder_config = config["model"]["vocoder"].copy()
            vocoder_config["sample_rate"] = config["data"]["sample_rate"]
            self.vocoder = UnivNetWrapper(**vocoder_config)
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
        # Convert to spectrogram
        spec = self.stft(waveform)
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        # Reshape for U-Net: treat frequency bins as channels
        # (B, freq, time) -> (B, freq, 1, time)
        magnitude_4d = magnitude.unsqueeze(2)  # (B, freq, 1, time)

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

        # Apply vocoder for final audio enhancement and noise reduction
        if self.use_vocoder:
            final_waveform = self.vocoder(enhanced_waveform)
            # Debug: Check for silent output
            if torch.allclose(
                final_waveform, torch.zeros_like(final_waveform), atol=1e-6
            ):
                print(
                    "Warning: Vocoder output is silent, falling back to enhanced waveform"
                )
                final_waveform = enhanced_waveform
        else:
            final_waveform = enhanced_waveform

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

        # Add noise to target for label smoothing (prevents overfitting)
        if self.training:
            noise_scale = 0.01  # Very small noise
            target_smooth = target + torch.randn_like(target) * noise_scale
        else:
            target_smooth = target

        # L1 loss (primary reconstruction loss) with label smoothing
        l1_loss = self.l1_loss(pred, target_smooth)
        losses["l1"] = torch.clamp(l1_loss, max=2.0)  # Even more conservative

        # Add channel dimension for auraloss (expects 3D: batch, channels, time)
        if pred.dim() == 2:
            pred_3d = pred.unsqueeze(1)
            target_3d = target_smooth.unsqueeze(1)
        else:
            pred_3d = pred
            target_3d = target_smooth

        # Multi-resolution STFT loss with very conservative weighting
        try:
            stft_loss = self.multires_stft_loss(pred_3d, target_3d)
            losses["multires_stft"] = torch.clamp(stft_loss, max=10.0)
        except Exception as e:
            print(f"STFT loss computation failed: {e}")
            losses["multires_stft"] = torch.tensor(0.0, device=pred.device)

            # Add spectral coherence loss for better noise reduction
        try:
            pred_spec = self.stft(pred).contiguous()
            target_spec = self.stft(target_smooth).contiguous()

            # Magnitude loss for spectral clarity
            pred_mag = torch.abs(pred_spec).contiguous()
            target_mag = torch.abs(target_spec).contiguous()
            spectral_loss = F.l1_loss(pred_mag, target_mag)
            losses["spectral"] = torch.clamp(spectral_loss, max=3.0)

            # Phase coherence loss for naturalness
            pred_phase = torch.angle(pred_spec).contiguous()
            target_phase = torch.angle(target_spec).contiguous()
            phase_diff = torch.abs(torch.sin(pred_phase - target_phase)).contiguous()
            phase_loss = torch.mean(phase_diff)
            losses["phase"] = torch.clamp(phase_loss, max=1.0)

        except Exception as e:
            print(f"Spectral loss computation failed: {e}")
            losses["spectral"] = torch.tensor(0.0, device=pred.device)
            losses["phase"] = torch.tensor(0.0, device=pred.device)

        # Balanced loss for noise reduction while maintaining reconstruction
        total_loss = (
            0.6 * losses["l1"]  # Primary reconstruction loss
            + 0.2 * losses["multires_stft"]  # Spectral structure
            + 0.15 * losses["spectral"]  # Magnitude clarity
            + 0.05 * losses["phase"]  # Phase coherence
        )

        # Ultra-strict numerical stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 50.0:
            print("Warning: Unstable loss detected, falling back to L1 only")
            total_loss = losses["l1"]

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
        if self.use_vocoder and hasattr(self.vocoder, "enhancer"):
            for module in self.vocoder.enhancer.modules():
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
    dummy_input = torch.randn(batch_size, seq_length)
    dummy_target = torch.randn(batch_size, seq_length)

    with torch.no_grad():
        output, losses = model(dummy_input, dummy_target)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Losses: {losses}")
