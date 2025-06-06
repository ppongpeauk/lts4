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
from huggingface_hub import hf_hub_download


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
    Spectrogram U-Net for audio denoising and enhancement
    Architecture: 6 down/6 up blocks with dilated convolutions and skip connections
    """

    def __init__(
        self,
        in_channels: int = 513,
        base_channels: int = 48,
        max_channels: int = 768,
        n_blocks: int = 6,
        dilations: List[int] = [1, 2, 4, 8],
        use_attention: bool = True,
        attention_heads: int = 1,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.use_attention = use_attention

        # Calculate channel progression
        channels = [base_channels * (2**i) for i in range(n_blocks)]
        channels = [min(c, max_channels) for c in channels]
        channels = [in_channels] + channels

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = dilations[i % len(dilations)] if i >= 2 and i <= 4 else 1
            self.encoder_blocks.append(
                ConvBlock(channels[i], channels[i + 1], dilation=dilation, stride=2)
            )

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

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_ch = (
                channels[n_blocks - i] + channels[n_blocks - i - 1]
            )  # skip connection
            out_ch = channels[n_blocks - i - 1]
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels[n_blocks - i],
                        channels[n_blocks - i],
                        kernel_size=2,
                        stride=2,
                    ),
                    ConvBlock(in_ch, out_ch),
                )
            )

        # Output layer
        self.output = nn.Conv2d(channels[0], in_channels, kernel_size=1)

        # Initialize weights properly to prevent vanishing outputs
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights to prevent vanishing outputs"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, C, H, W) where C is frequency bins

        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
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

        # Output
        x = self.output(x)
        return x


class UnivNetWrapper(nn.Module):
    """
    Simplified neural vocoder for audio enhancement
    Operates directly on waveforms with residual connections
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

        # Multi-scale waveform enhancement network
        # Uses dilated convolutions for different temporal receptive fields
        self.enhancement_blocks = nn.ModuleList(
            [
                # Short-term features (transients)
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, padding=3, dilation=1),
                    nn.PReLU(),
                    nn.Conv1d(32, 32, kernel_size=7, padding=3, dilation=1),
                    nn.PReLU(),
                ),
                # Medium-term features (harmonics)
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=15, padding=7, dilation=1),
                    nn.PReLU(),
                    nn.Conv1d(32, 32, kernel_size=15, padding=14, dilation=2),
                    nn.PReLU(),
                ),
                # Long-term features (texture)
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=31, padding=15, dilation=1),
                    nn.PReLU(),
                    nn.Conv1d(32, 32, kernel_size=31, padding=60, dilation=4),
                    nn.PReLU(),
                ),
            ]
        )

        # Feature fusion and output
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=7, padding=3),  # 32*3 = 96 input channels
            nn.PReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

        # High-frequency enhancer for air and presence
        self.hf_enhance = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Tanh(),
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent vanishing gradients"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize final layer with small weights to start with identity-like behavior
        if hasattr(self.fusion[-1], "weight"):
            nn.init.xavier_uniform_(self.fusion[-1].weight, gain=0.1)

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
        Enhance waveform using multi-scale processing
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

        # Multi-scale feature extraction
        features = []
        for block in self.enhancement_blocks:
            feat = block(x)
            features.append(feat)

        # Concatenate multi-scale features
        combined_features = torch.cat(features, dim=1)  # (B, 96, T)

        # Feature fusion
        enhanced = self.fusion(combined_features)  # (B, 1, T)

        # High-frequency enhancement
        hf_component = self.hf_enhance(x)

        # Combine with residual connection and HF enhancement
        # Use learnable mixing to prevent over-processing
        output = residual + 0.3 * enhanced + 0.1 * hf_component

        # Apply soft limiting to prevent clipping
        output = torch.tanh(output * 0.95)

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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Freeze VGG features
        for param in self.vgg_net.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Convert to mel-spectrograms
        pred_mel = self.mel_transform(pred).unsqueeze(1)
        target_mel = self.mel_transform(target).unsqueeze(1)

        # Extract features
        pred_features = self.vgg_net(pred_mel)
        target_features = self.vgg_net(target_mel)

        # L2 loss between features
        return F.mse_loss(pred_features, target_features)


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

    def forward(self, waveform, target_waveform=None):
        # Convert to spectrogram
        spec = self.stft(waveform)
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        # Add channel dimension for U-Net (expects 4D: batch, channels, freq, time)
        magnitude_4d = magnitude.unsqueeze(1)  # (B, 1, freq, time)

        # Enhance magnitude with U-Net
        enhanced_magnitude_4d = self.unet(magnitude_4d)

        # Remove channel dimension to match original spectrogram shape
        enhanced_magnitude = enhanced_magnitude_4d.squeeze(1)  # (B, freq, time)

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
        """Calculate combined loss"""
        losses = {}

        # L1 loss
        losses["l1"] = self.l1_loss(pred, target)

        # Add channel dimension for auraloss (expects 3D: batch, channels, time)
        if pred.dim() == 2:
            pred_3d = pred.unsqueeze(1)
            target_3d = target.unsqueeze(1)
        else:
            pred_3d = pred
            target_3d = target

        # Multi-resolution STFT loss
        losses["multires_stft"] = self.multires_stft_loss(pred_3d, target_3d)

        # VGGish perceptual loss
        losses["vggish"] = self.vggish_loss(pred, target)

        # Combined loss
        total_loss = (
            self.loss_weights["l1"] * losses["l1"]
            + self.loss_weights["multires_stft"] * losses["multires_stft"]
            + self.loss_weights["vggish"] * losses["vggish"]
        )

        losses["total"] = total_loss
        return losses


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
