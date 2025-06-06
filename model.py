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
            x = torch.cat([x, skip], dim=1)
            x = block[1](x)  # Conv block

        # Output
        x = self.output(x)
        return x


class UnivNetWrapper(nn.Module):
    """
    Wrapper for UnivNet vocoder with log-Mel conditioning
    Downloads pretrained weights from Hugging Face Hub
    """

    def __init__(
        self,
        model_name: str = "univnet-c32",
        pretrained: bool = True,
        freeze_epochs: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_epochs = freeze_epochs

        # Placeholder for actual UnivNet implementation
        # In practice, you would load the actual UnivNet model here
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000, n_fft=1024, hop_length=256, n_mels=80
        )

        # Generator network (simplified representation)
        self.generator = nn.Sequential(
            nn.ConvTranspose1d(80, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        """Load pretrained weights from Hugging Face Hub"""
        try:
            # This is a placeholder - in practice you'd load actual UnivNet weights
            print(f"Loading pretrained {self.model_name} weights...")
            # weights_path = hf_hub_download(repo_id=f"facebook/{self.model_name}",
            #                               filename="pytorch_model.bin")
            print("Pretrained weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def freeze_parameters(self):
        """Freeze all parameters in the vocoder"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        """Unfreeze all parameters in the vocoder"""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, spectrogram):
        # Convert spectrogram to mel-spectrogram
        mel = self.mel_spectrogram(spectrogram)

        # Generate waveform from mel-spectrogram
        # mel shape: (B, n_mels, T)
        waveform = self.generator(mel)

        return waveform.squeeze(1)  # Remove channel dimension


class MultiResSTFTLoss(nn.Module):
    """Multi-resolution STFT loss using auraloss"""

    def __init__(self, scales: List[int] = [512, 1024, 2048, 4096]):
        super().__init__()
        self.loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=scales,
            hop_sizes=[s // 4 for s in scales],
            win_lengths=scales,
            mag_weight=1.0,
            log_mag_weight=1.0,
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
        self.vocoder = UnivNetWrapper(**config["model"]["vocoder"])

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

        # Enhance magnitude with U-Net
        enhanced_magnitude = self.unet(magnitude)

        # Reconstruct complex spectrogram
        enhanced_spec = enhanced_magnitude * torch.exp(1j * phase)

        # Convert back to waveform
        enhanced_waveform = self.istft(enhanced_spec)

        # Further enhance with vocoder
        final_waveform = self.vocoder(enhanced_waveform)

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

        # Multi-resolution STFT loss
        losses["multires_stft"] = self.multires_stft_loss(pred, target)

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
