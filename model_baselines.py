from typing import Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from config import NORMALIZE_OUTPUTS, SEQUENCE_LENGTH
except ImportError:
    NORMALIZE_OUTPUTS = False
    SEQUENCE_LENGTH = 1

class SpatialCrossAttention(nn.Module):
    """
    Spatial Cross-Attention module.
    Uses dynamic features (Query) to highlight relevant regions in static features (Key/Value).
    """
    def __init__(self, static_channels: int, dynamic_channels: int):
        super().__init__()

        inter_channels = max(static_channels // 2, 1)

        self.W_g = nn.Sequential(
            nn.Conv2d(dynamic_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(static_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, static_feat: torch.Tensor, dynamic_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_feat: High-res static features (B, C_s, H, W)
            dynamic_feat: Upsampled dynamic features (B, C_d, H, W)
        Returns:
            Fused features (B, C_s + C_d, H, W)
        """
        g = self.W_g(dynamic_feat)
        x = self.W_x(static_feat)

        psi = self.relu(g + x)
        attn_map = self.psi(psi)

        weighted_static = static_feat * attn_map

        return torch.cat([weighted_static, dynamic_feat], dim=1)


class AttentionGate(nn.Module):
    """Attention Gate for skip connections in Attention U-Net."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in gating signal (from decoder)
            F_l: Number of channels in skip connection (from encoder)
            F_int: Number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder (lower resolution, upsampled)
            x: Skip connection from encoder
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class NaiveCNN(nn.Module):
    """
    A configurable convolutional neural network that consumes multiple image layers (channels)
    and predicts a single image layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: Sequence[int] = (64, 128, 64),
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the NaiveCNN.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: 1)
            hidden_channels: Sequence of hidden layer channel sizes
            kernel_size: Size of convolutional kernels
            padding: Padding for convolutions (use 1 for same-size output with kernel_size=3)
            dropout_rate: Dropout probability for regularization (default: 0.3)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate

        layers = []
        prev_channels = in_channels

        for hidden_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_ch, kernel_size=kernel_size, padding=padding, padding_mode='reflect'),
                nn.LeakyReLU(0.02, inplace=True),
                nn.BatchNorm2d(hidden_ch),
                nn.Dropout2d(p=dropout_rate)
            ])
            prev_channels = hidden_ch

        layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.model(x)

class UNet(nn.Module):
    """
    UNet architecture for image-to-image tasks.
    Features encoder-decoder structure with skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        use_transpose_conv: bool = False,
        norm_layer: Optional[type] = None,
    ):
        """
        Initialize the UNet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: 1)
            base_channels: Number of channels in first layer (doubled at each downsampling)
            depth: Number of downsampling/upsampling stages (default: 4)
            use_transpose_conv: If True, use ConvTranspose2d for upsampling instead of Upsample + Conv2d
            norm_layer: Normalization layer to use (default: nn.BatchNorm2d). Can pass nn.GroupNorm, etc.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.use_transpose_conv = use_transpose_conv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = base_channels
        prev_channels = in_channels

        for i in range(depth):
            self.encoders.append(self._make_conv_block(prev_channels, channels, dropout_rate=0.05))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = channels
            channels *= 2

        self.bottleneck = self._make_conv_block(prev_channels, channels, dropout_rate=0.2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        for i in range(depth):
            if use_transpose_conv:
                self.upsamples.append(nn.Sequential(
                    nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2),
                    nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, padding_mode='reflect')
                ))
            else:
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, padding_mode='reflect')
                ))
            self.decoders.append(self._make_conv_block(channels, channels // 2, dropout_rate=0.00))
            self.attention_gates.append(AttentionGate(F_g=channels // 2, F_l=channels // 2, F_int=channels // 4))
            channels //= 2

        self.shared_features = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            norm_layer(channels),
            nn.SiLU(inplace=True),
        )

        self.output_head1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1)
        )

        self._init_weights()

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float
    ) -> nn.Module:
        """Create a convolutional block with two conv layers, using LeakyReLU activation."""
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        layers.append(self.norm_layer(out_channels))
        layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        layers.append(self.norm_layer(out_channels))
        layers.append(nn.SiLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        else:
            layers.append(nn.Identity())

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for LeakyReLU."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, landmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the UNet.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            landmask: Optional landmask tensor of shape (batch_size, 1, height, width) or (height, width)
                      to gate the output. Values should be 0 (water) or 1 (land).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        skip_connections = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i, (upsample, decoder, attn_gate) in enumerate(zip(self.upsamples, self.decoders, self.attention_gates)):
            x = upsample(x)

            skip = skip_connections[i]
            if x.shape != skip.shape:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2])

            skip = attn_gate(g=x, x=skip)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        x = self.shared_features(x)

        out1 = self.output_head1(x)

        if NORMALIZE_OUTPUTS:
            out1 = torch.sigmoid(out1)
        else:
            out1 = F.softplus(out1, beta=4.0)

        if landmask is not None:
            # Ensure landmask has the right shape (batch_size, 1, height, width)
            if landmask.dim() == 2:
                # Shape: (height, width) -> (1, 1, height, width)
                landmask = landmask.unsqueeze(0).unsqueeze(0)
            elif landmask.dim() == 3:
                # Shape: (batch_size, height, width) -> (batch_size, 1, height, width)
                landmask = landmask.unsqueeze(1)

            landmask = landmask.to(out1.device)

            # This will zero out predictions over water (landmask=0)
            out1 = out1 * landmask

        return out1


class R2D2ResidualBlock(nn.Module):
    """
    Residual Block for R2D2Unet based on the provided structure.
    Formula: Ri(h) = Conv ◦ Do ◦ swish ◦ FiLM(e) ◦ GN ◦ Conv ◦ swish ◦ GN(h) + h
    """
    def __init__(self, channels, condition_dim=None, dropout_rate=0.1):
        super().__init__()
        num_groups = min(32, channels)

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')

        # FiLM conditioning projection (only one, applied after 2nd GN)
        self.film_proj = None
        if condition_dim is not None:
            self.film_proj = nn.Linear(condition_dim, channels * 2)

            # Initialize to identity modulation (scale=0 -> 1+scale=1, shift=0)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, condition=None):
        """
        Forward pass.
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)

        # Apply FiLM (after 2nd GN, before 2nd swish)
        if self.film_proj is not None and condition is not None:
            emb = self.film_proj(condition) # (B, 2*C)
            emb = emb.unsqueeze(-1).unsqueeze(-1) # (B, 2*C, 1, 1)
            scale, shift = torch.chunk(emb, 2, dim=1) # (B, C, 1, 1)
            h = h * (1 + scale) + shift

        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


class R2D2UNet(nn.Module):
    """
    R2D2 U-Net architecture based on the paper (Section S1.4).

    Architecture flow:
    1. Encoder: Input → 128 channels
    2. Downsampling: 128 → 256 → 512 → 768 (3 levels, 4 residual blocks each)
    3. Upsampling: 768 → 512 → 256 → 128 (3 levels, 4 residual blocks each)
    4. Decoder: 128 → 64 → output_channels

    Uses same API as UNet for easy drop-in replacement.
    """

    class PixelShuffleUpsample(nn.Module):
        """
        Pixel shuffle upsampling as described in R2D2 paper (S1.4.3).
        UConv3: applies 3×3 conv that multiplies channels by 4, then reshapes to double spatial dimensions.
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            # Multiply channels by 4 for 2x upsampling in both dimensions
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, padding_mode='reflect')
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input tensor (B, C_in, H, W)
            Returns:
                Upsampled tensor (B, C_out, 2H, 2W)
            """
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            return x

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        use_transpose_conv: bool = False,
        dropout_rate: float = 0.1,
        condition_dim: int = None,
    ):
        """
        Initialize the R2D2Unet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: 1)
            base_channels: Kept for API compatibility with UNet (not used)
            depth: Kept for API compatibility with UNet (not used)
            use_transpose_conv: Kept for API compatibility with UNet (not used)
            dropout_rate: Dropout rate in residual blocks (default: 0.1)
            condition_dim: Dimension of conditioning vector (for FiLM)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.condition_dim = condition_dim

        # Channel progression as per R2D2 paper: 128 → 256 → 512 → 768
        self.channel_progression = [128, 256, 512, 768]

        # ====================================================================
        # ENCODING (S1.4.1)
        # ====================================================================
        # Dealiasing convolutional layer - project input to 128 channels
        self.encoder = nn.Conv2d(in_channels, 128, kernel_size=7, padding=3, padding_mode='reflect')

        # ====================================================================
        # DOWNSAMPLING STACK (S1.4.2)
        # ====================================================================
        self.down_convs = nn.ModuleList()
        self.down_residual_blocks = nn.ModuleList()

        for i in range(3):
            in_ch = self.channel_progression[i]
            out_ch = self.channel_progression[i + 1]

            down_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
            self.down_convs.append(down_conv)

            residual_blocks = nn.ModuleList([
                R2D2ResidualBlock(out_ch, condition_dim=condition_dim, dropout_rate=dropout_rate)
                for _ in range(4)
            ])
            self.down_residual_blocks.append(residual_blocks)

        # ====================================================================
        # UPSAMPLING STACK (S1.4.3)
        # ====================================================================
        self.up_convs = nn.ModuleList()
        self.up_residual_blocks = nn.ModuleList()

        for i in range(3):
            in_ch = self.channel_progression[3 - i]
            out_ch = self.channel_progression[2 - i]

            up_conv = self.PixelShuffleUpsample(in_ch, out_ch)
            self.up_convs.append(up_conv)

            residual_blocks = nn.ModuleList([
                R2D2ResidualBlock(out_ch, condition_dim=condition_dim, dropout_rate=dropout_rate)
                for _ in range(4)
            ])
            self.up_residual_blocks.append(residual_blocks)

        # ====================================================================
        # DECODING (S1.4.4)
        # ====================================================================
        self.decoder = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=7, padding=3, padding_mode='reflect')

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, landmask: Optional[torch.Tensor] = None, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the R2D2Unet.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            landmask: Optional landmask tensor of shape (batch_size, 1, height, width) or (height, width)
                      to gate the output. Values should be 0 (water) or 1 (land).
            condition: Optional conditioning tensor of shape (batch_size, condition_dim)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # ====================================================================
        # ENCODING
        # ====================================================================
        h0 = self.encoder(x)  # (B, 128, H, W)

        # ====================================================================
        # DOWNSAMPLING STACK
        # ====================================================================
        skip_connections = []
        h = h0

        for i, (down_conv, residual_blocks) in enumerate(
            zip(self.down_convs, self.down_residual_blocks)
        ):
            h = down_conv(h)

            for res_block in residual_blocks:
                h = res_block(h, condition=condition)

            # Store output for skip connection (r_{i,4} from paper)
            skip_connections.append(h)

        # ====================================================================
        # UPSAMPLING STACK
        # ====================================================================
        # Reverse skip connections for decoder (deepest to shallowest)
        # Drop the last skip connection (which is the input itself)
        # And append None for the last upsampling block (which uses h0 after the loop)
        skip_connections = skip_connections[:-1][::-1] + [None]

        for i, (up_conv, residual_blocks, skip) in enumerate(
            zip(self.up_convs, self.up_residual_blocks, skip_connections)
        ):
            # First upsampling block: multiply by 2 as per equation S15
            if i == 0:
                h = 2 * h

            h = up_conv(h)

            if skip is not None:
                if h.shape != skip.shape:
                    diff_h = skip.shape[2] - h.shape[2]
                    diff_w = skip.shape[3] - h.shape[3]
                    h = F.pad(h, [diff_w // 2, diff_w - diff_w // 2,
                                 diff_h // 2, diff_h - diff_h // 2])

                # Add skip connection (u_{i,4} = R(UConv3(u_{i+1,1}) + r_{i,4}) from S14)
                h = h + skip

            for res_block in residual_blocks:
                h = res_block(h, condition=condition)

        # Add final skip connection from encoder (equation S16)
        if h.shape != h0.shape:
            diff_h = h0.shape[2] - h.shape[2]
            diff_w = h0.shape[3] - h.shape[3]
            h = F.pad(h, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        h = h + h0

        # ====================================================================
        # DECODING
        # ====================================================================
        h = self.decoder(h)  # 128 → 64
        out = self.final_conv(h)  # 64 → out_channels

        if NORMALIZE_OUTPUTS:
            out = torch.sigmoid(out)
        else:
            out = F.softplus(out, beta=4.0)

        if landmask is not None:
            # Ensure landmask has the right shape (batch_size, 1, height, width)
            if landmask.dim() == 2:
                # Shape: (height, width) -> (1, 1, height, width)
                landmask = landmask.unsqueeze(0).unsqueeze(0)
            elif landmask.dim() == 3:
                # Shape: (batch_size, height, width) -> (batch_size, 1, height, width)
                landmask = landmask.unsqueeze(1)

            landmask = landmask.to(out.device)

            # This will zero out predictions over water (landmask=0)
            out = out * landmask

        return out


class FusionBlock(nn.Module):
    """
    Fusion module to combine features from static (high-res) and dynamic (low-res) branches.
    Supports multiple fusion strategies: concatenation, addition, attention-based, or cross-attention.
    """
    def __init__(self, static_channels: int, dynamic_channels: int, fusion_type: str = "concat"):
        """
        Args:
            static_channels: Number of channels in static features
            dynamic_channels: Number of channels in dynamic features
            fusion_type: Type of fusion - "concat", "add", "attention", or "cross_attention"
        """
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # Simple concatenation - no learnable parameters
            self.out_channels = static_channels + dynamic_channels
            self.fusion = nn.Identity()

        elif fusion_type == "add":
            # Element-wise addition - project to same dimension first
            self.out_channels = static_channels
            if dynamic_channels != static_channels:
                self.dynamic_proj = nn.Conv2d(dynamic_channels, static_channels, kernel_size=1, bias=False)
            else:
                self.dynamic_proj = nn.Identity()
            self.fusion = None

        elif fusion_type == "attention":
            # Attention-based fusion - learn importance weights
            self.out_channels = static_channels + dynamic_channels
            self.attention = nn.Sequential(
                nn.Conv2d(static_channels + dynamic_channels, (static_channels + dynamic_channels) // 2,
                         kernel_size=1, bias=False),
                nn.BatchNorm2d((static_channels + dynamic_channels) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d((static_channels + dynamic_channels) // 2, 2, kernel_size=1),  # 2 weights
                nn.Softmax(dim=1)
            )
            self.fusion = None

        elif fusion_type == "cross_attention":
            self.out_channels = static_channels + dynamic_channels
            self.cross_attn = SpatialCrossAttention(static_channels, dynamic_channels)
            self.fusion = None

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'concat', 'add', 'attention', or 'cross_attention'")

    def forward(self, static_feat: torch.Tensor, dynamic_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_feat: High-resolution static features (B, C_s, H, W)
            dynamic_feat: Low-resolution dynamic features (B, C_d, H', W')
                          Will be upsampled to match static_feat size

        Returns:
            Fused features (B, out_channels, H, W)
        """
        if dynamic_feat.shape[-2:] != static_feat.shape[-2:]:
            dynamic_feat = F.interpolate(
                dynamic_feat,
                size=static_feat.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        if self.fusion_type == "concat":
            return torch.cat([static_feat, dynamic_feat], dim=1)

        elif self.fusion_type == "add":
            dynamic_proj = self.dynamic_proj(dynamic_feat)
            return static_feat + dynamic_proj

        elif self.fusion_type == "attention":
            combined = torch.cat([static_feat, dynamic_feat], dim=1)
            attention_weights = self.attention(combined)  # (B, 2, H, W)

            w_static = attention_weights[:, 0:1, :, :]  # (B, 1, H, W)
            w_dynamic = attention_weights[:, 1:2, :, :]  # (B, 1, H, W)

            weighted_static = static_feat * w_static
            weighted_dynamic = dynamic_feat * w_dynamic
            return torch.cat([weighted_static, weighted_dynamic], dim=1)

        elif self.fusion_type == "cross_attention":
            return self.cross_attn(static_feat, dynamic_feat)


class DualEncoderUNet(nn.Module):
    """
    Dual-Encoder U-Net for multi-resolution input fusion.

    Architecture:
        - Static Encoder: Processes high-resolution static data (terrain, elevation, etc.)
        - Dynamic Encoder: Processes low-resolution dynamic data (weather variables)
        - Fusion Modules: Combine features at each scale
        - Decoder: Generates high-resolution output from fused features

    This architecture allows the model to handle inputs at different native resolutions
    without upsampling low-resolution data beforehand.
    """

    def __init__(
        self,
        static_channels: int,
        dynamic_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dynamic_depth: int = None,
        fusion_type: str = "add",
        use_transpose_conv: bool = False,
        norm_layer: Optional[type] = None,
    ):
        """
        Args:
            static_channels: Number of input channels for static data (high-res)
            dynamic_channels: Number of input channels for dynamic data (low-res)
            out_channels: Number of output channels
            base_channels: Number of channels in first layer (doubled at each downsampling)
            depth: Number of downsampling/upsampling stages for static encoder
            dynamic_depth: Number of downsampling stages for dynamic encoder (defaults to depth if None)
                          Use lower value when dynamic input is already at lower resolution
            fusion_type: Type of fusion - "concat", "add", or "attention"
            use_transpose_conv: If True, use ConvTranspose2d for upsampling instead of Upsample + Conv2d
            norm_layer: Normalization layer to use (default: nn.BatchNorm2d). Can pass nn.GroupNorm, etc.
        """
        super().__init__()

        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.out_channels = out_channels
        self.depth = depth
        self.dynamic_depth = dynamic_depth if dynamic_depth is not None else depth
        self.fusion_type = fusion_type
        self.use_transpose_conv = use_transpose_conv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        # ====================================================================
        # STATIC ENCODER (High-Resolution Path)
        # ====================================================================
        self.static_encoders = nn.ModuleList()
        self.static_pools = nn.ModuleList()

        channels = base_channels
        prev_channels = static_channels

        for i in range(depth):
            self.static_encoders.append(
                self._make_conv_block(prev_channels, channels, dropout_rate=0.05)
            )
            self.static_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = channels
            channels *= 2

        self.static_bottleneck = self._make_conv_block(prev_channels, channels, dropout_rate=0.3)

        # ====================================================================
        # DYNAMIC ENCODER (Low-Resolution Path)
        # ====================================================================
        self.dynamic_encoders = nn.ModuleList()
        self.dynamic_pools = nn.ModuleList()

        channels = base_channels
        prev_channels = dynamic_channels

        for i in range(self.dynamic_depth):
            self.dynamic_encoders.append(
                self._make_conv_block(prev_channels, channels, dropout_rate=0.05)
            )
            self.dynamic_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = channels
            channels *= 2

        self.dynamic_bottleneck = self._make_conv_block(prev_channels, channels, dropout_rate=0.3)

        # ====================================================================
        # FUSION MODULES (at each scale)
        # ====================================================================
        self.fusion_blocks = nn.ModuleList()

        static_bottleneck_channels = base_channels * (2 ** depth)
        dynamic_bottleneck_channels = base_channels * (2 ** self.dynamic_depth)
        self.fusion_bottleneck = FusionBlock(
            static_channels=static_bottleneck_channels,
            dynamic_channels=dynamic_bottleneck_channels,
            fusion_type=fusion_type
        )

        for i in range(depth):
            static_ch = base_channels * (2 ** i)
            if i < self.dynamic_depth:
                dynamic_ch = base_channels * (2 ** i)
            else:
                # For deeper static scales, use the deepest dynamic features
                dynamic_ch = base_channels * (2 ** (self.dynamic_depth - 1))
            self.fusion_blocks.append(
                FusionBlock(static_channels=static_ch, dynamic_channels=dynamic_ch, fusion_type=fusion_type)
            )

        self.dynamic_upsamplers = nn.ModuleList()
        for i in range(depth):
            if i >= self.dynamic_depth:
                scale_factor = 2 ** (i - self.dynamic_depth + 1)
                self.dynamic_upsamplers.append(
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                )
            else:
                self.dynamic_upsamplers.append(nn.Identity())


        # ====================================================================
        # DECODER (Shared path for fused features)
        # ====================================================================
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        channels = self.fusion_bottleneck.out_channels

        for i in range(depth):
            if use_transpose_conv:
                self.upsamples.append(nn.Sequential(
                    nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2),
                    nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, padding_mode='reflect')
                ))
            else:
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, padding_mode='reflect')
                ))

            skip_channels = self.fusion_blocks[depth - 1 - i].out_channels
            decoder_in_channels = channels // 2 + skip_channels

            self.decoders.append(
                self._make_conv_block(decoder_in_channels, channels // 2, dropout_rate=0.00)
            )

            self.attention_gates.append(
                AttentionGate(F_g=channels // 2, F_l=skip_channels, F_int=channels // 4)
            )

            channels //= 2

        # ====================================================================
        # OUTPUT HEAD
        # ====================================================================
        self.shared_features = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            norm_layer(channels),
            nn.SiLU(inplace=True),
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, out_channels, kernel_size=1)
        )

        self._init_weights()

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float
    ) -> nn.Module:
        """Create a convolutional block with two conv layers."""
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        layers.append(self.norm_layer(out_channels))
        layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False))
        layers.append(self.norm_layer(out_channels))
        layers.append(nn.SiLU(inplace=True))

        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        else:
            layers.append(nn.Identity())

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        static_highres: torch.Tensor,
        dynamic_lowres: torch.Tensor,
        landmask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Dual-Encoder UNet.

        Args:
            static_highres: High-resolution static input (B, C_static, H_high, W_high)
            dynamic_lowres: Low-resolution dynamic input (B, C_dynamic, H_low, W_low)
            landmask: Optional landmask tensor to gate output (B, 1, H, W) or (H, W)

        Returns:
            Output tensor of shape (B, out_channels, H_high, W_high)
        """
        # ====================================================================
        # ENCODER PASS - Extract multi-scale features from both branches
        # ====================================================================

        static_skip_connections = []
        x_static = static_highres

        for encoder, pool in zip(self.static_encoders, self.static_pools):
            x_static = encoder(x_static)
            static_skip_connections.append(x_static)
            x_static = pool(x_static)

        x_static = self.static_bottleneck(x_static)

        dynamic_skip_connections = []
        x_dynamic = dynamic_lowres

        for encoder, pool in zip(self.dynamic_encoders, self.dynamic_pools):
            x_dynamic = encoder(x_dynamic)
            dynamic_skip_connections.append(x_dynamic)
            x_dynamic = pool(x_dynamic)

        x_dynamic = self.dynamic_bottleneck(x_dynamic)

        # ====================================================================
        # FUSION - Combine features from both branches at each scale
        # ====================================================================

        x_fused = self.fusion_bottleneck(x_static, x_dynamic)

        fused_skip_connections = []
        for i in range(self.depth):
            static_skip = static_skip_connections[i]

            if i < self.dynamic_depth:
                dynamic_skip = dynamic_skip_connections[i]
            else:
                # Use the deepest dynamic features and upsample
                dynamic_skip = dynamic_skip_connections[-1]
                dynamic_skip = self.dynamic_upsamplers[i](dynamic_skip)

            fused_skip = self.fusion_blocks[i](static_skip, dynamic_skip)
            fused_skip_connections.append(fused_skip)

        fused_skip_connections = fused_skip_connections[::-1]

        # ====================================================================
        # DECODER - Generate high-resolution output from fused features
        # ====================================================================

        x = x_fused

        for i, (upsample, decoder, attn_gate) in enumerate(
            zip(self.upsamples, self.decoders, self.attention_gates)
        ):
            x = upsample(x)

            skip = fused_skip_connections[i]

            if x.shape != skip.shape:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2])

            skip = attn_gate(g=x, x=skip)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # ====================================================================
        # OUTPUT HEAD
        # ====================================================================

        x = self.shared_features(x)
        out = self.output_head(x)

        out = F.relu(out)

        if landmask is not None:
            # Ensure landmask has the right shape (batch_size, 1, height, width)
            if landmask.dim() == 2:
                landmask = landmask.unsqueeze(0).unsqueeze(0)
            elif landmask.dim() == 3:
                landmask = landmask.unsqueeze(1)

            landmask = landmask.to(out.device)
            out = out * landmask

        return out
