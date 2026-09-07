from typing import Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from config import NORMALIZE_OUTPUTS, SEQUENCE_LENGTH
except ImportError:
    NORMALIZE_OUTPUTS = False
    SEQUENCE_LENGTH = 1


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: bool = True):
        """
        Args:
            dim: Number of input channels
            window_size: The height and width of the window
            num_heads: Number of attention heads
            qkv_bias: If True, add a learnable bias to query, key, value
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features with shape (num_windows*B, N, C)
            mask: (0/-inf) mask with shape (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: Window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Number of input channels
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size for SW-MSA
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input features with shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, H*W, embed_dim)
            H, W: Height and width after patching
        """
        B, C, H, W = x.shape
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch Merging Layer (downsampling).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple:
        """
        Args:
            x: (B, H*W, C)
            H, W: Height and width of feature map
        Returns:
            x: (B, H/2*W/2, 2*C)
            H, W: New height and width
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer (upsampling for decoder).
    """
    def __init__(self, dim: int, dim_scale: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple:
        """
        Args:
            x: (B, H*W, C)
            H, W: Height and width of feature map
        Returns:
            x: (B, 2H*2W, C/2)
            H, W: New height and width
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)

        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)

        return x, H * 2, W * 2


class SwinUNet(nn.Module):
    """
    Swin-UNet: UNet architecture with Swin Transformer blocks.
    Combines the hierarchical encoder-decoder structure of UNet with
    the powerful representation learning of Swin Transformers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 4,
    ):
        """
        Args:
            in_channels: Number of input image channels
            out_channels: Number of output channels
            embed_dim: Patch embedding dimension
            depths: Depths of each Swin Transformer stage
            num_heads: Number of attention heads in different layers
            window_size: Window size
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            dropout: Dropout rate
            patch_size: Patch size for patch embedding
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )

        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim = embed_dim * (2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for i in range(depths[i_layer])
            ])
            self.encoder_layers.append(layer)

            if i_layer < self.num_layers - 1:
                self.downsample_layers.append(PatchMerging(dim=dim))

        bottleneck_dim = embed_dim * (2 ** (self.num_layers - 1))
        self.bottleneck = nn.ModuleList([
            SwinTransformerBlock(
                dim=bottleneck_dim,
                num_heads=num_heads[-1],
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for i in range(2)
        ])

        self.upsample_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers - 1, -1, -1):
            dim = embed_dim * (2 ** i_layer)
            self.upsample_layers.append(PatchExpanding(dim=dim * 2 if i_layer < self.num_layers - 1 else dim))
            self.concat_back_dim.append(nn.Linear(dim * 2, dim))

            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for i in range(depths[i_layer])
            ])
            self.decoder_layers.append(layer)

        self.final_expand = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
        )

        self.shared_features = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU()
        )

        # Head 1: Large scale output
        self.output_head1 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, 1, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, landmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Swin-UNet.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            landmask: Optional landmask tensor of shape (batch_size, 1, height, width) or (height, width)
                      to gate the output. Values should be 0 (water) or 1 (land).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        _, _, orig_H, orig_W = x.shape

        # Calculate padding to ensure dimensions are compatible with patch_size and window_size
        # After patch embedding, H and W must be divisible by window_size at each downsample stage
        # With (num_layers-1) downsample operations, we need H,W divisible by 2^(num_layers-1) * window_size
        # So input H and W must be divisible by (patch_size * 2^(num_layers-1) * window_size)
        patch_size = self.patch_embed.patch_size
        window_size = self.encoder_layers[0][0].window_size
        downsample_factor = 2 ** (self.num_layers - 1)
        required_multiple = patch_size * downsample_factor * window_size

        pad_H = (required_multiple - orig_H % required_multiple) % required_multiple
        pad_W = (required_multiple - orig_W % required_multiple) % required_multiple

        if pad_H > 0 or pad_W > 0:
            # Pad on right and bottom
            x = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')

        x, H, W = self.patch_embed(x)

        skip_connections = []
        skip_sizes = []

        for i_layer, (encoder, downsample) in enumerate(
            zip(self.encoder_layers, self.downsample_layers + [None])
        ):
            for block in encoder:
                x = block(x, H, W)

            # Only save skip connections for layers that will have corresponding decoder upsampling
            # The last encoder layer output goes directly to bottleneck, no skip needed
            if i_layer < self.num_layers - 1:
                skip_connections.append(x)
                skip_sizes.append((H, W))

            if downsample is not None:
                x, H, W = downsample(x, H, W)

        for block in self.bottleneck:
            x = block(x, H, W)

        skip_connections = skip_connections[::-1]
        skip_sizes = skip_sizes[::-1]

        # Use decoder layers 0 through (num_layers-2) which corresponds to (num_layers-1) upsampling operations
        # This matches the number of skip connections saved during encoding
        for i_decoder in range(self.num_layers - 1):
            upsample = self.upsample_layers[i_decoder]
            concat_back = self.concat_back_dim[i_decoder]
            decoder = self.decoder_layers[i_decoder]

            x, H, W = upsample(x, H, W)

            skip = skip_connections[i_decoder]
            x = torch.cat([skip, x], dim=-1)
            x = concat_back(x)

            for block in decoder:
                x = block(x, H, W)

        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.final_expand(x)

        if x.shape[2] != orig_H or x.shape[3] != orig_W:
            x = F.interpolate(x, size=(orig_H, orig_W), mode='bilinear', align_corners=False)

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
