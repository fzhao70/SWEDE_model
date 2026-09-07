from typing import Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from config import NORMALIZE_OUTPUTS
except ImportError:
    NORMALIZE_OUTPUTS = False


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This implementation drops entire residual paths during training with a probability `drop_prob`.
    During inference, it applies an identity mapping.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, ...) to broadcast across all feature dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        """
        Args:
            dim: Number of input channels
            window_size: The height and width of the window
            num_heads: Number of attention heads
            qkv_bias: If True, add a learnable bias to query, key, value
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Got dim // num_heads = {dim / num_heads}"
            )
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

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

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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


class SwinTransformerLayer(nn.Module):
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
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim: Number of input channels
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size for SW-MSA
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            dropout: Dropout rate for MLP
            drop_path: Stochastic depth rate
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        # Compute cyclic-shift attention mask for SW-MSA.
        # Without this mask, shifted-window blocks behave identically to regular
        # window blocks, losing all cross-window connectivity.
        if self.shift_size > 0:
            img_mask = torch.zeros(1, H, W, 1, device=shifted_x.device, dtype=shifted_x.dtype)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)   # (nW, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

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
            nn.GroupNorm(num_groups=min(32, max(1, F_int // 4)), num_channels=F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=min(32, max(1, F_int // 4)), num_channels=F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


class LatitudeBias(nn.Module):
    """
    Learnable latitude-dependent bias correction.

    Applies a linear bias that varies with the y-coordinate (latitude):
        bias(y) = slope * y + intercept
    where y ∈ [-1, 1] represents south to north.
    """
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.slope = nn.Parameter(torch.zeros(out_channels))
        self.intercept = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        lat = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        bias_map = self.slope.view(1, -1, 1, 1) * lat + self.intercept.view(1, -1, 1, 1)
        return x + bias_map

    def get_bias_stats(self):
        return {
            'slope': self.slope.detach().cpu().item() if self.slope.numel() == 1 else self.slope.detach().cpu().numpy(),
            'intercept': self.intercept.detach().cpu().item() if self.intercept.numel() == 1 else self.intercept.detach().cpu().numpy(),
            'north_bias': (self.slope + self.intercept).detach().cpu().item() if self.slope.numel() == 1 else None,
            'south_bias': (-self.slope + self.intercept).detach().cpu().item() if self.slope.numel() == 1 else None,
        }


class ResConvBlock(nn.Module):
    """Two-layer conv block with a residual shortcut connection.

    When in_channels != out_channels a 1×1 Conv+Norm projection aligns the
    shortcut, matching standard ResNet practice.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_rate: float, norm_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect', bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect', bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + self.shortcut(x)


class DualEncoderSwinUNet(nn.Module):
    """
    Hybrid Dual-Encoder U-Net with Swin Transformer for Dynamic Input.

    Same architecture as model_desu_reg_lessskip.DualEncoderSwinUNet, with three
    backbone fixes ported from model_desu_full_hist (but no history encoder):
      1. SW-MSA cyclic-shift attention mask is correctly applied (was missing in
         lessskip — shifted blocks were silently equivalent to non-shifted W-MSA).
      2. CNN conv blocks use residual shortcuts (ResConvBlock).
      3. Conv2d weight init uses SiLU-calibrated gain (gain ≈ 2.099) instead of
         ReLU-calibrated Kaiming-He.

    Architecture:
        - Static Encoder: CNN (High-Res), residual blocks
        - Dynamic Encoder: Swin Transformer (Low-Res), with proper SW-MSA mask
        - Fusion:
            - Bottleneck: 4x Swin Transformer Blocks with Residual Connection
            - Skips: CNN Concatenation
        - Decoder: CNN, residual blocks
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
        swin_embed_dim: Optional[int] = None,
        swin_num_heads: Sequence[int] = (3, 6, 12, 24),
        swin_window_size: int = 7,
        swin_mlp_ratio: float = 4.0,
        swin_patch_size: int = 4,  # dynamic input usually lower res, so small patch size
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.12,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        decoder_dropout_rate: float = 0.0,
        num_bottleneck_fusion_blocks: int = 4,
        # Per-stage learnable gate on static skip connections. Deep stages
        # are mostly redundant with the bottleneck fusion; init small there
        # and at full strength at high-res stages. Length should match `depth`.
        static_skip_gate_init: Sequence[float] = (0.1, 0.1, 1.0, 1.0),
        # Final output activation. True (default) applies ReLU so the model can
        # only emit non-negative values (correct for raw SWE). Set False when the
        # target is a signed quantity such as a SWE anomaly relative to a
        # climatology, so negative predictions are allowed.
        output_relu: bool = True,
    ):
        super().__init__()

        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.out_channels = out_channels
        self.output_relu = output_relu
        self.depth = depth
        self.dynamic_depth = dynamic_depth if dynamic_depth is not None else depth
        self.fusion_type = fusion_type
        self.use_transpose_conv = use_transpose_conv
        self.num_bottleneck_fusion_blocks = num_bottleneck_fusion_blocks

        if norm_layer is None:
            # Use GroupNorm for better stability with small batches and sample independence
            norm_layer = lambda channels: nn.GroupNorm(
                num_groups=min(32, max(1, channels // 4)),
                num_channels=channels
            )
        self.norm_layer = norm_layer

        # ====================================================================
        # STATIC ENCODER (CNN)
        # ====================================================================
        self.static_encoders = nn.ModuleList()
        self.static_pools = nn.ModuleList()

        channels = base_channels
        prev_channels = static_channels

        for i in range(depth):
            self.static_encoders.append(
                self._make_conv_block(prev_channels, channels, dropout_rate=0.0)
            )
            self.static_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = channels
            channels *= 2

        self.static_bottleneck_channels = prev_channels
        self.static_bottleneck = self._make_conv_block(prev_channels, prev_channels, dropout_rate=0.1)
        self.static_bottleneck_out_channels = prev_channels

        # ====================================================================
        # DYNAMIC ENCODER (Swin Transformer)
        # ====================================================================
        self.swin_embed_dim = swin_embed_dim if swin_embed_dim is not None else base_channels

        self.swin_patch_embed = PatchEmbed(
            patch_size=swin_patch_size, in_channels=dynamic_channels, embed_dim=self.swin_embed_dim
        )

        self.swin_layers = nn.ModuleList()
        self.swin_downsamples = nn.ModuleList()

        swin_depths_per_stage = [2] * self.dynamic_depth

        current_swin_num_heads = list(swin_num_heads)
        if len(current_swin_num_heads) < self.dynamic_depth:
            while len(current_swin_num_heads) < self.dynamic_depth:
                current_swin_num_heads.append(current_swin_num_heads[-1] * 2)
        current_swin_num_heads = current_swin_num_heads[:self.dynamic_depth]

        total_swin_blocks = sum(swin_depths_per_stage)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_swin_blocks)]

        block_idx = 0
        for i_layer in range(self.dynamic_depth):
            dim = self.swin_embed_dim * (2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerLayer(
                    dim=dim,
                    num_heads=current_swin_num_heads[i_layer],
                    window_size=swin_window_size,
                    shift_size=0 if (i % 2 == 0) else swin_window_size // 2,
                    mlp_ratio=swin_mlp_ratio,
                    dropout=0.1,
                    drop_path=dpr[block_idx + i],
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                )
                for i in range(swin_depths_per_stage[i_layer])
            ])
            self.swin_layers.append(layer)
            block_idx += swin_depths_per_stage[i_layer]

            if i_layer < self.dynamic_depth - 1:
                self.swin_downsamples.append(PatchMerging(dim=dim))

        self.swin_out_dim = self.swin_embed_dim * (2 ** (self.dynamic_depth - 1))
        self.swin_norm = nn.LayerNorm(self.swin_out_dim)

        # ====================================================================
        # FUSION (BOTTLENECK) - Deep Residual Swin
        # ====================================================================
        bottleneck_drop_path = drop_path_rate * 1.5 if drop_path_rate > 0 else 0.0
        self.bottleneck_fusion_blocks = nn.ModuleList([
            SwinTransformerLayer(
                dim=self.swin_out_dim,
                num_heads=current_swin_num_heads[-1],
                window_size=swin_window_size,
                shift_size=0 if (i % 2 == 0) else swin_window_size // 2,
                mlp_ratio=swin_mlp_ratio,
                dropout=dropout_rate,
                drop_path=bottleneck_drop_path,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
            )
            for i in range(self.num_bottleneck_fusion_blocks)
        ])

        if self.static_bottleneck_out_channels != self.swin_out_dim:
            self.static_bot_proj = nn.Sequential(
                nn.Conv2d(self.static_bottleneck_out_channels, self.swin_out_dim, kernel_size=1),
                self.norm_layer(self.swin_out_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.static_bot_proj = nn.Identity()

        # ====================================================================
        # DECODER (CNN)
        # ====================================================================
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        decoder_channels = self.static_bottleneck_out_channels

        self.bottleneck_back_proj = nn.Sequential(
            nn.Conv2d(self.swin_out_dim, decoder_channels, kernel_size=1),
            self.norm_layer(decoder_channels),
            nn.SiLU(inplace=True)
        )

        for i in range(depth):
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(decoder_channels, 2 * decoder_channels, kernel_size=1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
                nn.Conv2d(decoder_channels // 2, decoder_channels // 2, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
                self.norm_layer(decoder_channels // 2),
                nn.SiLU(inplace=True)
            ))

            static_skip_ch = base_channels * (2 ** (depth - 1 - i))

            if i < self.dynamic_depth:
                swin_skip_idx = self.dynamic_depth - 1 - i
                dynamic_skip_ch = self.swin_embed_dim * (2 ** swin_skip_idx)
                skip_total_ch = static_skip_ch + dynamic_skip_ch
            else:
                skip_total_ch = static_skip_ch

            self.decoders.append(
                self._make_conv_block(decoder_channels // 2 + skip_total_ch, decoder_channels // 2, dropout_rate=decoder_dropout_rate)
            )

            self.attention_gates.append(
                AttentionGate(F_g=decoder_channels // 2, F_l=skip_total_ch, F_int=decoder_channels // 4)
            )

            decoder_channels //= 2

        # ====================================================================
        # OUTPUT HEAD
        # ====================================================================
        self.shared_features = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            self.norm_layer(decoder_channels),
            nn.SiLU(inplace=True),
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, out_channels, kernel_size=1, bias=True)
        )

        _gate = list(static_skip_gate_init)
        if len(_gate) < depth:
            _gate = _gate + [1.0] * (depth - len(_gate))
        elif len(_gate) > depth:
            _gate = _gate[:depth]
        self.static_skip_gate = nn.Parameter(torch.tensor(_gate, dtype=torch.float32))

        self._init_weights()

    def _make_conv_block(self, in_channels, out_channels, dropout_rate):
        """Helper to create CNN blocks with residual shortcut."""
        return ResConvBlock(in_channels, out_channels, dropout_rate, self.norm_layer)

    def _init_weights(self):
        # SiLU/Swish gain: Var[SiLU(N(0,1))] ≈ 0.454, so gain = sqrt(2/0.454) ≈ 2.099.
        # Kaiming fan_in std = gain / sqrt(fan_in). PyTorch's kaiming_normal_ doesn't
        # support 'silu', so we compute the fan manually and apply the correct std.
        _SILU_GAIN = 2.099
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
                std = _SILU_GAIN / (fan_in ** 0.5)
                with torch.no_grad():
                    m.weight.normal_(0, std)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _resize_if_needed(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        if x.shape[2:] != target_size:
            return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x

    def forward(self, static_highres, dynamic_lowres, landmask=None):
        """
        Forward pass matching DualEncoderUNet signature.
        """
        # ==================== STATIC ENCODER (CNN) ====================
        static_skips = []
        x_s = static_highres
        for encoder, pool in zip(self.static_encoders, self.static_pools):
            x_s = encoder(x_s)
            static_skips.append(x_s)
            x_s = pool(x_s)

        x_s_bot = self.static_bottleneck(x_s)

        # ==================== DYNAMIC ENCODER (SWIN) ====================
        x_d = dynamic_lowres
        _, _, H_d, W_d = x_d.shape

        window_size = self.swin_layers[0][0].window_size
        patch_size = self.swin_patch_embed.patch_size
        downsample_factor = 2 ** (self.dynamic_depth - 1)
        required_mult = patch_size * downsample_factor * window_size

        pad_h = (required_mult - H_d % required_mult) % required_mult
        pad_w = (required_mult - W_d % required_mult) % required_mult
        if pad_h > 0 or pad_w > 0:
            x_d = F.pad(x_d, (0, pad_w, 0, pad_h), mode='reflect')

        x_d, H_swin, W_swin = self.swin_patch_embed(x_d)

        dynamic_skips = []

        for i, (layer, downsample) in enumerate(zip(self.swin_layers, self.swin_downsamples + [None])):
            for block in layer:
                x_d = block(x_d, H_swin, W_swin)

            B, L, C = x_d.shape
            curr_skip = x_d.view(B, H_swin, W_swin, C).permute(0, 3, 1, 2).contiguous()
            dynamic_skips.append(curr_skip)

            if downsample:
                x_d, H_swin, W_swin = downsample(x_d, H_swin, W_swin)

        x_d = self.swin_norm(x_d)

        B, L, C = x_d.shape
        x_d_bot = x_d.view(B, H_swin, W_swin, C).permute(0, 3, 1, 2).contiguous()

        # ==================== FUSION (BOTTLENECK) ====================
        x_s_bot_resized = self._resize_if_needed(x_s_bot, x_d_bot.shape[2:])
        x_s_bot_resized = self.static_bot_proj(x_s_bot_resized)
        x_fused = x_d_bot + x_s_bot_resized

        x_fused_flat = x_fused.flatten(2).transpose(1, 2)

        residual = x_fused_flat
        for block in self.bottleneck_fusion_blocks:
            x_fused_flat = block(x_fused_flat, H_swin, W_swin)

        x_fused_flat = x_fused_flat + residual

        x_fused = x_fused_flat.view(B, H_swin, W_swin, -1).permute(0, 3, 1, 2).contiguous()

        # ==================== DECODER (CNN) ====================
        x = self.bottleneck_back_proj(x_fused)

        static_skips = static_skips[::-1]

        for i, (upsample, decoder, attn_gate) in enumerate(zip(self.upsamples, self.decoders, self.attention_gates)):
            x = upsample(x)

            s_skip = static_skips[i]
            s_skip = s_skip * self.static_skip_gate[i]

            if i < self.dynamic_depth:
                swin_skip_idx = self.dynamic_depth - 1 - i
                d_skip = dynamic_skips[swin_skip_idx]
                d_skip = self._resize_if_needed(d_skip, s_skip.shape[2:])
                combined_skip = torch.cat([s_skip, d_skip], dim=1)
            else:
                combined_skip = s_skip

            if x.shape[2:] != combined_skip.shape[2:]:
                diff_h = combined_skip.shape[2] - x.shape[2]
                diff_w = combined_skip.shape[3] - x.shape[3]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

            combined_skip = attn_gate(g=x, x=combined_skip)

            x = torch.cat([combined_skip, x], dim=1)
            x = decoder(x)

        x = self.shared_features(x)
        out = self.output_head(x)

        if landmask is not None:
            if landmask.dim() == 2:
                landmask = landmask.unsqueeze(0).unsqueeze(0)
            elif landmask.dim() == 3:
                landmask = landmask.unsqueeze(1)
            landmask = landmask.to(out.device)
            out = out * landmask

        # ReLU only when the target is non-negative (raw SWE). For a signed
        # anomaly target, output_relu=False lets the head emit negatives; the
        # landmask multiply above still forces exactly 0 over water (anomaly=0).
        if self.output_relu:
            out = F.relu(out)

        return out

    def get_min_input_size(self):
        """
        Calculate the minimum input size required for the model.
        Returns: (min_height, min_width) for both static and dynamic inputs
        """
        window_size = 7
        patch_size = self.swin_patch_embed.patch_size
        downsample_factor = 2 ** (self.dynamic_depth - 1)
        required_mult = patch_size * downsample_factor * window_size

        static_min = 2 ** self.depth

        return max(required_mult, static_min), max(required_mult, static_min)
