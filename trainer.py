"""
Trainer for ERA5 snow prediction models.

Configuration: Modify constants in config.py to change training settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import time

from config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_MAX_NORM,
    CHECKPOINT_DIR,
    SAVE_EVERY_N_EPOCHS,
    LOG_EVERY_N_BATCHES,
    USE_TORCH_COMPILE,
    COMPILE_MODE,
    NORMALIZE_OUTPUTS,
    USE_POS_WEIGHT
)

try:
    from config import DECODER_WEIGHT_DECAY
    _HAS_DECODER_WEIGHT_DECAY = True
except ImportError:
    _HAS_DECODER_WEIGHT_DECAY = False

try:
    from config import CNN_ENCODER_WEIGHT_DECAY
    _HAS_CNN_ENCODER_WEIGHT_DECAY = True
except ImportError:
    _HAS_CNN_ENCODER_WEIGHT_DECAY = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

_NORM_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


def _build_no_decay_names(model):
    """
    Return a set of fully-qualified parameter names that must NOT receive
    weight decay:
      - all parameters of any normalisation layer (LN / GN / BN)
      - all bias parameters
      - Swin relative_position_bias_table entries

    Uses isinstance() on module types so parameters inside nn.Sequential
    stored by integer index (e.g. 'static_bottleneck.1.weight') are still
    correctly identified even when the name contains no 'norm' substring.
    """
    no_decay_names = set()

    for module_name, module in model.named_modules():
        prefix = module_name + '.' if module_name else ''

        if isinstance(module, _NORM_TYPES):
            for param_name in module._parameters:
                if module._parameters[param_name] is not None:
                    no_decay_names.add(prefix + param_name)

        for param_name, param in module.named_parameters(recurse=False):
            if param is None:
                continue
            if param_name == 'bias' or 'relative_position_bias_table' in param_name:
                no_decay_names.add(prefix + param_name)

    return no_decay_names


def separate_weight_decay_params(model):
    """
    Separate model parameters into two weight-decay groups:
      - decay_params:    Conv/Linear weights  → weight_decay = WEIGHT_DECAY
      - no_decay_params: Norm weights/biases, all biases, position-bias tables
                         → weight_decay = 0.0
    """
    no_decay_names = _build_no_decay_names(model)

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in no_decay_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


def separate_weight_decay_params_by_component(model):
    """
    Separate DualEncoderSwinUNet parameters into four weight-decay groups:

    1. swin_decay        – Swin encoder & all SwinTransformerLayer (STL) blocks
                           (swin_patch_embed, swin_layers, swin_downsamples, swin_norm,
                            bottleneck_fusion_blocks, bottleneck_proj)
                           → weight_decay = WEIGHT_DECAY
    2. cnn_encoder_decay – Static CNN encoder (static_encoders, static_bottleneck)
                           → weight_decay = CNN_ENCODER_WEIGHT_DECAY
    3. decoder_decay     – CNN decoder (bottleneck_back_proj, upsamples, decoders,
                           attention_gates, shared_features, output_head)
                           → weight_decay = DECODER_WEIGHT_DECAY
    4. no_decay          – ALL norm layer params (LN/GN/BN weight+bias), ALL biases,
                           and Swin relative_position_bias_table — everywhere in the model
                           → weight_decay = 0.0

    Uses isinstance() checks so norms inside nn.Sequential (indexed by number)
    are correctly identified regardless of parameter name.
    Only called when DECODER_WEIGHT_DECAY is defined in config.
    """
    SWIN_PREFIXES = (
        'swin_patch_embed.',
        'swin_layers.',
        'swin_downsamples.',
        'swin_norm.',
        'bottleneck_fusion_blocks.',
        'bottleneck_proj.',
    )
    CNN_ENCODER_PREFIXES = (
        'static_encoders.',
        'static_bottleneck.',
    )

    no_decay_names = _build_no_decay_names(model)

    swin_decay = []
    cnn_encoder_decay = []
    decoder_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in no_decay_names:
            no_decay.append(param)
        elif any(name.startswith(prefix) for prefix in SWIN_PREFIXES):
            swin_decay.append(param)
        elif any(name.startswith(prefix) for prefix in CNN_ENCODER_PREFIXES):
            cnn_encoder_decay.append(param)
        else:
            decoder_decay.append(param)

    return swin_decay, cnn_encoder_decay, decoder_decay, no_decay


# ============================================================================
# CUSTOM LOSS FUNCTIONS
# ============================================================================


class CombinedLoss(nn.Module):
    """
    Combined loss with regression and binary classification:
    1. Binary classification using regression outputs as logits
       (no separate classification head needed)
    2. Regression on pixels with significant snow values
    3. Per-channel weighting to balance different scales (if not using output normalization)

    The regression outputs are reused as logits for binary classification,
    creating a unified representation where higher regression values indicate
    higher probability of snow presence.

    Args:
        alpha: Weight balance between regression and classification (0-1)
               alpha=1.0 means only regression, alpha=0.0 means only classification
        threshold: Minimum value to consider as "snow present"
        regression_loss_type: Type of regression loss ('smooth_l1' or 'mse')
        channel_weights: Weights for each channel to balance different scales
                        If NORMALIZE_OUTPUTS=True, this is ignored and set to [1.0, 1.0]
        pos_weight: Per-channel positive class weights for binary cross-entropy (optional)
                   List of weights, one per channel. Higher values = more attention to positive class
        dynamic_pos_weight: If True, calculate pos_weight from actual batch distribution
                          pos_weight = (num_negative_pixels) / (num_positive_pixels)
                          If False, use static pos_weight from initialization
        dice_weight: Weight for Dice loss (default: 0.01). Dice loss improves segmentation quality.
        ssim_weight: Weight for SSIM loss (default: 0.0). SSIM captures structural similarity.
        sobel_weight: Weight for Sobel gradient loss (default: 0.01). Sobel loss preserves edge/gradient information.
        tweedie_weight: Weight for Tweedie loss (default: 0.0). Tweedie loss is suitable for zero-inflated continuous data.
        tweedie_p: Power parameter for Tweedie loss (default: 1.5). Range (1, 2) for compound Poisson-Gamma.
        variance_weight: Weight for variance matching loss (default: 0.0). Helps maintain temporal variability.
        pinball_weight: Weight for pinball (quantile) loss (default: 0.0). Can be a scalar or a list of weights
                       for multiple quantiles. Pinball loss is useful for quantile regression.
        pinball_quantile: Quantile for pinball loss (default: 0.5). Can be a scalar or a list of quantiles
                         matching pinball_weight. Range [0, 1], where 0.5 is the median.
        log1p_weight: Weight for log(1+y) loss (default: 0.5). Log-space loss helps with zero-inflated data.
        temporal_agg_weight: Weight for temporal aggregate consistency loss (default: 0.0). Ensures temporal mass conservation.
        mse_weight: Weight for MSE loss (default: 0.0). Standard mean squared error loss.
    """
    def __init__(self, alpha=0.5, channel_weights=[1.0], pos_weight=None, dynamic_pos_weight=False, dice_weight=0.01, ssim_weight=0.0, sobel_weight=0.01, tweedie_weight=0.0, tweedie_p=1.5, variance_weight=0.0, pinball_weight=0.0, pinball_quantile=0.5, log1p_weight=0.5, temporal_agg_weight=0.0, mse_weight=0.0, threshold=1e-4, temp_constraint_weight=0.0, temp_threshold=15.0, temp_combined_threshold=None, temp_lag_threshold=None, smooth_l1_beta=5.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.regression_loss_type = 'smooth_l1'
        self.smooth_l1_beta = smooth_l1_beta
        self.pos_weight = pos_weight
        self.dynamic_pos_weight = dynamic_pos_weight
        self.temperature = 1
        self.eps = 1e-10
        self.dice_weight = dice_weight
        self.max_range = 5000

        if NORMALIZE_OUTPUTS:
            self.channel_weights = [1.0]
            print("  Output normalization enabled: Using channel weight [1.0]")
        else:
            self.channel_weights = channel_weights

        if self.dynamic_pos_weight:
            print(f"  Using dynamic pos_weight calculated per-batch from actual class distribution")
        elif self.pos_weight is not None:
            print(f"  Using static pos_weight for BCE loss: {self.pos_weight}")

        if self.dice_weight > 0:
            print(f"  Dice weight: {self.dice_weight}")

        self.ssim_weight = ssim_weight
        if self.ssim_weight > 0:
            print(f"  SSIM weight: {self.ssim_weight}")

        self.sobel_weight = sobel_weight
        if self.sobel_weight > 0:
            print(f"  Sobel gradient weight: {self.sobel_weight}")

        self.tweedie_weight = tweedie_weight
        self.tweedie_p = tweedie_p
        if self.tweedie_weight > 0:
            print(f"  Tweedie weight: {self.tweedie_weight} (p={self.tweedie_p})")

        self.variance_weight = variance_weight
        if self.variance_weight > 0:
            print(f"  Variance matching weight: {self.variance_weight}")

        self.pinball_weight = pinball_weight
        self.pinball_quantile = pinball_quantile
        _pinball_active = (
            isinstance(self.pinball_weight, (list, tuple)) and any(w > 0 for w in self.pinball_weight)
        ) or (
            not isinstance(self.pinball_weight, (list, tuple)) and self.pinball_weight > 0
        )
        if _pinball_active:
            print(f"  Pinball (quantile) loss weight: {self.pinball_weight} (quantile={self.pinball_quantile})")

        self.log1p_weight = log1p_weight
        if self.log1p_weight > 0:
            print(f"  Log(1+y) loss weight: {self.log1p_weight}")

        self.temporal_agg_weight = temporal_agg_weight
        if self.temporal_agg_weight > 0:
            print(f"  Temporal aggregate consistency loss weight: {self.temporal_agg_weight}")

        self.mse_weight = mse_weight
        if self.mse_weight > 0:
            print(f"  MSE loss weight: {self.mse_weight}")

        # Temperature constraint loss weight and thresholds (all in normalized space)
        self.temp_constraint_weight = temp_constraint_weight
        self.temp_threshold = temp_threshold              # standalone: t2min > this → no snow
        self.temp_combined_threshold = temp_combined_threshold  # combined: t2min > this AND lag > lag_threshold → no snow
        self.temp_lag_threshold = temp_lag_threshold      # combined: t2_lag_mean > this AND t2min > combined_threshold → no snow
        if self.temp_constraint_weight > 0:
            print(f"  Temperature constraint weight: {self.temp_constraint_weight}")
            print(f"    - Standalone:  t2min > {self.temp_threshold} (normalized) → no snow")
            if self.temp_combined_threshold is not None and self.temp_lag_threshold is not None:
                print(f"    - Combined:    t2_lag_mean > {self.temp_lag_threshold} (normalized) AND t2min > {self.temp_combined_threshold} (normalized) → no snow")

    def _compute_ssim(self, pred, target, window_size=11):
        """Compute SSIM loss (1 - SSIM) between pred and target."""
        C1 = (0.01 * self.max_range) ** 2
        C2 = (0.03 * self.max_range) ** 2

        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=pred.dtype, device=pred.device) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

        window = gaussian_window(window_size)
        window = window.expand(pred.shape[1], 1, window_size, window_size)

        padding = window_size // 2

        mu_pred = nn.functional.conv2d(pred, window, padding=padding, groups=pred.shape[1])
        mu_target = nn.functional.conv2d(target, window, padding=padding, groups=target.shape[1])

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = nn.functional.conv2d(pred ** 2, window, padding=padding, groups=pred.shape[1]) - mu_pred_sq
        sigma_target_sq = nn.functional.conv2d(target ** 2, window, padding=padding, groups=target.shape[1]) - mu_target_sq
        sigma_pred_target = nn.functional.conv2d(pred * target, window, padding=padding, groups=pred.shape[1]) - mu_pred_target

        sigma_pred_sq = sigma_pred_sq.clamp_min(0.0)
        sigma_target_sq = sigma_target_sq.clamp_min(0.0)

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))


        return 1 - ssim_map.mean()

    def _compute_dice_loss(self, pred, target):
        """
        Compute Dice loss for segmentation quality.
        Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
        Dice loss = 1 - Dice coefficient

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Dice loss (scalar)
        """
        smooth = 1e-5  # Smoothing to avoid division by zero

        # Flatten spatial dimensions for each batch and channel
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)

        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        pred_sum = pred_flat.sum(dim=2)  # (B, C)
        target_sum = target_flat.sum(dim=2)  # (B, C)

        dice_coef = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

        # Clamp Dice coefficient to valid range [0, 1] to prevent numerical instabilities
        dice_coef = torch.clamp(dice_coef, 0.0, 1.0)

        return 1.0 - dice_coef.mean()

    def _compute_sobel_loss(self, pred, target):
        """
        Compute Sobel gradient loss between pred and target.
        Measures difference in spatial gradients (edges).

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Sobel gradient loss (scalar)
        """
        # Sobel kernels for horizontal and vertical gradients
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # Expand kernels to match number of channels
        num_channels = pred.shape[1]
        sobel_x = sobel_x.expand(num_channels, 1, 3, 3)
        sobel_y = sobel_y.expand(num_channels, 1, 3, 3)

        pred_grad_x = nn.functional.conv2d(pred, sobel_x, padding=1, groups=num_channels)
        pred_grad_y = nn.functional.conv2d(pred, sobel_y, padding=1, groups=num_channels)

        target_grad_x = nn.functional.conv2d(target, sobel_x, padding=1, groups=num_channels)
        target_grad_y = nn.functional.conv2d(target, sobel_y, padding=1, groups=num_channels)

        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + self.eps)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + self.eps)

        sobel_loss = torch.abs(pred_grad_mag - target_grad_mag).mean()

        return sobel_loss

    def _compute_tweedie_loss(self, pred, target):
        """
        Compute Tweedie loss for zero-inflated continuous data.
        Tweedie distributions are useful for modeling non-negative data with many zeros.

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Tweedie loss (scalar)
        """
        eps = 1e-8
        p = self.tweedie_p

        # Ensure positive predictions
        pred = pred.clamp(min=eps)

        # Tweedie deviance (negative log-likelihood)
        if abs(p - 1.0) < 1e-6:  # Poisson case
            loss = -target * torch.log(pred + eps) + pred
        elif abs(p - 2.0) < 1e-6:  # Gamma case
            loss = torch.log(pred + eps) + target / (pred + eps)
        else:  # General case (1 < p < 2)
            # Tweedie deviance formula
            loss = (target ** (2 - p)) / ((1 - p) * (2 - p)) - \
                   (target * (pred ** (1 - p))) / (1 - p) + \
                   (pred ** (2 - p)) / (2 - p)

        return loss.mean()

    def _compute_variance_loss(self, pred, target):
        """
        Compute variance matching loss (second moment matching).
        This helps preserve temporal variability in predictions.

        The loss penalizes the difference between the variance of predictions
        and the variance of targets across the spatial dimensions.

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Variance matching loss (scalar)
        """
        # Compute variance across spatial dimensions (H, W) for each batch and channel
        # var(X) = E[X^2] - E[X]^2
        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        target_mean = target.mean(dim=(2, 3), keepdim=True)

        pred_var = ((pred - pred_mean) ** 2).mean(dim=(2, 3))
        target_var = ((target - target_mean) ** 2).mean(dim=(2, 3))

        # L1 loss on variance difference (more robust than L2)
        variance_loss = torch.abs(pred_var - target_var).mean()

        return variance_loss

    def _compute_pinball_loss(self, pred, target):
        """
        Compute pinball loss (quantile loss) for quantile regression.
        The pinball loss is asymmetric and useful for estimating conditional quantiles.

        Formula: L(y, ŷ, τ) = max(τ(y - ŷ), (τ - 1)(y - ŷ))
                            = { τ(y - ŷ)     if y >= ŷ (underestimation)
                              { (1-τ)(ŷ - y) if y < ŷ  (overestimation)

        Supports multi-quantile mode: if pinball_quantile is a list, computes a
        weighted sum of pinball losses over each (weight, quantile) pair.

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Pinball loss (scalar). In multi-quantile mode the result is already
            the weighted combination; the caller should add it directly to total_loss.
        """
        snow_mask = target > self.threshold
        if snow_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        errors = target - pred

        if isinstance(self.pinball_quantile, (list, tuple)):
            weights = (
                self.pinball_weight
                if isinstance(self.pinball_weight, (list, tuple))
                else [self.pinball_weight] * len(self.pinball_quantile)
            )
            total = torch.tensor(0.0, device=pred.device)
            for q, w in zip(self.pinball_quantile, weights):
                if w > 0:
                    loss_q = torch.where(errors >= 0, q * errors, (q - 1) * errors)
                    total = total + w * loss_q[snow_mask].mean()
            return total
        else:
            loss = torch.where(
                errors >= 0,
                self.pinball_quantile * errors,
                (self.pinball_quantile - 1) * errors
            )
            return loss[snow_mask].mean()

    def _compute_log1p_loss(self, pred, target):
        """
        Compute log(1+y) loss - MSE in log space.
        This loss is particularly useful for zero-inflated data with large value ranges,
        as it treats relative errors more uniformly across different scales.

        Formula: L = mean((log(1 + pred) - log(1 + target))^2)

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Log(1+y) loss (scalar)
        """
        # Ensure non-negative values before applying log
        pred_pos = torch.clamp(pred, min=0.0)
        target_pos = torch.clamp(target, min=0.0)

        log_pred = torch.log1p(pred_pos)
        log_target = torch.log1p(target_pos)

        loss = ((log_pred - log_target) ** 2).mean()

        return loss

    def _compute_temporal_aggregate_loss(self, pred, target):
        """
        Compute temporal aggregate consistency loss.

        This ensures that the sum/mean of predictions over the batch dimension
        (representing temporal samples) matches the sum/mean of targets.
        Useful for ensuring mass conservation in precipitation/snow predictions.

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            Temporal aggregate consistency loss (scalar)
        """
        # Sum over batch dimension (temporal)
        pred_sum = pred.sum(dim=0)  # (C, H, W)
        target_sum = target.sum(dim=0)  # (C, H, W)

        sum_loss = ((pred_sum - target_sum) ** 2).mean()

        pred_mean = pred.mean(dim=0)  # (C, H, W)
        target_mean = target.mean(dim=0)  # (C, H, W)
        mean_loss = ((pred_mean - target_mean) ** 2).mean()

        return 0.00001 * sum_loss + 0.5 * mean_loss

    def _compute_mse_loss(self, pred, target):
        """
        Compute standard MSE (Mean Squared Error) loss.

        Formula: L = mean((pred - target)^2)

        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, C, H, W)

        Returns:
            MSE loss (scalar)
        """
        mse_loss = ((pred - target) ** 2).mean()
        return mse_loss

    def _compute_temperature_constraint_loss(self, pred, t2min, t2_lag_mean=None):
        """
        Temperature constraint loss: penalize snow predictions in physically warm regions.

        Two conditions trigger the no-snow constraint (OR logic):
          1. Standalone:  t2min > temp_threshold
          2. Combined:    t2_lag_mean > temp_lag_threshold  AND  t2min > temp_combined_threshold

        All thresholds are in normalized space (pre-converted from °C in main.py).

        Args:
            pred:        Predicted snow values (B, C, H, W)
            t2min:       Daily minimum temperature, normalized (B, 1, H, W) or (B, H, W)
            t2_lag_mean: Lagged mean temperature, normalized (B, 1, H, W) or (B, H, W); optional

        Returns:
            Temperature constraint loss (scalar)
        """
        if t2min.dim() == 3:
            t2min = t2min.unsqueeze(1)

        if t2min.shape[-2:] != pred.shape[-2:]:
            t2min = F.interpolate(t2min, size=pred.shape[-2:], mode='bilinear', align_corners=True)

        warm_mask = (t2min > self.temp_threshold)

        if (t2_lag_mean is not None
                and self.temp_lag_threshold is not None
                and self.temp_combined_threshold is not None):
            if t2_lag_mean.dim() == 3:
                t2_lag_mean = t2_lag_mean.unsqueeze(1)
            if t2_lag_mean.shape[-2:] != pred.shape[-2:]:
                t2_lag_mean = F.interpolate(t2_lag_mean, size=pred.shape[-2:], mode='bilinear', align_corners=True)
            warm_mask = warm_mask | ((t2_lag_mean > self.temp_lag_threshold) & (t2min > self.temp_combined_threshold))

        warm_mask = warm_mask.float()  # (B, 1, H, W)

        if warm_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_pos = torch.clamp(pred, min=0.0)
        n_warm = warm_mask.sum()
        constraint_loss = (warm_mask * pred_pos ** 2).sum() / (n_warm + 1e-8)

        return constraint_loss

    def forward(self, pred, target, temperature=None, temperature_lag=None):
        """
        Args:
            pred:            Model regression predictions (B, C, H, W)
            target:          Ground truth (B, C, H, W)
            temperature:     Optional t2min, normalized (B, 1, H, W) or (B, H, W).
            temperature_lag: Optional t2_lag_mean, normalized (B, 1, H, W) or (B, H, W).
        """
        total_loss = 0.0
        num_channels = pred.shape[1]
        B, C, H, W = target.shape

        _diag = {
            'regression_weighted': 0.0,
            'binary_weighted': 0.0,
        }

        for ch_idx in range(num_channels):
            pred_ch = pred[:, ch_idx:ch_idx+1, :, :]
            target_ch = target[:, ch_idx:ch_idx+1, :, :]
            weight = self.channel_weights[ch_idx] if ch_idx < len(self.channel_weights) else 1.0

            has_snow = (target_ch > self.threshold).float()

            logits = torch.log((pred_ch + self.eps) / (self.threshold + self.eps)) / self.temperature
            logits = logits.clamp(-20.0, 20.0)

            if self.dynamic_pos_weight:
                pos = has_snow.sum()
                neg = (has_snow == 0).sum()
                pw = (neg / (pos + self.eps)).clamp_(0.0, 15.0) if pos > 0 else torch.tensor(1.0, device=target.device, dtype=target.dtype)
            else:
                pw = None

            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, has_snow, pos_weight=pw, reduction='mean'
            )

            mask = has_snow
            if mask.sum() > 0:
                if self.regression_loss_type == 'smooth_l1':
                    loss_values = nn.functional.smooth_l1_loss(pred_ch, target_ch, beta=self.smooth_l1_beta, reduction='none')
                else:
                    loss_values = (pred_ch - target_ch) ** 2

                regression_loss = (mask * loss_values).sum() / (mask.sum() + 1e-8)
            else:
                regression_loss = torch.tensor(0.0, device=pred.device)

            channel_loss = self.alpha * regression_loss + (1 - self.alpha) * binary_loss

            total_loss += weight * channel_loss

            _diag['regression_weighted'] += float(weight * self.alpha * regression_loss.detach().item() if torch.is_tensor(regression_loss) else weight * self.alpha * regression_loss)
            _diag['binary_weighted'] += float(weight * (1 - self.alpha) * binary_loss.detach().item())

        _ch_norm = sum(self.channel_weights[:num_channels])
        total_loss = total_loss / _ch_norm
        _diag['regression_weighted'] /= _ch_norm
        _diag['binary_weighted'] /= _ch_norm

        if self.dice_weight > 0:
            dice_loss = self._compute_dice_loss(pred, target)
            total_loss = total_loss + self.dice_weight * dice_loss

        if self.ssim_weight > 0:
            ssim_loss = self._compute_ssim(pred, target)
            total_loss = total_loss + self.ssim_weight * ssim_loss

        if self.sobel_weight > 0:
            sobel_loss = self._compute_sobel_loss(pred, target)
            total_loss = total_loss + self.sobel_weight * sobel_loss

        if self.tweedie_weight > 0:
            tweedie_loss = self._compute_tweedie_loss(pred, target)
            total_loss = total_loss + self.tweedie_weight * tweedie_loss

        if self.variance_weight > 0:
            variance_loss = self._compute_variance_loss(pred, target)
            total_loss = total_loss + self.variance_weight * variance_loss

        _pinball_active = (
            isinstance(self.pinball_weight, (list, tuple)) and any(w > 0 for w in self.pinball_weight)
        ) or (
            not isinstance(self.pinball_weight, (list, tuple)) and self.pinball_weight > 0
        )
        if _pinball_active:
            pinball_loss = self._compute_pinball_loss(pred, target)
            if isinstance(self.pinball_weight, (list, tuple)):
                # Multi-quantile: _compute_pinball_loss already applies per-quantile weights
                total_loss = total_loss + pinball_loss
                _diag['pinball_weighted'] = float(pinball_loss.detach().item())
            else:
                total_loss = total_loss + self.pinball_weight * pinball_loss
                _diag['pinball_weighted'] = float(self.pinball_weight * pinball_loss.detach().item())

        if self.log1p_weight > 0:
            log1p_loss = self._compute_log1p_loss(pred, target)
            total_loss = total_loss + self.log1p_weight * log1p_loss
            _diag['log1p_weighted'] = float(self.log1p_weight * log1p_loss.detach().item())

        if self.temporal_agg_weight > 0:
            temporal_agg_loss = self._compute_temporal_aggregate_loss(pred, target)
            total_loss = total_loss + self.temporal_agg_weight * temporal_agg_loss

        if self.mse_weight > 0:
            mse_loss = self._compute_mse_loss(pred, target)
            total_loss = total_loss + self.mse_weight * mse_loss

        if self.temp_constraint_weight > 0 and temperature is not None:
            temp_constraint_loss = self._compute_temperature_constraint_loss(pred, temperature, temperature_lag)
            total_loss = total_loss + self.temp_constraint_weight * temp_constraint_loss
            _diag['temp_constraint_weighted'] = float(self.temp_constraint_weight * temp_constraint_loss.detach().item())

        _diag['total'] = float(total_loss.detach().item()) if torch.is_tensor(total_loss) else float(total_loss)
        self._last_components = _diag

        return total_loss


class DirectLoss(nn.Module):
    """Plain regression loss in (transformed) target space.

    Drop-in replacement for CombinedLoss when config.USE_DIRECT_LOSS is True.
    Computes a single mse / l1 / smooth_l1 term over the FULL field (no
    snow-presence masking). Intended for use with a variance-stabilizing target
    transform (e.g. log1p) applied in the dataset: in that space a plain
    regression loss handles snow presence and magnitude jointly, so the BCE /
    pinball / tweedie / temperature-constraint terms are no longer needed.

    The forward signature matches CombinedLoss (accepts, and ignores, the
    optional temperature / temperature_lag kwargs) so the Trainer loops are
    unchanged.

    Optional snow-pixel weighting: zero-inflated targets (many no-snow pixels
    at 0) bias a plain mean toward under-predicting snow. With
    snow_pixel_weight > 0, pixels whose target exceeds snow_threshold get a
    per-pixel weight of (1 + snow_pixel_weight) while the rest stay at 1, and
    the loss is a WEIGHTED MEAN (normalized by the total weight). Because the
    normalization tracks the weights, snow_pixel_weight = 0 reduces exactly to
    the plain mean (== F.mse_loss / l1 / smooth_l1). snow_threshold is in the
    same (transformed) units as the target, e.g. log1p space.
    """

    def __init__(self, loss_type='mse', smooth_l1_beta=1.0,
                 snow_pixel_weight=0.0, snow_threshold=1e-3):
        super(DirectLoss, self).__init__()
        self.loss_type = loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.snow_pixel_weight = snow_pixel_weight
        self.snow_threshold = snow_threshold
        self._last_components = {}
        msg = f"  Direct loss: {loss_type}"
        if loss_type == 'smooth_l1':
            msg += f" (beta={smooth_l1_beta})"
        if snow_pixel_weight > 0:
            msg += f" | snow-pixel weight: +{snow_pixel_weight} for target>{snow_threshold:g}"
        print(msg)

    def forward(self, pred, target, temperature=None, temperature_lag=None):
        if self.loss_type == 'l1':
            per_pixel = torch.abs(pred - target)
        elif self.loss_type == 'smooth_l1':
            per_pixel = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta, reduction='none')
        else:
            per_pixel = (pred - target) ** 2

        if self.snow_pixel_weight > 0:
            w = 1.0 + self.snow_pixel_weight * (target > self.snow_threshold).to(per_pixel.dtype)
            loss = (w * per_pixel).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = per_pixel.mean()

        _v = float(loss.detach().item())
        self._last_components = {f'direct_{self.loss_type}': _v, 'total': _v}
        return loss


class Stage1Trainer:
    """
    Trainer for Stage 1 only: LowResSnowPriorNet.
    Trains the low-res prior network on downsampled targets.
    """

    def __init__(self, model, train_loader, val_loader, criterion=None):
        """
        Initialize Stage 1 trainer.

        Args:
            model: LowResSnowPriorNet model
            train_loader: Training dataloader (should provide low-res dynamic data)
            val_loader: Validation dataloader
            criterion: Loss function (default: CombinedLoss)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(CHECKPOINT_DIR) / "stage1_pretrain"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        decay_params, no_decay_params = separate_weight_decay_params(self.model)
        self.optimizer = NAdam([
            {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=LEARNING_RATE, decoupled_weight_decay=True)

        try:
            from config import WARMUP_EPOCHS, WARMUP_START_LR
            self.warmup_epochs = WARMUP_EPOCHS
            self.warmup_start_lr = WARMUP_START_LR
            self.use_warmup = WARMUP_EPOCHS > 0
        except ImportError:
            self.warmup_epochs = 0
            self.warmup_start_lr = LEARNING_RATE
            self.use_warmup = False

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR
        )

        if NORMALIZE_OUTPUTS:
            channel_weights = [1.0]
        else:
            channel_weights = self._compute_channel_weights_from_stats()

        pos_weight = self._compute_pos_weight_from_stats()

        if criterion is None:
            self.criterion = CombinedLoss(
                alpha=0.70,
                channel_weights=channel_weights,
                pos_weight=pos_weight,
                dynamic_pos_weight=False,
                dice_weight=0.050,
                ssim_weight=0.500,
                sobel_weight=0.01,
                tweedie_weight=1.0,
                tweedie_p=1.5,
                variance_weight=0.0,
                pinball_weight=0.0,
                pinball_quantile=0.5,
                log1p_weight=0.5,
                temporal_agg_weight=0.0,
                mse_weight=0.0
            )
        else:
            self.criterion = criterion

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae_ch0': [],
            'val_mae_ch0': [],
            'train_rmse_ch0': [],
            'val_rmse_ch0': [],
            'lr': [],
            'epoch_time': []
        }

        self.best_val_loss = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_mae = float('inf')
        self.epochs_without_improvement = 0

        print(f"Stage 1 Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"    - With weight decay: {sum(p.numel() for p in decay_params):,}")
        print(f"    - Without weight decay (norms, biases): {sum(p.numel() for p in no_decay_params):,}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        if self.use_warmup:
            print(f"  Warmup: {self.warmup_epochs} epochs ({self.warmup_start_lr:.2e} -> {LEARNING_RATE:.2e})")
        else:
            print(f"  Warmup: Disabled")

    def _update_lr_warmup(self, epoch):
        """Update learning rate during warmup period."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lr = self.warmup_start_lr + (LEARNING_RATE - self.warmup_start_lr) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False

    def _compute_channel_weights_from_stats(self):
        """Compute channel weights from normalization stats."""
        import json
        stats_file = Path("normalization_stats.json")
        if not stats_file.exists():
            return [1.0]
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            if 'target' in stats:
                target_stats = stats['target']
                target_mean = [target_stats[var]['mean'] for var in sorted(target_stats.keys())]
            elif 'target_mean' in stats:
                target_mean = stats['target_mean']
            else:
                return [1.0]
            max_of_all = max(target_mean)
            weights = [max_of_all / max_val if max_val > 0 else 1.0 for max_val in target_mean]
            print(f"  Stage 1 channel weights: {weights}")
            return weights
        except Exception as e:
            print(f"  Warning: Error loading stats: {e}, using default [1.0]")
            return [1.0]

    def _compute_pos_weight_from_stats(self):
        """Compute pos_weight from normalization stats."""
        import json
        stats_file = Path("normalization_stats.json")
        if not stats_file.exists():
            return [5.0]
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            if 'target' in stats:
                target_stats = stats['target']
                target_mean = [target_stats[var]['mean'] for var in sorted(target_stats.keys())]
                target_max = [target_stats[var]['max'] for var in sorted(target_stats.keys())]
            elif 'target_mean' in stats and 'target_max' in stats:
                target_mean = stats['target_mean']
                target_max = stats['target_max']
            else:
                return [5.0]
            pos_weights = []
            for mean_val, max_val in zip(target_mean, target_max):
                if max_val > 0 and mean_val > 0:
                    ratio = max_val / mean_val
                    pos_weight_ch = min(max(ratio / 10.0, 1.0), 50.0)
                    pos_weights.append(pos_weight_ch)
                else:
                    pos_weights.append(5.0)
            print(f"  Stage 1 pos_weight: {[f'{w:.2f}' for w in pos_weights]}")
            return pos_weights
        except Exception as e:
            print(f"  Warning: Error computing pos_weight: {e}, using default [5.0]")
            return [5.0]

    def compute_per_channel_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute per-channel MAE and RMSE metrics."""
        metrics = {}
        num_channels = outputs.shape[1]
        for ch_idx in range(num_channels):
            pred_ch = outputs[:, ch_idx, :, :]
            target_ch = targets[:, ch_idx, :, :]
            mae_ch = torch.abs(pred_ch - target_ch).mean()
            rmse_ch = torch.sqrt(((pred_ch - target_ch) ** 2).mean())
            metrics[f'mae_ch{ch_idx}'] = mae_ch.item()
            metrics[f'rmse_ch{ch_idx}'] = rmse_ch.item()
        metrics['mae'] = torch.abs(outputs - targets).mean().item()
        metrics['rmse'] = torch.sqrt(((outputs - targets) ** 2).mean()).item()
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {
            'mae': 0.0, 'rmse': 0.0,
            'mae_ch0': 0.0, 'rmse_ch0': 0.0
        }

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack: (static_highres, dynamic_lowres, targets, landmask)
            static_highres, dynamic_lowres, targets, landmask = batch_data
            dynamic_lowres = dynamic_lowres.to(self.device)
            targets = targets.to(self.device)

            # Downsample targets to match low-res dynamic data resolution
            lowres_targets = F.interpolate(
                targets,
                size=dynamic_lowres.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

            self.optimizer.zero_grad()

            outputs = self.model(dynamic_lowres)
            loss = self.criterion(outputs, lowres_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                GRAD_CLIP_MAX_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()
            batch_metrics = self.compute_per_channel_metrics(outputs, lowres_targets)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            _loss_val = loss.item()
            _diag_dump = (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0 or _loss_val > 40.0
            if _diag_dump:
                print(f"    Batch {batch_idx+1}/{len(self.train_loader)} - "
                      f"Loss: {_loss_val:.4f}, MAE: {batch_metrics['mae']:.4f}")
                comp = getattr(self.criterion, '_last_components', None)
                if comp is not None:
                    parts = ", ".join(f"{k}={v:.4f}" for k, v in comp.items() if k != 'total')
                    print(f"      [components] {parts}")
                with torch.no_grad():
                    t = lowres_targets
                    o = outputs
                    print(f"      [target]  min={t.min().item():.3f} max={t.max().item():.3f} "
                          f"mean={t.mean().item():.3f} p99={torch.quantile(t.flatten(), 0.99).item():.3f} "
                          f"frac_snow={(t>0.1).float().mean().item():.4f}")
                    print(f"      [pred]    min={o.min().item():.3f} max={o.max().item():.3f} "
                          f"mean={o.mean().item():.3f} p99={torch.quantile(o.flatten(), 0.99).item():.3f}")
                    # Pred over warm regions (channel 4 = t2min normalized)
                    if dynamic_lowres.shape[1] > 4:
                        t2min = dynamic_lowres[:, 4:5, :, :]
                        if t2min.shape[-2:] != o.shape[-2:]:
                            t2min_up = F.interpolate(t2min, size=o.shape[-2:], mode='bilinear', align_corners=True)
                        else:
                            t2min_up = t2min
                        warm = (t2min_up > self.criterion.temp_threshold).float()
                        if warm.sum() > 0:
                            pred_pos = o.clamp(min=0.0)
                            warm_pred_max = (pred_pos * warm).max().item()
                            warm_pred_mean = (pred_pos * warm).sum().item() / warm.sum().item()
                            print(f"      [warm]    n_warm_px={int(warm.sum().item())} "
                                  f"pred_mean_warm={warm_pred_mean:.4f} pred_max_warm={warm_pred_max:.4f}")

        n_batches = len(self.train_loader)
        result = {'loss': total_loss / n_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / n_batches

        return result

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {
            'mae': 0.0, 'rmse': 0.0,
            'mae_ch0': 0.0, 'rmse_ch0': 0.0
        }

        for batch_data in self.val_loader:
            # Unpack: (static_highres, dynamic_lowres, targets, landmask)
            static_highres, dynamic_lowres, targets, landmask = batch_data
            dynamic_lowres = dynamic_lowres.to(self.device)
            targets = targets.to(self.device)

            # Downsample targets to match low-res dynamic data resolution
            lowres_targets = F.interpolate(
                targets,
                size=dynamic_lowres.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

            outputs = self.model(dynamic_lowres)
            loss = self.criterion(outputs, lowres_targets)

            total_loss += loss.item()
            batch_metrics = self.compute_per_channel_metrics(outputs, lowres_targets)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

        n_batches = len(self.val_loader)
        result = {'loss': total_loss / n_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / n_batches

        return result

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_rmse': self.best_val_rmse,
            'best_val_mae': self.best_val_mae
        }

        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"    Saved Stage 1 checkpoint: {checkpoint_path.name}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"    ✓ New best Stage 1 model saved (val_rmse: {self.best_val_mae:.4f}, val_rmse: {self.best_val_rmse:.4f}, val_loss: {self.best_val_loss:.4f})")

    def train(self, epochs: int = None, start_epoch: int = 0) -> Dict:
        """
        Train Stage 1 model.

        Args:
            epochs: Number of epochs (default: use config EPOCHS)
            start_epoch: Epoch to start from (for resuming)
        """
        if epochs is None:
            epochs = EPOCHS

        print("\n" + "=" * 70)
        print(" " * 20 + "STAGE 1 PRETRAINING")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        if start_epoch > 0:
            print(f"Resuming from epoch: {start_epoch}")
        print("=" * 70 + "\n")

        training_status = "completed"
        final_epoch = start_epoch

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            print(f"Epoch {epoch+1}/{epochs}")
            print("-" * 70)

            in_warmup = False
            if self.use_warmup:
                in_warmup = self._update_lr_warmup(epoch)
                if in_warmup:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  [Warmup] LR: {current_lr:.6f}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start_time

            if not in_warmup:
                self.scheduler.step(val_metrics['loss'])

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['train_mae_ch0'].append(train_metrics['mae_ch0'])
            self.history['val_mae_ch0'].append(val_metrics['mae_ch0'])
            self.history['train_rmse_ch0'].append(train_metrics['rmse_ch0'])
            self.history['val_rmse_ch0'].append(val_metrics['rmse_ch0'])
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Epoch time: {epoch_time:.2f}s")

            is_best = val_metrics['mae'] < self.best_val_mae
            if is_best:
                self.best_val_rmse = val_metrics['rmse']
                self.best_val_mae = val_metrics['mae']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch + 1, is_best)

            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                training_status = "early_stopped"
                final_epoch = epoch + 1
                break

            print()
        else:
            training_status = "completed"
            final_epoch = epoch + 1

        print("=" * 70)
        print(f"Stage 1 pretraining completed! Best val RMSE: {self.best_val_rmse:.4f}")
        print("=" * 70)

        return self.history


# ============================================================================
# MAIN TRAINER CLASS (For Joint Training or Single-Stage Models)
# ============================================================================


class Trainer:
    """Simple trainer for ERA5 snow prediction models."""

    def __init__(self, model, train_loader, val_loader, criterion=None, loss_weights=None, scheduler="ReduceLROnPlateau", temp_min_channel_idx=None, temp_lag_channel_idx=None):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function (default: AsymmetricZeroPenaltyLoss with penalty=10.0)
            loss_weights: Dictionary of loss weights for CombinedLoss. Available keys:
                - alpha: Weight balance between regression and classification (default: 0.70)
                - dice_weight: Weight for Dice loss (default: 0.010)
                - ssim_weight: Weight for SSIM loss (default: 0.500)
                - sobel_weight: Weight for Sobel gradient loss (default: 0.050)
                - tweedie_weight: Weight for Tweedie loss (default: 1.2)
                - tweedie_p: Power parameter for Tweedie loss (default: 1.5)
                - variance_weight: Weight for variance matching loss (default: 0.0)
                - pinball_weight: Weight for pinball loss (default: 0.0)
                - pinball_quantile: Quantile for pinball loss (default: 0.95)
                - log1p_weight: Weight for log(1+y) loss (default: 2.0)
                - temporal_agg_weight: Weight for temporal aggregate loss (default: 0.0)
                - mse_weight: Weight for MSE loss (default: 0.0)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.is_dual_encoder = model.__class__.__name__ in ['DualEncoderUNet', 'DualEncoderSwinUNet', 'DualEncoderFNOUNet']

        self.is_two_stage = model.__class__.__name__ == 'TwoStageOTModel'

        # Channel indices for temperature constraint inputs (None = disabled)
        self.temp_min_channel_idx = temp_min_channel_idx
        self.temp_lag_channel_idx = temp_lag_channel_idx

        if USE_TORCH_COMPILE:
            print(f"  Compiling model with mode='{COMPILE_MODE}'...")
            self.model = torch.compile(self.model, mode=COMPILE_MODE)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if _HAS_DECODER_WEIGHT_DECAY and _HAS_CNN_ENCODER_WEIGHT_DECAY:
            swin_decay, cnn_encoder_decay, decoder_decay, no_decay_params = \
                separate_weight_decay_params_by_component(self.model)
            self.optimizer = NAdam([
                {'params': swin_decay,         'weight_decay': WEIGHT_DECAY},
                {'params': cnn_encoder_decay,  'weight_decay': CNN_ENCODER_WEIGHT_DECAY},
                {'params': decoder_decay,      'weight_decay': DECODER_WEIGHT_DECAY},
                {'params': no_decay_params,    'weight_decay': 0.0}
            ], lr=LEARNING_RATE, decoupled_weight_decay = True)
            print(f"    - Swin encoder      (WD={WEIGHT_DECAY}): {sum(p.numel() for p in swin_decay):,}")
            print(f"    - CNN encoder       (WD={CNN_ENCODER_WEIGHT_DECAY}): {sum(p.numel() for p in cnn_encoder_decay):,}")
            print(f"    - CNN decoder       (WD={DECODER_WEIGHT_DECAY}): {sum(p.numel() for p in decoder_decay):,}")
            print(f"    - No WD (norms, biases): {sum(p.numel() for p in no_decay_params):,}")
        else:
            all_params = list(self.model.parameters())
            self.optimizer = NAdam(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)
            print(f"    - All parameters (WD={WEIGHT_DECAY}): {sum(p.numel() for p in all_params):,}")

        try:
            from config import WARMUP_EPOCHS, WARMUP_START_LR
            self.warmup_epochs = WARMUP_EPOCHS
            self.warmup_start_lr = WARMUP_START_LR
            self.use_warmup = WARMUP_EPOCHS > 0
        except ImportError:
            self.warmup_epochs = 0
            self.warmup_start_lr = LEARNING_RATE
            self.use_warmup = False

        if scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=LR_SCHEDULER_PATIENCE,
                factor=LR_SCHEDULER_FACTOR
            )
        elif scheduler == 'CosineAnnealingLR':
            try:
                from config import COSINE_T_MAX
                t_max = COSINE_T_MAX
            except ImportError:
                t_max = 100
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max
            )
            print(f"  CosineAnnealingLR T_max = {t_max}")
        print(f"Use the {scheduler}")


        self.current_training_epoch = 0

        # Compute channel weights from normalization statistics (only if not using output normalization)
        if NORMALIZE_OUTPUTS:
            # Output normalization: single channel on [0, 1] scale
            channel_weights = [1.0]
        else:
            # No output normalization: use computed weights to balance different scales
            channel_weights = self._compute_channel_weights_from_stats()

        pos_weight = None
        pos_weight = self._compute_pos_weight_from_stats()

        default_loss_weights = {
            'alpha': 0.70,
            'dice_weight': 0.010,
            'ssim_weight': 0.500,
            'sobel_weight': 0.050,
            'tweedie_weight': 1.2,
            'tweedie_p': 1.5,
            'variance_weight': 0.0,
            'pinball_weight': 0.0,
            'pinball_quantile': 0.95,
            'log1p_weight': 2.0,
            'temporal_agg_weight': 0.0,
            'mse_weight': 0.0,
            'threshold': 1e-4,
            'dynamic_pos_weight': False,
            'temp_constraint_weight': 0.0,
            'temp_threshold': 15.0,
            'temp_combined_threshold': None,
            'temp_lag_threshold': None,
            'smooth_l1_beta': 5.0,
        }

        if loss_weights is not None:
            default_loss_weights.update(loss_weights)

        self.loss_weights = default_loss_weights

        # Optionally replace the composite CombinedLoss with a plain regression
        # loss in transformed target space (opt-in via config.USE_DIRECT_LOSS).
        try:
            import config as _project_config
            _use_direct = getattr(_project_config, 'USE_DIRECT_LOSS', False)
            _direct_type = getattr(_project_config, 'DIRECT_LOSS_TYPE', 'mse')
            _direct_beta = float(getattr(_project_config, 'DIRECT_SMOOTH_L1_BETA', 1.0))
            _snow_w = float(getattr(_project_config, 'DIRECT_SNOW_PIXEL_WEIGHT', 0.0))
            _snow_thr = float(getattr(_project_config, 'DIRECT_SNOW_THRESHOLD', 1e-3))
        except Exception:
            _use_direct, _direct_type, _direct_beta = False, 'mse', 1.0
            _snow_w, _snow_thr = 0.0, 1e-3

        if _use_direct:
            self.criterion = DirectLoss(loss_type=_direct_type, smooth_l1_beta=_direct_beta,
                                        snow_pixel_weight=_snow_w, snow_threshold=_snow_thr)
        else:
            self.criterion = CombinedLoss(
                alpha=self.loss_weights['alpha'],
                channel_weights=channel_weights,
                pos_weight=pos_weight,
                dynamic_pos_weight=self.loss_weights['dynamic_pos_weight'],
                dice_weight=self.loss_weights['dice_weight'],
                ssim_weight=self.loss_weights['ssim_weight'],
                sobel_weight=self.loss_weights['sobel_weight'],
                tweedie_weight=self.loss_weights['tweedie_weight'],
                tweedie_p=self.loss_weights['tweedie_p'],
                variance_weight=self.loss_weights['variance_weight'],
                pinball_weight=self.loss_weights['pinball_weight'],
                pinball_quantile=self.loss_weights['pinball_quantile'],
                log1p_weight=self.loss_weights['log1p_weight'],
                temporal_agg_weight=self.loss_weights['temporal_agg_weight'],
                mse_weight=self.loss_weights['mse_weight'],
                threshold=self.loss_weights['threshold'],
                temp_constraint_weight=self.loss_weights['temp_constraint_weight'],
                temp_threshold=self.loss_weights['temp_threshold'],
                temp_combined_threshold=self.loss_weights['temp_combined_threshold'],
                temp_lag_threshold=self.loss_weights['temp_lag_threshold'],
                smooth_l1_beta=self.loss_weights['smooth_l1_beta'],
            )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae_ch0': [],
            'val_mae_ch0': [],
            'train_rmse_ch0': [],
            'val_rmse_ch0': [],
            'lr': [],
            'epoch_time': []  # Time per epoch in seconds
        }

        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_rmse = float('inf')
        self.epochs_without_improvement = 0

        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_warmup:
            print(f"  Warmup: {self.warmup_epochs} epochs ({self.warmup_start_lr:.2e} -> {LEARNING_RATE:.2e})")
        else:
            print(f"  Warmup: Disabled")

    def _update_lr_warmup(self, epoch):
        """Update learning rate during warmup period."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lr = self.warmup_start_lr + (LEARNING_RATE - self.warmup_start_lr) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return True
        return False

    def _compute_channel_weights_from_stats(self):
        """Compute channel weights based on target max values from normalization stats."""
        import json

        stats_file = Path("normalization_stats.json")
        if not stats_file.exists():
            print(f"  Warning: {stats_file} not found, using default weight [1.0]")
            return [1.0]

        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            if 'target' in stats:
                target_stats = stats['target']
                target_mean = [target_stats[var]['mean'] for var in sorted(target_stats.keys())]
            elif 'target_mean' in stats:
                target_mean = stats['target_mean']
            else:
                print(f"  Warning: No target statistics found in {stats_file}, using default weight [1.0]")
                return [1.0]

            max_of_all = max(target_mean)

            # Compute weights: weight_i = max_of_all / max_i
            # This makes the loss contribution equal across channels when errors are at max scale
            weights = [max_of_all / max_val if max_val > 0 else 1.0 for max_val in target_mean]

            print(f"  Auto-computed channel weights from target_mean {target_mean}:")
            print(f"    Channel weights: {weights}")

            return weights

        except Exception as e:
            print(f"  Warning: Error loading stats file: {e}, using default weight [1.0]")
            return [1.0]

    def _compute_pos_weight_from_stats(self):
        """
        Compute pos_weight for binary cross-entropy based on class imbalance.
        pos_weight = (count of no-snow pixels) / (count of snow pixels)

        Higher values mean the model pays more attention to detecting snow presence.
        For sparse snow data, this is typically > 1.
        """
        import json

        stats_file = Path("normalization_stats.json")
        if not stats_file.exists():
            print(f"  Warning: {stats_file} not found, using default pos_weight [5.0]")
            return [5.0]

        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            threshold = 1e-8  # Same as loss threshold
            pos_weights = []

            if 'target' in stats:
                target_stats = stats['target']
                target_mean = [target_stats[var]['mean'] for var in sorted(target_stats.keys())]
                target_max = [target_stats[var]['max'] for var in sorted(target_stats.keys())]
            elif 'target_mean' in stats and 'target_max' in stats:
                target_mean = stats['target_mean']
                target_max = stats['target_max']
            else:
                print(f"  Warning: No target statistics found in {stats_file}, using default pos_weight [5.0]")
                return [5.0]

            # Estimate class imbalance from target statistics
            # Using mean/max ratio to estimate proportion of non-zero pixels
            for ch_idx, (mean_val, max_val) in enumerate(zip(target_mean, target_max)):
                if max_val > 0 and mean_val > 0:
                    # Estimate: if mean << max, data is sparse (few positive pixels)
                    # pos_weight ≈ (1 - sparsity) / sparsity where sparsity = mean/max
                    # Simple heuristic: pos_weight = max / mean (capped for stability)
                    ratio = max_val / mean_val
                    # Cap at reasonable values to avoid extreme weights
                    pos_weight_ch = min(max(ratio / 10.0, 1.0), 50.0)
                    pos_weights.append(pos_weight_ch)
                else:
                    pos_weights.append(5.0)  # Default moderate weight

            print(f"  Auto-computed pos_weight from target statistics:")
            print(f"    target_mean: {target_mean}")
            print(f"    target_max:  {target_max}")
            print(f"    pos_weight:  {[f'{w:.2f}' for w in pos_weights]}")
            print(f"  (Higher pos_weight = more attention to snow presence)")

            return pos_weights

        except Exception as e:
            print(f"  Warning: Error computing pos_weight: {e}, using default [5.0]")
            return [5.0]

    def compute_per_channel_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-channel MAE and RMSE metrics.

        Args:
            outputs: Model predictions (B, C, H, W)
            targets: Ground truth (B, C, H, W)

        Returns:
            Dictionary with per-channel metrics
        """
        metrics = {}
        num_channels = outputs.shape[1]

        for ch_idx in range(num_channels):
            pred_ch = outputs[:, ch_idx, :, :]
            target_ch = targets[:, ch_idx, :, :]

            mae_ch = torch.abs(pred_ch - target_ch).mean()
            rmse_ch = torch.sqrt(((pred_ch - target_ch) ** 2).mean())

            metrics[f'mae_ch{ch_idx}'] = mae_ch.item()
            metrics[f'rmse_ch{ch_idx}'] = rmse_ch.item()

        metrics['mae'] = torch.abs(outputs - targets).mean().item()
        metrics['rmse'] = torch.sqrt(((outputs - targets) ** 2).mean()).item()

        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch, returning per-channel metrics."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {
            'mae': 0.0, 'rmse': 0.0,
            'mae_ch0': 0.0, 'rmse_ch0': 0.0
        }

        for batch_idx, batch_data in enumerate(self.train_loader):
            if self.is_dual_encoder or self.is_two_stage:
                # DualEncoderUNet or TwoStageOTModel: (static_highres, dynamic_lowres, targets, landmask)
                static_highres, dynamic_lowres, targets, landmask = batch_data
                static_highres = static_highres.to(self.device)
                dynamic_lowres = dynamic_lowres.to(self.device)
            else:
                # Single encoder models: (inputs, targets, landmask)
                inputs, targets, landmask = batch_data
                inputs = inputs.to(self.device)

            targets = targets.to(self.device)
            landmask = landmask.to(self.device)

            # Extract temperature channels for constraint loss (if configured)
            temperature = None
            temperature_lag = None
            if self.temp_min_channel_idx is not None:
                src = dynamic_lowres if (self.is_dual_encoder or self.is_two_stage) else inputs
                temperature = src[:, self.temp_min_channel_idx:self.temp_min_channel_idx + 1, :, :]
                if self.temp_lag_channel_idx is not None:
                    temperature_lag = src[:, self.temp_lag_channel_idx:self.temp_lag_channel_idx + 1, :, :]

            self.optimizer.zero_grad()

            if self.is_two_stage:
                # Two-stage model: get both low-res prior and high-res output
                lowres_prior, outputs = self.model(static_highres, dynamic_lowres, landmask=landmask, return_prior=True)
                loss = self.criterion(lowres_prior, outputs, targets)
            elif self.is_dual_encoder:
                outputs = self.model(static_highres, dynamic_lowres, landmask=landmask)
                loss = self.criterion(outputs, targets, temperature=temperature, temperature_lag=temperature_lag)
            else:
                outputs = self.model(inputs, landmask=landmask)
                loss = self.criterion(outputs, targets, temperature=temperature, temperature_lag=temperature_lag)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                GRAD_CLIP_MAX_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()
            batch_metrics = self.compute_per_channel_metrics(outputs, targets)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            _loss_val = loss.item()
            _diag_dump = (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0 or _loss_val > 80.0
            if _diag_dump:
                print(f"    Batch {batch_idx+1}/{len(self.train_loader)} - "
                      f"Loss: {_loss_val:.4f}, MAE: {batch_metrics['mae']:.4f}, "
                      f"Ch0: {batch_metrics['mae_ch0']:.2f}")
                comp = getattr(self.criterion, '_last_components', None)
                if comp is not None:
                    parts = ", ".join(f"{k}={v:.4f}" for k, v in comp.items() if k != 'total')
                    print(f"      [components] {parts}")
                with torch.no_grad():
                    t = targets
                    o = outputs
                    print(f"      [target]  min={t.min().item():.3f} max={t.max().item():.3f} "
                          f"mean={t.mean().item():.3f} p99={torch.quantile(t.flatten(), 0.99).item():.3f} "
                          f"frac_snow={(t>0.1).float().mean().item():.4f}")
                    print(f"      [pred]    min={o.min().item():.3f} max={o.max().item():.3f} "
                          f"mean={o.mean().item():.3f} p99={torch.quantile(o.flatten(), 0.99).item():.3f}")
                    if temperature is not None and hasattr(self.criterion, 'temp_threshold'):
                        t2min_up = temperature
                        if t2min_up.shape[-2:] != o.shape[-2:]:
                            t2min_up = F.interpolate(t2min_up, size=o.shape[-2:], mode='bilinear', align_corners=True)
                        warm = (t2min_up > self.criterion.temp_threshold).float()
                        if warm.sum() > 0:
                            pred_pos = o.clamp(min=0.0)
                            warm_pred_max = (pred_pos * warm).max().item()
                            warm_pred_mean = (pred_pos * warm).sum().item() / warm.sum().item()
                            print(f"      [warm]    n_warm_px={int(warm.sum().item())} "
                                  f"pred_mean_warm={warm_pred_mean:.4f} pred_max_warm={warm_pred_max:.4f}")

        n_batches = len(self.train_loader)
        result = {'loss': total_loss / n_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / n_batches

        return result

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set, returning per-channel metrics."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {
            'mae': 0.0, 'rmse': 0.0,
            'mae_ch0': 0.0, 'rmse_ch0': 0.0
        }
        comp_sums: Dict[str, float] = {}
        comp_counts: Dict[str, int] = {}

        for batch_data in self.val_loader:
            if self.is_dual_encoder or self.is_two_stage:
                # DualEncoderUNet or TwoStageOTModel: (static_highres, dynamic_lowres, targets, landmask)
                static_highres, dynamic_lowres, targets, landmask = batch_data
                static_highres = static_highres.to(self.device)
                dynamic_lowres = dynamic_lowres.to(self.device)
            else:
                # Single encoder models: (inputs, targets, landmask)
                inputs, targets, landmask = batch_data
                inputs = inputs.to(self.device)

            targets = targets.to(self.device)
            landmask = landmask.to(self.device)

            # Extract temperature channels for constraint loss (if configured)
            temperature = None
            temperature_lag = None
            if self.temp_min_channel_idx is not None:
                src = dynamic_lowres if (self.is_dual_encoder or self.is_two_stage) else inputs
                temperature = src[:, self.temp_min_channel_idx:self.temp_min_channel_idx + 1, :, :]
                if self.temp_lag_channel_idx is not None:
                    temperature_lag = src[:, self.temp_lag_channel_idx:self.temp_lag_channel_idx + 1, :, :]

            if self.is_two_stage:
                # Two-stage model: get both low-res prior and high-res output
                lowres_prior, outputs = self.model(static_highres, dynamic_lowres, landmask=landmask, return_prior=True)
                loss = self.criterion(lowres_prior, outputs, targets)
            elif self.is_dual_encoder:
                outputs = self.model(static_highres, dynamic_lowres, landmask=landmask)
                loss = self.criterion(outputs, targets, temperature=temperature, temperature_lag=temperature_lag)
            else:
                outputs = self.model(inputs, landmask=landmask)
                loss = self.criterion(outputs, targets, temperature=temperature, temperature_lag=temperature_lag)

            total_loss += loss.item()
            batch_metrics = self.compute_per_channel_metrics(outputs, targets)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            comp = getattr(self.criterion, '_last_components', None)
            if comp is not None:
                for k, v in comp.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
                    comp_counts[k] = comp_counts.get(k, 0) + 1

        n_batches = len(self.val_loader)
        result = {'loss': total_loss / n_batches}
        for key in total_metrics:
            result[key] = total_metrics[key] / n_batches

        # Stash per-component validation loss breakdown on the result dict;
        # the train loop prints it next to the "Val - Loss: ..." line.
        if comp_sums:
            comp_means = {k: comp_sums[k] / comp_counts[k] for k in comp_sums}
            for k, v in comp_means.items():
                result[f'comp_{k}'] = v

        return result

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'best_val_rmse': self.best_val_rmse
        }

        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"    Saved checkpoint: {checkpoint_path.name}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"    ✓ New best model saved (val_rmse: {self.best_val_rmse:.4f}, val_loss: {self.best_val_loss:.4f}, val_mae: {self.best_val_mae:.4f})")

    def train(self, start_epoch: int = 0) -> Dict:
        """
        Train the model.

        Args:
            start_epoch: Epoch to start training from (for resuming)

        Returns:
            Training history dictionary
        """
        # Import config to get current EPOCHS value (may be modified by two_stage_train)
        import config
        current_epochs = config.EPOCHS

        print("\n" + "=" * 70)
        print(" " * 25 + "TRAINING")
        print("=" * 70)
        print(f"Epochs: {current_epochs}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        if start_epoch > 0:
            print(f"Resuming from epoch: {start_epoch}")
        print("=" * 70 + "\n")

        # Initialize status variables (in case loop doesn't run)
        training_status = "completed"
        final_epoch = start_epoch

        for epoch in range(start_epoch, current_epochs):
            epoch_start_time = time.time()

            print(f"Epoch {epoch+1}/{current_epochs}")
            print("-" * 70)

            in_warmup = False
            if self.use_warmup:
                in_warmup = self._update_lr_warmup(epoch)
                if in_warmup:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  [Warmup] LR: {current_lr:.6f}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start_time

            # Reset CosineAnnealingLR counter when warmup just finished
            if self.use_warmup and epoch == self.warmup_epochs and isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.last_epoch = -1

            # Update scheduler (only after warmup completes)
            if not in_warmup:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])

            self.history['train_mae_ch0'].append(train_metrics['mae_ch0'])
            self.history['val_mae_ch0'].append(val_metrics['mae_ch0'])
            self.history['train_rmse_ch0'].append(train_metrics['rmse_ch0'])
            self.history['val_rmse_ch0'].append(val_metrics['rmse_ch0'])

            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(f"          Ch0 MAE: {train_metrics['mae_ch0']:.2f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            print(f"          Ch0 MAE: {val_metrics['mae_ch0']:.2f}")
            _val_comp_parts = ", ".join(
                f"{k[len('comp_'):]}={v:.4f}"
                for k, v in val_metrics.items()
                if k.startswith('comp_') and k != 'comp_total'
            )
            if _val_comp_parts:
                print(f"          [components] {_val_comp_parts}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Epoch time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)")

            is_best = val_metrics['mae'] < self.best_val_mae
            if is_best:
                self.best_val_rmse = val_metrics['rmse']
                self.best_val_loss = val_metrics['loss']
                self.best_val_mae = val_metrics['mae']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch + 1, is_best)

            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                training_status = "early_stopped"
                final_epoch = epoch + 1
                break

            print()
        else:
            training_status = "completed"
            final_epoch = epoch + 1

        print("=" * 70)
        print(f"Training completed! Best val RMSE: {self.best_val_rmse:.4f} MAE: {self.best_val_mae:.4f}")

        if len(self.history['epoch_time']) > 0:
            total_time = sum(self.history['epoch_time'])
            avg_time = total_time / len(self.history['epoch_time'])
            print(f"\nTiming Summary:")
            print(f"  Total training time: {total_time:.2f}s ({total_time/60:.2f}m / {total_time/3600:.2f}h)")
            print(f"  Average time per epoch: {avg_time:.2f}s ({avg_time/60:.2f}m)")
            print(f"  Fastest epoch: {min(self.history['epoch_time']):.2f}s")
            print(f"  Slowest epoch: {max(self.history['epoch_time']):.2f}s")

        print("=" * 70)

        self.plot_training_curves(final_epoch=final_epoch, status=training_status)

        return self.history

    def plot_training_curves(self, final_epoch=None, status="completed"):
        """
        Plot and save training curves with per-channel metrics.

        Args:
            final_epoch: Final epoch number (if None, uses length of history)
            status: Status of training ("completed", "interrupted", "early_stopped")
        """
        from datetime import datetime

        if final_epoch is None:
            final_epoch = len(self.history['train_loss'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        fig.suptitle(f'Training Curves - Epoch {final_epoch} ({status})',
                    fontsize=14, fontweight='bold')

        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_mae'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_mae'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error (Overall)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['train_rmse'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history['val_rmse'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Root Mean Squared Error (Overall)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        versioned_filename = f'training_curves_epoch{final_epoch}_{status}_{timestamp}.png'
        versioned_path = self.checkpoint_dir / versioned_filename
        plt.savefig(versioned_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining curves saved to: {versioned_path}")

        latest_path = self.checkpoint_dir / 'training_curves_latest.png'
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        print(f"Training curves also saved to: {latest_path}")

        plt.close()

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        # Strip _orig_mod. prefix added by torch.compile if present
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        # Backward compatibility: if older checkpoints don't have best_val_rmse,
        # fall back to best_val_loss as a proxy to keep resume functional.
        self.best_val_rmse = checkpoint.get('best_val_rmse', self.best_val_loss)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val MAE:  {self.best_val_mae:.4f}")
        print(f"  Best val RMSE: {self.best_val_rmse:.4f}")
        return checkpoint.get('epoch', 0)

    def load_weights_only(self, checkpoint_path: str):
        """
        Load only model weights from checkpoint (no optimizer, scheduler, or history).
        Training will start from epoch 0 with these pretrained weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        # Strip _orig_mod. prefix added by torch.compile if present
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load with strict=False to handle architectural differences between checkpoint and current model
        # This allows loading pretrained weights even if some layers differ
        result = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from: {checkpoint_path}")
        print(f"  Missing keys ({len(result.missing_keys)}): {result.missing_keys}")
        print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")
        print(f"  Training will start from epoch 0 with pretrained weights")
        print(f"  Optimizer, scheduler, and history are NOT loaded (fresh start)")
        return 0


# ============================================================================
# TWO-STAGE TRAINING (Shuffled -> Unshuffled with temporal consistency)
# ============================================================================


def two_stage_train(
    model,
    create_dataloaders_fn,
    stage1_epochs: int = None,
    stage2_epochs: int = None,
    stage2_lr: float = None,
    stage2_temporal_agg_weight: float = None,
    resume_checkpoint: str = None,
    init_weights_from: str = None,
    skip_stage1: bool = None,
):
    """
    Two-stage training approach:

    Stage 1: Normal training with shuffled data
             - Uses regular learning rate
             - Uses standard loss weights
             - Data is randomly shuffled each epoch
             - Can be skipped if init_weights_from is provided and skip_stage1=True

    Stage 2: Fine-tuning with unshuffled (chronologically ordered) data
             - Uses much smaller learning rate
             - Uses larger temporal_agg_weight for accumulation consistency
             - Data is in chronological order to learn temporal patterns

    Args:
        model: PyTorch model to train
        create_dataloaders_fn: Function to create dataloaders (should accept shuffle_train parameter)
        stage1_epochs: Number of epochs for Stage 1 (default: from config EPOCHS)
        stage2_epochs: Number of epochs for Stage 2 (default: from config STAGE2_EPOCHS)
        stage2_lr: Learning rate for Stage 2 (default: from config STAGE2_LEARNING_RATE)
        stage2_temporal_agg_weight: Temporal aggregate loss weight for Stage 2
                                    (default: from config STAGE2_TEMPORAL_AGG_WEIGHT)
        resume_checkpoint: Path to checkpoint to resume training from
        init_weights_from: Path to checkpoint to initialize weights from (starts from epoch 0)
        skip_stage1: If True, skip Stage 1 training and go directly to Stage 2.
                     Requires init_weights_from to be set. (default: from config SKIP_STAGE1_TRAINING)

    Returns:
        Dictionary with training histories for both stages
    """
    try:
        from config import (
            STAGE2_EPOCHS as CONFIG_STAGE2_EPOCHS,
            STAGE2_LEARNING_RATE as CONFIG_STAGE2_LR,
            STAGE2_TEMPORAL_AGG_WEIGHT as CONFIG_STAGE2_AGG_WEIGHT
        )
    except ImportError:
        CONFIG_STAGE2_EPOCHS = 15
        CONFIG_STAGE2_LR = 1e-6
        CONFIG_STAGE2_AGG_WEIGHT = 5.0

    if skip_stage1 is None:
        try:
            from config import SKIP_STAGE1_TRAINING
            skip_stage1 = SKIP_STAGE1_TRAINING
        except ImportError:
            skip_stage1 = False

    if stage1_epochs is None:
        stage1_epochs = EPOCHS
    if stage2_epochs is None:
        stage2_epochs = CONFIG_STAGE2_EPOCHS
    if stage2_lr is None:
        stage2_lr = CONFIG_STAGE2_LR
    if stage2_temporal_agg_weight is None:
        stage2_temporal_agg_weight = CONFIG_STAGE2_AGG_WEIGHT

    print("\n" + "=" * 80)
    print(" " * 25 + "TWO-STAGE TRAINING")
    print("=" * 80)
    if skip_stage1 and init_weights_from:
        print(f"Stage 1: SKIPPED (using pretrained weights from {init_weights_from})")
    else:
        print(f"Stage 1: {stage1_epochs} epochs with shuffled data, LR={LEARNING_RATE}")
    print(f"Stage 2: {stage2_epochs} epochs with unshuffled data, LR={stage2_lr}")
    print(f"Stage 2 temporal_agg_weight: {stage2_temporal_agg_weight}")
    print("=" * 80 + "\n")

    results = {
        'stage1_history': None,
        'stage2_history': None
    }

    # Import config module for modifying EPOCHS during training stages
    import config
    original_epochs = EPOCHS

    # ========================================================================
    # STAGE 1: Normal training with shuffled data (or skip if using pretrained)
    # ========================================================================

    stage1_best_val_rmse = float('inf')
    stage1_best_val_loss = float('inf')
    stage1_best_val_mae = float('inf')

    if skip_stage1 and init_weights_from:
        print("\n" + "=" * 80)
        print(" " * 15 + "STAGE 1: SKIPPED (LOADING PRETRAINED WEIGHTS)")
        print("=" * 80 + "\n")

        weights_path = Path(init_weights_from)
        if weights_path.exists():
            print(f"Loading pretrained weights from: {init_weights_from}")
            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(str(weights_path), map_location=device)
            state_dict = checkpoint['model_state_dict']

            # Strip _orig_mod. prefix added by torch.compile if present
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)
            # Get best val RMSE from checkpoint if available (fallback to val loss for older checkpoints)
            stage1_best_val_rmse = checkpoint.get('best_val_rmse', checkpoint.get('best_val_loss', float('inf')))
            stage1_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            stage1_best_val_mae = checkpoint.get('best_val_mae', float('inf'))
            print(f"  Pretrained model best val RMSE: {stage1_best_val_rmse:.4f}")
            print("  Model weights loaded successfully")
        else:
            raise FileNotFoundError(f"Pretrained weights not found at: {init_weights_from}")

        results['stage1_history'] = None

        print("\n" + "=" * 80)
        print(" " * 20 + "STAGE 1 SKIPPED - PROCEEDING TO STAGE 2")
        print("=" * 80 + "\n")

    else:
        print("\n" + "=" * 80)
        print(" " * 20 + "STAGE 1: TRAINING WITH SHUFFLED DATA")
        print("=" * 80 + "\n")

        train_loader, val_loader, _ = create_dataloaders_fn(shuffle_train=True)

        stage1_trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_weights={'temporal_agg_weight': 0.0}  # No temporal aggregation in Stage 1
        )

        start_epoch = 0
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
            if resume_path.exists():
                print(f"Resuming from checkpoint: {resume_checkpoint}")
                start_epoch = stage1_trainer.load_checkpoint(str(resume_path))
                print(f"  Loaded from epoch {start_epoch}")
        elif init_weights_from:
            weights_path = Path(init_weights_from)
            if weights_path.exists():
                print(f"Initializing weights from: {init_weights_from}")
                stage1_trainer.load_weights_only(str(weights_path))

        # Temporarily override EPOCHS for Stage 1
        config.EPOCHS = stage1_epochs

        stage1_history = stage1_trainer.train(start_epoch=start_epoch)
        results['stage1_history'] = stage1_history
        stage1_best_val_rmse = stage1_trainer.best_val_rmse
        stage1_best_val_loss = stage1_trainer.best_val_loss
        stage1_best_val_mae = stage1_trainer.best_val_mae

        # Restore original EPOCHS
        config.EPOCHS = original_epochs

        print("\n" + "=" * 80)
        print(" " * 20 + "STAGE 1 COMPLETED")
        print(f"Best validation RMSE: {stage1_trainer.best_val_rmse:.4f}")
        print("=" * 80 + "\n")

    # ========================================================================
    # STAGE 2: Fine-tuning with unshuffled data and temporal consistency
    # ========================================================================

    print("\n" + "=" * 80)
    print(" " * 15 + "STAGE 2: FINE-TUNING WITH UNSHUFFLED DATA")
    print("=" * 80 + "\n")

    train_loader_unshuffled, val_loader, _ = create_dataloaders_fn(shuffle_train=False)

    stage2_loss_weights = {
        'temporal_agg_weight': stage2_temporal_agg_weight,  # Larger weight for temporal consistency
        'dice_weight': 0.000,
        'ssim_weight': 0.000,
        'sobel_weight': 0.000,
        'tweedie_weight': 0.000,
    }

    stage2_trainer = Trainer(
        model=model,  # Use the same model (already trained in Stage 1)
        train_loader=train_loader_unshuffled,
        val_loader=val_loader,
        loss_weights=stage2_loss_weights
    )

    for param_group in stage2_trainer.optimizer.param_groups:
        param_group['lr'] = stage2_lr

    # Copy best_val_rmse from Stage 1 so we continue from there
    stage2_trainer.best_val_rmse = stage1_best_val_rmse
    stage2_trainer.best_val_loss = stage1_best_val_loss
    stage2_trainer.best_val_mae = stage1_best_val_mae

    stage2_trainer.use_warmup = False
    stage2_trainer.warmup_epochs = 0

    config.EPOCHS = stage2_epochs
    stage2_history = stage2_trainer.train(start_epoch=0)
    results['stage2_history'] = stage2_history

    # Restore original EPOCHS
    config.EPOCHS = original_epochs

    print("\n" + "=" * 80)
    print(" " * 20 + "STAGE 2 COMPLETED")
    print(f"Best validation RMSE: {stage2_trainer.best_val_rmse:.4f}")
    print("=" * 80 + "\n")

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================

    print("\n" + "=" * 80)
    print(" " * 25 + "TWO-STAGE TRAINING COMPLETE")
    print("=" * 80)
    print(f"Stage 1 (Shuffled):   {len(results['stage1_history']['train_loss'])} epochs")
    print(f"Stage 2 (Unshuffled): {len(results['stage2_history']['train_loss'])} epochs")
    print(f"Final best validation RMSE: {stage2_trainer.best_val_rmse:.4f}")
    print("=" * 80 + "\n")

    return results
