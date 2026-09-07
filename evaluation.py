"""Evaluation of a trained SWEDE checkpoint.

Loads a checkpoint, runs inference on the test years defined in config.py and
writes predictions, metrics and the paper figures to the output directory.

The model itself is built by ``main.create_model()`` so that evaluation always
reproduces the architecture that was trained.

Usage:
    python evaluation.py
    python evaluation.py --checkpoint checkpoints/best_model.pth --output predictions/
    python evaluation.py --skip-inference      # reuse predictions.npy / targets.npy

Which diagnostics are produced is controlled by EVALUATE_TRAINING_DATA,
EVALUATE_STATES and EVALUATE_REGIONS below; everything else comes from config.py.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATE_TRAINING_DATA = False

EVALUATE_STATES = ['CA', 'WA', 'OR', 'ID', 'MT', 'CO', 'WY', 'UT', 'NV']

EVALUATE_REGIONS = {
    "PNW": {"lon_min": -125, "lon_max": -117, "lat_min": 42, "lat_max": 49},
    "NCR": {"lon_min": -117, "lon_max": -107, "lat_min": 42, "lat_max": 49},
    "SW":  {"lon_min": -121, "lon_max": -114, "lat_min": 35, "lat_max": 42},
    "SCR": {"lon_min": -114, "lon_max":  -105, "lat_min": 35, "lat_max": 42},
    "SN":  {"lon_min": -121, "lon_max":  -119, "lat_min": 37, "lat_max": 40},
    "NRM": {"lon_min": -111, "lon_max":  -108, "lat_min": 42, "lat_max": 46},
}

# Grid resolution in kilometers (adjust based on your WRF domain configuration)
GRID_RESOLUTION_KM = 9.0

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    main_module=None,
) -> Tuple[nn.Module, Dict]:
    """Load a checkpoint into the architecture built by ``main.create_model()``.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        main_module: The imported ``main`` module. Imported here if not given.

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if main_module is None:
        import main as main_module

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  ✓ Checkpoint loaded")

    state_dict = checkpoint['model_state_dict']

    # torch.compile() prefixes every parameter name with '_orig_mod.'
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print(f"  Detected torch.compile() wrapped model, unwrapping...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = main_module.create_model()
    print(f"  Model class: {model.__class__.__name__}")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, checkpoint


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    save_predictions: bool = True,
    output_dir: Optional[Path] = None,
    config_module = None,
    denormalize_func = None,
    inverse_transform_func = None
) -> Dict:
    """
    Evaluate model on dataset with per-channel metrics.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        save_predictions: Whether to save predictions to disk (default: True)
        output_dir: Directory to save predictions (if save_predictions=True)
        config_module: Configuration module (imported from case directory)
        denormalize_func: Denormalization function (from dataset_unires)
        inverse_transform_func: Inverse of a variance-stabilizing target
            transform (e.g. expm1 for a log1p target). When the case's config
            sets ``TARGET_TRANSFORM`` to something other than ``"none"``, the
            model predicts in transformed space and both predictions and
            targets must be mapped back to physical SWE (mm) before computing
            metrics. Pass ``dataset_module.inverse_transform_target`` here.
            When ``None`` (older cases without the flag) this is a no-op, so the
            script still works for cases whose config predates ``TARGET_TRANSFORM``.

    Returns:
        Dictionary containing evaluation metrics
    """
    if config_module is not None:
        NORMALIZE_OUTPUTS = getattr(config_module, 'NORMALIZE_OUTPUTS', False)
        TARGET_TRANSFORM = getattr(config_module, 'TARGET_TRANSFORM', 'none')
    else:
        NORMALIZE_OUTPUTS = False
        TARGET_TRANSFORM = 'none'

    # A target transform is only inverted if the case actually enables it AND
    # the dataset module exposes the inverse. Disable the no-op inverse call
    # otherwise so behavior (and this message) is correct for legacy cases.
    if TARGET_TRANSFORM == 'none' or inverse_transform_func is None:
        inverse_transform_func = None
        if TARGET_TRANSFORM != 'none':
            print(f"  Warning: config sets TARGET_TRANSFORM='{TARGET_TRANSFORM}' but no "
                  f"inverse_transform_target() was provided; predictions will NOT be "
                  f"inverted to physical SWE.")
    else:
        print(f"  Target transform '{TARGET_TRANSFORM}' enabled: predictions and "
              f"targets will be inverted to physical SWE (mm) for metrics/plots.")

    model.eval()
    criterion = nn.MSELoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters:")
    print(f"    Trainable:   {trainable_params:,}")
    print(f"    Total:       {total_params:,}")
    if total_params != trainable_params:
        print(f"    Frozen:      {total_params - trainable_params:,}")

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mae_ch0 = 0.0
    total_rmse_ch0 = 0.0
    n_samples = 0

    batch_inference_times = []  # time per batch (seconds)

    all_predictions = []
    all_targets = []
    all_times = []

    model_in_channels = None
    model_per_timestep_channels = None
    is_temporal_model = hasattr(model, 'sequence_length')
    # DualEncoderSwinUNet has swin_patch_embed instead of dynamic_encoders
    is_dual_encoder = (hasattr(model, 'static_encoders') and
                       (hasattr(model, 'dynamic_encoders') or hasattr(model, 'swin_patch_embed')))

    if hasattr(model, 'in_channels'):
        model_in_channels = model.in_channels
        # For TemporalUNet, multiply by sequence length since data is flattened
        if is_temporal_model:
            model_in_channels = model_in_channels * model.sequence_length
    elif hasattr(model, 'encoders') and len(model.encoders) > 0:
        first_layer = model.encoders[0]
        if isinstance(first_layer, nn.Sequential) and len(first_layer) > 0:
            first_conv = first_layer[0]
            if isinstance(first_conv, nn.Conv2d):
                model_in_channels = first_conv.in_channels
    elif hasattr(model, 'patch_embed'):
        model_in_channels = model.patch_embed.in_channels
    elif is_dual_encoder:
        pass

    norm_stats = None
    if NORMALIZE_OUTPUTS:
        import json
        from pathlib import Path as PathLib
        stats_file = PathLib("normalization_stats.json")
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                norm_stats = json.load(f)
            print(f"  Output normalization enabled: will denormalize for metrics and visualization")
        else:
            print(f"  Warning: NORMALIZE_OUTPUTS=True but normalization_stats.json not found")

    print(f"\nEvaluating on {len(dataloader)} batches...")

    dataset = dataloader.dataset
    time_index = getattr(dataset, 'time_index', None)
    batch_size = dataloader.batch_size if dataloader.batch_size else 1

    channel_mismatch_warned = False

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 4:
                # DualEncoderUNet format: (static, dynamic, targets, landmask)
                static_inputs, dynamic_inputs, targets, landmask = batch_data
                static_inputs = static_inputs.to(device)
                dynamic_inputs = dynamic_inputs.to(device)
                targets = targets.to(device)
                landmask = landmask.to(device)

                inputs = static_inputs

            elif len(batch_data) == 3:
                # Standard format: (inputs, targets, landmask)
                inputs, targets, landmask = batch_data
                inputs = inputs.to(device)
                targets = targets.to(device)
                landmask = landmask.to(device)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch_data)}")

            if time_index is not None:
                current_batch_size = targets.size(0)
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + current_batch_size, len(time_index))
                batch_times = [time_index[i] for i in range(start_idx, end_idx)]
                all_times.extend(batch_times)

            if not is_dual_encoder and model_in_channels is not None and inputs.size(1) != model_in_channels:
                print(f"\n{'='*70}")
                print(f"  ❌ CHANNEL MISMATCH ERROR")
                print(f"{'='*70}")
                print(f"  Model expects:    {model_in_channels} input channels")
                print(f"  Dataset provides: {inputs.size(1)} input channels")
                print(f"  Difference:       {abs(inputs.size(1) - model_in_channels)} channels")
                print(f"\n{'='*70}")
                print(f"  POSSIBLE CAUSES:")
                print(f"{'='*70}")
                print(f"  1. Config was modified after training (variables added/removed)")
                print(f"  2. Checkpoint is from a different configuration")
                print(f"  3. Model architecture mismatch")
                print(f"\n{'='*70}")
                print(f"  SOLUTIONS:")
                print(f"{'='*70}")
                print(f"  Option 1: Modify config.py to match the checkpoint")
                print(f"    - Current config has {inputs.size(1)} channels")
                print(f"    - Need to reduce to {model_in_channels} channels")
                print(f"    - Check STATIC_VARS, DYNAMIC_VARS, and DYNAMIC_3D_VARS in config.py")
                print(f"\n  Option 2: Retrain the model with current config")
                print(f"    - Backup old checkpoints: mv checkpoints checkpoints_old")
                print(f"    - Remove normalization stats: rm normalization_stats.json")
                print(f"    - Start training: python main.py")
                print(f"{'='*70}\n")
                raise ValueError(
                    f"Channel mismatch: Model expects {model_in_channels} channels "
                    f"but dataset provides {inputs.size(1)} channels. "
                    f"Please align config.py with the checkpoint or retrain the model."
                )

            if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if is_dual_encoder and len(batch_data) == 4:
                outputs = model(static_inputs, dynamic_inputs, landmask=landmask)
            else:
                outputs = model(inputs, landmask=landmask)
            if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                torch.cuda.synchronize()
            batch_inference_times.append(time.perf_counter() - t0)

            # Map outputs and targets back to physical SWE (mm) so metrics and
            # saved predictions are always in original units. Two independent,
            # opt-in stages that mirror the forward pipeline in the dataset
            # (normalize -> transform), inverted in reverse order:
            #   1. Undo the variance-stabilizing target transform (e.g. expm1
            #      for a log1p target). No-op when TARGET_TRANSFORM == "none"
            #      or when inverse_transform_func is None (older configs).
            #   2. Undo output [0, 1] normalization. Only active when
            #      NORMALIZE_OUTPUTS is set and stats are available.
            outputs_denorm = outputs
            targets_denorm = targets

            if inverse_transform_func is not None:
                outputs_denorm = inverse_transform_func(outputs_denorm)
                targets_denorm = inverse_transform_func(targets_denorm)

            if NORMALIZE_OUTPUTS and norm_stats is not None and denormalize_func is not None:
                outputs_denorm = denormalize_func(outputs_denorm, norm_stats)
                targets_denorm = denormalize_func(targets_denorm, norm_stats)

            loss = criterion(outputs_denorm, targets_denorm)
            mae = torch.abs(outputs_denorm - targets_denorm).mean()
            rmse = torch.sqrt(((outputs_denorm - targets_denorm) ** 2).mean())

            total_loss += loss.item() * targets.size(0)
            total_mae += mae.item() * targets.size(0)
            total_rmse += rmse.item() * targets.size(0)

            if outputs_denorm.shape[1] >= 1:
                mae_ch0 = torch.abs(outputs_denorm[:, 0] - targets_denorm[:, 0]).mean()
                rmse_ch0 = torch.sqrt(((outputs_denorm[:, 0] - targets_denorm[:, 0]) ** 2).mean())

                total_mae_ch0 += mae_ch0.item() * targets.size(0)
                total_rmse_ch0 += rmse_ch0.item() * targets.size(0)

            n_samples += targets.size(0)

            if save_predictions:
                all_predictions.append(outputs_denorm.cpu().numpy())
                all_targets.append(targets_denorm.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    if batch_inference_times:
        total_inference_time = sum(batch_inference_times)
        avg_batch_time = total_inference_time / len(batch_inference_times)
        # Each sample is one full spatial domain
        time_per_domain = total_inference_time / n_samples if n_samples > 0 else 0.0
        print(f"\n  Inference timing:")
        print(f"    Total inference time:      {total_inference_time:.3f} s  ({n_samples} samples, {len(batch_inference_times)} batches)")
        print(f"    Time per batch:            {avg_batch_time * 1e3:.2f} ms  (batch size ~{batch_size})")
        print(f"    Time per full domain:      {time_per_domain * 1e3:.2f} ms")
    else:
        avg_batch_time = 0.0
        time_per_domain = 0.0

    metrics = {
        'loss': total_loss / n_samples,
        'mae': total_mae / n_samples,
        'rmse': total_rmse / n_samples,
        'n_samples': n_samples,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'inference_time_total_s': sum(batch_inference_times) if batch_inference_times else 0.0,
        'inference_time_per_batch_ms': avg_batch_time * 1e3,
        'inference_time_per_domain_ms': time_per_domain * 1e3,
    }

    if total_mae_ch0 > 0:
        metrics['mae_ch0'] = total_mae_ch0 / n_samples
        metrics['rmse_ch0'] = total_rmse_ch0 / n_samples

    if save_predictions and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        np.save(output_dir / "predictions.npy", predictions)
        np.save(output_dir / "targets.npy", targets)

        if all_times:
            times_array = np.array(all_times)  # Shape: (n_samples, 2) for (year, time_idx)
            np.save(output_dir / "times.npy", times_array)
            print(f"    times.npy: {times_array.shape}")

        print(f"\n  Predictions saved to {output_dir}/")
        print(f"    predictions.npy: {predictions.shape}")
        print(f"    targets.npy: {targets.shape}")
        metrics['predictions'] = predictions
        metrics['targets'] = targets
        if all_times:
            metrics['times'] = np.array(all_times)

    return metrics


# ============================================================================
# REGIONAL METRICS TABLE
# ============================================================================


def print_regional_metrics_table(predictions, targets, times, region_defs,
                                  target_names=None, snow_threshold=1.0,
                                  save_path=None):
    """
    Print a summary metrics table per region (and whole domain).

    Columns
    -------
    Region | r (interannual) | Max SWE Bias (%) | Seasonal r |
    Spatial r (mean) | Pixel MAE (mm) | F1 (snow presence)

    Args:
        predictions : np.ndarray  (N, C, H, W)
        targets     : np.ndarray  (N, C, H, W)
        times       : np.ndarray  (N, 2+)  col-0 = year, col-1 = intra-year idx
        region_defs : dict  {name: {lon_min,lon_max,lat_min,lat_max}}
        target_names: list[str]  channel names
        snow_threshold: float  SWE threshold (mm) for snow-presence F1
        save_path   : str or Path  if provided, write the table to this file
    """
    from plot import get_lat_lon, _build_month_index

    if target_names is None:
        target_names = [f"Ch{i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times is None or times.ndim < 2:
        print("  Cannot compute metrics table: times array not available or 1-D.")
        return

    lat, lon = get_lat_lon()
    if lat is None:
        print("  Cannot compute metrics table: lat/lon not available.")
        return

    years = times[:, 0].astype(int)
    unique_years = np.unique(years)

    month_map, ordered_months = _build_month_index(times)

    masks = {}
    for rname, bounds in region_defs.items():
        masks[rname] = (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
    masks['Domain'] = np.ones(lat.shape, dtype=bool)

    def _compute_metrics(mask, ch):
        pred_px = predictions[:, ch][:, mask]   # (N, n_pixels)
        targ_px = targets[:, ch][:, mask]

        # ── 1. Interannual r ──────────────────────────────────────────────────
        pred_yr, targ_yr = [], []
        for yr in unique_years:
            ym = years == yr
            if not np.any(ym):
                continue
            pred_yr.append(np.mean(pred_px[ym]))
            targ_yr.append(np.mean(targ_px[ym]))
        pred_yr = np.array(pred_yr)
        targ_yr = np.array(targ_yr)
        if len(pred_yr) > 1:
            r_inter = np.corrcoef(targ_yr, pred_yr)[0, 1]
        else:
            r_inter = float('nan')

        # ── 2. Max SWE Bias (%) ───────────────────────────────────────────────
        pred_max_yr, targ_max_yr = [], []
        for yr in unique_years:
            ym = years == yr
            if not np.any(ym):
                continue
            pred_max_yr.append(np.max(pred_px[ym]))
            targ_max_yr.append(np.max(targ_px[ym]))
        pred_mean_max = np.mean(pred_max_yr)
        targ_mean_max = np.mean(targ_max_yr)
        if targ_mean_max != 0:
            max_bias_pct = (pred_mean_max - targ_mean_max) / targ_mean_max * 100.0
        else:
            max_bias_pct = float('nan')

        # ── 3. Seasonal r (monthly climatology) ──────────────────────────────
        if len(ordered_months) >= 3:
            pred_seas, targ_seas = [], []
            for m in ordered_months:
                idx = month_map[m]
                if len(idx) == 0:
                    continue
                pred_seas.append(np.mean(pred_px[idx]))
                targ_seas.append(np.mean(targ_px[idx]))
            pred_seas = np.array(pred_seas)
            targ_seas = np.array(targ_seas)
            if len(pred_seas) > 1:
                r_seas = np.corrcoef(targ_seas, pred_seas)[0, 1]
            else:
                r_seas = float('nan')
        else:
            r_seas = float('nan')

        # ── 4 & 5. Spatial r (mean over time) and Pixel MAE ─────────────────
        # pred_px shape: (N, n_pixels)
        # Compute spatial correlation at each time step, then average over time.
        # For each t: r_t = corr(pred_px[t, :], targ_px[t, :]) across pixels.
        p_anom = pred_px - pred_px.mean(axis=1, keepdims=True)   # subtract spatial mean per timestep
        t_anom = targ_px - targ_px.mean(axis=1, keepdims=True)
        p_std = pred_px.std(axis=1)   # (N,) spatial std per timestep
        t_std = targ_px.std(axis=1)
        valid = (p_std > 0) & (t_std > 0)
        if np.any(valid):
            n_pix = pred_px.shape[1]
            r_t = np.sum(p_anom[valid] * t_anom[valid], axis=1) / (
                n_pix * p_std[valid] * t_std[valid]
            )
            pixel_r = float(np.mean(r_t))
        else:
            pixel_r = float('nan')

        pixel_mae = float(np.mean(np.abs(pred_px - targ_px)))

        # ── 6. F1 (snow presence) ─────────────────────────────────────────────
        pred_bin = (pred_px >= snow_threshold).ravel()
        targ_bin = (targ_px >= snow_threshold).ravel()
        tp = np.sum(pred_bin & targ_bin)
        fp = np.sum(pred_bin & ~targ_bin)
        fn = np.sum(~pred_bin & targ_bin)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else float('nan')

        return r_inter, max_bias_pct, r_seas, pixel_r, pixel_mae, f1

    col_w = [8, 16, 18, 12, 16, 16, 20]
    header = (
        f"{'Region':<{col_w[0]}}  "
        f"{'r (interann.)':>{col_w[1]}}  "
        f"{'MaxSWE Bias (%)':>{col_w[2]}}  "
        f"{'Seasonal r':>{col_w[3]}}  "
        f"{'Spatial r (mean)':>{col_w[4]}}  "
        f"{'Pixel MAE (mm)':>{col_w[5]}}  "
        f"{'F1 (snow)':>{col_w[6]}}"
    )
    sep = "-" * len(header)

    all_lines = []

    for ch_idx in range(n_channels):
        ch_lines = []
        ch_lines.append("\n" + "=" * 70)
        ch_lines.append(f"  REGIONAL METRICS TABLE — {target_names[ch_idx]}")
        ch_lines.append("=" * 70)
        ch_lines.append(header)
        ch_lines.append(sep)

        for rname, mask in masks.items():
            if not np.any(mask):
                ch_lines.append(f"  {rname}: no pixels in mask, skipping")
                continue
            r_inter, bias_pct, r_seas, pix_r, pix_mae, f1 = _compute_metrics(mask, ch_idx)

            def _fmt(v, fmt='.3f'):
                return f'{v:{fmt}}' if not (isinstance(v, float) and np.isnan(v)) else '  N/A'

            ch_lines.append(
                f"{rname:<{col_w[0]}}  "
                f"{_fmt(r_inter):>{col_w[1]}}  "
                f"{_fmt(bias_pct, '.1f'):>{col_w[2]}}  "
                f"{_fmt(r_seas):>{col_w[3]}}  "
                f"{_fmt(pix_r):>{col_w[4]}}  "
                f"{_fmt(pix_mae, '.2f'):>{col_w[5]}}  "
                f"{_fmt(f1):>{col_w[6]}}"
            )
        ch_lines.append(sep)
        all_lines.extend(ch_lines)
        for line in ch_lines:
            print(line)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(all_lines) + '\n')
        print(f"\n  Regional metrics table saved to: {save_path}")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate trained snow prediction model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: from config.py)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for predictions (default: from config.py)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: from config.py)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save predictions to disk (default: True)')
    parser.add_argument('--skip-inference', action='store_true', default=False,
                        help='Skip inference if predictions already exist (default: False)')

    args = parser.parse_args()
    print("=" * 70)
    print(" " * 20 + "MODEL EVALUATION")
    print("=" * 70)

    import config as config_module
    import dataset as dataset_module
    import main as main_module

    from plot import (
        select_interesting_samples,
        visualize_predictions,
        plot_total_mean_max_scatter,
        plot_total_mean_max_scatter_spatial,
        plot_mean_max_spatial_maps,
        plot_mean_max_spatial_maps_yearly,
        plot_temporal_variability_spatial_maps,
        plot_yearly_max_variability_spatial_maps,
        plot_yearly_max_comparison,
        plot_state_evaluation,
        plot_state_evaluation_yearly,
        plot_region_evaluation,
        plot_region_evaluation_yearly,
        plot_time_series_seasonality,
        plot_time_series_seasonality_spatial_max,
        plot_time_series_seasonality_regions,
        plot_seasonality_regions_combined,
        plot_seasonality_comparison_regions,
        plot_mean_of_yearly_max_scatter_whole_domain,
        plot_mean_of_yearly_max_scatter_state,
        plot_mean_of_yearly_max_scatter_regional,
        plot_regional_metrics_bar,
        plot_regional_yearly_max_bar,
        plot_regional_yearly_mae_bar,
        plot_state_metrics_bar,
        plot_state_yearly_max_bar,
        plot_state_yearly_mae_bar,
        plot_whole_domain_time_series,
        plot_long_term_time_series_regions,
        plot_yearly_mean_time_series_regions,
        plot_yearly_mean_time_series_regions_combined,
        plot_april1_swe_time_series_regions_combined,
        plot_yearly_snow_volume_regions_combined,
        plot_regional_snow_sum_time_series,
        plot_whole_domain_snow_volume_time_series,
        plot_whole_domain_yearly_total_snow,
        plot_annual_snow_volume_scatter_domain,
        plot_annual_snow_volume_scatter_regions,
        plot_avg_snow_volume_scatter_regions,
        plot_qq,
        plot_qq_regions_combined,
        plot_regional_state_yearly_max_bar_combined,
        plot_regional_state_yearly_sum_bar_combined,
        plot_error_vs_terrain
    )

    try:
        from plot import (
            plot_whole_domain_time_series_accumulated,
            plot_long_term_time_series_regions_accumulated,
            plot_long_term_time_series_regions_accumulated_yearly_reset,
            plot_yearly_mean_time_series_regions_accumulated,
            plot_yearly_mean_time_series_regions_accumulated_yearly_reset
        )
        accumulated_plots_available = True
    except ImportError:
        accumulated_plots_available = False
        if EVALUATE_TRAINING_DATA:
            print("\n" + "=" * 70)
            print("WARNING: Accumulated plotting functions not available")
            print("Training data evaluation will use non-accumulated functions")
            print("=" * 70)

    DEFAULT_CHECKPOINT = getattr(config_module, 'DEFAULT_CHECKPOINT', 'checkpoints/best_model.pth')
    DEFAULT_OUTPUT_DIR = getattr(config_module, 'DEFAULT_OUTPUT_DIR', 'predictions')
    DEFAULT_DEVICE = getattr(config_module, 'DEVICE', 'cpu')
    TARGET_VARS = getattr(config_module, 'TARGET_VARS', ['SNOW'])
    N_VIS_SAMPLES = getattr(config_module, 'N_VIS_SAMPLES', 10)
    N_HOLO = getattr(config_module, 'N_HOLO', 0)

    checkpoint = args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINT
    output = args.output if args.output else DEFAULT_OUTPUT_DIR
    device_str = args.device if args.device else DEFAULT_DEVICE
    n_vis_samples =  N_VIS_SAMPLES

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    model, checkpoint_dict = load_model_from_checkpoint(
        checkpoint,
        device=device,
        main_module=main_module,
    )

    if 'epoch' in checkpoint_dict:
        print(f"  Checkpoint epoch: {checkpoint_dict['epoch']}")
    if 'best_val_loss' in checkpoint_dict:
        print(f"  Best validation loss: {checkpoint_dict['best_val_loss']:.4f}")

    print("\n" + "=" * 70)
    print("Loading test data...")
    print("=" * 70)

    import inspect
    dataloader_kwargs = {'shuffle_train': False}
    HOLO = getattr(config_module, 'HOLO', 0)
    HOLO_HIGH_RES_SCALE = getattr(config_module, 'HOLO_HIGH_RES_SCALE', 1)
    # HOLO may be an int (symmetric) or a per-side (south, north, west, east)
    # 4-tuple. Normalize to per-side sides so the active check and display
    # work for both without assuming a scalar.
    _holo_sides = tuple(HOLO) if isinstance(HOLO, (tuple, list)) else (HOLO,) * 4
    _holo_active = any(h > 0 for h in _holo_sides)
    sig = inspect.signature(dataset_module.create_dataloaders)
    if 'holo' in sig.parameters:
        dataloader_kwargs['holo'] = HOLO
        dataloader_kwargs['holo_high_res_scale'] = HOLO_HIGH_RES_SCALE
        if _holo_active:
            _holo_hr = tuple(h * HOLO_HIGH_RES_SCALE for h in _holo_sides)
            print(f"  Using boundary cropping (south, north, west, east): "
                  f"low-res {_holo_sides}px, high-res {_holo_hr}px")
        else:
            print(f"  No boundary cropping (HOLO=0)")
    elif _holo_active:
        print(f"  Warning: config defines HOLO={HOLO} but create_dataloaders "
              f"does not accept a 'holo' argument; ignoring.")

    # Disable shuffling for evaluation to preserve temporal order
    train_loader, _, test_loader = dataset_module.create_dataloaders(**dataloader_kwargs)

    predictions_file = output_dir / "predictions.npy"
    targets_file = output_dir / "targets.npy"
    times_file = output_dir / "times.npy"

    if args.skip_inference and predictions_file.exists() and targets_file.exists():
        print("\n" + "=" * 70)
        print("SKIPPING INFERENCE - Loading existing predictions...")
        print("=" * 70)
        print(f"  Loading from: {output_dir}/")

        predictions = np.load(predictions_file)
        targets = np.load(targets_file)
        times = np.load(times_file) if times_file.exists() else None

        print(f"    predictions.npy: {predictions.shape}")
        print(f"    targets.npy: {targets.shape}")
        if times is not None:
            print(f"    times.npy: {times.shape}")

        n_samples = predictions.shape[0]
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        metrics = {
            'loss': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': n_samples,
            'predictions': predictions,
            'targets': targets
        }

        if predictions.shape[1] >= 1:
            mae_ch0 = np.mean(np.abs(predictions[:, 0] - targets[:, 0]))
            rmse_ch0 = np.sqrt(np.mean((predictions[:, 0] - targets[:, 0]) ** 2))
            metrics['mae_ch0'] = float(mae_ch0)
            metrics['rmse_ch0'] = float(rmse_ch0)

        if times is not None:
            metrics['times'] = times

        print(f"  ✓ Predictions loaded successfully")
    else:
        print("\n" + "=" * 70)
        print("Evaluating model on test set...")
        print("=" * 70)

        metrics = evaluate_model(
            model,
            test_loader,
            device=str(device),
            save_predictions=True,
            output_dir=output_dir,
            config_module=config_module,
            denormalize_func=getattr(dataset_module, 'denormalize_outputs', None),
            inverse_transform_func=getattr(dataset_module, 'inverse_transform_target', None)
        )

    print("\n" + "=" * 70)
    print(" " * 25 + "RESULTS")
    print("=" * 70)
    print(f"\nTest Set Metrics (n={metrics['n_samples']} samples):")
    print(f"  Loss (MSE): {metrics['loss']:.6f}")
    print(f"  MAE:        {metrics['mae']:.6f}")
    print(f"  RMSE:       {metrics['rmse']:.6f}")

    if 'mae_ch0' in metrics:
        print(f"\nPer-Channel Metrics:")
        print(f"  Snow:")
        print(f"    MAE:  {metrics['mae_ch0']:.6f}")
        print(f"    RMSE: {metrics['rmse_ch0']:.6f}")

    print("=" * 70)

    metrics_file = output_dir / "evaluation_metrics.json"
    metrics_to_save = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                      for k, v in metrics.items() if k not in ['predictions', 'targets']}
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    if EVALUATE_TRAINING_DATA:
        output_dir_train = output_dir.parent / (output_dir.name + "_train")
        predictions_file_train = output_dir_train / "predictions.npy"
        targets_file_train = output_dir_train / "targets.npy"
        times_file_train = output_dir_train / "times.npy"

        if args.skip_inference and predictions_file_train.exists() and targets_file_train.exists():
            print("\n" + "=" * 70)
            print("SKIPPING TRAINING INFERENCE - Loading existing predictions...")
            print("=" * 70)
            print(f"  Loading from: {output_dir_train}/")

            predictions_train = np.load(predictions_file_train)
            targets_train = np.load(targets_file_train)
            times_train = np.load(times_file_train) if times_file_train.exists() else None

            print(f"    predictions.npy: {predictions_train.shape}")
            print(f"    targets.npy: {targets_train.shape}")
            if times_train is not None:
                print(f"    times.npy: {times_train.shape}")

            n_samples_train = predictions_train.shape[0]
            mse_train = np.mean((predictions_train - targets_train) ** 2)
            mae_train = np.mean(np.abs(predictions_train - targets_train))
            rmse_train = np.sqrt(mse_train)

            metrics_train = {
                'loss': float(mse_train),
                'mae': float(mae_train),
                'rmse': float(rmse_train),
                'n_samples': n_samples_train,
                'predictions': predictions_train,
                'targets': targets_train
            }

            if predictions_train.shape[1] >= 1:
                mae_ch0_train = np.mean(np.abs(predictions_train[:, 0] - targets_train[:, 0]))
                rmse_ch0_train = np.sqrt(np.mean((predictions_train[:, 0] - targets_train[:, 0]) ** 2))
                metrics_train['mae_ch0'] = float(mae_ch0_train)
                metrics_train['rmse_ch0'] = float(rmse_ch0_train)

            if times_train is not None:
                metrics_train['times'] = times_train

            print(f"  ✓ Training predictions loaded successfully")
        else:
            print("\n" + "=" * 70)
            print("Evaluating model on training set...")
            print("=" * 70)

            metrics_train = evaluate_model(
                model,
                train_loader,
                device=str(device),
                save_predictions=True,
                output_dir=output_dir_train,
                config_module=config_module,
                denormalize_func=getattr(dataset_module, 'denormalize_outputs', None),
                inverse_transform_func=getattr(dataset_module, 'inverse_transform_target', None)
            )

        print("\n" + "=" * 70)
        print(" " * 22 + "TRAINING RESULTS")
        print("=" * 70)
        print(f"\nTraining Set Metrics (n={metrics_train['n_samples']} samples):")
        print(f"  Loss (MSE): {metrics_train['loss']:.6f}")
        print(f"  MAE:        {metrics_train['mae']:.6f}")
        print(f"  RMSE:       {metrics_train['rmse']:.6f}")

        if 'mae_ch0' in metrics_train:
            print(f"\nPer-Channel Metrics:")
            print(f"  Snow:")
            print(f"    MAE:  {metrics_train['mae_ch0']:.6f}")
            print(f"    RMSE: {metrics_train['rmse_ch0']:.6f}")

        print("=" * 70)

        metrics_file_train = output_dir_train / "evaluation_metrics.json"
        metrics_to_save_train = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                                for k, v in metrics_train.items() if k not in ['predictions', 'targets']}
        with open(metrics_file_train, 'w') as f:
            json.dump(metrics_to_save_train, f, indent=2)
        print(f"\nTraining metrics saved to: {metrics_file_train}")
    else:
        metrics_train = None
        output_dir_train = None
        print("\n" + "=" * 70)
        print("Training data evaluation DISABLED (EVALUATE_TRAINING_DATA = False)")
        print("=" * 70)

    if 'predictions' in metrics:
        print("\n" + "=" * 70)
        print("Creating visualizations...")
        print("=" * 70)

        visualize_predictions(
            metrics['predictions'],
            metrics['targets'],
            output_dir / 'visualizations',
            n_samples=n_vis_samples,
            target_names=TARGET_VARS,
            select_interesting=True
        )

        print("\n" + "=" * 70)
        print("Creating total/mean/max scatter plots (temporal)...")
        print("=" * 70)
        plot_total_mean_max_scatter(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            target_names=TARGET_VARS
        )

        print("\n" + "=" * 70)
        print("Creating total/mean/max scatter plots (spatial)...")
        print("=" * 70)
        plot_total_mean_max_scatter_spatial(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            target_names=TARGET_VARS,
            times=metrics.get('times')
        )

        print("\n" + "=" * 70)
        print("Creating spatial comparison maps (zero snow masked)...")
        print("=" * 70)
        plot_mean_max_spatial_maps(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            target_names=TARGET_VARS,
            mask_zero_snow=True,
            vmin = 0,
            vmax = 1000,
        )

        print("\n" + "=" * 70)
        print("Creating temporal variability spatial maps (std and range, zero snow masked)...")
        print("=" * 70)
        plot_temporal_variability_spatial_maps(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            target_names=TARGET_VARS,
            mask_zero_snow=True
        )

        if 'times' in metrics:
            print("\n" + "=" * 70)
            print("Creating yearly max variability spatial maps (std and range, zero snow masked)...")
            print("=" * 70)
            plot_yearly_max_variability_spatial_maps(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS,
                mask_zero_snow=True
            )

        if 'times' in metrics:
            print("\n" + "=" * 70)
            print("Creating seasonality time series...")
            print("=" * 70)
            plot_time_series_seasonality(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS
            )

            print("\n" + "=" * 70)
            print("Creating seasonality time series (spatial max)...")
            print("=" * 70)
            plot_time_series_seasonality_spatial_max(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS
            )

        if EVALUATE_STATES is not None:
            print("\n" + "=" * 70)
            print("Creating state-level evaluation...")
            print("=" * 70)
            plot_state_evaluation(
                metrics['predictions'],
                metrics['targets'],
                output_dir,
                state_abbrevs=EVALUATE_STATES,
                target_names=TARGET_VARS
            )

        if EVALUATE_REGIONS is not None:
            print("\n" + "=" * 70)
            print("Creating region-level evaluation...")
            print("=" * 70)
            plot_region_evaluation(
                metrics['predictions'],
                metrics['targets'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS
            )

        if EVALUATE_REGIONS is not None and 'times' in metrics:
            print("\n" + "=" * 70)
            print("Creating region seasonality plots (spatial mean)...")
            print("=" * 70)
            plot_time_series_seasonality_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating region seasonality plots (spatial max)...")
            print("=" * 70)
            plot_time_series_seasonality_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating combined seasonality figure (all regions, spatial mean)...")
            print("=" * 70)
            plot_seasonality_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating combined seasonality figure (all regions, spatial mean, day-to-day)...")
            print("=" * 70)
            plot_seasonality_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean',
                time_res='day'
            )

            print("\n" + "=" * 70)
            print("Creating combined seasonality figure (all regions, spatial max)...")
            print("=" * 70)
            plot_seasonality_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating region seasonality comparison plots (spatial mean)...")
            print("=" * 70)
            plot_seasonality_comparison_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating regional snow sum time series...")
            print("=" * 70)
            plot_regional_snow_sum_time_series(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS
            )

            print("\n" + "=" * 70)
            print("Creating long-term regional time series (spatial mean)...")
            print("=" * 70)
            plot_long_term_time_series_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating long-term regional time series (spatial max)...")
            print("=" * 70)
            plot_long_term_time_series_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating yearly mean regional time series (spatial mean)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating yearly mean regional time series (spatial max)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating combined yearly mean regional time series (spatial mean)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating combined yearly mean regional time series (spatial max)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating combined April 1 SWE regional time series (spatial mean)...")
            print("=" * 70)
            plot_april1_swe_time_series_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating combined April 1 SWE regional time series (spatial max)...")
            print("=" * 70)
            plot_april1_swe_time_series_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating combined yearly regional snow volume time series (yearly mean)...")
            print("=" * 70)
            plot_yearly_snow_volume_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS,
                yearly_stat='mean'
            )

            print("\n" + "=" * 70)
            print("Creating combined yearly regional snow volume time series (annual total)...")
            print("=" * 70)
            plot_yearly_snow_volume_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS,
                yearly_stat='sum'
            )

        if 'times' in metrics:
            print("\n" + "=" * 70)
            print("Creating whole domain time series (spatial mean)...")
            print("=" * 70)
            plot_whole_domain_time_series(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating whole domain time series (spatial max)...")
            print("=" * 70)
            plot_whole_domain_time_series(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating whole domain total snow water volume time series...")
            print("=" * 70)
            plot_whole_domain_snow_volume_time_series(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS
            )

            print("\n" + "=" * 70)
            print("Creating whole domain yearly total snow figure...")
            print("=" * 70)
            plot_whole_domain_yearly_total_snow(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS
            )

            print("\n" + "=" * 70)
            print("Creating Figure 5: annual snow volume scatter (whole domain)...")
            print("=" * 70)
            plot_annual_snow_volume_scatter_domain(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                resolution_km=GRID_RESOLUTION_KM,
                target_names=TARGET_VARS
            )

            if EVALUATE_REGIONS is not None:
                print("\n" + "=" * 70)
                print("Creating Figure 5: annual snow volume scatter (regions)...")
                print("=" * 70)
                plot_annual_snow_volume_scatter_regions(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    resolution_km=GRID_RESOLUTION_KM,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating Figure 5 (avg snow volume scatter by region)...")
                print("=" * 70)
                plot_avg_snow_volume_scatter_regions(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    resolution_km=GRID_RESOLUTION_KM,
                    target_names=TARGET_VARS
                )

        if 'times' in metrics:
            print("\n" + "=" * 70)
            print("Creating yearly max comparison...")
            print("=" * 70)
            plot_yearly_max_comparison(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS
            )

            print("\n" + "=" * 70)
            print("Creating mean of yearly maxima spatial maps (zero snow masked)...")
            print("=" * 70)
            plot_mean_max_spatial_maps_yearly(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS,
                mask_zero_snow=True,
                vmin = 0,
                vmax = 1000,
            )

            print("\n" + "=" * 70)
            print("Creating mean of yearly max scatter plot (whole domain)...")
            print("=" * 70)
            plot_mean_of_yearly_max_scatter_whole_domain(
                metrics['predictions'],
                metrics['targets'],
                metrics['times'],
                output_dir,
                target_names=TARGET_VARS
            )

            if EVALUATE_STATES is not None:
                print("\n" + "=" * 70)
                print("Creating state-level evaluation with yearly maxima...")
                print("=" * 70)
                plot_state_evaluation_yearly(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    state_abbrevs=EVALUATE_STATES,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating mean of yearly max scatter plot (state-level)...")
                print("=" * 70)
                plot_mean_of_yearly_max_scatter_state(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    state_abbrevs=EVALUATE_STATES,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating state metrics bar charts...")
                print("=" * 70)
                plot_state_metrics_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    output_dir,
                    state_abbrevs=EVALUATE_STATES,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating state yearly max bar charts...")
                print("=" * 70)
                plot_state_yearly_max_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    state_abbrevs=EVALUATE_STATES,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating state yearly MAE bar charts...")
                print("=" * 70)
                plot_state_yearly_mae_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    state_abbrevs=EVALUATE_STATES,
                    target_names=TARGET_VARS
                )

            if EVALUATE_REGIONS is not None:
                print("\n" + "=" * 70)
                print("Creating region-level evaluation with yearly maxima...")
                print("=" * 70)
                plot_region_evaluation_yearly(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating mean of yearly max scatter plot (regional)...")
                print("=" * 70)
                plot_mean_of_yearly_max_scatter_regional(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    regions=EVALUATE_REGIONS,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating region metrics bar charts...")
                print("=" * 70)
                plot_regional_metrics_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating regional yearly max bar charts...")
                print("=" * 70)
                plot_regional_yearly_max_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    target_names=TARGET_VARS
                )

                print("\n" + "=" * 70)
                print("Creating regional yearly MAE bar charts...")
                print("=" * 70)
                plot_regional_yearly_mae_bar(
                    metrics['predictions'],
                    metrics['targets'],
                    metrics['times'],
                    output_dir,
                    region_defs=EVALUATE_REGIONS,
                    target_names=TARGET_VARS
                )

                if EVALUATE_STATES is not None:
                    print("\n" + "=" * 70)
                    print("Creating combined regional + state yearly max bar chart...")
                    print("=" * 70)
                    plot_regional_state_yearly_max_bar_combined(
                        metrics['predictions'],
                        metrics['targets'],
                        metrics['times'],
                        output_dir,
                        region_defs=EVALUATE_REGIONS,
                        state_abbrevs=EVALUATE_STATES,
                        target_names=TARGET_VARS,
                        diff_ylim=(-20, 20)
                    )

                    print("\n" + "=" * 70)
                    print("Creating combined regional + state yearly peak volume (km^3) bar chart...")
                    print("=" * 70)
                    plot_regional_state_yearly_sum_bar_combined(
                        metrics['predictions'],
                        metrics['targets'],
                        metrics['times'],
                        output_dir,
                        region_defs=EVALUATE_REGIONS,
                        state_abbrevs=EVALUATE_STATES,
                        resolution_km=GRID_RESOLUTION_KM,
                        target_names=TARGET_VARS,
                        diff_ylim=(-20, 20)
                    )

    if 'predictions' in metrics:
        print("\n" + "=" * 70)
        print("Creating QQ plots (whole domain + per region)...")
        print("=" * 70)
        plot_qq(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            region_defs=EVALUATE_REGIONS,
            target_names=TARGET_VARS
        )

        if EVALUATE_REGIONS is not None:
            print("\n" + "=" * 70)
            print("Creating combined QQ plot (all regions in one figure)...")
            print("=" * 70)
            plot_qq_regions_combined(
                metrics['predictions'],
                metrics['targets'],
                output_dir,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS
            )

    if 'predictions' in metrics:
        print("\n" + "=" * 70)
        print("Creating error vs terrain height plots...")
        print("=" * 70)
        plot_error_vs_terrain(
            metrics['predictions'],
            metrics['targets'],
            output_dir,
            times=metrics.get('times'),
            region_defs=EVALUATE_REGIONS,
            target_names=TARGET_VARS
        )

    if 'predictions' in metrics and 'times' in metrics and EVALUATE_REGIONS is not None:
        print_regional_metrics_table(
            metrics['predictions'],
            metrics['targets'],
            metrics['times'],
            region_defs=EVALUATE_REGIONS,
            target_names=TARGET_VARS,
            save_path=output_dir / "regional_metrics_table.txt"
        )

    if EVALUATE_TRAINING_DATA and metrics_train is not None and 'predictions' in metrics_train and 'times' in metrics_train and accumulated_plots_available:
        print("\n" + "=" * 70)
        print("Creating TRAINING DATA time series plots...")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("Creating whole domain time series for TRAINING data (spatial mean)...")
        print("=" * 70)
        plot_whole_domain_time_series(
            metrics_train['predictions'],
            metrics_train['targets'],
            metrics_train['times'],
            output_dir_train,
            target_names=TARGET_VARS,
            metric='mean'
        )

        print("\n" + "=" * 70)
        print("Creating whole domain time series for TRAINING data (spatial max)...")
        print("=" * 70)
        plot_whole_domain_time_series(
            metrics_train['predictions'],
            metrics_train['targets'],
            metrics_train['times'],
            output_dir_train,
            target_names=TARGET_VARS,
            metric='max'
        )

        if EVALUATE_REGIONS is not None:
            print("\n" + "=" * 70)
            print("Creating long-term regional time series for TRAINING data (spatial mean)...")
            print("=" * 70)
            plot_long_term_time_series_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating long-term regional time series for TRAINING data (spatial max)...")
            print("=" * 70)
            plot_long_term_time_series_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating yearly mean regional time series for TRAINING data (spatial mean)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating yearly mean regional time series for TRAINING data (spatial max)...")
            print("=" * 70)
            plot_yearly_mean_time_series_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

        print("\n" + "=" * 70)
        print("Creating seasonality time series for TRAINING data...")
        print("=" * 70)
        plot_time_series_seasonality(
            metrics_train['predictions'],
            metrics_train['targets'],
            metrics_train['times'],
            output_dir_train,
            target_names=TARGET_VARS
        )

        print("\n" + "=" * 70)
        print("Creating seasonality time series for TRAINING data (spatial max)...")
        print("=" * 70)
        plot_time_series_seasonality_spatial_max(
            metrics_train['predictions'],
            metrics_train['targets'],
            metrics_train['times'],
            output_dir_train,
            target_names=TARGET_VARS
        )

        if EVALUATE_REGIONS is not None:
            print("\n" + "=" * 70)
            print("Creating region seasonality time series for TRAINING data (spatial mean)...")
            print("=" * 70)
            plot_time_series_seasonality_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

            print("\n" + "=" * 70)
            print("Creating region seasonality time series for TRAINING data (spatial max)...")
            print("=" * 70)
            plot_time_series_seasonality_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='max'
            )

            print("\n" + "=" * 70)
            print("Creating region seasonality comparison for TRAINING data (spatial mean)...")
            print("=" * 70)
            plot_seasonality_comparison_regions(
                metrics_train['predictions'],
                metrics_train['targets'],
                metrics_train['times'],
                output_dir_train,
                region_defs=EVALUATE_REGIONS,
                target_names=TARGET_VARS,
                metric='mean'
            )

    print("\n" + "=" * 70)
    print(" " * 20 + "EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nTest results saved to: {output_dir}/")
    if EVALUATE_TRAINING_DATA and output_dir_train is not None:
        print(f"Training results saved to: {output_dir_train}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
