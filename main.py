"""Training entry point for SWEDE.

Usage:
    python main.py

All settings are read from config.py.
"""

import gc
import math
from pathlib import Path

import torch
import torch._dynamo

from config import (
    MODEL_TYPE,
    BASE_CHANNELS,
    DEPTH,
    DEVICE,
    CHECKPOINT_DIR,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_STATIC_CHANNELS,
    N_DYNAMIC_CHANNELS,
    SWIN_EMBED_DIM,
    SWIN_DEPTHS,
    SWIN_NUM_HEADS,
    SWIN_WINDOW_SIZE,
    SWIN_PATCH_SIZE,
    SWIN_MLP_RATIO,
    DROPOUT_RATE,
    ATTN_DROP_RATE,
    PROJ_DROP_RATE,
    DECODER_DROPOUT_RATE,
    DROP_PATH_RATE,
    RESUME_FROM_CHECKPOINT,
    INIT_WEIGHTS_FROM,
    EPOCHS,
    SAVE_EVERY_N_EPOCHS,
    DYNAMIC_DOWNSAMPLE_FACTOR,
    HOLO,
    HOLO_HIGH_RES_SCALE,
)
from dataset import create_dataloaders
from trainer import Trainer
from model import DualEncoderSwinUNet
from model_baselines import UNet
from model_swin import SwinUNet

# Normalization constants for t2min and t2_lag90d_mean, taken from
# normalization_stats.json. Used to express the temperature constraint
# thresholds (in deg C) in the normalized space the model sees.
T2MIN_MEAN, T2MIN_STD = 280.3428039550781, 10.456090927125024
T2LAG_MEAN, T2LAG_STD = 284.19073486328125, 9.026534080506371

# Standalone threshold: t2min above this value implies no snow.
TEMP_THRESHOLD_C = 20.0
# Combined thresholds: t2min above TMIN_COMBO_C *and* t2_lag90d_mean above
# LAG_THRESHOLD_C implies no snow. Set to None in LOSS_WEIGHTS to disable.
TMIN_COMBO_C = 10.0
LAG_THRESHOLD_C = 20.0

# Channel indices into the low-resolution dynamic stack (see DYNAMIC_VARS in config.py).
TEMP_MIN_CHANNEL_IDX = 4
TEMP_LAG_CHANNEL_IDX = 8


def create_model():
    """Build the model described by config.py.

    Also used by evaluation.py so that training and evaluation always
    instantiate the exact same architecture.
    """
    print(f"  Model type:      {MODEL_TYPE}")

    if MODEL_TYPE.lower() == "dualencoderswinunet":
        downsample_log2 = int(math.log2(DYNAMIC_DOWNSAMPLE_FACTOR))
        dynamic_depth = max(0, DEPTH - downsample_log2)

        model = DualEncoderSwinUNet(
            static_channels=N_STATIC_CHANNELS,
            dynamic_channels=N_DYNAMIC_CHANNELS,
            out_channels=N_OUTPUT_CHANNELS,
            base_channels=BASE_CHANNELS,
            depth=DEPTH,
            dynamic_depth=dynamic_depth,
            fusion_type="add",
            swin_embed_dim=SWIN_EMBED_DIM,
            swin_window_size=SWIN_WINDOW_SIZE,
            swin_num_heads=SWIN_NUM_HEADS,
            dropout_rate=DROPOUT_RATE,
            drop_path_rate=DROP_PATH_RATE,
            attn_drop_rate=ATTN_DROP_RATE,
            proj_drop_rate=PROJ_DROP_RATE,
            decoder_dropout_rate=DECODER_DROPOUT_RATE,
        )
        print(f"  Static channels:  {N_STATIC_CHANNELS} (high-res)")
        print(f"  Dynamic channels: {N_DYNAMIC_CHANNELS} (low-res)")
        print(f"  Output channels:  {N_OUTPUT_CHANNELS}")
        print(f"  Base channels:   {BASE_CHANNELS}")
        print(f"  Static depth:    {DEPTH}")
        print(f"  Dynamic depth:   {dynamic_depth} (adjusted for {DYNAMIC_DOWNSAMPLE_FACTOR}x downsample)")
        print(f"  Fusion type:     add")
    elif MODEL_TYPE.lower() == "unet":
        model = UNet(
            in_channels=N_INPUT_CHANNELS,
            out_channels=N_OUTPUT_CHANNELS,
            base_channels=BASE_CHANNELS,
            depth=DEPTH,
        )
        print(f"  Input channels:  {N_INPUT_CHANNELS}")
        print(f"  Output channels: {N_OUTPUT_CHANNELS}")
        print(f"  Base channels:   {BASE_CHANNELS}")
        print(f"  Depth:           {DEPTH}")
    elif MODEL_TYPE.lower() == "swinunet":
        model = SwinUNet(
            in_channels=N_INPUT_CHANNELS,
            out_channels=N_OUTPUT_CHANNELS,
            embed_dim=SWIN_EMBED_DIM,
            depths=SWIN_DEPTHS,
            num_heads=SWIN_NUM_HEADS,
            window_size=SWIN_WINDOW_SIZE,
            patch_size=SWIN_PATCH_SIZE,
            mlp_ratio=SWIN_MLP_RATIO,
            dropout=DROPOUT_RATE,
        )
        print(f"  Input channels:  {N_INPUT_CHANNELS}")
        print(f"  Output channels: {N_OUTPUT_CHANNELS}")
        print(f"  Embed dim:       {SWIN_EMBED_DIM}")
        print(f"  Depths:          {SWIN_DEPTHS}")
        print(f"  Num heads:       {SWIN_NUM_HEADS}")
        print(f"  Window size:     {SWIN_WINDOW_SIZE}")
        print(f"  Patch size:      {SWIN_PATCH_SIZE}")
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    return model


def build_loss_weights():
    """Return the composite loss configuration passed to the Trainer."""
    temp_threshold = (TEMP_THRESHOLD_C + 273.15 - T2MIN_MEAN) / T2MIN_STD
    temp_combined_threshold = (TMIN_COMBO_C + 273.15 - T2MIN_MEAN) / T2MIN_STD
    temp_lag_threshold = (LAG_THRESHOLD_C + 273.15 - T2LAG_MEAN) / T2LAG_STD
    print(f"Temperature thresholds (normalized): standalone={temp_threshold:.3f}, "
          f"combined_tmin={temp_combined_threshold:.3f}, lag={temp_lag_threshold:.3f}")

    return {
        'alpha': 0.70,
        'dice_weight': 0.000,
        'ssim_weight': 0.000,
        'sobel_weight': 0.000,
        'pinball_weight': [0.005, 0.030, 0.010],
        'pinball_quantile': [0.95, 0.90, 0.10],
        'tweedie_weight': 0.000,
        'tweedie_p': 1.5,
        'log1p_weight': 0.500,
        'temporal_agg_weight': 0.000,
        'mse_weight': 0.000,
        'threshold': 0.100,
        'dynamic_pos_weight': True,
        'smooth_l1_beta': 5.0,
        'temp_constraint_weight': 5.0,
        'temp_threshold': temp_threshold,
        'temp_combined_threshold': None,
        'temp_lag_threshold': None,
    }


def main(resume_checkpoint=None, init_weights_from=None):
    """Run the full training pipeline and report test-set metrics.

    Args:
        resume_checkpoint: Checkpoint to resume from. Restores model, optimizer,
            scheduler, history and epoch counter.
        init_weights_from: Checkpoint to load model weights from only. Training
            restarts at epoch 0. Ignored if resume_checkpoint is set.
    """
    print("\n" + "=" * 70)
    print(" " * 25 + "SWEDE TRAINING")
    print("=" * 70)

    if resume_checkpoint:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
    elif init_weights_from:
        print(f"\nInitializing weights from: {init_weights_from}")

    print("\n[STEP 1/3] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        holo=HOLO, holo_high_res_scale=HOLO_HIGH_RES_SCALE
    )

    print("\n[STEP 2/3] Creating model...")
    model = create_model()

    print("\n[STEP 3/3] Training model...")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        loss_weights=build_loss_weights(),
        scheduler='CosineAnnealingLR',
        temp_min_channel_idx=TEMP_MIN_CHANNEL_IDX,
        temp_lag_channel_idx=TEMP_LAG_CHANNEL_IDX,
    )

    start_epoch = 0
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            print(f"\n  WARNING: Checkpoint not found at {resume_path}")
            print("    Starting training from scratch instead.\n")
        else:
            print("\n" + "=" * 70)
            print("Loading checkpoint for full resume...")
            print("=" * 70)
            start_epoch = trainer.load_checkpoint(str(resume_path))
            print(f"  Loaded from epoch {start_epoch}")
            print(f"  Will continue training from epoch {start_epoch + 1} to epoch {EPOCHS}")
            if start_epoch >= EPOCHS:
                print(f"\n  NOTE: Checkpoint is at epoch {start_epoch}, but EPOCHS is {EPOCHS}.")
                print("        No additional training will occur; increase EPOCHS in config.py.")
            print("=" * 70 + "\n")
    elif init_weights_from:
        weights_path = Path(init_weights_from)
        if not weights_path.exists():
            print(f"\n  WARNING: Weights file not found at {weights_path}")
            print("    Starting training with random initialization instead.\n")
        else:
            print("\n" + "=" * 70)
            print("Loading model weights only...")
            print("=" * 70)
            start_epoch = trainer.load_weights_only(str(weights_path))
            print("=" * 70 + "\n")

    history = trainer.train(start_epoch=start_epoch)
    best_val_loss = trainer.best_val_loss

    # Release optimizer state, the AMP scaler and the torch.compile CUDA graph
    # pools before evaluation; otherwise tens of GB stay resident on the GPU.
    del trainer
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Loading best model for evaluation...")
    checkpoint = torch.load(Path(CHECKPOINT_DIR) / "best_model.pth", map_location=DEVICE)
    state_dict = {k.replace('_orig_mod.', ''): v
                  for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)

    print("\nEvaluating on test set...")
    model.eval()
    model = model.to(torch.device(DEVICE))

    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for static_highres, dynamic_lowres, targets, landmask in test_loader:
            static_highres = static_highres.to(torch.device(DEVICE))
            dynamic_lowres = dynamic_lowres.to(torch.device(DEVICE))
            targets = targets.to(torch.device(DEVICE))
            landmask = landmask.to(torch.device(DEVICE))

            outputs = model(static_highres, dynamic_lowres, landmask=landmask)
            total_loss += criterion(outputs, targets).item()
            total_mae += torch.abs(outputs - targets).mean().item()

    test_loss = total_loss / len(test_loader)
    test_mae = total_mae / len(test_loader)
    test_rmse = test_loss ** 0.5

    print("\n" + "=" * 70)
    print(" " * 25 + "RESULTS")
    print("=" * 70)
    print(f"\nBest validation loss:  {best_val_loss:.4f}")
    print("\nTest metrics:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  MAE:        {test_mae:.4f}")
    print(f"  RMSE:       {test_rmse:.4f}")
    print("\n" + "=" * 70)
    print(f"Checkpoints written to: {CHECKPOINT_DIR}/ (every {SAVE_EVERY_N_EPOCHS} epochs)")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    main(
        resume_checkpoint=RESUME_FROM_CHECKPOINT,
        init_weights_from=INIT_WEIGHTS_FROM,
    )
