"""Configuration for the SWEDE snow-water-equivalent downscaling model.

Every module in this repository reads its settings from here. Paths point at
the ERA5-driven WRF archive used in the paper; change them to point at your own
copy of the data.
"""

from pathlib import Path

import torch

# ============================================================================
# DATA PATHS
# ============================================================================

# Daily low-resolution predictors, one file per variable per year
# (*.daily.era5.d02.YYYY.nc).
DYNAMIC_PATH = "/glade/campaign/uwyo/wyom0169/fzhao/wus_gcm_downscale_input/era5/all_nan_fill/all/"

# High-resolution target fields (snow variables).
TARGET_PATH = "/glade/campaign/uwyo/wyom0169/fzhao/wus_gcm_downscale_input/era5/target/"

# WRF static file supplying terrain and land-surface fields.
STATIC_PATH = "/glade/work/fzhao/snow_prediction/static/wrfinput_d02_highfreq"

# Filename template for the dynamic variables.
DYNAMIC_FILE_TEMPLATE = "{var_name}.daily.era5.d02.{year}.nc"

# ============================================================================
# DOMAIN GEOMETRY
# ============================================================================

# Number of encoder stages the dynamic (coarse) branch drops relative to the
# static branch: dynamic_depth = DEPTH - log2(DYNAMIC_DOWNSAMPLE_FACTOR).
DYNAMIC_DOWNSAMPLE_FACTOR = 4

# Boundary halo: number of low-res pixels cropped from each side of the dynamic
# inputs. Static and target arrays are cropped by HOLO * HOLO_HIGH_RES_SCALE.
# At HOLO=1, dynamic 34x27 -> 32x25 and static/target 340x270 -> 320x250; the
# resulting H=32 is divisible by 8, giving clean Swin window tiling in both stages.
HOLO = 1

# High-res to low-res spatial ratio (340/34 = 270/27 = 10).
HOLO_HIGH_RES_SCALE = 10

# ============================================================================
# TIME SPLITS
# ============================================================================

TRAIN_YEARS = [i for i in range(1951, 2014)]
VAL_YEARS = [2017, 2018, 2019, 2020, 2021]
TEST_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# ============================================================================
# VARIABLE CATALOG
# Statistics are computed for every variable listed here, so the normalization
# file does not have to be regenerated when the selection below changes.
# ============================================================================

ALL_STATIC_VARS = ['LANDMASK', 'LU_INDEX', 'HGT', 'VAR', 'IVGTYP', 'XLONG', 'XLAT',
                   'XLONG_HIGHFREQ', 'XLAT_HIGHFREQ', 'HGT_HIGHFREQ',
                   'GRADHGT', 'GRADHGT_X', 'GRADHGT_Y']

ALL_DYNAMIC_VARS_2D = ['q2', 't2', 'lwp', 'prec', 'psfc', 'rh', 'ivt_u', 'ivt_v',
                       'tdd_30d', 'tdd_7d', 'fdd_30d', 'fdd_7d', 'prec_accum_7d', 'prec_accum_30d',
                       'q3d_850', 'q3d_500', 't3d_850', 't3d_500',
                       'ivtu_accum_30d', 'ivtv_accum_30d', 'ivtu_max_30d', 'ivtv_max_30d',
                       't2min', 't2max', 'lw_dwn', 'sw_dwn',
                       'sw_dwn_lag80d', 't2_lag40d', 't2max_lag40d',
                       't2min_lag30d', 't2_lag90d_mean', 'rh_lag100d_mean',
                       't2max_lag90d_mean', 'prec_lag110d_mean', 'lw_dwn_lag80d_mean',
                       'rh_lag120d_mean', 't2max_lag100d_mean', 't2min_lag80d_mean',
                       'sw_dwn_lag160d_mean',
                       'rh_lag120d_max', 'sw_dwn_lag140d_max', 't2max_lag90d_max', 't2min_lag80d_max',
                       'sw_sfc_lag130d_mean', 'pblh_lag140d_mean', 'q2_lag45d_mean']

ALL_DYNAMIC_VARS_MULTIDIM = []
ALL_DYNAMIC_3D_VARS = []
ALL_TARGET_VARS = ['snow', 'prec_snow']

# Variables carrying a 'component' dimension; each component becomes its own
# feature channel.
MULTIDIM_VARS = {}

# ============================================================================
# SELECTED VARIABLES
# ============================================================================

# Static predictors. XLONG and XLAT, if selected, expand into cos/sin pairs.
STATIC_VARS = ['LANDMASK', 'LU_INDEX', 'HGT', 'VAR', 'GRADHGT_X', 'GRADHGT_Y']

# Dynamic predictors: instantaneous fields plus the lagged means that carry the
# seasonal memory the snowpack integrates over.
DYNAMIC_VARS = ['q2', 't2', 'prec', 'psfc', 't2min', 't2max', 'lw_dwn', 'sw_dwn',
                't2_lag90d_mean', 'prec_lag110d_mean', 'lw_dwn_lag80d_mean',
                'sw_sfc_lag130d_mean', 'q2_lag45d_mean']

DYNAMIC_3D_VARS = []
DYNAMIC_3D_LEVELS = []

TARGET_VARS = ['snow']

# ============================================================================
# DATA LOADING
# ============================================================================

BATCH_SIZE = 32
NUM_WORKERS = 8

# Keep only training samples whose snow coverage (fraction of pixels with
# snow > 0) exceeds this value. None disables the filter.
MIN_SNOW_COVERAGE_THRESHOLD = None

# Rescale targets to [0, 1]. Delete normalization_stats.json after changing this.
NORMALIZE_OUTPUTS = False

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# One of "dualencoderswinunet" (SWEDE), "unet", or "swinunet".
MODEL_TYPE = "dualencoderswinunet"

BASE_CHANNELS = 64
DEPTH = 4

SWIN_EMBED_DIM = 256
# One entry per Swin stage; each must divide the stage's channel dimension.
SWIN_NUM_HEADS = (4, 8)
SWIN_WINDOW_SIZE = 4

# Only read by the "swinunet" baseline; SWEDE derives its stage count from
# DEPTH and DYNAMIC_DOWNSAMPLE_FACTOR.
SWIN_DEPTHS = (2, 2)
SWIN_PATCH_SIZE = 1
SWIN_MLP_RATIO = 4.0

DROPOUT_RATE = 0.15
# Applied to the attention weights after softmax.
ATTN_DROP_RATE = 0.1
# Applied after the attention output projection.
PROJ_DROP_RATE = 0.1
# Applied inside the CNN decoder conv blocks.
DECODER_DROPOUT_RATE = 0.01
# Stochastic depth, scaled linearly from 0 across Swin blocks; the bottleneck uses 1.5x.
DROP_PATH_RATE = 0.20

# ============================================================================
# TRAINING
# ============================================================================

EPOCHS = 100
LEARNING_RATE = 1e-4

# Linear warmup from WARMUP_START_LR to LEARNING_RATE. Set WARMUP_EPOCHS = 0 to disable.
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-7

# Weight decay is set per parameter group: Swin blocks, the static CNN encoder,
# and the decoder are regularized differently because upsampling layers benefit
# from lighter regularization.
WEIGHT_DECAY = 0.08
CNN_ENCODER_WEIGHT_DECAY = 0.06
DECODER_WEIGHT_DECAY = 0.04

GRAD_CLIP_MAX_NORM = 1.0

# Derive the BCE pos_weight from the training-data statistics.
USE_POS_WEIGHT = False

LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 20

# ============================================================================
# CHECKPOINTS
# ============================================================================

CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_N_EPOCHS = 10
LOG_EVERY_N_BATCHES = 10

# Resume training with the full state (model, optimizer, scheduler, epoch,
# history) from this checkpoint, e.g. "./checkpoints/latest.pth". Training then
# continues from the saved epoch up to EPOCHS. None trains from scratch.
RESUME_FROM_CHECKPOINT = None

# Load model weights only and restart at epoch 0. Ignored when
# RESUME_FROM_CHECKPOINT is set.
INIT_WEIGHTS_FROM = None

# ============================================================================
# EVALUATION
# ============================================================================

DEFAULT_CHECKPOINT = "./checkpoints/best_model.pth"
DEFAULT_OUTPUT_DIR = "./predictions"
N_VIS_SAMPLES = 5

# ============================================================================
# RUNTIME
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch.compile typically gives a 20-50% speedup; disable on incompatible stacks.
USE_TORCH_COMPILE = True
COMPILE_MODE = "reduce-overhead"

# ============================================================================
# DERIVED CHANNEL COUNTS
# ============================================================================

_n_static_channels = 0
for var in STATIC_VARS:
    _n_static_channels += 2 if var in ['XLONG', 'XLAT'] else 1

_n_dynamic_channels = 0
for var in DYNAMIC_VARS:
    _n_dynamic_channels += MULTIDIM_VARS[var]['n_components'] if var in MULTIDIM_VARS else 1

_n_3d_channels = len(DYNAMIC_3D_VARS) * len(DYNAMIC_3D_LEVELS)

N_STATIC_CHANNELS = _n_static_channels
N_DYNAMIC_CHANNELS = _n_dynamic_channels + _n_3d_channels
N_INPUT_CHANNELS = _n_static_channels + _n_dynamic_channels + _n_3d_channels
N_OUTPUT_CHANNELS = len(TARGET_VARS)

DYNAMIC_PATH = Path(DYNAMIC_PATH)
TARGET_PATH = Path(TARGET_PATH)
STATIC_PATH = Path(STATIC_PATH)
CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
DEFAULT_OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "CONFIGURATION")
    print("=" * 70)
    print("\n[DATA]")
    print(f"  Dynamic path:    {DYNAMIC_PATH}")
    print(f"  Target path:     {TARGET_PATH}")
    print(f"  Static path:     {STATIC_PATH}")
    print(f"  Train years:     {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} ({len(TRAIN_YEARS)} years)")
    print(f"  Val years:       {VAL_YEARS}")
    print(f"  Test years:      {TEST_YEARS}")
    print("\n[VARIABLES]")
    print(f"  Static vars:     {len(STATIC_VARS)} base -> {_n_static_channels} channels")
    print(f"  Dynamic vars:    {len(DYNAMIC_VARS)} -> {N_DYNAMIC_CHANNELS} channels")
    print(f"  Target vars:     {len(TARGET_VARS)} ({', '.join(TARGET_VARS)})")
    print(f"  Output channels: {N_OUTPUT_CHANNELS}")
    print("\n[MODEL]")
    print(f"  Type:            {MODEL_TYPE}")
    print(f"  Base channels:   {BASE_CHANNELS}")
    print(f"  Depth:           {DEPTH}")
    print(f"  Embed dim:       {SWIN_EMBED_DIM}")
    print(f"  Num heads:       {SWIN_NUM_HEADS}")
    print(f"  Window size:     {SWIN_WINDOW_SIZE}")
    print(f"  Dropout rate:    {DROPOUT_RATE}")
    print("\n[TRAINING]")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Warmup epochs:   {WARMUP_EPOCHS} ({WARMUP_START_LR:.2e} -> {LEARNING_RATE:.2e})")
    print(f"  Weight decay:    {WEIGHT_DECAY}")
    print(f"  Grad clip:       {GRAD_CLIP_MAX_NORM}")
    print(f"  Early stopping:  {EARLY_STOPPING_PATIENCE} epochs")
    print(f"  Boundary halo:   low-res {HOLO} px / high-res {HOLO * HOLO_HIGH_RES_SCALE} px each side")
    print(f"  Snow coverage:   {MIN_SNOW_COVERAGE_THRESHOLD if MIN_SNOW_COVERAGE_THRESHOLD is not None else 'no filter'}")
    print(f"  Resume from:     {RESUME_FROM_CHECKPOINT or INIT_WEIGHTS_FROM or 'None (from scratch)'}")
    print("\n[RUNTIME]")
    print(f"  Num workers:     {NUM_WORKERS}")
    print(f"  torch.compile:   {USE_TORCH_COMPILE} (mode='{COMPILE_MODE}')")
    print(f"  Device:          {DEVICE}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    print("=" * 70)
