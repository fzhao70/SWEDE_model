# SWEDE: Snow Water Equivalent Downscaling with a dual-encoder Swin U-Net

SWEDE learns the mapping from coarse daily atmospheric forcing to high-resolution daily snow water equivalent (SWE) over western North America. It replaces the dynamical downscaling step that would otherwise be needed to translate a global reanalysis or an Earth system model onto a convection-permitting grid.

## Overview

The model reads two inputs at different spatial resolutions without interpolation:

| Branch  | Source                          | Grid            | Channels |
|---------|---------------------------------|-----------------|----------|
| Static  | WRF `wrfinput_d02` terrain      | 340 x 270, 9 km | 6        |
| Dynamic | ERA5-driven daily atmosphere    | 34 x 27, ~90 km | 13       |
| Target  | Daily SWE (`snow`, mm)          | 340 x 270, 9 km | 1        |

A CNN encoder processes the high-resolution terrain, where signal is local and strongly spatially correlated. A Swin Transformer encoder handles the coarse atmospheric fields, where signal is synoptic and long-range. The two branches are fused at the bottleneck and decoded back to the 9 km grid with a CNN decoder. The configuration in the paper contains 37.7 M parameters.

## Repository structure

```
config.py                   All settings: paths, variables, splits, architecture, training
compute_statistics.py       One-pass Welford statistics → normalization_stats.json
dataset.py                  Multi-resolution Dataset and dataloaders
model.py                    DualEncoderSwinUNet, the SWEDE architecture
model_baselines.py          NaiveCNN, UNet, DualEncoderUNet, R2D2UNet baselines
model_swin.py               Single-encoder SwinUNet baseline
trainer.py                  Training loop, composite loss, checkpointing
main.py                     Training entry point
evaluation.py               Inference, metrics and paper figures from a checkpoint
plot.py                     Figure library used by evaluation.py
normalization_stats.json    Statistics for the variables in config.py
run_model_*.pbs             PBS job scripts (V100 / A100 / H100)
stat_cpu.pbs                PBS job script for compute_statistics.py
```

Trained weights are not tracked in the repository: `checkpoints/` and `*.pth` are ignored by `.gitignore` because a SWEDE checkpoint is roughly 450 MB.

## Installation

Python 3.11+ with PyTorch 2.x (CUDA build). Install PyTorch for your CUDA version first, then the remaining dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`torch.compile` is enabled by default (`USE_TORCH_COMPILE` in `config.py`); set it to `False` on stacks where compilation is unavailable.

On NCAR Casper, the environment used for the paper is loaded with:

```bash
module load conda
conda activate torch-gpu-cu124
```

## Data preparation

Three inputs are required; point them to from `config.py`:

- **`DYNAMIC_PATH`** — one NetCDF file per variable per year, named by `DYNAMIC_FILE_TEMPLATE` (e.g., `{var_name}.daily.era5.d02.{year}.nc`). Each file holds a `(day, lat2d, lon2d)` array on the coarse grid.
- **`TARGET_PATH`** — same layout for the target variable `snow` on the 9 km grid.
- **`STATIC_PATH`** — a WRF `wrfinput` file supplying `LANDMASK`, `LU_INDEX`, `HGT`, `VAR`, `GRADHGT_X` and `GRADHGT_Y`.

**Data splits** are by year: training 1951–2013, validation 2017–2021, test 2014–2024.

**Dynamic predictors** (13 channels):
- Instantaneous atmospheric state: `t2`, `t2min`, `t2max`, `q2`, `psfc`, `prec`, `lw_dwn`, `sw_dwn`
- Lagged seasonal means (5): `t2_lag90d_mean`, `prec_lag110d_mean`, `lw_dwn_lag80d_mean`, `sw_sfc_lag130d_mean`, `q2_lag45d_mean`

The lagged terms provide the seasonal memory that SWE integrates over; without them the instantaneous state cannot determine snowpack state.

**Boundary handling:** One low-resolution pixel is cropped from every boundary (`HOLO = 1`, equivalent to ten high-resolution pixels) so that the interior grid tiles cleanly under the Swin window partitioning: 34 x 27 becomes 32 x 25, and 340 x 270 becomes 320 x 250. This ensures H=32 is divisible by 8 for clean Swin window tiling.

### Compute normalization statistics

```bash
python compute_statistics.py
```

This streams every training file once and writes `normalization_stats.json`. A precomputed file for the paper configuration is included, so this step is needed only when the data or the variable catalog in `config.py` changes.

## Training

### Basic usage

```bash
python main.py
```

All settings are read from `config.py`. Checkpoints are written to `checkpoints/`:
- `best_model.pth` — lowest validation loss
- `latest.pth` — most recent epoch
- `checkpoint_epoch_N.pth` — every `SAVE_EVERY_N_EPOCHS` epochs

Training curves are written alongside checkpoints.

### Resume training

Set `RESUME_FROM_CHECKPOINT = "./checkpoints/latest.pth"` in `config.py` and raise `EPOCHS`. To start from pretrained weights with a fresh optimizer, set `INIT_WEIGHTS_FROM` instead.

### On a PBS cluster

```bash
qsub stat_cpu.pbs        # compute_statistics.py on CPU nodes
qsub run_model_h100.pbs  # main.py on one H100 (also _a100 / _v100 variants)
```

Edit the `#PBS -A` account string before submitting.

## Model architecture

`DualEncoderSwinUNet` in `model.py`.

**Static encoder** — four CNN stages (64, 128, 256, 512 channels), each two 3×3 convolutions with reflect padding, GroupNorm and SiLU inside a residual block, followed by 2×2 max-pooling. A bottleneck block with 10% `Dropout2d` closes the branch. GroupNorm is used throughout instead of BatchNorm: batch composition varies strongly with season, making batch statistics unreliable.

**Dynamic encoder** — patch embedding followed by two Swin stages (256 and 512 channels; 4 and 8 heads). Each stage is two blocks that alternate regular and shifted window attention over 4×4 windows, with learnable relative position bias. `PatchMerging` halves the resolution between stages. The stage count is derived as `DEPTH - log2(DYNAMIC_DOWNSAMPLE_FACTOR)`, so the two branches meet at a common bottleneck resolution.

**Bottleneck fusion** — the static bottleneck is projected to the Swin channel width, resized if needed, and added to the dynamic bottleneck. Four further Swin blocks run over the sum with a global residual connection spanning all four.

**Decoder** — four PixelShuffle upsampling stages (512 → 256 → 128 → 64 → 32 channels). PixelShuffle avoids the checkerboard artifacts that transposed convolutions produce. Each stage concatenates matching encoder skips, gated by an attention gate; the two deepest stages receive both static and dynamic skips, the two shallowest receive static skips only. Static skips additionally carry a per-stage learnable gate, initialized small at depths where the bottleneck fusion already carries the information.

**Output head** — 32 → 32 → 16 → 1 channels, multiplied by the landmask and passed through ReLU so that predicted SWE is non-negative by construction.

## Configuration

All settings are in `config.py`: data paths, train/val/test year splits, variable selection, model architecture, training hyperparameters, checkpointing, evaluation, runtime options (torch.compile, device, number of workers).

Derived values (channel counts, model architecture parameters) are computed at module import time. The configuration can also be printed:

```bash
python config.py
```

## Citation

If you use this code, please cite the accompanying paper:

[Citation information to be added]

## License

Released under the MIT License. See [LICENSE](LICENSE).
