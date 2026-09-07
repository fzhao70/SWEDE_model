"""
Unified ERA5 NetCDF Dataset Loader.

Configuration: Modify constants in config.py to change data settings.

Normalization Statistics:
    This module requires normalization_stats.json to exist.
    Run compute_statistics.py first to generate this file before using create_dataloaders().

This unified version supports both:
1. Year-based files (e.g., snow.daily.era5.d02.2020.nc)
2. Single files without year (e.g., snow_monthly.nc)

The file naming is controlled by DYNAMIC_FILE_TEMPLATE in config.py:
- For yearly files: "{var_name}.daily.era5.d02.{year}.nc"
- For single files: "{var_name}_monthly.nc"
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import xarray as xr
import json
from tqdm import tqdm

from config import (
    DYNAMIC_PATH,
    TARGET_PATH,
    STATIC_PATH,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
    STATIC_VARS,
    DYNAMIC_VARS,
    TARGET_VARS,
    BATCH_SIZE,
    NUM_WORKERS,
    DYNAMIC_FILE_TEMPLATE,
    NORMALIZE_OUTPUTS,
    MIN_SNOW_COVERAGE_THRESHOLD
)

try:
    from config import DYNAMIC_3D_VARS, DYNAMIC_3D_LEVELS
except ImportError:
    DYNAMIC_3D_VARS = []
    DYNAMIC_3D_LEVELS = []

try:
    from config import MULTIDIM_VARS
except ImportError:
    MULTIDIM_VARS = {}

USE_YEARLY_FILES = "{year}" in DYNAMIC_FILE_TEMPLATE

# ============================================================================
# TARGET TRANSFORM (opt-in via config; default "none" = legacy raw-SWE target)
# ----------------------------------------------------------------------------
# When enabled, the model predicts a variance-stabilizing transform of SWE
# (e.g. log1p) instead of raw SWE. The forward transform is applied to the
# target here in the dataset; inverse_transform_target() converts predictions
# (and transformed targets) back to physical SWE (mm) for metrics/plots.
# ============================================================================
try:
    import config as _project_config
    TARGET_TRANSFORM = getattr(_project_config, 'TARGET_TRANSFORM', 'none')
    BOXCOX_LAMBDA = float(getattr(_project_config, 'BOXCOX_LAMBDA', 0.5))
except Exception:
    TARGET_TRANSFORM = 'none'
    BOXCOX_LAMBDA = 0.5


def transform_target_np(x):
    """Forward transform raw SWE -> transformed space (numpy array in, array out)."""
    if TARGET_TRANSFORM == 'log1p':
        return np.log1p(np.maximum(x, 0.0))
    if TARGET_TRANSFORM == 'boxcox':
        xp = np.maximum(x, 0.0)
        if abs(BOXCOX_LAMBDA) < 1e-8:
            return np.log1p(xp)
        return ((xp + 1.0) ** BOXCOX_LAMBDA - 1.0) / BOXCOX_LAMBDA
    return x


def inverse_transform_target(y):
    """Inverse transform transformed space -> raw SWE (mm).

    Accepts a torch.Tensor or numpy array and returns the same type. A no-op
    when TARGET_TRANSFORM == 'none'. Result is clamped to be non-negative.
    """
    is_torch = torch.is_tensor(y)
    if TARGET_TRANSFORM == 'log1p' or (TARGET_TRANSFORM == 'boxcox' and abs(BOXCOX_LAMBDA) < 1e-8):
        if is_torch:
            return torch.expm1(torch.clamp(y, min=0.0))
        return np.expm1(np.maximum(y, 0.0))
    if TARGET_TRANSFORM == 'boxcox':
        if is_torch:
            base = torch.clamp(BOXCOX_LAMBDA * torch.clamp(y, min=0.0) + 1.0, min=0.0)
            return torch.clamp(base ** (1.0 / BOXCOX_LAMBDA) - 1.0, min=0.0)
        base = np.maximum(BOXCOX_LAMBDA * np.maximum(y, 0.0) + 1.0, 0.0)
        return np.maximum(base ** (1.0 / BOXCOX_LAMBDA) - 1.0, 0.0)
    return y


# ============================================================================
# DATASET CLASS
# ============================================================================

def _normalize_holo(holo):
    """Return the per-side crop (south, north, west, east) from an int or 4-sequence.

    An int crops all four sides equally, which is the original behaviour and the
    only form every existing case uses. A 4-tuple crops each side independently,
    e.g. (1, 1, 2, 1) turns a 34x27 low-res grid into 32x24 by taking the extra
    column off the western edge. Index 0 of each axis is the south / west edge of
    the WRF grid, matching the convention in the CONUS_6km_era5_*_holo_multiple
    case-local loaders.
    """
    if holo is None:
        return (0, 0, 0, 0)
    if isinstance(holo, (tuple, list)):
        if len(holo) != 4:
            raise ValueError(
                f"holo must be an int or a (south, north, west, east) 4-tuple, got {holo}")
        return tuple(int(h) for h in holo)
    return (int(holo),) * 4


class ERA5Dataset(Dataset):
    """Unified ERA5 dataset for snow prediction.

    Supports both year-based and single-file formats.
    """

    def __init__(self, years, normalization_stats, apply_snow_coverage_filter=False,
                 use_sequences=False, sequence_length=1, holo=0, holo_high_res_scale=1,
                 preload_dynamic=True, preload_static=True):
        """
        Initialize dataset.

        Args:
            years: List of years to include
            normalization_stats: Pre-computed normalization statistics (required)
            apply_snow_coverage_filter: If True and MIN_SNOW_COVERAGE_THRESHOLD is set,
                                        filter samples based on snow coverage
            use_sequences: If True, load sequences of consecutive time steps (default: False)
            sequence_length: Number of consecutive time steps to load when use_sequences=True (default: 1)
            holo: Low-res boundary pixels to crop from the dynamic inputs (default: 0).
                  An int crops all four sides equally (original behaviour); a
                  (south, north, west, east) 4-tuple crops each side independently,
                  e.g. (1, 1, 2, 1) turns 34x27 into 32x24 by taking the extra column
                  off the western edge. The static and target arrays (high-res) are
                  cropped by the same per-side values times holo_high_res_scale.
            holo_high_res_scale: Multiplier applied to `holo` for high-res arrays (static,
                  target). Set to the high-res / low-res ratio (e.g. 10 for a 340x270 vs
                  34x27 setup). Default 1 keeps the original single-resolution behaviour.
            preload_dynamic: If True, load all dynamic data into memory at init for faster
                             __getitem__ access (default: True)
            preload_static: If True, pre-compute the normalized static array once at init
                            instead of rebuilding it on every __getitem__ (default: True)
        """
        self.dynamic_path = Path(DYNAMIC_PATH)
        self.target_path = Path(TARGET_PATH)
        self.years = sorted(years)
        self.norm_stats = normalization_stats
        self.use_sequences = use_sequences
        self.sequence_length = sequence_length if use_sequences else 1
        self.holo = holo
        self.holo_lr = _normalize_holo(holo)
        self.holo_hr = tuple(h * holo_high_res_scale for h in self.holo_lr)
        # Retained for reporting/backward compatibility; all cropping goes
        # through holo_lr / holo_hr.
        self.holo_high = self.holo_hr[0] if len(set(self.holo_hr)) == 1 else self.holo_hr
        self.preload_dynamic = preload_dynamic
        self.preload_static = preload_static

        self._load_static_data()

        if self.preload_static:
            self._preload_static_data()

        self._build_time_index()

        if self.preload_dynamic:
            self._preload_all_dynamic()

        if apply_snow_coverage_filter and MIN_SNOW_COVERAGE_THRESHOLD is not None:
            self._filter_by_snow_coverage()

    def _crop_lowres(self, data):
        """Crop the per-side low-res halo from the last two (row, col) dims."""
        s, n, w, e = self.holo_lr
        if s == n == w == e == 0:
            return data
        return data[..., s:data.shape[-2] - n, w:data.shape[-1] - e]

    def _crop_highres(self, data):
        """Crop the per-side high-res halo from the last two (row, col) dims."""
        s, n, w, e = self.holo_hr
        if s == n == w == e == 0:
            return data
        return data[..., s:data.shape[-2] - n, w:data.shape[-1] - e]

    def _fill_dynamic_invalid(self, data, key, group='dynamic'):
        """Replace NaN/Inf in a dynamic input with the per-variable mean.

        After downstream normalization ``(x - mean) / std``, filled cells
        become 0, avoiding the constant ``-mean/std`` offset that plain
        NaN->0 fill produced at coastlines and at the western boundary
        (e.g. ``sw_dwn``, ``sw_sfc_lag130d_mean``). Falls back to 0-fill
        if no mean is available in ``norm_stats`` for ``key``.
        """
        stats = self.norm_stats.get(group, {}).get(key, {})
        fill = float(stats['mean']) if 'mean' in stats else 0.0
        return np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

    def _load_static_data(self):
        """Load static geoinformation from wrfinput file."""
        static_path = Path(STATIC_PATH)

        print(f"Loading static data from {static_path}")
        self.static_data = {}

        ds = xr.open_dataset(static_path)
        for var_name in STATIC_VARS:
            if var_name in ds.data_vars or var_name in ds.coords:
                var_data = ds[var_name].values
                if var_data.ndim > 2:
                    var_data = var_data.squeeze()
                    if var_data.ndim > 2:
                        var_data = var_data[0]

                var_data = np.nan_to_num(var_data, nan=0.0, posinf=1e10, neginf=-1e10)

                var_data = self._crop_highres(var_data)

                self.static_data[var_name] = var_data
                print(f"  Loaded {var_name}: {var_data.shape}")
        ds.close()

        first_var = list(self.static_data.values())[0]
        self.height, self.width = first_var.shape[-2:]

    def _preload_static_data(self):
        """Pre-compute the fully processed and normalized static array.

        The result (C_static, H, W) is identical for every sample, so we
        compute it once here and reuse it in _load_single_timestep.
        """
        static_list = []
        for var_name in STATIC_VARS:
            var_data = self.static_data[var_name].copy()
            var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
            if var_name == 'XLONG':
                lon_rad = np.deg2rad(var_data)
                static_list.append(np.cos(lon_rad))
                static_list.append(np.sin(lon_rad))
            elif var_name == 'XLAT':
                lat_rad = np.deg2rad(var_data)
                static_list.append(np.cos(lat_rad))
                static_list.append(np.sin(lat_rad))
            else:
                static_list.append(var_data)

        static = np.stack(static_list, axis=0).astype(np.float32)

        # Apply normalization (mirrors the inline logic in _load_single_timestep)
        channel_idx = 0
        for var_name in STATIC_VARS:
            if var_name == 'XLONG':
                if 'cos_lon' in self.norm_stats['static']:
                    static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['cos_lon']['mean']) / \
                                         self.norm_stats['static']['cos_lon']['std']
                channel_idx += 1
                if 'sin_lon' in self.norm_stats['static']:
                    static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['sin_lon']['mean']) / \
                                         self.norm_stats['static']['sin_lon']['std']
                channel_idx += 1
            elif var_name == 'XLAT':
                if 'cos_lat' in self.norm_stats['static']:
                    static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['cos_lat']['mean']) / \
                                         self.norm_stats['static']['cos_lat']['std']
                channel_idx += 1
                if 'sin_lat' in self.norm_stats['static']:
                    static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['sin_lat']['mean']) / \
                                         self.norm_stats['static']['sin_lat']['std']
                channel_idx += 1
            else:
                if var_name in self.norm_stats['static']:
                    static[channel_idx] = (static[channel_idx] - self.norm_stats['static'][var_name]['mean']) / \
                                         self.norm_stats['static'][var_name]['std']
                channel_idx += 1

        self._static_preload = static
        print(f"Static preloaded: shape={static.shape}, {static.nbytes / (1024 ** 2):.2f} MB")

    def _build_time_index(self):
        """Build index of all time steps.

        Handles both year-based and single-file formats automatically.
        """
        self.time_index = []

        if USE_YEARLY_FILES:
            print(f"Indexing years (year-based files): {self.years}")
            for year in self.years:
                # Find sample file to get number of time steps
                pattern = DYNAMIC_FILE_TEMPLATE.replace("{var_name}", "*").replace("{year}", str(year))
                year_files = list(self.dynamic_path.glob(pattern))
                if not year_files:
                    print(f"  Warning: No files found for year {year}")
                    continue

                ds = xr.open_dataset(year_files[0])
                if 'time' in ds.dims:
                    n_times = len(ds.dims['time'])
                elif 'Time' in ds.dims:
                    n_times = len(ds.dims['Time'])
                else:
                    first_var = list(ds.data_vars.keys())[-1]
                    n_times = ds[first_var].shape[0]
                ds.close()

                for t in range(n_times):
                    self.time_index.append((year, t))

                print(f"  Year {year}: {n_times} time steps")
        else:
            print(f"Indexing time steps for years: {self.years} (single-file format)")

            pattern = DYNAMIC_FILE_TEMPLATE.replace("{var_name}", "*")
            sample_files = list(self.dynamic_path.glob(pattern))
            if not sample_files:
                print(f"  Warning: No files found matching pattern {pattern}")
                return

            ds = xr.open_dataset(sample_files[0])

            if 'time' in ds.coords:
                time_coord = ds.coords['time']
            elif 'Time' in ds.coords:
                time_coord = ds.coords['Time']
            else:
                print(f"  Warning: No time coordinate found in file")
                ds.close()
                return

            import pandas as pd
            time_values = pd.to_datetime(time_coord.values)
            n_times = len(time_values)

            if self.years:
                for t_idx, time_val in enumerate(time_values):
                    year = time_val.year
                    if year in self.years:
                        # Store as (year, time_idx_within_file) tuple
                        # to be consistent with year-based format
                        self.time_index.append((year, t_idx))

                print(f"  Selected {len(self.time_index)} time steps from {n_times} total")
            else:
                for t_idx, time_val in enumerate(time_values):
                    year = time_val.year
                    self.time_index.append((year, t_idx))
                print(f"  Using all {n_times} time steps")

            ds.close()

        if self.use_sequences and self.sequence_length > 1:
            original_count = len(self.time_index)
            filtered_index = []
            for i, (year, t) in enumerate(self.time_index):
                # For year-based: consecutive timesteps must be in same year
                # For single-file: consecutive timesteps can span years
                valid = True
                for offset in range(1, self.sequence_length):
                    if i + offset >= len(self.time_index):
                        valid = False
                        break
                    next_year, next_t = self.time_index[i + offset]
                    if USE_YEARLY_FILES:
                        if next_year != year or next_t != t + offset:
                            valid = False
                            break
                    else:
                        if next_t != t + offset:
                            valid = False
                            break
                if valid:
                    filtered_index.append((year, t))
            self.time_index = filtered_index

            print(f"  Filtered for sequences (length={self.sequence_length}): {original_count} -> {len(self.time_index)} samples")

        print(f"Total: {len(self.time_index)} samples")

    def _preload_all_dynamic(self):
        """Preload all dynamic and 3D dynamic data into memory.

        Stores full time-series arrays per variable so __getitem__ can
        index directly without any file I/O.  Layout:
          - year-based files:  cache[key][year] = np.ndarray(T, H, W)
          - single file:       cache[key]        = np.ndarray(T, H, W)
        """
        import re
        print("Preloading dynamic data into memory (this may take a while)...")
        self._dynamic_preload = {}
        self._dynamic_3d_preload = {}

        years_to_iter = self.years if USE_YEARLY_FILES else [None]

        for year in tqdm(years_to_iter, desc="Preloading dynamic data"):
            # ---- DYNAMIC_VARS ----
            for var_name in DYNAMIC_VARS:
                filename = (DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=year)
                            if USE_YEARLY_FILES else
                            DYNAMIC_FILE_TEMPLATE.format(var_name=var_name))
                var_file = self.dynamic_path / filename
                try:
                    ds = xr.open_dataset(var_file)
                    valid_vars = [v for v in ds.data_vars.keys()
                                  if v not in ['time', 'Time', 'lat', 'day', 'lon',
                                               'latitude', 'longitude', 'pressure']]
                    if not valid_vars:
                        ds.close()
                        continue
                    the_var = ds[valid_vars[0]]
                    var_dims = the_var.dims
                    full_data = the_var.values  # (T, ...)
                    pressure_coords = ds['pressure'].values if 'pressure' in ds.coords else None
                    ds.close()

                    if var_name in MULTIDIM_VARS:
                        n_comp = MULTIDIM_VARS[var_name]['n_components']
                        comp_names = MULTIDIM_VARS[var_name]['component_names']
                        if full_data.ndim == 4 and full_data.shape[1] == n_comp:
                            for i, comp_name in enumerate(comp_names):
                                cdata = full_data[:, i, :, :]
                                cdata = self._crop_lowres(cdata)
                                cdata = self._fill_dynamic_invalid(cdata, comp_name, group='dynamic').astype(np.float32)
                                self._store_preload(self._dynamic_preload, comp_name, year, cdata)

                    elif 'pressure' in var_dims and pressure_coords is not None:
                        match = re.search(r'_(\d+)$', var_name)
                        if match:
                            target_pressure = float(match.group(1))
                            if target_pressure in pressure_coords:
                                pidx = list(pressure_coords).index(target_pressure)
                                pdata = full_data[:, pidx, :, :]
                                pdata = self._crop_lowres(pdata)
                                pdata = self._fill_dynamic_invalid(pdata, var_name, group='dynamic').astype(np.float32)
                                self._store_preload(self._dynamic_preload, var_name, year, pdata)

                    else:
                        if full_data.ndim == 4:
                            full_data = full_data[:, 0, :, :]  # squeeze singleton dim
                        if full_data.ndim != 3:
                            continue
                        full_data = self._crop_lowres(full_data)
                        full_data = self._fill_dynamic_invalid(full_data, var_name, group='dynamic').astype(np.float32)
                        self._store_preload(self._dynamic_preload, var_name, year, full_data)

                except Exception:
                    continue

            # ---- DYNAMIC_3D_VARS ----
            for var_name in DYNAMIC_3D_VARS:
                filename = (DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=year)
                            if USE_YEARLY_FILES else
                            DYNAMIC_FILE_TEMPLATE.format(var_name=var_name))
                var_file = self.dynamic_path / filename
                try:
                    ds = xr.open_dataset(var_file)
                    valid_vars = [v for v in ds.data_vars.keys()
                                  if v not in ['time', 'Time', 'lat', 'day', 'lon',
                                               'latitude', 'longitude', 'pressure']]
                    if not valid_vars:
                        ds.close()
                        continue
                    full_data = ds[valid_vars[0]].values  # (T, level, H, W)
                    ds.close()

                    for level in DYNAMIC_3D_LEVELS:
                        if full_data.ndim >= 3 and level < full_data.shape[1]:
                            ldata = full_data[:, level, :, :]
                            if ldata.ndim > 3:
                                ldata = ldata.squeeze()
                            if ldata.ndim != 3:
                                continue
                            ldata = self._crop_lowres(ldata)
                            key = f"{var_name}_level_{level}"
                            ldata = self._fill_dynamic_invalid(ldata, key, group='dynamic_3d').astype(np.float32)
                            self._store_preload(self._dynamic_3d_preload, key, year, ldata)

                except Exception:
                    continue

        total_bytes = 0
        for entry in list(self._dynamic_preload.values()) + list(self._dynamic_3d_preload.values()):
            if isinstance(entry, np.ndarray):
                total_bytes += entry.nbytes
            elif isinstance(entry, dict):
                total_bytes += sum(a.nbytes for a in entry.values())
        print(f"Preloading complete: {len(self._dynamic_preload)} dynamic + "
              f"{len(self._dynamic_3d_preload)} 3D arrays, "
              f"{total_bytes / (1024 ** 2):.1f} MB in memory")

    def _store_preload(self, cache_dict, key, year, data):
        """Store a (T, H, W) array in the preload cache."""
        if USE_YEARLY_FILES:
            if key not in cache_dict:
                cache_dict[key] = {}
            cache_dict[key][year] = data
        else:
            cache_dict[key] = data

    def _get_from_preload(self, cache_dict, key, year, time_idx):
        """Return one (H, W) slice from preloaded cache, or None if missing."""
        if key not in cache_dict:
            return None
        entry = cache_dict[key]
        if USE_YEARLY_FILES:
            if not isinstance(entry, dict) or year not in entry:
                return None
            return entry[year][time_idx]
        else:
            return entry[time_idx]

    def _load_dynamic_from_cache(self, file_identifier, time_idx):
        """Return dynamic variable dict using preloaded arrays."""
        dynamic_data = {}
        year = file_identifier
        for var_name in DYNAMIC_VARS:
            if var_name in MULTIDIM_VARS:
                for comp_name in MULTIDIM_VARS[var_name]['component_names']:
                    data = self._get_from_preload(self._dynamic_preload, comp_name, year, time_idx)
                    if data is not None:
                        dynamic_data[comp_name] = data
            else:
                data = self._get_from_preload(self._dynamic_preload, var_name, year, time_idx)
                if data is not None:
                    dynamic_data[var_name] = data
        return dynamic_data

    def _load_3d_from_cache(self, file_identifier, time_idx):
        """Return 3D variable dict using preloaded arrays."""
        data_3d = {}
        year = file_identifier
        for var_name in DYNAMIC_3D_VARS:
            for level in DYNAMIC_3D_LEVELS:
                key = f"{var_name}_level_{level}"
                data = self._get_from_preload(self._dynamic_3d_preload, key, year, time_idx)
                if data is not None:
                    data_3d[key] = data
        return data_3d

    def _filter_by_snow_coverage(self):
        """Filter time index to only include samples with snow coverage above threshold.

        Only runs if MIN_SNOW_COVERAGE_THRESHOLD is not None.
        Snow coverage is calculated as the percentage of pixels with snow > 0.
        """
        print(f"\nFiltering samples by snow coverage (threshold: {MIN_SNOW_COVERAGE_THRESHOLD*100:.1f}%)")

        filtered_index = []

        # Get snow file - should be first target variable
        target_var = TARGET_VARS[0]

        if USE_YEARLY_FILES:
            for year in tqdm(self.years, desc="Filtering by snow coverage"):
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=target_var, year=year)
                target_file = self.dynamic_path / filename

                if not target_file.exists():
                    print(f"\n  Warning: Target file not found: {target_file}")
                    continue

                try:
                    ds = xr.open_dataset(target_file)
                    var_names = [v for v in ds.data_vars.keys()
                                if v not in ['time', 'Time', 'lat', 'lon', 'latitude', 'longitude', 'day', 'pressure']]
                    if not var_names:
                        print(f"\n  Warning: No valid variables found in {target_file}")
                        ds.close()
                        continue

                    snow_var = ds[var_names[0]]

                    for year_idx, time_idx in [(y, t) for y, t in self.time_index if y == year]:
                        snow_data = snow_var.isel({snow_var.dims[0]: time_idx}).values

                        if snow_data.ndim > 2:
                            snow_data = snow_data.squeeze()

                        snow_data = np.nan_to_num(snow_data, nan=0.0, posinf=0.0, neginf=0.0)
                        snow_data = np.where(np.abs(snow_data) > 1e29, 0.0, snow_data)

                        total_pixels = snow_data.size
                        snow_pixels = np.sum(snow_data > 0)
                        snow_coverage = snow_pixels / total_pixels

                        if snow_coverage >= MIN_SNOW_COVERAGE_THRESHOLD:
                            filtered_index.append((year_idx, time_idx))

                    ds.close()

                except Exception as e:
                    print(f"\n  Warning: Failed to read snow data for year {year}: {e}")
                    continue
        else:
            filename = DYNAMIC_FILE_TEMPLATE.format(var_name=target_var)
            target_file = self.dynamic_path / filename

            if not target_file.exists():
                print(f"\n  Warning: Target file not found: {target_file}")
                return

            try:
                ds = xr.open_dataset(target_file)
                var_names = [v for v in ds.data_vars.keys()
                            if v not in ['time', 'Time', 'lat', 'lon', 'latitude', 'longitude', 'day', 'pressure']]
                if not var_names:
                    print(f"\n  Warning: No valid variables found in {target_file}")
                    ds.close()
                    return

                snow_var = ds[var_names[0]]

                for year_idx, time_idx in tqdm(self.time_index, desc="Filtering by snow coverage"):
                    snow_data = snow_var.isel({snow_var.dims[0]: time_idx}).values

                    if snow_data.ndim > 2:
                        snow_data = snow_data.squeeze()

                    snow_data = np.nan_to_num(snow_data, nan=0.0, posinf=0.0, neginf=0.0)
                    snow_data = np.where(np.abs(snow_data) > 1e29, 0.0, snow_data)

                    total_pixels = snow_data.size
                    snow_pixels = np.sum(snow_data > 0)
                    snow_coverage = snow_pixels / total_pixels

                    if snow_coverage >= MIN_SNOW_COVERAGE_THRESHOLD:
                        filtered_index.append((year_idx, time_idx))

                ds.close()

            except Exception as e:
                print(f"\n  Warning: Failed to read snow data: {e}")
                return

        original_count = len(self.time_index)
        self.time_index = filtered_index
        filtered_count = len(self.time_index)

        print(f"Filtered {original_count - filtered_count} samples ({(original_count - filtered_count)/original_count*100:.1f}%)")
        print(f"Remaining: {filtered_count} samples with snow coverage >= {MIN_SNOW_COVERAGE_THRESHOLD*100:.1f}%")

    def _load_dynamic_data(self, file_identifier, time_idx):
        """Load dynamic variables for a specific time. Returns dict keyed by variable/component name.

        Args:
            file_identifier: For yearly files, this is the year. For single files, unused.
            time_idx: Time index within the file
        """
        if self.preload_dynamic and hasattr(self, '_dynamic_preload'):
            return self._load_dynamic_from_cache(file_identifier, time_idx)

        dynamic_data = {}

        for var_name in DYNAMIC_VARS:
            if USE_YEARLY_FILES:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=file_identifier)
            else:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name)

            var_file = self.dynamic_path / filename

            try:
                ds = xr.open_dataset(var_file)
                var_names = [v for v in ds.data_vars.keys()
                            if v not in ['time', 'Time', 'lat', 'day', 'lon', 'latitude', 'longitude', 'pressure']]
                if not var_names:
                    ds.close()
                    continue

                var_data = ds[var_names[0]].isel({ds[var_names[0]].dims[0]: time_idx}).values

                if var_name in MULTIDIM_VARS:
                    n_components = MULTIDIM_VARS[var_name]['n_components']
                    component_names = MULTIDIM_VARS[var_name]['component_names']
                    # Expected shape: (n_components, lat, lon)
                    if var_data.ndim == 3 and var_data.shape[0] == n_components:
                        # Data is already in correct format (component, lat, lon)
                        pass
                    else:
                        ds.close()
                        continue

                    expected_shape = (self.height, self.width)
                    var_data = self._crop_lowres(var_data)

                    for i, comp_name in enumerate(component_names):
                        dynamic_data[comp_name] = self._fill_dynamic_invalid(
                            var_data[i], comp_name, group='dynamic'
                        )

                else:
                    if var_data.ndim > 2:
                        if 'pressure' in ds.dims or 'pressure' in ds[var_names[0]].dims:
                            # Extract pressure level from variable name (e.g., q3d_850 -> 850)
                            import re
                            match = re.search(r'_(\d+)$', var_name)
                            if match:
                                target_pressure = float(match.group(1))
                                if 'pressure' in ds.coords:
                                    pressure_levels = ds.pressure.values
                                    if target_pressure in pressure_levels:
                                        pressure_idx = list(pressure_levels).index(target_pressure)
                                        var_data = var_data[pressure_idx, :, :]
                                    else:
                                        ds.close()
                                        continue
                                else:
                                    ds.close()
                                    continue
                            else:
                                ds.close()
                                continue
                        else:
                            var_data = var_data.squeeze()

                    if var_data.ndim != 2:
                        ds.close()
                        continue

                    var_data = self._crop_lowres(var_data)

                    # Mean-fill so post-normalization filled cells become 0
                    var_data = self._fill_dynamic_invalid(var_data, var_name, group='dynamic')

                    dynamic_data[var_name] = var_data

                ds.close()

            except Exception as e:
                continue

        return dynamic_data

    def _load_3d_data(self, file_identifier, time_idx):
        """Load 3D variables for specific levels. Returns dict keyed by 'varname_level_N'.

        Args:
            file_identifier: For yearly files, this is the year. For single files, unused.
            time_idx: Time index within the file
        """
        if self.preload_dynamic and hasattr(self, '_dynamic_3d_preload'):
            return self._load_3d_from_cache(file_identifier, time_idx)

        data_3d = {}
        expected_shape = (self.height, self.width)

        for var_name in DYNAMIC_3D_VARS:
            if USE_YEARLY_FILES:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=file_identifier)
            else:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name)

            var_file = self.dynamic_path / filename

            try:
                ds = xr.open_dataset(var_file)
                var_names = [v for v in ds.data_vars.keys()
                            if v not in ['time', 'Time', 'lat', 'day', 'lon', 'latitude', 'longitude', 'pressure']]
                if not var_names:
                    ds.close()
                    continue

                var_data = ds[var_names[0]].isel({ds[var_names[0]].dims[0]: time_idx}).values

                # var_data should have shape (level, lat, lon) or similar
                for level in DYNAMIC_3D_LEVELS:
                    if var_data.ndim >= 3 and level < var_data.shape[0]:
                        # Assume first dimension is vertical level
                        level_data = var_data[level]
                        if level_data.ndim > 2:
                            level_data = level_data.squeeze()
                    else:
                        continue

                    if level_data.ndim != 2:
                        continue

                    level_data = self._crop_lowres(level_data)

                    # Mean-fill so post-normalization filled cells become 0
                    key = f"{var_name}_level_{level}"
                    level_data = self._fill_dynamic_invalid(level_data, key, group='dynamic_3d')

                    data_3d[key] = level_data

                ds.close()

            except Exception as e:
                continue

        return data_3d

    def _load_target(self, file_identifier, time_idx):
        """Load target variables. Returns dict keyed by variable name.

        Args:
            file_identifier: For yearly files, this is the year. For single files, unused.
            time_idx: Time index within the file
        """
        target_data = {}
        expected_shape = (self.height, self.width)

        for target_var in TARGET_VARS:
            if USE_YEARLY_FILES:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=target_var, year=file_identifier)
            else:
                filename = DYNAMIC_FILE_TEMPLATE.format(var_name=target_var)

            target_file = self.target_path / filename

            if not target_file.exists():
                raise FileNotFoundError(f"Target file not found: {target_file}")

            try:
                ds = xr.open_dataset(target_file)
                var_names = [v for v in ds.data_vars.keys()
                            if v not in ['time', 'Time', 'lat', 'lon', 'latitude', 'longitude', 'day', 'pressure']]
                if not var_names:
                    ds.close()
                    raise ValueError(f"No valid variables found in target file: {target_file}")

                var_data = ds[var_names[0]].isel({ds[var_names[0]].dims[0]: time_idx}).values
                if var_data.ndim > 2:
                    var_data = var_data.squeeze()

                if var_data.ndim != 2:
                    ds.close()
                    raise ValueError(f"Target variable '{target_var}' has unexpected dimensions: {var_data.shape}")

                var_data = self._crop_highres(var_data)

                var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                var_data = np.nan_to_num(var_data, nan=0.0, posinf=0.0, neginf=0.0)

                target_data[target_var] = var_data

                ds.close()

            except Exception as e:
                raise RuntimeError(f"Error loading target variable '{target_var}': {e}")

        return target_data

    def __len__(self):
        return len(self.time_index)

    def _load_single_timestep(self, file_identifier, time_idx):
        """Load data for a single timestep.

        Args:
            file_identifier: Year for year-based files, None for single-file format
            time_idx: Time index within the file

        Returns:
            inputs: Combined input array (static + dynamic + dynamic_3d), shape (C, H, W)
            target: Target array, shape (num_targets, H, W)
            landmask: Landmask array, shape (H, W)
        """
        dynamic_dict = self._load_dynamic_data(file_identifier, time_idx)
        dynamic_3d_dict = self._load_3d_data(file_identifier, time_idx)
        target_dict = self._load_target(file_identifier, time_idx)

        if self.preload_static and hasattr(self, '_static_preload'):
            static = self._static_preload
        else:
            # XLONG and XLAT expand to 4 channels: cos_lon, sin_lon, cos_lat, sin_lat
            static_list = []
            for var_name in STATIC_VARS:
                var_data = self.static_data[var_name]
                var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                if var_name == 'XLONG':
                    lon_rad = np.deg2rad(var_data)
                    static_list.append(np.cos(lon_rad))
                    static_list.append(np.sin(lon_rad))
                elif var_name == 'XLAT':
                    lat_rad = np.deg2rad(var_data)
                    static_list.append(np.cos(lat_rad))
                    static_list.append(np.sin(lat_rad))
                else:
                    static_list.append(var_data)
            static = np.stack(static_list, axis=0).astype(np.float32)

        dynamic_list = []
        for var_name in DYNAMIC_VARS:
            if var_name in MULTIDIM_VARS:
                for comp_name in MULTIDIM_VARS[var_name]['component_names']:
                    if comp_name in dynamic_dict:
                        var_data = dynamic_dict[comp_name]
                        var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                        dynamic_list.append(var_data)
            else:
                if var_name in dynamic_dict:
                    var_data = dynamic_dict[var_name]
                    var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                    dynamic_list.append(var_data)

        dynamic = np.stack(dynamic_list, axis=0).astype(np.float32) if dynamic_list else np.zeros((0, self.height, self.width), dtype=np.float32)

        dynamic_3d_list = []
        for var_name in DYNAMIC_3D_VARS:
            for level in DYNAMIC_3D_LEVELS:
                key = f"{var_name}_level_{level}"
                if key in dynamic_3d_dict:
                    var_data = dynamic_3d_dict[key]
                    var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                    dynamic_3d_list.append(var_data)

        dynamic_3d = np.stack(dynamic_3d_list, axis=0).astype(np.float32) if dynamic_3d_list else np.zeros((0, self.height, self.width), dtype=np.float32)

        target_list = []
        for var_name in TARGET_VARS:
            if var_name in target_dict:
                var_data = target_dict[var_name]
                var_data = np.where(np.abs(var_data) > 1e29, 0.0, var_data)
                target_list.append(var_data)

        target = np.stack(target_list, axis=0).astype(np.float32)

        if not (self.preload_static and hasattr(self, '_static_preload')):
            channel_idx = 0
            for var_name in STATIC_VARS:
                if var_name == 'XLONG':
                    if 'cos_lon' in self.norm_stats['static']:
                        static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['cos_lon']['mean']) / \
                                             self.norm_stats['static']['cos_lon']['std']
                    channel_idx += 1
                    if 'sin_lon' in self.norm_stats['static']:
                        static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['sin_lon']['mean']) / \
                                             self.norm_stats['static']['sin_lon']['std']
                    channel_idx += 1
                elif var_name == 'XLAT':
                    if 'cos_lat' in self.norm_stats['static']:
                        static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['cos_lat']['mean']) / \
                                             self.norm_stats['static']['cos_lat']['std']
                    channel_idx += 1
                    if 'sin_lat' in self.norm_stats['static']:
                        static[channel_idx] = (static[channel_idx] - self.norm_stats['static']['sin_lat']['mean']) / \
                                             self.norm_stats['static']['sin_lat']['std']
                    channel_idx += 1
                else:
                    if var_name in self.norm_stats['static']:
                        static[channel_idx] = (static[channel_idx] - self.norm_stats['static'][var_name]['mean']) / \
                                             self.norm_stats['static'][var_name]['std']
                    channel_idx += 1

        channel_idx = 0
        for var_name in DYNAMIC_VARS:
            if var_name in MULTIDIM_VARS:
                for comp_name in MULTIDIM_VARS[var_name]['component_names']:
                    if comp_name in dynamic_dict:
                        if comp_name in self.norm_stats['dynamic']:
                            dynamic[channel_idx] = (dynamic[channel_idx] - self.norm_stats['dynamic'][comp_name]['mean']) / \
                                                  self.norm_stats['dynamic'][comp_name]['std']
                        channel_idx += 1
            else:
                if var_name in dynamic_dict:
                    if var_name in self.norm_stats['dynamic']:
                        dynamic[channel_idx] = (dynamic[channel_idx] - self.norm_stats['dynamic'][var_name]['mean']) / \
                                              self.norm_stats['dynamic'][var_name]['std']
                    channel_idx += 1

        channel_idx = 0
        for var_name in DYNAMIC_3D_VARS:
            for level in DYNAMIC_3D_LEVELS:
                key = f"{var_name}_level_{level}"
                if key in dynamic_3d_dict:
                    if key in self.norm_stats.get('dynamic_3d', {}):
                        dynamic_3d[channel_idx] = (dynamic_3d[channel_idx] - self.norm_stats['dynamic_3d'][key]['mean']) / \
                                                 self.norm_stats['dynamic_3d'][key]['std']
                    channel_idx += 1

        # Normalize targets if enabled (using minmax: value / max)
        if NORMALIZE_OUTPUTS:
            for i, var_name in enumerate(TARGET_VARS):
                if var_name in self.norm_stats['target']:
                    if self.norm_stats['target'][var_name]['max'] > 0:
                        target[i] = target[i] / self.norm_stats['target'][var_name]['max']

        # Apply variance-stabilizing target transform (opt-in via config.TARGET_TRANSFORM).
        # The model then predicts the transformed SWE directly; invert at eval time
        # with inverse_transform_target(). No-op when TARGET_TRANSFORM == 'none'.
        if TARGET_TRANSFORM != 'none':
            target = transform_target_np(target).astype(np.float32)

        landmask_idx = 0
        for i, var_name in enumerate(STATIC_VARS):
            if var_name == 'LANDMASK':
                landmask_idx = i
                break

        landmask = self.static_data['LANDMASK'].astype(np.float32)
        landmask = np.where(np.abs(landmask) > 1e29, 0.0, landmask)

        dynamic_combined = np.concatenate([dynamic, dynamic_3d], axis=0).astype(np.float32) if dynamic_3d.shape[0] > 0 else dynamic.astype(np.float32)

        # Return numpy arrays: (static_highres, dynamic_lowres, target, landmask)
        return static.astype(np.float32), dynamic_combined, target, landmask

    def __getitem__(self, idx):
        """Get a single sample or sequence of samples.

        Returns:
            For multi-resolution DualEncoderUNet:
                static_highres: (C_static, H_high, W_high) tensor - high-resolution static features
                dynamic_lowres: (C_dynamic, H_low, W_low) tensor - low-resolution dynamic features
                target: (num_targets, H_high, W_high) tensor - target at high resolution
                landmask: (H_high, W_high) tensor - landmask at high resolution

            Note: Sequences are not yet supported for multi-resolution format
        """
        year, time_idx = self.time_index[idx]
        if USE_YEARLY_FILES:
            file_identifier = year
        else:
            file_identifier = None

        if not self.use_sequences or self.sequence_length == 1:
            static_highres, dynamic_lowres, target, landmask = self._load_single_timestep(file_identifier, time_idx)
            return (torch.from_numpy(static_highres),
                    torch.from_numpy(dynamic_lowres),
                    torch.from_numpy(target),
                    torch.from_numpy(landmask))
        else:
            static_highres, dynamic_lowres, target, landmask = self._load_single_timestep(file_identifier, time_idx)
            return (torch.from_numpy(static_highres),
                    torch.from_numpy(dynamic_lowres),
                    torch.from_numpy(target),
                    torch.from_numpy(landmask))


# ============================================================================
# DENORMALIZATION UTILITIES
# ============================================================================


def denormalize_outputs(outputs: torch.Tensor, norm_stats: dict) -> torch.Tensor:
    """
    Denormalize model outputs back to original scale using minmax denormalization.

    Args:
        outputs: Normalized outputs (B, C, H, W) in range [0, 1]
        norm_stats: Dictionary containing 'target' key with dict of variable stats

    Returns:
        Denormalized outputs in original scale
    """
    if not NORMALIZE_OUTPUTS:
        return outputs

    target_stats = norm_stats.get('target', None)
    if target_stats is None:
        raise ValueError("target stats not found in normalization stats")

    is_torch = isinstance(outputs, torch.Tensor)
    if is_torch:
        device = outputs.device
        outputs = outputs.cpu().numpy()

    # Denormalize each channel: value = normalized * max
    denorm_outputs = np.zeros_like(outputs)
    for i, var_name in enumerate(TARGET_VARS):
        if var_name in target_stats:
            target_max = target_stats[var_name]['max']
            if target_max > 0:
                denorm_outputs[:, i] = outputs[:, i] * target_max

    if is_torch:
        denorm_outputs = torch.from_numpy(denorm_outputs).to(device)

    return denorm_outputs


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(use_sequences=False, sequence_length=1, holo=0, holo_high_res_scale=1,
                       shuffle_train=True, preload_dynamic=True, preload_static=True):
    """Create train, val, and test dataloaders.

    Args:
        use_sequences: If True, load sequences of consecutive time steps (default: False)
        sequence_length: Number of consecutive time steps to load when use_sequences=True (default: 1)
        holo: Low-res boundary pixels to crop from the dynamic inputs (default: 0).
              An int crops all four sides equally (original behaviour); a
              (south, north, west, east) 4-tuple crops each side independently.
              High-res arrays (static, target) are cropped by the same per-side
              values times holo_high_res_scale.
        holo_high_res_scale: Multiplier applied to `holo` for high-res arrays. Set to the
              high-res / low-res spatial ratio (e.g. 10 for 340x270 vs 34x27). Default 1
              preserves the original single-resolution behaviour.
        shuffle_train: If True, shuffle training data (default: True).
                      Set to False for Stage 2 training with chronologically ordered data.
        preload_dynamic: If True, load all dynamic data into memory at dataset init (default: True).
                        Set to False to reduce memory usage at the cost of per-sample file I/O.
        preload_static: If True, pre-compute the normalized static array once at init (default: True).
                        Static data is the same for every sample so this avoids redundant computation.

    Note: This function requires normalization_stats.json to exist.
          Run compute_statistics.py first to generate this file.
    """
    print("=" * 70)
    print("Creating Dataloaders")
    print("=" * 70)
    print(f"File format: {'Year-based' if USE_YEARLY_FILES else 'Single-file'}")
    print(f"Template: {DYNAMIC_FILE_TEMPLATE}")
    print(f"Shuffle training data: {shuffle_train}")
    print(f"Preload dynamic data:  {preload_dynamic}")
    print(f"Preload static data:   {preload_static}")
    _holo_sides = _normalize_holo(holo)
    if any(h > 0 for h in _holo_sides):
        _holo_hr = tuple(h * holo_high_res_scale for h in _holo_sides)
        print(f"Boundary cropping (south, north, west, east): "
              f"low-res {_holo_sides}px, high-res {_holo_hr}px")

    stats_file = Path("normalization_stats.json")

    if not stats_file.exists():
        raise FileNotFoundError(f"Normalization statistics file not found: {stats_file}. Run 'python compute_statistics.py' first.")

    print(f"\nLoading normalization statistics from {stats_file}")
    try:
        with open(stats_file, 'r') as f:
            norm_stats = json.load(f)

        if 'static' not in norm_stats or not isinstance(norm_stats['static'], dict):
            print("\n" + "=" * 70)
            print("ERROR: Old statistics file format detected!")
            print("=" * 70)
            print(f"\nThe statistics file {stats_file} uses the old array-based format.")
            print(f"Please regenerate it to use the new dictionary-based format:\n")
            print(f"  python compute_statistics.py")
            print("=" * 70 + "\n")
            raise ValueError(f"Statistics file {stats_file} uses old format. Please regenerate.")

        missing_static = [v for v in STATIC_VARS if v not in norm_stats['static'] and v not in ['XLONG', 'XLAT']]
        if missing_static:
            print(f"\nWarning: Missing statistics for static variables: {missing_static}")
            print("These variables may not have been in the dataset when stats were computed.")

        missing_dynamic = []
        for var_name in DYNAMIC_VARS:
            if var_name in MULTIDIM_VARS:
                for comp_name in MULTIDIM_VARS[var_name]['component_names']:
                    if comp_name not in norm_stats['dynamic']:
                        missing_dynamic.append(comp_name)
            else:
                if var_name not in norm_stats['dynamic']:
                    missing_dynamic.append(var_name)

        if missing_dynamic:
            print(f"\nWarning: Missing statistics for dynamic variables: {missing_dynamic}")
            print("Please recompute statistics if you want to use these variables.")

        missing_3d = []
        for var_name in DYNAMIC_3D_VARS:
            for level in DYNAMIC_3D_LEVELS:
                key = f"{var_name}_level_{level}"
                if key not in norm_stats.get('dynamic_3d', {}):
                    missing_3d.append(key)

        if missing_3d:
            print(f"\nWarning: Missing statistics for 3D variables: {missing_3d}")
            print("Please recompute statistics if you want to use these variables.")

        missing_targets = [v for v in TARGET_VARS if v not in norm_stats['target']]
        if missing_targets:
            print(f"\nERROR: Missing statistics for target variables: {missing_targets}")
            raise ValueError(f"Target variables {missing_targets} not found in statistics. Please recompute.")

        print("✓ Successfully loaded dictionary-based normalization statistics")
        print(f"  Available static vars:    {len(norm_stats['static'])}")
        print(f"  Available dynamic vars:   {len(norm_stats['dynamic'])}")
        print(f"  Available 3D var-levels:  {len(norm_stats.get('dynamic_3d', {}))}")
        print(f"  Available target vars:    {len(norm_stats['target'])}")

    except Exception as e:
        if "Old statistics file format" in str(e) or "uses old format" in str(e):
            raise
        raise RuntimeError(f"Failed to load normalization statistics from {stats_file}: {e}")

    print("\n" + "=" * 70)
    print("Creating Datasets")
    print("=" * 70)

    if MIN_SNOW_COVERAGE_THRESHOLD is not None:
        print(f"\nSnow coverage filtering ENABLED for training set (threshold: {MIN_SNOW_COVERAGE_THRESHOLD*100:.1f}%)")
        print("Note: Only training samples with sufficient snow coverage will be used")
    else:
        print("\nSnow coverage filtering DISABLED (using all samples)")
        print("Note: To filter by snow coverage, set MIN_SNOW_COVERAGE_THRESHOLD in config.py")

    train_dataset = ERA5Dataset(TRAIN_YEARS, normalization_stats=norm_stats, apply_snow_coverage_filter=False,
                                use_sequences=use_sequences, sequence_length=sequence_length,
                                holo=holo, holo_high_res_scale=holo_high_res_scale,
                                preload_dynamic=preload_dynamic, preload_static=preload_static)
    val_dataset = ERA5Dataset(VAL_YEARS, normalization_stats=norm_stats, apply_snow_coverage_filter=False,
                              use_sequences=use_sequences, sequence_length=sequence_length,
                              holo=holo, holo_high_res_scale=holo_high_res_scale,
                              preload_dynamic=preload_dynamic, preload_static=preload_static)
    test_dataset = ERA5Dataset(TEST_YEARS, normalization_stats=norm_stats, apply_snow_coverage_filter=False,
                               use_sequences=use_sequences, sequence_length=sequence_length,
                               holo=holo, holo_high_res_scale=holo_high_res_scale,
                               preload_dynamic=preload_dynamic, preload_static=preload_static)

    if len(train_dataset) == 0:
        print("\n⚠️  WARNING: Training dataset is empty! Check TRAIN_YEARS and data file availability.")
    if len(val_dataset) == 0:
        print("\n⚠️  WARNING: Validation dataset is empty! Check VAL_YEARS and data file availability.")
    if len(test_dataset) == 0:
        print("\n⚠️  WARNING: Test dataset is empty! Check TEST_YEARS and data file availability.")

    if len(train_dataset) == 0 and len(val_dataset) == 0 and len(test_dataset) == 0:
        raise ValueError("All datasets are empty! Please check your year ranges and data files.")

    # Only shuffle if dataset is not empty (to avoid RandomSampler error with 0 samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_train if len(train_dataset) > 0 else False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    # Calculate static channels (XLONG and XLAT expand to 2 channels each)
    n_static_channels = 0
    for var in STATIC_VARS:
        if var in ['XLONG', 'XLAT']:
            n_static_channels += 2
        else:
            n_static_channels += 1

    n_dynamic_channels = 0
    for var in DYNAMIC_VARS:
        if var in MULTIDIM_VARS:
            n_dynamic_channels += MULTIDIM_VARS[var]['n_components']
        else:
            n_dynamic_channels += 1

    n_3d_channels = len(DYNAMIC_3D_VARS) * len(DYNAMIC_3D_LEVELS)

    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Training:   {len(train_dataset):5d} samples from years {TRAIN_YEARS}")
    print(f"Validation: {len(val_dataset):5d} samples from years {VAL_YEARS}")
    print(f"Test:       {len(test_dataset):5d} samples from years {TEST_YEARS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Spatial:    {train_dataset.height} x {train_dataset.width}")
    print(f"Static:     {len(STATIC_VARS)} base variables -> {n_static_channels} channels")
    if 'XLONG' in STATIC_VARS or 'XLAT' in STATIC_VARS:
        print(f"  └─ XLONG, XLAT expand to cos/sin encodings")
    print(f"Dynamic:    {len(DYNAMIC_VARS)} base variables -> {n_dynamic_channels} channels")
    if MULTIDIM_VARS:
        for var_name, var_info in MULTIDIM_VARS.items():
            if var_name in DYNAMIC_VARS:
                print(f"  └─ {var_name}: {var_info['n_components']} components")
    if DYNAMIC_3D_VARS:
        print(f"Dynamic 3D: {len(DYNAMIC_3D_VARS)} variables × {len(DYNAMIC_3D_LEVELS)} levels -> {n_3d_channels} channels")
        for var_name in DYNAMIC_3D_VARS:
            print(f"  └─ {var_name}: levels {DYNAMIC_3D_LEVELS}")
    print(f"Targets:    {len(TARGET_VARS)} variables")
    print(f"Total input channels: {n_static_channels + n_dynamic_channels + n_3d_channels}")
    print("=" * 70)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = create_dataloaders()

        print("\nTesting data loading...")
        for inputs, targets, landmask in train_loader:
            print(f"Inputs shape:   {inputs.shape}")
            print(f"Targets shape:  {targets.shape}")
            print(f"Landmask shape: {landmask.shape}")
            print(f"Inputs range:   [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"Targets range:  [{targets.min():.3f}, {targets.max():.3f}]")
            break

        print("\n✓ Dataloader test successful!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
