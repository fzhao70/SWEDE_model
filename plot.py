import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import string
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, shape
from shapely.prepared import prep
from pathlib import Path
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta
import sys
import os

try:
    import config
    STATIC_PATH = getattr(config, 'STATIC_PATH', None)
    N_HOLO = getattr(config, 'N_HOLO', 0)
    HOLO = getattr(config, 'HOLO', 0)
    HOLO_HIGH_RES_SCALE = getattr(config, 'HOLO_HIGH_RES_SCALE', 1)
    DYNAMIC_PATH = getattr(config, 'DYNAMIC_PATH', None)
    DYNAMIC_FILE_TEMPLATE = getattr(config, 'DYNAMIC_FILE_TEMPLATE', None)
    DYNAMIC_VARS = getattr(config, 'DYNAMIC_VARS', [])
except ImportError:
    STATIC_PATH = None
    N_HOLO = 0
    HOLO = 0
    HOLO_HIGH_RES_SCALE = 1
    DYNAMIC_PATH = None
    DYNAMIC_FILE_TEMPLATE = None
    DYNAMIC_VARS = []

# Effective crop applied to high-res arrays (lat/lon, HGT), as a per-side
# (south, north, west, east) tuple. Holo cases use the HOLO * HOLO_HIGH_RES_SCALE
# crop, where HOLO may be an int (symmetric) or a per-side 4-tuple; older cases
# use N_HOLO. All default to 0, returning the high-res arrays uncropped.
_HOLO_SIDES = tuple(HOLO) if isinstance(HOLO, (tuple, list)) else (HOLO,) * 4
if any(h > 0 for h in _HOLO_SIDES):
    HIGH_RES_CROP = tuple(h * HOLO_HIGH_RES_SCALE for h in _HOLO_SIDES)
else:
    HIGH_RES_CROP = (N_HOLO,) * 4


def compute_metrics(predictions, targets, mask=None):
    """Return MAE, RMSE, MSE, R2, correlation, bias and error std.

    Args:
        predictions: Array of predictions (any shape)
        targets: Ground truth, same shape as predictions
        mask: Optional binary mask (1 = include, 0 = exclude)
    """
    pred_flat = np.asarray(predictions).flatten()
    target_flat = np.asarray(targets).flatten()

    if mask is not None:
        mask_flat = np.asarray(mask).flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    if len(pred_flat) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mse': np.nan, 'r2': np.nan,
                'correlation': np.nan, 'bias': np.nan, 'std_error': np.nan,
                'n_samples': 0}

    mae = np.mean(np.abs(pred_flat - target_flat))
    mse = np.mean((pred_flat - target_flat) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    if np.std(pred_flat) > 1e-10 and np.std(target_flat) > 1e-10:
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    else:
        correlation = np.nan

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'r2': float(r2),
        'correlation': float(correlation),
        'bias': float(np.mean(pred_flat - target_flat)),
        'std_error': float(np.std(pred_flat - target_flat)),
        'n_samples': int(len(pred_flat)),
    }


def accumulate_predictions(predictions, times, reset_month_day=None):
    """Turn per-step increments into running totals, in chronological order.

    Only used by the ``accumulate=True`` plotting paths, which apply to models
    that predict day-to-day changes rather than absolute SWE.

    Args:
        predictions: Array (n_samples, channels, height, width) of increments
        times: Array (n_samples, 2) of (year, day_of_year)
        reset_month_day: (month, day) at which the running total resets, or None
    """
    n_samples = predictions.shape[0]
    accumulated = np.zeros_like(predictions)

    time_sort_idx = np.lexsort((times[:, 1], times[:, 0]))
    reverse_idx = np.argsort(time_sort_idx)
    sorted_predictions = predictions[time_sort_idx]
    sorted_times = times[time_sort_idx]

    current_sum = np.zeros_like(sorted_predictions[0])
    days_per_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    for i in range(n_samples):
        if reset_month_day is not None and i > 0:
            curr_year, curr_doy = int(sorted_times[i, 0]), int(sorted_times[i, 1])
            prev_year, prev_doy = int(sorted_times[i - 1, 0]), int(sorted_times[i - 1, 1])
            reset_month, reset_day = reset_month_day
            reset_doy = days_per_month[reset_month - 1] + reset_day - 1

            if curr_year > prev_year:
                if curr_doy >= reset_doy or prev_doy < reset_doy:
                    current_sum = np.zeros_like(current_sum)
            elif curr_doy >= reset_doy and prev_doy < reset_doy:
                current_sum = np.zeros_like(current_sum)

        current_sum = current_sum + sorted_predictions[i]
        accumulated[time_sort_idx[i]] = current_sum.copy()

    return accumulated[reverse_idx]


def _crop_highres(arr):
    """Crop the per-side (south, north, west, east) halo from a 2D high-res array."""
    s, n, w, e = HIGH_RES_CROP
    if s == n == w == e == 0:
        return arr
    return arr[s:arr.shape[0] - n, w:arr.shape[1] - e]

FIGURE_DPI = 200

# ============================================================================
# IPCC-STYLE DISCRETE COLORMAPS
# ============================================================================
# The IPCC visual style guide asks for colour scales split into a small number
# of discrete classes rather than a continuously blended ramp, so that a colour
# read off a map maps back to one bounded interval. Every colour-mapped field
# in this module therefore goes through `discretize_cmap()`, which converts a
# named matplotlib colormap into a ListedColormap of N classes plus a
# BoundaryNorm whose class edges are "nice" round numbers.

# Default number of colour classes. Diverging fields get an even count so that
# zero always falls on a class boundary (symmetric colours either side of zero).
IPCC_N_LEVELS = 10
IPCC_N_LEVELS_DIVERGING = 10

# Colormaps treated as diverging (centred scales, e.g. percentage differences).
_DIVERGING_CMAPS = {
    'RdBu', 'RdBu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
    'BrBG', 'BrBG_r', 'PuOr', 'PuOr_r', 'PiYG', 'PiYG_r', 'PRGn', 'PRGn_r',
    'RdGy', 'RdGy_r', 'coolwarm', 'coolwarm_r', 'bwr', 'bwr_r',
    'seismic', 'seismic_r', 'Spectral', 'Spectral_r',
}


def _resolve_cmap(cmap):
    """Return a Colormap object from a name or a Colormap."""
    if isinstance(cmap, str):
        return matplotlib.colormaps[cmap]
    return cmap


def _is_diverging_cmap(cmap):
    name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', '')
    return name in _DIVERGING_CMAPS


def discretize_cmap(cmap, vmin=None, vmax=None, n_levels=None, extend='neither'):
    """
    Build an IPCC-style discretized colormap and matching BoundaryNorm.

    Class edges are chosen with MaxNLocator so they land on round numbers and
    always span [vmin, vmax]. When the colorbar is extended, one extra colour
    is sampled off each extended end and used as the under/over colour, so the
    in-range classes keep the full colour span and out-of-range values remain
    visually distinct.

    Args:
        cmap:     colormap name or Colormap instance
        vmin:     low end of the data range
        vmax:     high end of the data range
        n_levels: requested number of colour classes (default: IPCC_N_LEVELS,
                  or IPCC_N_LEVELS_DIVERGING for diverging colormaps)
        extend:   'neither' | 'min' | 'max' | 'both' — must match the value
                  passed to plt.colorbar()

    Returns:
        (cmap, norm, levels). If vmin/vmax are unusable (None, non-finite, or
        vmax <= vmin) the original colormap is returned with norm=None and
        levels=None so callers can fall back to plain vmin/vmax scaling.
    """
    base = _resolve_cmap(cmap)

    if vmin is None or vmax is None:
        return base, None, None
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return base, None, None

    if n_levels is None:
        n_levels = IPCC_N_LEVELS_DIVERGING if _is_diverging_cmap(cmap) else IPCC_N_LEVELS

    # tick_values() always widens to round numbers covering [vmin, vmax], so
    # the returned edges never leave part of the data range unclassified.
    levels = MaxNLocator(nbins=n_levels, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    levels = np.asarray(levels, dtype=float)
    if levels.size < 2:
        return base, None, None

    n_classes = levels.size - 1
    n_under = 1 if extend in ('min', 'both') else 0
    n_over = 1 if extend in ('max', 'both') else 0

    sampled = base(np.linspace(0.0, 1.0, n_classes + n_under + n_over))
    main = sampled[n_under: sampled.shape[0] - n_over] if n_over else sampled[n_under:]

    disc = mcolors.ListedColormap(main)
    if n_under:
        disc.set_under(sampled[0])
    if n_over:
        disc.set_over(sampled[-1])

    norm = mcolors.BoundaryNorm(levels, ncolors=disc.N, clip=False)
    return disc, norm, levels


def colorbar_ticks(levels, max_ticks=11):
    """Thin class edges down to at most `max_ticks` labelled colorbar ticks."""
    levels = np.asarray(levels)
    if levels.size <= max_ticks:
        return levels
    step = int(np.ceil(levels.size / max_ticks))
    return levels[::step]


def year_cmap_norm(unique_years, cmap='plasma'):
    """
    Discrete colour class per water year for year-coloured scatter plots.

    Years are a categorical axis, so each year gets exactly one colour instead
    of a position on a continuous ramp.

    Returns:
        (cmap, norm, years)
    """
    yrs = np.unique(np.asarray(unique_years, dtype=float))
    base = _resolve_cmap(cmap)
    disc = mcolors.ListedColormap(base(np.linspace(0.0, 1.0, max(yrs.size, 1))))
    if yrs.size < 2:
        edges = np.array([yrs[0] - 0.5, yrs[0] + 0.5]) if yrs.size else np.array([0.0, 1.0])
    else:
        edges = np.concatenate([[yrs[0] - 0.5], (yrs[:-1] + yrs[1:]) / 2.0, [yrs[-1] + 0.5]])
    return disc, mcolors.BoundaryNorm(edges, ncolors=disc.N, clip=True), yrs


def panel_label(index, name=None):
    """
    Journal-style panel label for multi-panel figures: 0 -> '(a)', 1 -> '(b)', …

    With `name`, returns the labelled panel title, e.g. panel_label(0, 'PNW')
    -> '(a) PNW'. Indices past 'z' continue as '(aa)', '(ab)', …
    """
    letters = string.ascii_lowercase
    if index < len(letters):
        tag = letters[index]
    else:
        tag = letters[index // len(letters) - 1] + letters[index % len(letters)]
    label = f'({tag})'
    return f'{label} {name}' if name else label


def get_lat_lon():
    """Load lat/lon from static file."""
    if STATIC_PATH and os.path.exists(STATIC_PATH):
        try:
            with nc.Dataset(str(STATIC_PATH), "r") as fin:
                lon = fin['XLONG'][0, :, :]
                lat = fin['XLAT'][0, :, :]
                lon = _crop_highres(lon)
                lat = _crop_highres(lat)
                return lat, lon
        except Exception as e:
            print(f"Error loading static file: {e}")
    return None, None

def get_years(times):
    if times is not None and times.ndim > 1:
        return times[:, 0].astype(int)
    return None


# A pixel whose TARGET mean-of-yearly-maximum SWE is below this (mm) carries so
# little snow that its percentage difference is meaningless and it only stretches
# the colour scale. Every spatial map blanks those pixels, on all three panels,
# so all maps share one footprint. The threshold is always evaluated against the
# mean of yearly maxima -- never against whatever quantity the map itself shows,
# whose scale (a temporal std, a sum over all timesteps) is not comparable to mm
# of peak SWE.
SNOW_PRESENCE_THRESHOLD_MM = 20.0


def _target_mean_yearly_max(targets, times, ch):
    """Target mean-of-yearly-maxima SWE for one channel, or None if the years
    cannot be derived from `times`."""
    years = get_years(times)
    if years is None:
        return None
    maxs = []
    for y in np.unique(years):
        m = years == y
        if np.any(m):
            maxs.append(np.max(targets[m, ch], axis=0))
    if not maxs:
        return None
    return np.mean(np.array(maxs), axis=0)


def _snow_presence_mask(targets, times, ch, threshold=None):
    """Boolean map of pixels with enough snow to plot. None when `times` is
    unavailable, in which case callers skip the filter and behave as before."""
    ref = _target_mean_yearly_max(targets, times, ch)
    if ref is None:
        return None
    return ref >= (SNOW_PRESENCE_THRESHOLD_MM if threshold is None else threshold)


def _apply_snow_mask(mask, *arrays):
    """NaN out the barely-snow pixels of each array; a no-op when mask is None.
    Applied BEFORE the colour scale is derived so the scale reflects only the
    pixels that remain visible."""
    if mask is None:
        return arrays if len(arrays) > 1 else arrays[0]
    out = tuple(np.where(mask, a, np.nan) for a in arrays)
    return out if len(out) > 1 else out[0]

_DYNAMIC_TIME_CACHE = {}
_SINGLE_FILE_TIMES = None

def _infer_use_yearly_files():
    return DYNAMIC_FILE_TEMPLATE is not None and "{year}" in DYNAMIC_FILE_TEMPLATE

def _open_time_variable(path):
    """Return datetime array from a NetCDF file if available."""
    if not path.exists():
        return None
    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            for cand in ['time', 'Time', 'XTIME', 'day']:
                if cand in ds.variables:
                    time_var = ds[cand]
                    # xarray automatically converts to numpy datetime64 or cftime
                    return time_var.values
    except Exception as e:
        print(f"Warning: could not read time coordinate from {path}: {e}")
    return None

def _load_time_coordinates():
    """Load datetime arrays from dynamic data for mapping indices to real dates."""
    global _SINGLE_FILE_TIMES
    if DYNAMIC_PATH is None or DYNAMIC_FILE_TEMPLATE is None:
        return

    data_path = Path(DYNAMIC_PATH)
    use_yearly = _infer_use_yearly_files()
    var_name = DYNAMIC_VARS[0] if DYNAMIC_VARS else None
    if var_name is None:
        return

    if use_yearly:
        # Defer per-year loading; cache as needed
        return

    pattern = DYNAMIC_FILE_TEMPLATE.replace("{var_name}", var_name)
    sample_files = sorted(data_path.glob(pattern))
    if sample_files:
        times = _open_time_variable(sample_files[0])
        if times is not None:
            _SINGLE_FILE_TIMES = times

def _get_time_for_year_index(year, idx):
    """Map (year, idx) to datetime using cached coordinates."""
    if DYNAMIC_PATH is None or DYNAMIC_FILE_TEMPLATE is None:
        return None

    data_path = Path(DYNAMIC_PATH)
    use_yearly = _infer_use_yearly_files()
    var_name = DYNAMIC_VARS[0] if DYNAMIC_VARS else None
    if var_name is None:
        return None

    if use_yearly:
        if year not in _DYNAMIC_TIME_CACHE:
            fname = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=year)
            times = _open_time_variable(data_path / fname)
            if times is not None:
                _DYNAMIC_TIME_CACHE[year] = times
        if year in _DYNAMIC_TIME_CACHE:
            times = _DYNAMIC_TIME_CACHE[year]
            if 0 <= idx < len(times):
                return times[idx]
        return None

    if _SINGLE_FILE_TIMES is None:
        _load_time_coordinates()
    if _SINGLE_FILE_TIMES is not None and 0 <= idx < len(_SINGLE_FILE_TIMES):
        return _SINGLE_FILE_TIMES[idx]
    return None

def get_datetimes_from_data(times):
    """Return list of datetimes for each sample using data time coordinate if available."""
    if times is None or times.ndim < 2:
        return None

    _load_time_coordinates()
    datetimes = []
    use_yearly = _infer_use_yearly_files()
    for row in times:
        year = int(row[0])
        time_idx = int(row[1])
        dt = _get_time_for_year_index(year, time_idx)
        if dt is None:
            # Fallback: assume Jan 1 start for given year
            dt = datetime(year, 10, 1) + timedelta(days=time_idx)
        datetimes.append(dt)
    return datetimes

def convert_to_plot_dates(dates):
    """Convert various datetime types to matplotlib-compatible datetime objects.

    Handles cftime objects, numpy.datetime64, and standard datetime objects.
    """
    if not dates:
        return []

    converted = []
    for d in dates:
        if hasattr(d, 'year') and hasattr(d, 'month') and hasattr(d, 'day'):
            try:
                converted_date = datetime(d.year, d.month, d.day,
                                         getattr(d, 'hour', 0),
                                         getattr(d, 'minute', 0),
                                         getattr(d, 'second', 0))
                converted.append(converted_date)
            except (ValueError, AttributeError):
                converted.append(pd.Timestamp(d).to_pydatetime())
        else:
            converted.append(pd.Timestamp(d).to_pydatetime())

    return converted

def calculate_snow_accumulation(changes):
    """
    Calculate snow accumulation from changes with non-negativity constraint.

    This function takes predicted/target changes and calculates the cumulative
    snow accumulation pixel-by-pixel, ensuring that accumulated snow cannot be negative.

    Args:
        changes: Array of shape (n_times, height, width) representing snow changes

    Returns:
        accumulation: Array of shape (n_times, height, width) with accumulated snow
    """
    accumulation = np.zeros_like(changes)

    for i in range(changes.shape[0]):
        if i == 0:
            accumulation[i] = np.maximum(0, changes[i])
        else:
            accumulation[i] = np.maximum(0, accumulation[i-1] + changes[i])

    return accumulation


def calculate_snow_accumulation_yearly_reset(changes, times, reset_month=9, reset_day=1):
    """
    Calculate snow accumulation from changes with yearly reset on a specific date.

    This function resets the accumulated snow to 0 on the specified date each year
    (default: September 1st), which is useful for analyzing seasonal snow patterns.

    Args:
        changes: Array of shape (n_times, height, width) representing snow changes
        times: Time information array of shape (n_times, 2+) with [year, day_of_year, ...]
        reset_month: Month to reset accumulation (1-12), default=9 (September)
        reset_day: Day of month to reset, default=1

    Returns:
        accumulation: Array of shape (n_times, height, width) with accumulated snow
    """
    accumulation = np.zeros_like(changes)

    for i in range(changes.shape[0]):
        if i == 0:
            accumulation[i] = np.maximum(0, changes[i])
        else:
            year = int(times[i, 0])
            day_of_year = int(times[i, 1])

            reset_date = datetime(year, reset_month, reset_day)
            jan1 = datetime(year, 1, 1)
            reset_day_of_year = (reset_date - jan1).days

            if i > 0:
                prev_year = int(times[i-1, 0])
                prev_day = int(times[i-1, 1])

                if (year > prev_year and day_of_year >= reset_day_of_year) or \
                   (year == prev_year and prev_day < reset_day_of_year and day_of_year >= reset_day_of_year):
                    accumulation[i] = np.maximum(0, changes[i])
                else:
                    accumulation[i] = np.maximum(0, accumulation[i-1] + changes[i])
            else:
                accumulation[i] = np.maximum(0, accumulation[i-1] + changes[i])

    return accumulation


def _month_from_intra_index(idx: int, is_daily: bool) -> int:
    """Convert intra-year index to calendar month assuming Jan 1 start."""
    if is_daily:
        # Daily indices are day-of-year offsets starting Jan 1
        ref_date = datetime(2000, 1, 1) + timedelta(days=int(idx))
        return ref_date.month
    # Monthly indices are month offsets starting Jan (0 -> Jan)
    return (int(idx) % 12) + 1

def _water_year_month_order(months):
    """Return months ordered to start in September for plotting."""
    water_year = list(range(9, 13)) + list(range(1, 9))
    ordered = [m for m in water_year if m in months]
    return ordered if ordered else sorted(months)

def _water_year_day_order(keys):
    """Return (month, day) keys ordered to start on 1 September."""
    ordered = sorted([k for k in keys if k[0] >= 9]) + sorted([k for k in keys if k[0] < 9])
    return ordered if ordered else sorted(keys)


def _build_dayofyear_index(times):
    """Return mapping of (month, day) -> sample indices, ordered to start in September.

    Day-of-year analogue of `_build_month_index`. Keys are (month, day) pairs
    rather than a day-of-year number so that leap and non-leap years line up on
    the same calendar date.
    """
    if times is None or times.ndim < 2:
        return {}, []

    dates = get_datetimes_from_data(times)
    if not dates:
        return {}, []

    first_date = dates[0]
    if hasattr(first_date, 'month'):
        md_arr = np.array([(d.month, d.day) for d in dates])
    else:
        md_arr = np.array([(pd.Timestamp(d).month, pd.Timestamp(d).day) for d in dates])

    # Drop the leap day (and the 30th of February a 360-day calendar carries):
    # it is sampled in only a fraction of the years, so its climatology is built
    # from far fewer samples than its neighbours and shows up as a break.
    keys = [(int(m), int(d)) for m, d in np.unique(md_arr, axis=0)
            if not (int(m) == 2 and int(d) >= 29)]
    ordered = _water_year_day_order(keys)
    day_map = {
        k: np.where((md_arr[:, 0] == k[0]) & (md_arr[:, 1] == k[1]))[0]
        for k in ordered
    }
    return day_map, ordered


def _build_month_index(times):
    """Return mapping of month -> sample indices, ordered to start in September."""
    if times is None or times.ndim < 2:
        return {}, []

    dates = get_datetimes_from_data(times)
    if dates is not None:
        # Handle different datetime types:
        # - cftime objects: have .month but cannot convert to pandas
        # - standard datetime: have .month
        # - numpy.datetime64: no .month, need pandas conversion
        if dates:
            first_date = dates[0]
            if hasattr(first_date, 'month'):
                month_arr = np.array([d.month for d in dates])
            else:
                month_arr = np.array([pd.Timestamp(d).month for d in dates])
        else:
            month_arr = np.array([])
        months = np.unique(month_arr)
        ordered_months = _water_year_month_order(months)
        month_map = {m: np.where(month_arr == m)[0] for m in ordered_months}
        return month_map, ordered_months

    has_season_idx = times.shape[1] >= 3
    intra_year_idx = times[:, 2] if has_season_idx else times[:, 1]
    is_daily = np.max(intra_year_idx) > 20
    unique_indices = np.unique(intra_year_idx)

    month_map = {}
    for idx in unique_indices:
        target_month = _month_from_intra_index(idx, is_daily)
        indices = np.where(intra_year_idx == idx)[0]
        month_map.setdefault(target_month, []).extend(indices.tolist())

    ordered_months = _water_year_month_order(month_map.keys())
    month_map = {m: np.array(month_map[m]) for m in ordered_months}
    return month_map, ordered_months

def select_interesting_samples(predictions, targets, n_samples=10):
    """Select interesting samples based on spatial variation."""
    if len(targets) <= n_samples:
        return np.arange(len(targets))

    target_std = np.std(targets[:, 0, :, :], axis=(1, 2))
    target_mean = np.mean(targets[:, 0, :, :], axis=(1, 2))

    scores = target_std * (target_mean > 0.01)

    indices = np.argsort(scores)[::-1][:n_samples]
    return indices

def visualize_predictions(predictions, targets, output_dir, n_samples=10, target_names=None, select_interesting=True):
    """Visualize predictions vs targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if select_interesting:
        indices = select_interesting_samples(predictions, targets, n_samples)
    else:
        indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for idx in indices:
        for ch in range(n_channels):
            pred = predictions[idx, ch]
            targ = targets[idx, ch]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            vmin = min(pred.min(), targ.min())
            vmax = max(pred.max(), targ.max())

            val_cmap, val_norm, val_levels = discretize_cmap('viridis', vmin, vmax)
            val_kw = {'norm': val_norm} if val_norm is not None else {'vmin': vmin, 'vmax': vmax}

            im0 = axes[0].imshow(targ, cmap=val_cmap, **val_kw)
            axes[0].set_title(f'Target {target_names[ch]}')
            cb0 = plt.colorbar(im0, ax=axes[0], spacing='uniform')

            im1 = axes[1].imshow(pred, cmap=val_cmap, **val_kw)
            axes[1].set_title(f'Prediction {target_names[ch]}')
            cb1 = plt.colorbar(im1, ax=axes[1], spacing='uniform')

            if val_levels is not None:
                cb0.set_ticks(colorbar_ticks(val_levels))
                cb1.set_ticks(colorbar_ticks(val_levels))

            diff = pred - targ
            abs_max_diff = max(abs(diff.min()), abs(diff.max()))
            if abs_max_diff == 0: abs_max_diff = 1e-5 # Avoid singular value

            diff_cmap, diff_norm, diff_levels = discretize_cmap(
                'coolwarm', -abs_max_diff, abs_max_diff)
            diff_kw = ({'norm': diff_norm} if diff_norm is not None
                       else {'vmin': -abs_max_diff, 'vmax': abs_max_diff})

            im2 = axes[2].imshow(diff, cmap=diff_cmap, **diff_kw)
            axes[2].set_title('Difference (Pred - Target)')
            cb2 = plt.colorbar(im2, ax=axes[2], spacing='uniform')
            if diff_levels is not None:
                cb2.set_ticks(colorbar_ticks(diff_levels))

            plt.suptitle(f'Sample {idx}')
            plt.tight_layout()
            plt.savefig(output_dir / f'sample_{idx}_ch{ch}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_scatter(x, y, ax, title, xlabel, ylabel):
    x_flat = x.flatten()
    y_flat = y.flatten()

    ax.scatter(x_flat, y_flat, alpha=0.9, s=3, c='#2E86AB',
              edgecolors='none', rasterized=True)

    min_val = min(np.min(x_flat), np.min(y_flat))
    max_val = max(np.max(x_flat), np.max(y_flat))
    lims = [min_val, max_val]

    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=2, zorder=5, label='1:1 line')

    ax.set_aspect('equal')
    ax.set_title(title, fontsize='xx-large', fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize='large', fontweight='semibold')
    ax.set_ylabel(ylabel, fontsize='large', fontweight='semibold')

    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    ax.set_axisbelow(True)

    corr = np.corrcoef(x_flat, y_flat)[0, 1] if len(x_flat) > 1 else 0
    mae = np.mean(np.abs(x_flat - y_flat))
    rmse = np.sqrt(np.mean((x_flat - y_flat)**2))

    metrics_text = f'R = {corr:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    ax.legend(loc='lower right', frameon=True, fancybox=True,
             shadow=True, fontsize='large', framealpha=0.9)

def plot_total_mean_max_scatter(predictions, targets, output_dir, target_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        pred_mean = np.mean(predictions[:, ch, :, :], axis=(1, 2))
        targ_mean = np.mean(targets[:, ch, :, :], axis=(1, 2))
        plot_scatter(targ_mean, pred_mean, axes[0], f'Spatial Mean Snow', 'Target [mm]', 'Prediction [mm]')

        pred_max = np.max(predictions[:, ch, :, :], axis=(1, 2))
        targ_max = np.max(targets[:, ch, :, :], axis=(1, 2))
        plot_scatter(targ_max, pred_max, axes[1], f'Spatial Max Snow', 'Target [mm]', 'Prediction [mm]')

        pred_sum = np.sum(predictions[:, ch, :, :], axis=(1, 2))
        targ_sum = np.sum(targets[:, ch, :, :], axis=(1, 2))
        plot_scatter(targ_sum, pred_sum, axes[2], f'Spatial Sum Snow', 'Target [mm]', 'Prediction [mm]')

        plt.tight_layout()
        plt.savefig(output_dir / f'scatter_temporal_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_total_mean_max_scatter_spatial(predictions, targets, output_dir, target_names=None, times=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    years = get_years(times)
    has_years = years is not None and len(years) == predictions.shape[0]

    for ch in range(n_channels):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout = 'compressed')

        pred_mean = np.mean(predictions[:, ch, :, :], axis=0).flatten()
        targ_mean = np.mean(targets[:, ch, :, :], axis=0).flatten()
        plot_scatter(targ_mean, pred_mean, axes[0], f'Temporal Mean Snow', 'Target [mm]', 'Prediction [mm]')

        # Mean yearly max per pixel when times are available; otherwise fall back
        # to a temporal max so the right panel remains well-defined.
        if has_years:
            unique_years = np.unique(years)
            pred_maxs, targ_maxs = [], []
            for y in unique_years:
                mask = years == y
                if not np.any(mask): continue
                pred_maxs.append(np.max(predictions[mask, ch], axis=0))
                targ_maxs.append(np.max(targets[mask, ch], axis=0))
            if pred_maxs and targ_maxs:
                pred_max = np.mean(pred_maxs, axis=0).flatten()
                targ_max = np.mean(targ_maxs, axis=0).flatten()
                max_title = 'Mean Yearly Max Snow'
            else:
                pred_max = np.max(predictions[:, ch, :, :], axis=0).flatten()
                targ_max = np.max(targets[:, ch, :, :], axis=0).flatten()
                max_title = 'Temporal Max Snow'
        else:
            pred_max = np.max(predictions[:, ch, :, :], axis=0).flatten()
            targ_max = np.max(targets[:, ch, :, :], axis=0).flatten()
            max_title = 'Temporal Max Snow'
        plot_scatter(targ_max, pred_max, axes[1], max_title, 'Target [mm]', 'Prediction [mm]')

        plt.savefig(output_dir / f'scatter_spatial_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_map(data, lat, lon, ax, title, vmin=None, vmax=None, cmap='PuBu', draw_gridlines=False, extend='neither', filter_array=None, filter_threshold=SNOW_PRESENCE_THRESHOLD_MM, n_levels=None):
    if vmin is None or vmax is None:
        finite = np.asarray(data)[np.isfinite(data)]
        if finite.size:
            vmin = float(np.min(finite)) if vmin is None else vmin
            vmax = float(np.max(finite)) if vmax is None else vmax
    disc_cmap, norm, levels = discretize_cmap(cmap, vmin, vmax, n_levels=n_levels, extend=extend)
    # A BoundaryNorm already carries the limits; matplotlib rejects norm and
    # vmin/vmax together.
    scale_kw = {'norm': norm} if norm is not None else {'vmin': vmin, 'vmax': vmax}

    if lat is None or lon is None:
        im = ax.imshow(data, cmap=disc_cmap, **scale_kw)
    else:
        if filter_array is not None:
            data_copy = np.where(filter_array < filter_threshold, np.nan, data)
        if hasattr(ax, 'projection'):
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
            if filter_array is not None:
                im = ax.pcolormesh(lon, lat, data_copy, cmap=disc_cmap, shading='nearest', transform=ccrs.PlateCarree(), **scale_kw)
            else:
                im = ax.pcolormesh(lon, lat, data, cmap=disc_cmap, shading='nearest', transform=ccrs.PlateCarree(), **scale_kw)
            if draw_gridlines:
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
        else:
            im = ax.pcolormesh(lon, lat, data, cmap=disc_cmap, shading='nearest', **scale_kw)

    ax.set_title(title, fontsize='x-large', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend,
                        spacing='uniform')
    if levels is not None:
        cbar.set_ticks(colorbar_ticks(levels))

def plot_mean_max_spatial_maps(predictions, targets, output_dir, target_names=None, mask_zero_snow=False, vmin=None, vmax=None, times=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    lat, lon = get_lat_lon()

    for ch in range(n_channels):
        snow = _snow_presence_mask(targets, times, ch)
        for metric_name, func in [('Mean', np.mean), ('Max', np.max), ('Sum', np.sum)]:
            pred_agg = func(predictions[:, ch, :, :], axis=0)
            targ_agg = func(targets[:, ch, :, :], axis=0)

            if mask_zero_snow:
                mask = targ_agg > 0
                pred_agg = np.where(mask, pred_agg, np.nan)
                targ_agg = np.where(mask, targ_agg, np.nan)
            pred_agg, targ_agg = _apply_snow_mask(snow, pred_agg, targ_agg)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
            plot_vmin = vmin if vmin is not None else min(np.nanmin(pred_agg), np.nanmin(targ_agg))
            plot_vmax = vmax if vmax is not None else max(np.nanmax(pred_agg), np.nanmax(targ_agg))

            plot_map(targ_agg, lat, lon, axes[0], f'Target {metric_name}', plot_vmin, plot_vmax, draw_gridlines=True, extend='max')
            plot_map(pred_agg, lat, lon, axes[1], f'Pred {metric_name}', plot_vmin, plot_vmax, draw_gridlines=True, extend='max')

            with np.errstate(divide='ignore', invalid='ignore'):
                diff = (pred_agg - targ_agg) / targ_agg * 100

            plot_map(diff, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

            plt.suptitle(f'{target_names[ch]} {metric_name} Maps {"(Masked)" if mask_zero_snow else ""}', fontsize='xx-large', fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'map_{metric_name.lower()}_{target_names[ch]}{"_masked" if mask_zero_snow else ""}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_mean_max_spatial_maps_yearly(predictions, targets, times, output_dir, target_names=None, mask_zero_snow=False, vmin=None, vmax=None, accumulate=False, reset_month_day=None, filename_suffix=''):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
        filename_suffix: Suffix to append to output filenames
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    output_dir = Path(output_dir)
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    n_channels = predictions.shape[1]
    lat, lon = get_lat_lon()
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        pred_yearly_maxs = []
        targ_yearly_maxs = []
        for y in unique_years:
            mask = years == y
            if not np.any(mask): continue
            pred_yearly_maxs.append(np.max(predictions[mask, ch], axis=0))
            targ_yearly_maxs.append(np.max(targets[mask, ch], axis=0))

        pred_mean_max = np.mean(pred_yearly_maxs, axis=0)
        targ_mean_max = np.mean(targ_yearly_maxs, axis=0)
        # This map's own target field IS the snow-presence reference.
        snow = targ_mean_max >= SNOW_PRESENCE_THRESHOLD_MM

        if mask_zero_snow:
            mask = targ_mean_max > 0
            pred_mean_max = np.where(mask, pred_mean_max, np.nan)
            targ_mean_max = np.where(mask, targ_mean_max, np.nan)
        pred_mean_max, targ_mean_max = _apply_snow_mask(snow, pred_mean_max, targ_mean_max)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_vmin = vmin if vmin is not None else min(np.nanmin(pred_mean_max), np.nanmin(targ_mean_max))
        plot_vmax = vmax if vmax is not None else max(np.nanmax(pred_mean_max), np.nanmax(targ_mean_max))

        plot_map(targ_mean_max, lat, lon, axes[0], panel_label(0, 'WRF Target'), plot_vmin, plot_vmax, draw_gridlines=True, extend='max')
        plot_map(pred_mean_max, lat, lon, axes[1], panel_label(1, 'SWEDE Prediction'), plot_vmin, plot_vmax, draw_gridlines=True, extend='max')

        with np.errstate(divide='ignore', invalid='ignore'):
            diff = (pred_mean_max - targ_mean_max) / targ_mean_max * 100

        plot_map(diff, lat, lon, axes[2], panel_label(2, 'Diff (%)'), vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

        plt.suptitle(f'Mean of Yearly Maxima SWE', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'map_mean_yearly_max_{target_names[ch]}{"_masked" if mask_zero_snow else ""}{filename_suffix}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_temporal_variability_spatial_maps(predictions, targets, output_dir, target_names=None, mask_zero_snow=False, vmin_std=None, vmax_std=None, vmin_range=None, vmax_range=None, times=None):
    """
    Create spatial maps of temporal variability metrics (std and range).

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        output_dir: Directory to save plots
        target_names: List of target variable names
        mask_zero_snow: If True, mask out pixels where target mean is zero
        vmin_std: Minimum value for std colorbar (auto scale if None)
        vmax_std: Maximum value for std colorbar (auto scale if None)
        vmin_range: Minimum value for range colorbar (auto scale if None)
        vmax_range: Maximum value for range colorbar (auto scale if None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    lat, lon = get_lat_lon()

    for ch in range(n_channels):
        pred_std = np.std(predictions[:, ch, :, :], axis=0)
        targ_std = np.std(targets[:, ch, :, :], axis=0)

        pred_range = np.max(predictions[:, ch, :, :], axis=0) - np.min(predictions[:, ch, :, :], axis=0)
        targ_range = np.max(targets[:, ch, :, :], axis=0) - np.min(targets[:, ch, :, :], axis=0)

        if mask_zero_snow:
            targ_mean = np.mean(targets[:, ch, :, :], axis=0)
            mask = targ_mean > 0
            pred_std = np.where(mask, pred_std, np.nan)
            targ_std = np.where(mask, targ_std, np.nan)
            pred_range = np.where(mask, pred_range, np.nan)
            targ_range = np.where(mask, targ_range, np.nan)

        pred_std, targ_std, pred_range, targ_range = _apply_snow_mask(
            _snow_presence_mask(targets, times, ch),
            pred_std, targ_std, pred_range, targ_range)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_vmin_std = vmin_std if vmin_std is not None else min(np.nanmin(pred_std), np.nanmin(targ_std))
        plot_vmax_std = vmax_std if vmax_std is not None else max(np.nanmax(pred_std), np.nanmax(targ_std))

        plot_map(targ_std, lat, lon, axes[0], 'Target Temporal Std', plot_vmin_std, plot_vmax_std, draw_gridlines=True, extend='max')
        plot_map(pred_std, lat, lon, axes[1], 'Pred Temporal Std', plot_vmin_std, plot_vmax_std, draw_gridlines=True, extend='max')

        with np.errstate(divide='ignore', invalid='ignore'):
            diff_std = (pred_std - targ_std) / targ_std * 100

        plot_map(diff_std, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

        plt.suptitle(f'{target_names[ch]} Temporal Standard Deviation {"(Masked)" if mask_zero_snow else ""}', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'map_temporal_std_{target_names[ch]}{"_masked" if mask_zero_snow else ""}.png', dpi=FIGURE_DPI)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_vmin_range = vmin_range if vmin_range is not None else min(np.nanmin(pred_range), np.nanmin(targ_range))
        plot_vmax_range = vmax_range if vmax_range is not None else max(np.nanmax(pred_range), np.nanmax(targ_range))

        plot_map(targ_range, lat, lon, axes[0], 'Target Temporal Range (Max-Min)', plot_vmin_range, plot_vmax_range, draw_gridlines=True, extend='max')
        plot_map(pred_range, lat, lon, axes[1], 'Pred Temporal Range (Max-Min)', plot_vmin_range, plot_vmax_range, draw_gridlines=True, extend='max')

        with np.errstate(divide='ignore', invalid='ignore'):
            diff_range = (pred_range - targ_range) / targ_range * 100

        plot_map(diff_range, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

        plt.suptitle(f'{target_names[ch]} Temporal Range (Max-Min) {"(Masked)" if mask_zero_snow else ""}', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'map_temporal_range_{target_names[ch]}{"_masked" if mask_zero_snow else ""}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_yearly_max_variability_spatial_maps(predictions, targets, times, output_dir, target_names=None, mask_zero_snow=False, vmin_std=None, vmax_std=None, vmin_range=None, vmax_range=None):
    """
    Create spatial maps of yearly maximum variability metrics (std and range).

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Time array to extract years
        output_dir: Directory to save plots
        target_names: List of target variable names
        mask_zero_snow: If True, mask out pixels where mean of yearly maxes is zero
        vmin_std: Minimum value for std colorbar (auto scale if None)
        vmax_std: Maximum value for std colorbar (auto scale if None)
        vmin_range: Minimum value for range colorbar (auto scale if None)
        vmax_range: Maximum value for range colorbar (auto scale if None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    years = get_years(times)
    if years is None:
        print("Warning: Could not extract years from times. Skipping yearly max variability plots.")
        return

    unique_years = np.unique(years)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    lat, lon = get_lat_lon()

    for ch in range(n_channels):
        pred_yearly_maxs = []
        targ_yearly_maxs = []
        for y in unique_years:
            mask = years == y
            if not np.any(mask): continue
            pred_yearly_maxs.append(np.max(predictions[mask, ch], axis=0))
            targ_yearly_maxs.append(np.max(targets[mask, ch], axis=0))

        pred_yearly_maxs = np.array(pred_yearly_maxs)  # Shape: (n_years, height, width)
        targ_yearly_maxs = np.array(targ_yearly_maxs)  # Shape: (n_years, height, width)

        pred_std_yearly_max = np.std(pred_yearly_maxs, axis=0)
        targ_std_yearly_max = np.std(targ_yearly_maxs, axis=0)

        pred_range_yearly_max = np.max(pred_yearly_maxs, axis=0) - np.min(pred_yearly_maxs, axis=0)
        targ_range_yearly_max = np.max(targ_yearly_maxs, axis=0) - np.min(targ_yearly_maxs, axis=0)

        snow = np.mean(targ_yearly_maxs, axis=0) >= SNOW_PRESENCE_THRESHOLD_MM

        if mask_zero_snow:
            targ_mean_yearly_max = np.mean(targ_yearly_maxs, axis=0)
            mask = targ_mean_yearly_max > 0
            pred_std_yearly_max = np.where(mask, pred_std_yearly_max, np.nan)
            targ_std_yearly_max = np.where(mask, targ_std_yearly_max, np.nan)
            pred_range_yearly_max = np.where(mask, pred_range_yearly_max, np.nan)
            targ_range_yearly_max = np.where(mask, targ_range_yearly_max, np.nan)

        (pred_std_yearly_max, targ_std_yearly_max,
         pred_range_yearly_max, targ_range_yearly_max) = _apply_snow_mask(
            snow, pred_std_yearly_max, targ_std_yearly_max,
            pred_range_yearly_max, targ_range_yearly_max)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_vmin_std = vmin_std if vmin_std is not None else min(np.nanmin(pred_std_yearly_max), np.nanmin(targ_std_yearly_max))
        plot_vmax_std = vmax_std if vmax_std is not None else max(np.nanmax(pred_std_yearly_max), np.nanmax(targ_std_yearly_max))

        plot_map(targ_std_yearly_max, lat, lon, axes[0], 'Target Std of Yearly Max', plot_vmin_std, plot_vmax_std, draw_gridlines=True, extend='max')
        plot_map(pred_std_yearly_max, lat, lon, axes[1], 'Pred Std of Yearly Max', plot_vmin_std, plot_vmax_std, draw_gridlines=True, extend='max')

        with np.errstate(divide='ignore', invalid='ignore'):
            diff_std = (pred_std_yearly_max - targ_std_yearly_max) / targ_std_yearly_max * 100

        plot_map(diff_std, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

        plt.suptitle(f'{target_names[ch]} Std of Yearly Max {"(Masked)" if mask_zero_snow else ""}', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'map_yearly_max_std_{target_names[ch]}{"_masked" if mask_zero_snow else ""}.png', dpi=FIGURE_DPI)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_vmin_range = vmin_range if vmin_range is not None else min(np.nanmin(pred_range_yearly_max), np.nanmin(targ_range_yearly_max))
        plot_vmax_range = vmax_range if vmax_range is not None else max(np.nanmax(pred_range_yearly_max), np.nanmax(targ_range_yearly_max))

        plot_map(targ_range_yearly_max, lat, lon, axes[0], 'Target Range of Yearly Max', plot_vmin_range, plot_vmax_range, draw_gridlines=True, extend='max')
        plot_map(pred_range_yearly_max, lat, lon, axes[1], 'Pred Range of Yearly Max', plot_vmin_range, plot_vmax_range, draw_gridlines=True, extend='max')

        with np.errstate(divide='ignore', invalid='ignore'):
            diff_range = (pred_range_yearly_max - targ_range_yearly_max) / targ_range_yearly_max * 100

        plot_map(diff_range, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

        plt.suptitle(f'{target_names[ch]} Range of Yearly Max {"(Masked)" if mask_zero_snow else ""}', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'map_yearly_max_range_{target_names[ch]}{"_masked" if mask_zero_snow else ""}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_yearly_max_comparison(predictions, targets, times, output_dir, target_names=None, vmin=None, vmax=None):
    output_dir = Path(output_dir) / "yearly_max_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    lat, lon = get_lat_lon()
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        snow = _snow_presence_mask(targets, times, ch)
        for y in unique_years:
            mask = years == y
            if not np.any(mask): continue
            pred_max = np.max(predictions[mask, ch], axis=0)
            targ_max = np.max(targets[mask, ch], axis=0)
            pred_max, targ_max = _apply_snow_mask(snow, pred_max, targ_max)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
            plot_vmin = vmin if vmin is not None else min(np.nanmin(pred_max), np.nanmin(targ_max))
            plot_vmax = vmax if vmax is not None else max(np.nanmax(pred_max), np.nanmax(targ_max))

            plot_map(targ_max, lat, lon, axes[0], f'Target Max {y}', plot_vmin, plot_vmax, draw_gridlines=True, extend='max')
            plot_map(pred_max, lat, lon, axes[1], f'Pred Max {y}', plot_vmin, plot_vmax, draw_gridlines=True, extend='max')

            with np.errstate(divide='ignore', invalid='ignore'):
                diff = (pred_max - targ_max) / targ_max * 100

            plot_map(diff, lat, lon, axes[2], 'Diff (%)', vmin=-100, vmax=100, cmap='RdBu', draw_gridlines=True, extend='both')

            plt.suptitle(f'{target_names[ch]} Yearly Max {y}', fontsize='xx-large', fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'map_max_{y}_{target_names[ch]}.png', dpi=FIGURE_DPI)
            plt.close()

def get_state_mask(state_abbrev, lat, lon):
    try:
        reader = shpreader.Reader(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_1_states_provinces'))
        states = list(reader.records())
        state = next((s for s in states if s.attributes['postal'] == state_abbrev), None)
        if not state: return None
        state_geom = prep(state.geometry)

        mask = np.zeros(lat.shape, dtype=bool)
        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                if state_geom.contains(Point(lon[i, j], lat[i, j])):
                    mask[i, j] = True
        return mask
    except Exception:
        return None

def plot_state_evaluation(predictions, targets, output_dir, state_abbrevs, target_names=None):
    lat, lon = get_lat_lon()
    if lat is None: return
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)

    for state in state_abbrevs:
        mask = get_state_mask(state, lat, lon)
        if mask is None or not np.any(mask): continue

        state_pred = predictions[:, :, mask] # (N, C, Pixels)
        state_targ = targets[:, :, mask]

        output_subdir = output_dir / state
        output_subdir.mkdir(exist_ok=True)

        plot_total_mean_max_scatter_spatial(
            np.expand_dims(state_pred, 2), # Reshape back to pseudo-image
            np.expand_dims(state_targ, 2),
            output_subdir,
            target_names
        )

        n_channels = predictions.shape[1]
        for ch in range(n_channels):
            pred_pixel_mean = np.mean(state_pred[:, ch, :], axis=0)
            targ_pixel_mean = np.mean(state_targ[:, ch, :], axis=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_scatter(targ_pixel_mean, pred_pixel_mean, ax, f'{state} Pixel Mean', 'Target', 'Prediction')
            plt.savefig(output_subdir / f'scatter_pixel_mean_{state}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_state_evaluation_yearly(predictions, targets, times, output_dir, state_abbrevs, target_names=None, filename_suffix='', accumulate=False, reset_month_day=None):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for state in state_abbrevs:
        mask = get_state_mask(state, lat, lon)
        if mask is None or not np.any(mask): continue

        output_subdir = output_dir / state
        output_subdir.mkdir(exist_ok=True)

        for ch in range(n_channels):
            all_pred_maxs = []
            all_targ_maxs = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                p_max = np.max(p, axis=0)
                t_max = np.max(t, axis=0)

                all_pred_maxs.append(p_max)
                all_targ_maxs.append(t_max)

            if not all_pred_maxs: continue

            flat_pred = np.concatenate(all_pred_maxs)
            flat_targ = np.concatenate(all_targ_maxs)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_scatter(flat_targ, flat_pred, ax, f'{state} Yearly Max (All Years)', 'Target', 'Prediction')
            plt.savefig(output_subdir / f'scatter_yearly_max_pooled_{target_names[ch]}{filename_suffix}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_region_evaluation(predictions, targets, output_dir, region_defs, target_names=None):
    lat, lon = get_lat_lon()
    if lat is None: return
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        if not np.any(mask): continue

        region_pred = predictions[:, :, mask]
        region_targ = targets[:, :, mask]

        plot_total_mean_max_scatter_spatial(
            np.expand_dims(region_pred, 2),
            np.expand_dims(region_targ, 2),
            output_dir / name,
            target_names
        )

def plot_region_evaluation_yearly(predictions, targets, times, output_dir, region_defs, target_names=None, filename_suffix='', accumulate=False, reset_month_day=None):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for region_name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        if not np.any(mask): continue

        output_subdir = output_dir / region_name
        output_subdir.mkdir(exist_ok=True)

        for ch in range(n_channels):
            all_pred_maxs = []
            all_targ_maxs = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                p_max = np.max(p, axis=0)
                t_max = np.max(t, axis=0)

                all_pred_maxs.append(p_max)
                all_targ_maxs.append(t_max)

            if not all_pred_maxs: continue

            flat_pred = np.concatenate(all_pred_maxs)
            flat_targ = np.concatenate(all_targ_maxs)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_scatter(flat_targ, flat_pred, ax, f'{region_name} Yearly Max (Pooled)', 'Target', 'Prediction')
            plt.savefig(output_subdir / f'scatter_yearly_max_pooled_{target_names[ch]}{filename_suffix}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_time_series_seasonality(predictions, targets, times, output_dir, target_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    month_map, month_order = _build_month_index(times)
    if not month_map: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_positions = np.arange(len(month_order))

    for ch in range(n_channels):
        season_pred_mean = []
        season_pred_std = []
        season_targ_mean = []
        season_targ_std = []

        for month in month_order:
            month_indices = month_map[month]
            if len(month_indices) == 0:
                season_pred_mean.append(0)
                season_pred_std.append(0)
                season_targ_mean.append(0)
                season_targ_std.append(0)
                continue

            pred_spatial_mean = np.mean(predictions[month_indices, ch], axis=(1, 2))
            targ_spatial_mean = np.mean(targets[month_indices, ch], axis=(1, 2))

            season_pred_mean.append(np.mean(pred_spatial_mean))
            season_pred_std.append(np.std(pred_spatial_mean))
            season_targ_mean.append(np.mean(targ_spatial_mean))
            season_targ_std.append(np.std(targ_spatial_mean))

        season_pred_mean = np.array(season_pred_mean)
        season_pred_std = np.array(season_pred_std)
        season_targ_mean = np.array(season_targ_mean)
        season_targ_std = np.array(season_targ_std)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_positions, season_targ_mean, 'k-', linewidth=2, label='Target')
        ax.fill_between(x_positions, season_targ_mean - season_targ_std, season_targ_mean + season_targ_std, color='k', alpha=0.2)

        ax.plot(x_positions, season_pred_mean, 'b--', linewidth=2, label='Prediction')
        ax.fill_between(x_positions, season_pred_mean - season_pred_std, season_pred_mean + season_pred_std, color='b', alpha=0.2)

        ax.set_title(f'Spatial Mean {target_names[ch]} Seasonality', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([month_names[m-1] for m in month_order])
        ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'seasonality_mean_{target_names[ch]}.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

def plot_time_series_seasonality_spatial_max(predictions, targets, times, output_dir, target_names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    month_map, month_order = _build_month_index(times)
    if not month_map: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_positions = np.arange(len(month_order))

    for ch in range(n_channels):
        season_pred_max = []
        season_pred_std = []
        season_targ_max = []
        season_targ_std = []

        for month in month_order:
            month_indices = month_map[month]
            if len(month_indices) == 0:
                season_pred_max.append(0)
                season_pred_std.append(0)
                season_targ_max.append(0)
                season_targ_std.append(0)
                continue

            pred_spatial_max = np.max(predictions[month_indices, ch], axis=(1, 2))
            targ_spatial_max = np.max(targets[month_indices, ch], axis=(1, 2))

            season_pred_max.append(np.mean(pred_spatial_max))
            season_pred_std.append(np.std(pred_spatial_max))
            season_targ_max.append(np.mean(targ_spatial_max))
            season_targ_std.append(np.std(targ_spatial_max))

        season_pred_max = np.array(season_pred_max)
        season_pred_std = np.array(season_pred_std)
        season_targ_max = np.array(season_targ_max)
        season_targ_std = np.array(season_targ_std)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_positions, season_targ_max, 'k-', linewidth=2, label='Target')
        ax.fill_between(x_positions, season_targ_max - season_targ_std, season_targ_max + season_targ_std, color='k', alpha=0.2)

        ax.plot(x_positions, season_pred_max, 'b--', linewidth=2, label='Prediction')
        ax.fill_between(x_positions, season_pred_max - season_pred_std, season_pred_max + season_pred_std, color='b', alpha=0.2)

        ax.set_title(f'Spatial Max {target_names[ch]} Seasonality', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([month_names[m-1] for m in month_order])
        ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'seasonality_max_{target_names[ch]}.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

def plot_time_series_seasonality_regions(predictions, targets, times, output_dir, region_defs, target_names=None, metric='mean'):
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    month_map, month_order = _build_month_index(times)
    if not month_map: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_positions = np.arange(len(month_order))

    whole_stats = {}
    for ch in range(n_channels):
        w_pred_mean = []
        w_pred_std = []
        w_targ_mean = []
        w_targ_std = []

        for month in month_order:
            month_indices = month_map[month]
            if len(month_indices) == 0:
                w_pred_mean.append(0)
                w_pred_std.append(0)
                w_targ_mean.append(0)
                w_targ_std.append(0)
                continue

            p = predictions[month_indices, ch]
            t = targets[month_indices, ch]

            if metric == 'max':
                p_spatial = np.max(p, axis=(1, 2))
                t_spatial = np.max(t, axis=(1, 2))
            else:
                p_spatial = np.mean(p, axis=(1, 2))
                t_spatial = np.mean(t, axis=(1, 2))

            w_pred_mean.append(np.mean(p_spatial))
            w_pred_std.append(np.std(p_spatial))
            w_targ_mean.append(np.mean(t_spatial))
            w_targ_std.append(np.std(t_spatial))

        whole_stats[ch] = {
            'pred_mean': np.array(w_pred_mean),
            'pred_std': np.array(w_pred_std),
            'targ_mean': np.array(w_targ_mean),
            'targ_std': np.array(w_targ_std),
            'x_values': np.array(month_order)
        }

    for region_name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        if not np.any(mask): continue

        for ch in range(n_channels):
            vals_pred = []
            stds_pred = []
            vals_targ = []
            stds_targ = []

            for month in month_order:
                month_indices = month_map[month]
                if len(month_indices) == 0:
                    vals_pred.append(0)
                    stds_pred.append(0)
                    vals_targ.append(0)
                    stds_targ.append(0)
                    continue

                p = predictions[month_indices, ch][:, mask] # (N, Pixels)
                t = targets[month_indices, ch][:, mask]

                if metric == 'max':
                    p_metric = np.max(p, axis=1)
                    t_metric = np.max(t, axis=1)
                elif metric == 'sum':
                    p_metric = np.sum(p, axis=1)
                    t_metric = np.sum(t, axis=1)
                else:
                    p_metric = np.mean(p, axis=1)
                    t_metric = np.mean(t, axis=1)

                vals_pred.append(np.mean(p_metric))
                stds_pred.append(np.std(p_metric))
                vals_targ.append(np.mean(t_metric))
                stds_targ.append(np.std(t_metric))

            vals_pred = np.array(vals_pred)
            stds_pred = np.array(stds_pred)
            vals_targ = np.array(vals_targ)
            stds_targ = np.array(stds_targ)

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(x_positions, vals_targ, 'k-', linewidth=2, label=f'{region_name} Target')
            ax.fill_between(x_positions, vals_targ - stds_targ, vals_targ + stds_targ, color='k', alpha=0.2)

            ax.plot(x_positions, vals_pred, 'b-', linewidth=2, label=f'{region_name} Prediction')
            ax.fill_between(x_positions, vals_pred - stds_pred, vals_pred + stds_pred, color='b', alpha=0.2)

            metric_label = {'max': 'Spatial Max', 'sum': 'Spatial Sum'}.get(metric, 'Spatial Mean')
            ax.set_title(f'{region_name} {metric_label} {target_names[ch]} Seasonality', fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Month', fontsize='x-large')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([month_names[m-1] for m in month_order])
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'{region_name}_seasonality_{metric}_{target_names[ch]}.png', dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()

def plot_seasonality_regions_combined(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name: str = 'Target',
    prediction_label_name: str = 'Prediction',
    time_res: str = 'month',
):
    """
    Create a single combined figure with all regions in separate panels showing
    the seasonal cycle (monthly climatology, or daily climatology when
    time_res='day').

    Each panel shows:
      - Target and Prediction lines with ±1σ shading
      - Stats box with R, RMSE, MAE, Bias, peak-month error, and amplitude ratio
        — quantities suitable for reporting in a paper.

    Output filename: seasonality_combined_{metric}_{channel}.png
                     seasonality_combined_{metric}_{channel}_day_2_day.png (time_res='day')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None:
        return

    daily = (time_res == 'day')

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if daily:
        bin_map, bin_order = _build_dayofyear_index(times)
    else:
        bin_map, bin_order = _build_month_index(times)
    if not bin_map:
        return

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    x_positions = np.arange(len(bin_order))
    if daily:
        # Label only the first day of each month so the axis stays readable
        bin_labels = [f'{month_names[m - 1]} {d}' for m, d in bin_order]
        x_ticks = [i for i, (m, d) in enumerate(bin_order) if d == 1]
        x_labels = [month_names[bin_order[i][0] - 1] for i in x_ticks]
    else:
        bin_labels = [month_names[m - 1] for m in bin_order]
        x_ticks = list(x_positions)
        x_labels = bin_labels

    region_names = list(region_defs.keys())
    n_regions = len(region_names)
    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    region_masks = {
        rname: (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
        for rname, bounds in region_defs.items()
    }

    for ch in range(n_channels):
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        for ax_idx, region_name in enumerate(region_names):
            ax = axes[ax_idx]
            mask = region_masks[region_name]
            if not np.any(mask):
                ax.set_visible(False)
                continue

            vals_pred, stds_pred = [], []
            vals_targ, stds_targ = [], []

            for bin_key in bin_order:
                midx = bin_map[bin_key]
                if len(midx) == 0:
                    vals_pred.append(0); stds_pred.append(0)
                    vals_targ.append(0); stds_targ.append(0)
                    continue
                p = predictions[midx, ch][:, mask]
                t = targets[midx, ch][:, mask]
                if metric == 'max':
                    p_s = np.max(p, axis=1)
                    t_s = np.max(t, axis=1)
                else:
                    p_s = np.mean(p, axis=1)
                    t_s = np.mean(t, axis=1)
                vals_pred.append(np.mean(p_s)); stds_pred.append(np.std(p_s))
                vals_targ.append(np.mean(t_s)); stds_targ.append(np.std(t_s))

            vp = np.array(vals_pred)
            sp = np.array(stds_pred)
            vt = np.array(vals_targ)
            st = np.array(stds_targ)

            # ── compute paper statistics ────────────────────────────────────
            r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else float('nan')
            rmse = np.sqrt(np.mean((vp - vt) ** 2))
            mae  = np.mean(np.abs(vp - vt))
            bias = np.mean(vp - vt)                    # mm, signed

            peak_targ_idx = int(np.argmax(vt))
            peak_pred_idx = int(np.argmax(vp))
            peak_targ_mon = bin_labels[peak_targ_idx]
            peak_pred_mon = bin_labels[peak_pred_idx]
            peak_err_steps = peak_pred_idx - peak_targ_idx   # signed month/day offset

            # Seasonal amplitude = max − min of monthly climatology
            amp_targ = float(vt.max() - vt.min())
            amp_pred = float(vp.max() - vp.min())
            amp_ratio = (amp_pred / amp_targ) if amp_targ > 0 else float('nan')

            # ── plot ────────────────────────────────────────────────────────
            # Markers would swamp a ~365-point daily curve, so drop them there
            targ_style = 'k-' if daily else 'k-o'
            pred_style = 'b--' if daily else 'b--s'
            ax.plot(x_positions, vt, targ_style, linewidth=2, markersize=4,
                    label=target_label_name, alpha=0.9)
            ax.fill_between(x_positions, vt - st, vt + st, color='k', alpha=0.12)

            ax.plot(x_positions, vp, pred_style, linewidth=2, markersize=4,
                    label=prediction_label_name, alpha=0.9)
            ax.fill_between(x_positions, vp - sp, vp + sp, color='b', alpha=0.12)

            stats_lines = [
                f'R      = {r:.3f}',
                f'MAE    = {mae:.1f} mm',
                f'Peak {amp_targ:.0f}→{amp_pred:.0f}',
                f'Peak {"Date" if daily else "Month"} {peak_targ_mon}→{peak_pred_mon}',
            ]
            stats_text = '\n'.join(stats_lines)
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.88,
                         edgecolor='#8B7355', linewidth=1.2)
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize='large',
                    verticalalignment='top', bbox=props, family='monospace')

            ax.set_title(panel_label(ax_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_xlabel('Date' if daily else 'Month', fontsize='large')
            ax.set_ylabel('SWE [mm]', fontsize='large')
            ax.set_ylim(bottom=0)
            if daily:
                ax.set_xlim(x_positions[0], x_positions[-1])
            ax.legend(fontsize='large', loc='upper right')
            ax.grid(True, alpha=0.3)

        for ax_idx in range(n_regions, len(axes)):
            axes[ax_idx].set_visible(False)

        metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
        res_label = 'Daily ' if daily else ''
        plt.suptitle(
            f'Regional {res_label}Seasonal Cycle {metric_label} SWE',
            fontsize='xx-large', fontweight='bold'
        )
        plt.tight_layout()
        suffix = '_day_2_day' if daily else ''
        output_path = output_dir / f'seasonality_combined_{metric}_{target_names[ch]}{suffix}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_mean_of_yearly_max_scatter_whole_domain(predictions, targets, times, output_dir, target_names=None, filename_suffix='', accumulate=False, reset_month_day=None):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        pred_maxs = []
        targ_maxs = []
        for y in unique_years:
            mask = years == y
            if not np.any(mask): continue
            pred_maxs.append(np.max(predictions[mask, ch], axis=0))
            targ_maxs.append(np.max(targets[mask, ch], axis=0))

        pred_mean_max = np.mean(pred_maxs, axis=0).flatten()
        targ_mean_max = np.mean(targ_maxs, axis=0).flatten()

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_scatter(targ_mean_max, pred_mean_max, ax, f'Mean Yearly Max (Whole Domain)', 'Target', 'Prediction')
        plt.savefig(output_dir / f'scatter_mean_yearly_max_{target_names[ch]}{filename_suffix}.png', dpi=FIGURE_DPI)
        plt.close()

def plot_mean_of_yearly_max_scatter_state(predictions, targets, times, output_dir, state_abbrevs, target_names=None, filename_suffix='', accumulate=False, reset_month_day=None):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for state in state_abbrevs:
        mask = get_state_mask(state, lat, lon)
        if mask is None or not np.any(mask): continue

        for ch in range(n_channels):
            pred_maxs = []
            targ_maxs = []
            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue
                p = predictions[y_mask, ch][:, mask] # (N_y, Pixels)
                t = targets[y_mask, ch][:, mask]

                pred_maxs.append(np.max(p, axis=0))
                targ_maxs.append(np.max(t, axis=0))

            pred_mean_max = np.mean(pred_maxs, axis=0)
            targ_mean_max = np.mean(targ_maxs, axis=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_scatter(targ_mean_max, pred_mean_max, ax, f'{state} Mean Yearly Max', 'Target', 'Prediction')
            plt.savefig(output_dir / state / f'scatter_mean_yearly_max_{target_names[ch]}{filename_suffix}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_mean_of_yearly_max_scatter_regional(predictions, targets, times, output_dir, regions, target_names=None, filename_suffix='', accumulate=False, reset_month_day=None):
    """
    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return
    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for region_name, bounds in regions.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        if not np.any(mask): continue

        for ch in range(n_channels):
            pred_maxs = []
            targ_maxs = []
            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue
                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]
                pred_maxs.append(np.max(p, axis=0))
                targ_maxs.append(np.max(t, axis=0))

            pred_mean_max = np.mean(pred_maxs, axis=0)
            targ_mean_max = np.mean(targ_maxs, axis=0)

            output_subdir = output_dir / region_name
            output_subdir.mkdir(exist_ok=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            plot_scatter(targ_mean_max, pred_mean_max, ax, f'{region_name} Mean Yearly Max', 'Target', 'Prediction')
            plt.savefig(output_subdir / f'scatter_mean_yearly_max_{target_names[ch]}{filename_suffix}.png', dpi=FIGURE_DPI)
            plt.close()

def plot_whole_domain_time_series(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
):
    """
    Create time series plot for the whole domain.
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))

    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])
            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)
    dates = convert_to_plot_dates(dates)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating whole domain time series")

    for ch_idx in range(n_channels):
        fig, ax = plt.subplots(figsize=(20, 6))

        if metric == 'max':
            pred_agg = np.max(predictions_sorted[:, ch_idx], axis=(1, 2))
            target_agg = np.max(targets_sorted[:, ch_idx], axis=(1, 2))
        else:
            pred_agg = np.mean(predictions_sorted[:, ch_idx], axis=(1, 2))
            target_agg = np.mean(targets_sorted[:, ch_idx], axis=(1, 2))

        ax.plot(dates, target_agg, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
        ax.plot(dates, pred_agg, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

        metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
        ax.set_title(f'Whole Domain {metric_label} - {target_names[ch_idx]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if is_daily:
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        output_path = output_dir / f'whole_domain_timeseries_{metric}_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")

def plot_long_term_time_series_regions(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
    filename_suffix: str = '',
    accumulate: bool = False,
    reset_month_day: tuple = None,
):
    """
    Create four-panel long-term time series plot for four regions.

    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))

    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
        # Convert to matplotlib-compatible dates (handles cftime, numpy.datetime64, etc.)
        dates = convert_to_plot_dates(dates)
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])

            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating long-term region time series for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_region = predictions_sorted[:, ch_idx][:, mask]
            target_region = targets_sorted[:, ch_idx][:, mask]

            if metric == 'max':
                pred_agg = np.max(pred_region, axis=1)
                target_agg = np.max(target_region, axis=1)
                pred_min = np.min(pred_region, axis=1)
                pred_max = np.max(pred_region, axis=1)
                target_min = np.min(target_region, axis=1)
                target_max = np.max(target_region, axis=1)
            elif metric == 'sum':
                pred_agg = np.sum(pred_region, axis=1)
                target_agg = np.sum(target_region, axis=1)
                pred_min = np.min(pred_region, axis=1)
                pred_max = np.max(pred_region, axis=1)
                target_min = np.min(target_region, axis=1)
                target_max = np.max(target_region, axis=1)
            else:
                pred_agg = np.mean(pred_region, axis=1)
                target_agg = np.mean(target_region, axis=1)
                pred_min = np.min(pred_region, axis=1)
                pred_max = np.max(pred_region, axis=1)
                target_min = np.min(target_region, axis=1)
                target_max = np.max(target_region, axis=1)

            ax.plot(dates, target_agg, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)

            ax.plot(dates, pred_agg, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_agg, pred_agg)[0, 1] if len(target_agg) > 1 else 0

            # Calculate trends (mm/yr)
            years_float = np.array([d.year + (d.month - 1) / 12.0 + (d.day - 1) / 365.25 for d in dates])

            # Fit linear trend: y = slope * year + intercept
            if len(years_float) > 1:
                target_trend_coef = np.polyfit(years_float, target_agg, 1)
                pred_trend_coef = np.polyfit(years_float, pred_agg, 1)
                target_trend = target_trend_coef[0]  # mm/yr
                pred_trend = pred_trend_coef[0]  # mm/yr
            else:
                target_trend = 0
                pred_trend = 0

            stats_text = f'Corr = {corr:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = {'max': 'Spatial Max', 'sum': 'Spatial Sum'}.get(metric, 'Spatial Mean')
            ax.set_title(f'{region_name} {metric_label} - {target_names[ch_idx]}', fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Date', fontsize='x-large')
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if is_daily:
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            output_path = output_dir / f'long_term_timeseries_{metric}_{region_name}_{target_names[ch_idx]}{filename_suffix}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_yearly_mean_time_series_regions(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
    accumulate: bool = False,
    reset_month_day: tuple = None,
    resolution_km: float = None,
):
    """
    Create yearly mean time series plot for each region (one point per year).

    Args:
        accumulate: If True, accumulate the predictions/targets (for difference models)
        reset_month_day: Tuple of (month, day) to reset accumulation, or None for no reset
        resolution_km: Grid spacing in km. Only used when metric='sum', where the
            raw pixel sum of SWE in mm is not a physical unit; supplying the
            spacing converts it to snow water VOLUME in km^3
            (sum(mm) * (dx km)^2 * 1e-6) and relabels the axis and the trend
            accordingly. Left as None the raw pixel sum is plotted in mm, so
            existing callers are unaffected.
    """
    if accumulate:
        predictions = accumulate_predictions(predictions, times, reset_month_day=reset_month_day)
        targets = accumulate_predictions(targets, times, reset_month_day=None)

    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    unique_years = np.unique(years)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating yearly mean region time series for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_yearly_means = []
            pred_yearly_mins = []
            pred_yearly_maxs = []
            target_yearly_means = []
            target_yearly_mins = []
            target_yearly_maxs = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                pred_region = predictions[year_mask, ch_idx][:, mask]
                target_region = targets[year_mask, ch_idx][:, mask]

                if metric == 'max':
                    # For each timestep in the year, get spatial max, then aggregate across timesteps
                    pred_spatial_max = np.max(pred_region, axis=1)
                    target_spatial_max = np.max(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_max))
                    pred_yearly_mins.append(np.min(pred_spatial_max))
                    pred_yearly_maxs.append(np.max(pred_spatial_max))
                    target_yearly_means.append(np.mean(target_spatial_max))
                    target_yearly_mins.append(np.min(target_spatial_max))
                    target_yearly_maxs.append(np.max(target_spatial_max))
                elif metric == 'sum':
                    # For each timestep, get spatial SUM, then aggregate across timesteps
                    pred_spatial_sum = np.sum(pred_region, axis=1)
                    target_spatial_sum = np.sum(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_sum))
                    pred_yearly_mins.append(np.min(pred_spatial_sum))
                    pred_yearly_maxs.append(np.max(pred_spatial_sum))
                    target_yearly_means.append(np.mean(target_spatial_sum))
                    target_yearly_mins.append(np.min(target_spatial_sum))
                    target_yearly_maxs.append(np.max(target_spatial_sum))
                else:
                    # For each timestep in the year, get spatial mean, then aggregate across timesteps
                    pred_spatial_mean = np.mean(pred_region, axis=1)
                    target_spatial_mean = np.mean(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_mean))
                    pred_yearly_mins.append(np.min(pred_spatial_mean))
                    pred_yearly_maxs.append(np.max(pred_spatial_mean))
                    target_yearly_means.append(np.mean(target_spatial_mean))
                    target_yearly_mins.append(np.min(target_spatial_mean))
                    target_yearly_maxs.append(np.max(target_spatial_mean))

            pred_yearly_means = np.array(pred_yearly_means)
            pred_yearly_mins = np.array(pred_yearly_mins)
            pred_yearly_maxs = np.array(pred_yearly_maxs)
            target_yearly_means = np.array(target_yearly_means)
            target_yearly_mins = np.array(target_yearly_mins)
            target_yearly_maxs = np.array(target_yearly_maxs)

            # A spatial SUM of SWE in mm has no physical unit on its own — it is
            # only meaningful as a water volume. When the caller supplies the
            # grid spacing, convert to km^3 (sum(mm) * (dx km)^2 * 1e-6) and
            # relabel; without it the raw pixel sum in mm is kept, so existing
            # callers are unchanged.
            to_volume = (metric == 'sum' and resolution_km is not None)
            if to_volume:
                vol_factor = (float(resolution_km) ** 2) * 1e-6
                pred_yearly_means = pred_yearly_means * vol_factor
                pred_yearly_mins = pred_yearly_mins * vol_factor
                pred_yearly_maxs = pred_yearly_maxs * vol_factor
                target_yearly_means = target_yearly_means * vol_factor
                target_yearly_mins = target_yearly_mins * vol_factor
                target_yearly_maxs = target_yearly_maxs * vol_factor
            trend_unit = 'km^3' if to_volume else 'mm'
            trend_fmt = '.3f' if to_volume else '.2f'

            ax.plot(unique_years, target_yearly_means, 'k-o', linewidth=2, markersize=6,
                   label=target_label_name, alpha=0.8)

            ax.plot(unique_years, pred_yearly_means, 'b--s', linewidth=2, markersize=6,
                   label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_yearly_means, pred_yearly_means)[0, 1] if len(target_yearly_means) > 1 else 0

            # Calculate trends (per year, in whatever unit is plotted)
            if len(unique_years) > 1:
                target_trend_coef = np.polyfit(unique_years, target_yearly_means, 1)
                pred_trend_coef = np.polyfit(unique_years, pred_yearly_means, 1)
                target_trend = target_trend_coef[0]
                pred_trend = pred_trend_coef[0]
            else:
                target_trend = 0
                pred_trend = 0

            stats_text = (f'Corr = {corr:.3f}\n'
                          f'Target Trend = {target_trend:{trend_fmt}} {trend_unit}/yr\n'
                          f'Pred Trend = {pred_trend:{trend_fmt}} {trend_unit}/yr')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = {
                'max': 'Yearly Mean of Spatial Max',
                'sum': ('Yearly Mean Water Volume' if to_volume
                        else 'Yearly Mean of Spatial Sum'),
            }.get(metric, 'Yearly Mean of Spatial Mean')
            ax.set_title(f'{region_name} {metric_label} Snow', fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='x-large')
            ax.set_ylabel('Snow Water Volume (km$^3$)' if to_volume
                          else 'Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            plt.tight_layout()
            output_path = output_dir / f'yearly_mean_timeseries_{metric}_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_yearly_mean_time_series_regions_combined(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name='Target',
    prediction_label_name='Prediction',
):
    """
    Create a single combined figure with all regions in separate panels.
    Shows only R (Pearson correlation) and MAE (no trend lines).
    One figure per channel, each region occupies one subplot panel.
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    unique_years = np.unique(years)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None:
        return

    region_names = list(region_defs.keys())
    n_regions = len(region_names)

    region_masks = {}
    for region_name, bounds in region_defs.items():
        mask = (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
        region_masks[region_name] = mask

    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    for ch_idx in range(n_channels):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax_idx, region_name in enumerate(region_names):
            ax = axes[ax_idx]
            mask = region_masks[region_name]

            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_yearly_means = []
            target_yearly_means = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                pred_region = predictions[year_mask, ch_idx][:, mask]
                target_region = targets[year_mask, ch_idx][:, mask]

                if metric == 'max':
                    pred_spatial = np.max(pred_region, axis=1)
                    target_spatial = np.max(target_region, axis=1)
                else:
                    pred_spatial = np.mean(pred_region, axis=1)
                    target_spatial = np.mean(target_region, axis=1)

                pred_yearly_means.append(np.mean(pred_spatial))
                target_yearly_means.append(np.mean(target_spatial))

            pred_yearly_means = np.array(pred_yearly_means)
            target_yearly_means = np.array(target_yearly_means)

            ax.plot(unique_years, target_yearly_means, 'k-o', linewidth=2, markersize=5,
                    label=target_label_name, alpha=0.8)
            ax.plot(unique_years, pred_yearly_means, 'b--s', linewidth=2, markersize=5,
                    label=prediction_label_name, alpha=0.8)

            if len(target_yearly_means) > 1:
                corr = np.corrcoef(target_yearly_means, pred_yearly_means)[0, 1]
            else:
                corr = float('nan')
            mae = np.mean(np.abs(target_yearly_means - pred_yearly_means))

            stats_text = f'R = {corr:.3f}\nMAE = {mae:.2f} mm'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                         edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, family='monospace')

            metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
            ax.set_title(panel_label(ax_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='large')
            ax.set_ylabel('SWE [mm]', fontsize='large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize='large')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        for ax_idx in range(n_regions, len(axes)):
            axes[ax_idx].set_visible(False)

        metric_label = 'Yearly Mean of Spatial Max' if metric == 'max' else 'Yearly Mean of Spatial Mean'
        plt.suptitle(f'Regional {metric_label} SWE',
                     fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f'yearly_mean_timeseries_{metric}_all_regions_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def _select_april1_indices(times):
    """
    Select one sample index per year corresponding to April 1 SWE.

    Only exact April 1 samples are used; a year whose record contains no
    April 1 (e.g. a partial final year) is dropped from the series.

    Returns a tuple (years, indices, skipped_years) where `years` are the
    calendar years of the selected April 1 dates (not the file/water-year label
    in times[:, 0]), `indices` are the matching sample indices, and
    `skipped_years` are the calendar years that had no April 1 sample.
    """
    dates = get_datetimes_from_data(times)
    if dates is None or len(dates) == 0:
        return np.array([]), np.array([], dtype=int), []

    # Time coordinates may come back as numpy.datetime64 or cftime objects,
    # which do not expose .month/.day/.year uniformly.
    dates = convert_to_plot_dates(dates)

    years = times[:, 0]
    unique_years = np.unique(years)

    sel_years = []
    sel_indices = []
    skipped_years = []

    for year in unique_years:
        year_idx = np.where(years == year)[0]
        if len(year_idx) == 0:
            continue

        exact = [i for i in year_idx
                 if dates[i].month == 4 and dates[i].day == 1]
        if exact:
            sel_years.append(dates[exact[0]].year)
            sel_indices.append(exact[0])
        else:
            # times[:, 0] is a file/water-year label; report the calendar year
            # the missing April 1 would have fallen in.
            skipped_years.append(dates[year_idx[-1]].year)

    return np.array(sel_years), np.array(sel_indices, dtype=int), skipped_years


def plot_april1_swe_time_series_regions_combined(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name='Target',
    prediction_label_name='Prediction',
):
    """
    Create a single combined figure with all regions in separate panels showing
    the April 1 SWE time series (one value per year).

    Same layout as plot_yearly_mean_time_series_regions_combined, but instead of
    averaging over the whole year, only the April 1 sample of each year is used.
    Shows R (Pearson correlation) and MAE (no trend lines).
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    sel_years, sel_indices, skipped_years = _select_april1_indices(times)
    if len(sel_indices) == 0:
        print("Warning: no April 1 samples found in the data, skipping April 1 SWE figure.")
        return
    if skipped_years:
        print("  Note: no April 1 sample for " +
              ", ".join(str(y) for y in skipped_years) +
              "; these years are excluded from the figure.")

    order = np.argsort(sel_years)
    sel_years = sel_years[order]
    sel_indices = sel_indices[order]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None:
        return

    region_names = list(region_defs.keys())
    n_regions = len(region_names)

    region_masks = {}
    for region_name, bounds in region_defs.items():
        mask = (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
        region_masks[region_name] = mask

    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    for ch_idx in range(n_channels):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax_idx, region_name in enumerate(region_names):
            ax = axes[ax_idx]
            mask = region_masks[region_name]

            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_region = predictions[sel_indices, ch_idx][:, mask]
            target_region = targets[sel_indices, ch_idx][:, mask]

            if metric == 'max':
                pred_values = np.max(pred_region, axis=1)
                target_values = np.max(target_region, axis=1)
            else:
                pred_values = np.mean(pred_region, axis=1)
                target_values = np.mean(target_region, axis=1)

            ax.plot(sel_years, target_values, 'k-o', linewidth=2, markersize=5,
                    label=target_label_name, alpha=0.8)
            ax.plot(sel_years, pred_values, 'b--s', linewidth=2, markersize=5,
                    label=prediction_label_name, alpha=0.8)

            if len(target_values) > 1:
                corr = np.corrcoef(target_values, pred_values)[0, 1]
            else:
                corr = float('nan')
            mae = np.mean(np.abs(target_values - pred_values))

            stats_text = f'R = {corr:.3f}\nMAE = {mae:.2f} mm'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                         edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, family='monospace')

            ax.set_title(panel_label(ax_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='large')
            ax.set_ylabel('SWE [mm]', fontsize='large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize='large')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        for ax_idx in range(n_regions, len(axes)):
            axes[ax_idx].set_visible(False)

        metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
        plt.suptitle(f'Regional April 1 {metric_label} SWE',
                     fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f'april1_swe_timeseries_{metric}_all_regions_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_yearly_snow_volume_regions_combined(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    resolution_km: float,
    target_names: list = None,
    yearly_stat: str = 'mean',
    target_label_name='Target',
    prediction_label_name='Prediction',
):
    """
    Combined figure of yearly regional total snow water VOLUME (km³), one panel per region.

    Same layout/stats as plot_yearly_mean_time_series_regions_combined, but the spatial
    aggregate is the regional snow water volume instead of a spatial mean/max:
        volume(t) = sum over region of snow_mm * 1e-6 km/mm * resolution_km²   [km³]

    yearly_stat:
        'mean' -> yearly mean of the per-timestep regional volume (km³)
        'sum'  -> annual accumulated volume, summed over all timesteps in the year (km³)
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    unique_years = np.unique(years)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None:
        return

    region_names = list(region_defs.keys())
    n_regions = len(region_names)

    region_masks = {}
    for region_name, bounds in region_defs.items():
        mask = (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
        region_masks[region_name] = mask

    cell_area_km2 = resolution_km * resolution_km
    print(f"\nCreating combined regional snow volume figure (resolution: {resolution_km} km, "
          f"cell area: {cell_area_km2} km², yearly stat: {yearly_stat})")

    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    for ch_idx in range(n_channels):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax_idx, region_name in enumerate(region_names):
            ax = axes[ax_idx]
            mask = region_masks[region_name]

            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_yearly = []
            target_yearly = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                # Per-timestep regional snow water volume (km³)
                pred_vol = np.sum(predictions[year_mask, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2
                target_vol = np.sum(targets[year_mask, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2

                if yearly_stat == 'sum':
                    pred_yearly.append(np.sum(pred_vol))
                    target_yearly.append(np.sum(target_vol))
                else:
                    pred_yearly.append(np.mean(pred_vol))
                    target_yearly.append(np.mean(target_vol))

            pred_yearly = np.array(pred_yearly)
            target_yearly = np.array(target_yearly)

            ax.plot(unique_years, target_yearly, 'k-o', linewidth=2, markersize=5,
                    label=target_label_name, alpha=0.8)
            ax.plot(unique_years, pred_yearly, 'b--s', linewidth=2, markersize=5,
                    label=prediction_label_name, alpha=0.8)

            if len(target_yearly) > 1:
                corr = np.corrcoef(target_yearly, pred_yearly)[0, 1]
            else:
                corr = float('nan')
            mae = np.mean(np.abs(target_yearly - pred_yearly))

            stats_text = f'R = {corr:.3f}\nMAE = {mae:.2f} km$^3$'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                         edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, family='monospace')

            ax.set_title(panel_label(ax_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='large')
            ax.set_ylabel('Snow Water Volume [km$^3$]', fontsize='large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize='large')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        for ax_idx in range(n_regions, len(axes)):
            axes[ax_idx].set_visible(False)

        stat_label = ('Total Annual Snow Water Volume' if yearly_stat == 'sum'
                      else 'Yearly Mean Total Snow Water Volume')
        plt.suptitle(f'Regional {stat_label}', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f'yearly_{yearly_stat}_timeseries_volume_all_regions_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_bar_chart(names, values, output_path, title, ylabel, color='skyblue'):
    fig, ax = plt.subplots(figsize=(max(6, len(names)*0.5), 6))
    bars = ax.bar(names, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ============================================================================
# ACCUMULATED SNOW TIME SERIES FUNCTIONS (for change-based predictions)
# ============================================================================

def plot_whole_domain_time_series_accumulated(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
):
    """
    Create time series plot for the whole domain with accumulated snow.

    This function takes snow changes as input, calculates cumulative accumulation
    with non-negativity constraint, then plots the time series.

    Args:
        predictions: Snow changes (n_samples, n_channels, height, width)
        targets: Target snow changes (n_samples, n_channels, height, width)
        times: Time information array
        output_dir: Output directory path
        target_names: Names for each channel
        metric: 'mean' or 'max' for spatial aggregation
        target_label_name: Label for target in legend
        prediction_label_name: Label for prediction in legend
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))

    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])
            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)
    dates = convert_to_plot_dates(dates)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating whole domain time series (accumulated snow)")

    for ch_idx in range(n_channels):
        fig, ax = plt.subplots(figsize=(20, 6))

        pred_accum = calculate_snow_accumulation(predictions_sorted[:, ch_idx])
        target_accum = calculate_snow_accumulation(targets_sorted[:, ch_idx])

        if metric == 'max':
            pred_agg = np.max(pred_accum, axis=(1, 2))
            target_agg = np.max(target_accum, axis=(1, 2))
        else:
            pred_agg = np.mean(pred_accum, axis=(1, 2))
            target_agg = np.mean(target_accum, axis=(1, 2))

        ax.plot(dates, target_agg, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
        ax.plot(dates, pred_agg, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

        metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
        ax.set_title(f'Whole Domain {metric_label} (Accumulated Snow) - {target_names[ch_idx]}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if is_daily:
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        output_path = output_dir / f'whole_domain_timeseries_accumulated_{metric}_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_long_term_time_series_regions_accumulated(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
):
    """
    Create long-term time series plot for regions with accumulated snow.

    This function takes snow changes as input, calculates cumulative accumulation
    with non-negativity constraint, then plots regional time series.

    Args:
        predictions: Snow changes (n_samples, n_channels, height, width)
        targets: Target snow changes (n_samples, n_channels, height, width)
        times: Time information array
        output_dir: Output directory path
        region_defs: Dictionary of region definitions
        target_names: Names for each channel
        metric: 'mean' or 'max' for spatial aggregation
        target_label_name: Label for target in legend
        prediction_label_name: Label for prediction in legend
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))

    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
        # Convert to matplotlib-compatible dates (handles cftime, numpy.datetime64, etc.)
        dates = convert_to_plot_dates(dates)
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])

            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating long-term region time series (accumulated snow) for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        pred_accum_full = calculate_snow_accumulation(predictions_sorted[:, ch_idx])
        target_accum_full = calculate_snow_accumulation(targets_sorted[:, ch_idx])

        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_region = pred_accum_full[:, mask]
            target_region = target_accum_full[:, mask]

            if metric == 'max':
                pred_agg = np.max(pred_region, axis=1)
                target_agg = np.max(target_region, axis=1)
                pred_min = np.min(pred_region, axis=1)
                pred_max = np.max(pred_region, axis=1)
                target_min = np.min(target_region, axis=1)
                target_max = np.max(target_region, axis=1)
            else:
                pred_agg = np.mean(pred_region, axis=1)
                target_agg = np.mean(target_region, axis=1)
                pred_min = np.min(pred_region, axis=1)
                pred_max = np.max(pred_region, axis=1)
                target_min = np.min(target_region, axis=1)
                target_max = np.max(target_region, axis=1)

            ax.plot(dates, target_agg, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
            ax.plot(dates, pred_agg, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_agg, pred_agg)[0, 1] if len(target_agg) > 1 else 0

            # Calculate trends (mm/yr)
            years_float = np.array([d.year + (d.month - 1) / 12.0 + (d.day - 1) / 365.25 for d in dates])

            if len(years_float) > 1:
                target_trend_coef = np.polyfit(years_float, target_agg, 1)
                pred_trend_coef = np.polyfit(years_float, pred_agg, 1)
                target_trend = target_trend_coef[0]
                pred_trend = pred_trend_coef[0]
            else:
                target_trend = 0
                pred_trend = 0

            stats_text = f'Corr = {corr:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
            ax.set_title(f'{region_name} {metric_label} (Accumulated Snow) - {target_names[ch_idx]}',
                        fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Date', fontsize='x-large')
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if is_daily:
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            output_path = output_dir / f'long_term_timeseries_accumulated_{metric}_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_long_term_time_series_regions_accumulated_yearly_reset(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
    reset_month: int = 9,
    reset_day: int = 1,
):
    """
    Create long-term time series plot for regions with accumulated snow that resets yearly.

    This function resets the prediction snow accumulation to 0 on a specified date each year
    (default: September 1st), which is useful for analyzing seasonal snow patterns.
    Note: Only predictions are reset; targets use regular accumulation to show true values.

    Args:
        predictions: Snow changes (n_samples, n_channels, height, width)
        targets: Target snow changes (n_samples, n_channels, height, width)
        times: Time information array
        output_dir: Output directory path
        region_defs: Dictionary of region definitions
        target_names: Names for each channel
        metric: 'mean' or 'max' for spatial aggregation
        target_label_name: Label for target in legend
        prediction_label_name: Label for prediction in legend
        reset_month: Month to reset prediction accumulation (1-12), default=9 (September)
        reset_day: Day of month to reset, default=1
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))

    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]
    times_sorted = times[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
        # Convert to matplotlib-compatible dates (handles cftime, numpy.datetime64, etc.)
        dates = convert_to_plot_dates(dates)
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])

            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating long-term region time series (accumulated snow with yearly reset) for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        pred_accum_full = calculate_snow_accumulation_yearly_reset(
            predictions_sorted[:, ch_idx], times_sorted, reset_month, reset_day
        )
        target_accum_full = calculate_snow_accumulation(
            targets_sorted[:, ch_idx]
        )

        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_region = pred_accum_full[:, mask]
            target_region = target_accum_full[:, mask]

            if metric == 'max':
                pred_agg = np.max(pred_region, axis=1)
                target_agg = np.max(target_region, axis=1)
            else:
                pred_agg = np.mean(pred_region, axis=1)
                target_agg = np.mean(target_region, axis=1)

            ax.plot(dates, target_agg, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
            ax.plot(dates, pred_agg, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_agg, pred_agg)[0, 1] if len(target_agg) > 1 else 0

            stats_text = f'Corr = {corr:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
            ax.set_title(f'{region_name} {metric_label} (Accumulated Snow Yearly Reset) - {target_names[ch_idx]}',
                        fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Date', fontsize='x-large')
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if is_daily:
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            output_path = output_dir / f'long_term_timeseries_accumulated_sept_reset_{metric}_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_yearly_mean_time_series_regions_accumulated(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
):
    """
    Create yearly mean time series plot for regions with accumulated snow.

    This function takes snow changes as input, calculates cumulative accumulation
    with non-negativity constraint, then plots yearly statistics.

    Args:
        predictions: Snow changes (n_samples, n_channels, height, width)
        targets: Target snow changes (n_samples, n_channels, height, width)
        times: Time information array
        output_dir: Output directory path
        region_defs: Dictionary of region definitions
        target_names: Names for each channel
        metric: 'mean' or 'max' for spatial aggregation
        target_label_name: Label for target in legend
        prediction_label_name: Label for prediction in legend
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    unique_years = np.unique(years)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating yearly mean region time series (accumulated snow) for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        pred_accum_full = calculate_snow_accumulation(predictions[:, ch_idx])
        target_accum_full = calculate_snow_accumulation(targets[:, ch_idx])

        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_yearly_means = []
            pred_yearly_mins = []
            pred_yearly_maxs = []
            target_yearly_means = []
            target_yearly_mins = []
            target_yearly_maxs = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                pred_region = pred_accum_full[year_mask][:, mask]
                target_region = target_accum_full[year_mask][:, mask]

                if metric == 'max':
                    pred_spatial_max = np.max(pred_region, axis=1)
                    target_spatial_max = np.max(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_max))
                    pred_yearly_mins.append(np.min(pred_spatial_max))
                    pred_yearly_maxs.append(np.max(pred_spatial_max))
                    target_yearly_means.append(np.mean(target_spatial_max))
                    target_yearly_mins.append(np.min(target_spatial_max))
                    target_yearly_maxs.append(np.max(target_spatial_max))
                else:
                    pred_spatial_mean = np.mean(pred_region, axis=1)
                    target_spatial_mean = np.mean(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_mean))
                    pred_yearly_mins.append(np.min(pred_spatial_mean))
                    pred_yearly_maxs.append(np.max(pred_spatial_mean))
                    target_yearly_means.append(np.mean(target_spatial_mean))
                    target_yearly_mins.append(np.min(target_spatial_mean))
                    target_yearly_maxs.append(np.max(target_spatial_mean))

            pred_yearly_means = np.array(pred_yearly_means)
            pred_yearly_mins = np.array(pred_yearly_mins)
            pred_yearly_maxs = np.array(pred_yearly_maxs)
            target_yearly_means = np.array(target_yearly_means)
            target_yearly_mins = np.array(target_yearly_mins)
            target_yearly_maxs = np.array(target_yearly_maxs)

            ax.plot(unique_years, target_yearly_means, 'k-o', linewidth=2, markersize=6,
                   label=target_label_name, alpha=0.8)

            ax.plot(unique_years, pred_yearly_means, 'b--s', linewidth=2, markersize=6,
                   label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_yearly_means, pred_yearly_means)[0, 1] if len(target_yearly_means) > 1 else 0

            # Calculate trends (mm/yr)
            if len(unique_years) > 1:
                target_trend_coef = np.polyfit(unique_years, target_yearly_means, 1)
                pred_trend_coef = np.polyfit(unique_years, pred_yearly_means, 1)
                target_trend = target_trend_coef[0]
                pred_trend = pred_trend_coef[0]
            else:
                target_trend = 0
                pred_trend = 0

            stats_text = f'Corr = {corr:.3f}\nTarget Trend = {target_trend:.2f} mm/yr\nPred Trend = {pred_trend:.2f} mm/yr'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = {
                'max': 'Yearly Mean of Spatial Max',
                'sum': 'Yearly Mean of Spatial Sum',
            }.get(metric, 'Yearly Mean of Spatial Mean')
            ax.set_title(f'{region_name} {metric_label} (Accumulated Snow)',
                        fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='x-large')
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            plt.tight_layout()
            output_path = output_dir / f'yearly_mean_timeseries_accumulated_{metric}_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_yearly_mean_time_series_regions_accumulated_yearly_reset(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
    target_label_name = 'Target',
    prediction_label_name = 'Prediction',
    reset_month: int = 11,
    reset_day: int = 1,
):
    """
    Create yearly mean time series plot for regions with accumulated snow that resets yearly.

    This function resets the prediction snow accumulation to 0 on a specified date each year
    (default: November 1st), then calculates yearly statistics.
    Note: Only predictions are reset; targets use regular accumulation to show true values.

    Args:
        predictions: Snow changes (n_samples, n_channels, height, width)
        targets: Target snow changes (n_samples, n_channels, height, width)
        times: Time information array
        output_dir: Output directory path
        region_defs: Dictionary of region definitions
        target_names: Names for each channel
        metric: 'mean' or 'max' for spatial aggregation
        target_label_name: Label for target in legend
        prediction_label_name: Label for prediction in legend
        reset_month: Month to reset prediction accumulation (1-12), default=11 (November)
        reset_day: Day of month to reset, default=1
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    unique_years = np.unique(years)

    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))
    years_sorted = years[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]
    times_sorted = times[sort_idx]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = get_lat_lon()
    if lat is None: return

    print(f"\nCreating yearly mean region time series (accumulated snow with yearly reset) for: {', '.join(region_defs.keys())}")

    region_masks = {}
    for region_name, bounds in region_defs.items():
        lon_min = bounds['lon_min']
        lon_max = bounds['lon_max']
        lat_min = bounds['lat_min']
        lat_max = bounds['lat_max']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        region_masks[region_name] = mask

    for ch_idx in range(n_channels):
        pred_accum_full = calculate_snow_accumulation_yearly_reset(
            predictions_sorted[:, ch_idx], times_sorted, reset_month, reset_day
        )
        target_accum_full = calculate_snow_accumulation(
            targets_sorted[:, ch_idx]
        )

        for region_name, mask in region_masks.items():
            if not np.any(mask):
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            pred_yearly_means = []
            pred_yearly_mins = []
            pred_yearly_maxs = []
            target_yearly_means = []
            target_yearly_mins = []
            target_yearly_maxs = []

            for year in unique_years:
                year_mask = years_sorted == year
                if not np.any(year_mask):
                    continue

                pred_region = pred_accum_full[year_mask][:, mask]
                target_region = target_accum_full[year_mask][:, mask]

                if metric == 'max':
                    pred_spatial_max = np.max(pred_region, axis=1)
                    target_spatial_max = np.max(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_max))
                    pred_yearly_mins.append(np.min(pred_spatial_max))
                    pred_yearly_maxs.append(np.max(pred_spatial_max))
                    target_yearly_means.append(np.mean(target_spatial_max))
                    target_yearly_mins.append(np.min(target_spatial_max))
                    target_yearly_maxs.append(np.max(target_spatial_max))
                else:
                    pred_spatial_mean = np.mean(pred_region, axis=1)
                    target_spatial_mean = np.mean(target_region, axis=1)
                    pred_yearly_means.append(np.mean(pred_spatial_mean))
                    pred_yearly_mins.append(np.min(pred_spatial_mean))
                    pred_yearly_maxs.append(np.max(pred_spatial_mean))
                    target_yearly_means.append(np.mean(target_spatial_mean))
                    target_yearly_mins.append(np.min(target_spatial_mean))
                    target_yearly_maxs.append(np.max(target_spatial_mean))

            pred_yearly_means = np.array(pred_yearly_means)
            pred_yearly_mins = np.array(pred_yearly_mins)
            pred_yearly_maxs = np.array(pred_yearly_maxs)
            target_yearly_means = np.array(target_yearly_means)
            target_yearly_mins = np.array(target_yearly_mins)
            target_yearly_maxs = np.array(target_yearly_maxs)

            ax.plot(unique_years, target_yearly_means, 'k-o', linewidth=2, markersize=6,
                   label=target_label_name, alpha=0.8)

            ax.plot(unique_years, pred_yearly_means, 'b--s', linewidth=2, markersize=6,
                   label=prediction_label_name, alpha=0.8)

            corr = np.corrcoef(target_yearly_means, pred_yearly_means)[0, 1] if len(target_yearly_means) > 1 else 0

            # Calculate trends (mm/yr)
            if len(unique_years) > 1:
                target_trend_coef = np.polyfit(unique_years, target_yearly_means, 1)
                pred_trend_coef = np.polyfit(unique_years, pred_yearly_means, 1)
                target_trend = target_trend_coef[0]
                pred_trend = pred_trend_coef[0]
            else:
                target_trend = 0
                pred_trend = 0

            stats_text = f'Corr = {corr:.3f}\nTarget Trend = {target_trend:.2f} mm/yr\nPred Trend = {pred_trend:.2f} mm/yr'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='#8B7355', linewidth=1.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')

            metric_label = {
                'max': 'Yearly Mean of Spatial Max',
                'sum': 'Yearly Mean of Spatial Sum',
            }.get(metric, 'Yearly Mean of Spatial Mean')
            ax.set_title(f'{region_name} {metric_label} (Accumulated Snow Yearly Reset)',
                        fontsize='xx-large', fontweight='bold')
            ax.set_xlabel('Year', fontsize='x-large')
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize='x-large')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            plt.tight_layout()
            output_path = output_dir / f'yearly_mean_timeseries_accumulated_yearly_reset_{metric}_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_regional_metrics_bar(predictions, targets, output_dir, region_defs, target_names=None):
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    region_names = list(region_defs.keys())

    for ch in range(n_channels):
        maes = []
        rmses = []
        mean_yearly_maxs_pred = []

        valid_regions = []

        for name in region_names:
            bounds = region_defs[name]
            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
            if not np.any(mask): continue

            valid_regions.append(name)

            pred = predictions[:, ch][:, mask]
            targ = targets[:, ch][:, mask]

            mae = np.mean(np.abs(pred - targ))
            maes.append(mae)

            rmse = np.sqrt(np.mean((pred - targ)**2))
            rmses.append(rmse)

        if not valid_regions: continue

        plot_bar_chart(valid_regions, maes, output_dir / f'region_metrics_mae_{target_names[ch]}.png',
                      f'Regional MAE - {target_names[ch]}', 'MAE')
        plot_bar_chart(valid_regions, rmses, output_dir / f'region_metrics_rmse_{target_names[ch]}.png',
                      f'Regional RMSE - {target_names[ch]}', 'RMSE')


def plot_regional_yearly_max_bar(predictions, targets, times, output_dir, region_defs, target_names=None):
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    region_names = list(region_defs.keys())

    for ch in range(n_channels):
        region_vals_pred = []
        region_vals_targ = []
        valid_regions = []

        for name in region_names:
            bounds = region_defs[name]
            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
            if not np.any(mask): continue

            valid_regions.append(name)

            pred_maxs = []
            targ_maxs = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue
                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                pred_maxs.append(np.max(p))
                targ_maxs.append(np.max(t))

            region_vals_pred.append(np.mean(pred_maxs))
            region_vals_targ.append(np.mean(targ_maxs))

        if not valid_regions: continue

        x = np.arange(len(valid_regions))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_regions)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        rects1 = ax0.bar(x - width/2, region_vals_targ, width, label='Target')
        rects2 = ax0.bar(x + width/2, region_vals_pred, width, label='Prediction')

        ax0.set_ylabel('Snow Amount (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Maximum by Region - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        reg_targ_arr = np.array(region_vals_targ)
        reg_pred_arr = np.array(region_vals_pred)
        diff_pct = np.zeros_like(reg_targ_arr)
        mask_nz = reg_targ_arr != 0
        diff_pct[mask_nz] = (reg_pred_arr[mask_nz] - reg_targ_arr[mask_nz]) / reg_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        bars = ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_regions, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'region_yearly_max_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_regional_yearly_sum_bar(predictions, targets, times, output_dir, region_defs, target_names=None):
    """Create bar chart comparing the mean yearly PEAK SPATIAL SUM across regions.

    For each year, the spatial sum over the region's pixels is computed per
    timestep and the year's peak (max-over-time of that spatial sum) is taken;
    these per-year peaks are then averaged. Parallels the yearly-max bar but on
    the regional snow VOLUME (sum of mm over pixels) instead of single-pixel max.
    """
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    region_names = list(region_defs.keys())

    for ch in range(n_channels):
        region_vals_pred = []
        region_vals_targ = []
        valid_regions = []

        for name in region_names:
            bounds = region_defs[name]
            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
            if not np.any(mask): continue

            valid_regions.append(name)

            pred_sums = []
            targ_sums = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]   # (days, pix)
                t = targets[y_mask, ch][:, mask]

                # Peak spatial sum (snow volume) for the region in that year
                pred_sums.append(np.max(p.sum(axis=1)))
                targ_sums.append(np.max(t.sum(axis=1)))

            region_vals_pred.append(np.mean(pred_sums))
            region_vals_targ.append(np.mean(targ_sums))

        if not valid_regions: continue

        x = np.arange(len(valid_regions))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_regions)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        ax0.bar(x - width/2, region_vals_targ, width, label='Target')
        ax0.bar(x + width/2, region_vals_pred, width, label='Prediction')
        ax0.set_ylabel('Snow Sum (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Peak Spatial Sum by Region - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        reg_targ_arr = np.array(region_vals_targ)
        reg_pred_arr = np.array(region_vals_pred)
        diff_pct = np.zeros_like(reg_targ_arr)
        mask_nz = reg_targ_arr != 0
        diff_pct[mask_nz] = (reg_pred_arr[mask_nz] - reg_targ_arr[mask_nz]) / reg_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_regions, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'region_yearly_sum_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_regional_yearly_mae_bar(predictions, targets, times, output_dir, region_defs, target_names=None):
    """Create bar chart comparing yearly MAE across regions."""
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    region_names = list(region_defs.keys())

    for ch in range(n_channels):
        region_vals_pred = []
        region_vals_targ = []
        valid_regions = []

        for name in region_names:
            bounds = region_defs[name]
            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
            if not np.any(mask): continue

            valid_regions.append(name)

            pred_means = []
            targ_means = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                pred_means.append(np.mean(p))
                targ_means.append(np.mean(t))

            region_vals_pred.append(np.mean(pred_means))
            region_vals_targ.append(np.mean(targ_means))

        if not valid_regions: continue

        x = np.arange(len(valid_regions))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_regions)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        rects1 = ax0.bar(x - width/2, region_vals_targ, width, label='Target')
        rects2 = ax0.bar(x + width/2, region_vals_pred, width, label='Prediction')

        ax0.set_ylabel('Snow Amount (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Mean by Region - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        reg_targ_arr = np.array(region_vals_targ)
        reg_pred_arr = np.array(region_vals_pred)
        diff_pct = np.zeros_like(reg_targ_arr)
        mask_nz = reg_targ_arr != 0
        diff_pct[mask_nz] = (reg_pred_arr[mask_nz] - reg_targ_arr[mask_nz]) / reg_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        bars = ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_regions, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'region_yearly_mae_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_state_metrics_bar(predictions, targets, output_dir, state_abbrevs, target_names=None):
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        maes = []
        rmses = []
        valid_states = []

        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask): continue

            valid_states.append(state)

            pred = predictions[:, ch][:, mask]
            targ = targets[:, ch][:, mask]

            mae = np.mean(np.abs(pred - targ))
            maes.append(mae)

            rmse = np.sqrt(np.mean((pred - targ)**2))
            rmses.append(rmse)

        if not valid_states: continue

        plot_bar_chart(valid_states, maes, output_dir / f'state_metrics_comparison_mae_{target_names[ch]}.png',
                      f'State MAE - {target_names[ch]}', 'MAE')
        plot_bar_chart(valid_states, rmses, output_dir / f'state_metrics_comparison_rmse_{target_names[ch]}.png',
                      f'State RMSE - {target_names[ch]}', 'RMSE')


def plot_state_yearly_max_bar(predictions, targets, times, output_dir, state_abbrevs, target_names=None):
    """Create bar chart comparing yearly max across states."""
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        state_vals_pred = []
        state_vals_targ = []
        valid_states = []

        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask): continue

            valid_states.append(state)

            pred_maxs = []
            targ_maxs = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                pred_maxs.append(np.max(p))
                targ_maxs.append(np.max(t))

            state_vals_pred.append(np.mean(pred_maxs))
            state_vals_targ.append(np.mean(targ_maxs))

        if not valid_states: continue

        x = np.arange(len(valid_states))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_states)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        rects1 = ax0.bar(x - width/2, state_vals_targ, width, label='Target')
        rects2 = ax0.bar(x + width/2, state_vals_pred, width, label='Prediction')

        ax0.set_ylabel('Snow Amount (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Maximum by State - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        state_targ_arr = np.array(state_vals_targ)
        state_pred_arr = np.array(state_vals_pred)
        diff_pct = np.zeros_like(state_targ_arr)
        mask_nz = state_targ_arr != 0
        diff_pct[mask_nz] = (state_pred_arr[mask_nz] - state_targ_arr[mask_nz]) / state_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        bars = ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_states, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'state_yearly_max_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_state_yearly_sum_bar(predictions, targets, times, output_dir, state_abbrevs, target_names=None):
    """Create bar chart comparing the mean yearly PEAK SPATIAL SUM across states.

    For each year, the spatial sum over the state's pixels is computed per
    timestep and the year's peak (max-over-time of that spatial sum) is taken;
    these per-year peaks are then averaged. Parallels the yearly-max bar but on
    the state snow VOLUME (sum of mm over pixels) instead of single-pixel max.
    """
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        state_vals_pred = []
        state_vals_targ = []
        valid_states = []

        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask): continue

            valid_states.append(state)

            pred_sums = []
            targ_sums = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]   # (days, pix)
                t = targets[y_mask, ch][:, mask]

                # Peak spatial sum (snow volume) for the state in that year
                pred_sums.append(np.max(p.sum(axis=1)))
                targ_sums.append(np.max(t.sum(axis=1)))

            state_vals_pred.append(np.mean(pred_sums))
            state_vals_targ.append(np.mean(targ_sums))

        if not valid_states: continue

        x = np.arange(len(valid_states))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_states)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        ax0.bar(x - width/2, state_vals_targ, width, label='Target')
        ax0.bar(x + width/2, state_vals_pred, width, label='Prediction')
        ax0.set_ylabel('Snow Sum (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Peak Spatial Sum by State - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        state_targ_arr = np.array(state_vals_targ)
        state_pred_arr = np.array(state_vals_pred)
        diff_pct = np.zeros_like(state_targ_arr)
        mask_nz = state_targ_arr != 0
        diff_pct[mask_nz] = (state_pred_arr[mask_nz] - state_targ_arr[mask_nz]) / state_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_states, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'state_yearly_sum_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_state_yearly_mae_bar(predictions, targets, times, output_dir, state_abbrevs, target_names=None):
    """Create bar chart comparing yearly MAE across states."""
    years = get_years(times)
    if years is None: return
    unique_years = np.unique(years)
    output_dir = Path(output_dir) / "states"
    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = get_lat_lon()
    if lat is None: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        state_vals_pred = []
        state_vals_targ = []
        valid_states = []

        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask): continue

            valid_states.append(state)

            pred_means = []
            targ_means = []

            for y in unique_years:
                y_mask = years == y
                if not np.any(y_mask): continue

                p = predictions[y_mask, ch][:, mask]
                t = targets[y_mask, ch][:, mask]

                pred_means.append(np.mean(p))
                targ_means.append(np.mean(t))

            state_vals_pred.append(np.mean(pred_means))
            state_vals_targ.append(np.mean(targ_means))

        if not valid_states: continue

        x = np.arange(len(valid_states))
        width = 0.35

        fig = plt.figure(figsize=(max(8, len(valid_states)*1), 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax0 = plt.subplot(gs[0])
        rects1 = ax0.bar(x - width/2, state_vals_targ, width, label='Target')
        rects2 = ax0.bar(x + width/2, state_vals_pred, width, label='Prediction')

        ax0.set_ylabel('Snow Amount (mm)', fontsize=12)
        ax0.set_title(f'Mean Yearly Mean by State - {target_names[ch]}', fontsize=14, fontweight='bold')
        ax0.set_xticks(x)
        ax0.set_xticklabels([])
        ax0.legend()
        ax0.grid(axis='y', linestyle='--', alpha=0.5)

        state_targ_arr = np.array(state_vals_targ)
        state_pred_arr = np.array(state_vals_pred)
        diff_pct = np.zeros_like(state_targ_arr)
        mask_nz = state_targ_arr != 0
        diff_pct[mask_nz] = (state_pred_arr[mask_nz] - state_targ_arr[mask_nz]) / state_targ_arr[mask_nz] * 100

        ax1 = plt.subplot(gs[1])
        bars = ax1.bar(x, diff_pct, color=['red' if d > 0 else 'blue' for d in diff_pct])
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylabel('Diff (%)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_states, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f'state_yearly_mae_bar_{target_names[ch]}.png', dpi=FIGURE_DPI)
        plt.close()


def plot_seasonality_comparison_regions(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    target_names: list = None,
    metric: str = 'mean',
):
    """
    Create four-panel seasonality comparison plot for regions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    month_map, month_order = _build_month_index(times)
    if not month_map: return

    n_channels = predictions.shape[1]
    if target_names is None: target_names = [f"Ch{i}" for i in range(n_channels)]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_positions = np.arange(len(month_order))

    lat, lon = get_lat_lon()
    if lat is None: return

    region_masks = {}
    for region_name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        region_masks[region_name] = mask

    for ch in range(n_channels):
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        for ax_idx, (region_name, mask) in enumerate(region_masks.items()):
            if ax_idx >= len(axes): break
            ax = axes[ax_idx]

            if not np.any(mask):
                continue

            vals_pred = []
            stds_pred = []
            vals_targ = []
            stds_targ = []

            for month in month_order:
                month_indices = month_map[month]
                if len(month_indices) == 0:
                    vals_pred.append(0)
                    stds_pred.append(0)
                    vals_targ.append(0)
                    stds_targ.append(0)
                    continue

                p = predictions[month_indices, ch][:, mask]
                t = targets[month_indices, ch][:, mask]

                if metric == 'max':
                    p_spatial = np.max(p, axis=1)
                    t_spatial = np.max(t, axis=1)
                else:
                    p_spatial = np.mean(p, axis=1)
                    t_spatial = np.mean(t, axis=1)

                vals_pred.append(np.mean(p_spatial))
                stds_pred.append(np.std(p_spatial))
                vals_targ.append(np.mean(t_spatial))
                stds_targ.append(np.std(t_spatial))

            vals_pred = np.array(vals_pred)
            stds_pred = np.array(stds_pred)
            vals_targ = np.array(vals_targ)
            stds_targ = np.array(stds_targ)

            ax.plot(x_positions, vals_targ, 'k-', linewidth=2, label='Target')
            ax.fill_between(x_positions, vals_targ - stds_targ, vals_targ + stds_targ, color='k', alpha=0.2)

            ax.plot(x_positions, vals_pred, 'b-', linewidth=2, label='Prediction')
            ax.fill_between(x_positions, vals_pred - stds_pred, vals_pred + stds_pred, color='b', alpha=0.2)

            metric_label = 'Spatial Max' if metric == 'max' else 'Spatial Mean'
            ax.set_title(panel_label(ax_idx, f'{region_name} {metric_label} Seasonality'),
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([month_names[m-1] for m in month_order])
            ax.set_ylabel('Snow Water Equivalent (mm)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Regional Seasonality Comparison - {target_names[ch]}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_path = output_dir / f'seasonality_comparison_{metric}_{target_names[ch]}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_day_of_max_snow(predictions, targets, times, output_dir, target_names=None):
    """
    Plot the day of year when maximum snow occurs for each year.
    Creates spatial maps comparing target vs prediction for when peak snow happens.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of shape (n_samples, 2) with (year, time_idx)
        output_dir: Directory to save plots
        target_names: Names of target variables (default: ["Ch0", "Ch1", ...])
    """
    output_dir = Path(output_dir) / "day_of_max_snow"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = get_years(times)
    if years is None:
        print("  Warning: No time information available, skipping day of max snow plots")
        return

    unique_years = np.unique(years)
    lat, lon = get_lat_lon()

    datetimes = get_datetimes_from_data(times)
    if datetimes is None:
        print("  Warning: Could not convert times to datetimes, skipping day of max snow plots")
        return

    def get_day_of_year(dt):
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.timetuple().tm_yday
    day_of_year = np.array([get_day_of_year(dt) for dt in datetimes])

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        print(f"  Processing {target_names[ch]}...")

        for year in unique_years:
            year_mask = years == year
            if not np.any(year_mask):
                continue

            year_preds = predictions[year_mask, ch]  # (n_times, height, width)
            year_targs = targets[year_mask, ch]      # (n_times, height, width)
            year_doys = day_of_year[year_mask]        # (n_times,)

            if len(year_doys) < 2:
                print(f"    Skipping year {year} (insufficient data)")
                continue

            # Shape: (height, width)
            pred_max_idx = np.argmax(year_preds, axis=0)
            targ_max_idx = np.argmax(year_targs, axis=0)

            # Use advanced indexing to get DOY for each pixel's max time
            pred_doy_max = year_doys[pred_max_idx]
            targ_doy_max = year_doys[targ_max_idx]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                    subplot_kw={'projection': ccrs.PlateCarree()})

            doy_vmin = 1
            doy_vmax = 366
            doy_cmap = 'twilight_shifted'  # Circular colormap for seasonal data

            plot_map(targ_doy_max, lat, lon, axes[0],
                    f'Target - Day of Max Snow {year}',
                    vmin=doy_vmin, vmax=doy_vmax, cmap=doy_cmap,
                    draw_gridlines=True, extend='neither')

            plot_map(pred_doy_max, lat, lon, axes[1],
                    f'Prediction - Day of Max Snow {year}',
                    vmin=doy_vmin, vmax=doy_vmax, cmap=doy_cmap,
                    draw_gridlines=True, extend='neither')

            doy_diff = pred_doy_max - targ_doy_max

            # Handle circular nature of DOY (e.g., DOY 365 vs DOY 1)
            # If difference > 180 days, it's likely wrapping around the year
            doy_diff = np.where(doy_diff > 183, doy_diff - 365, doy_diff)
            doy_diff = np.where(doy_diff < -183, doy_diff + 365, doy_diff)

            plot_map(doy_diff, lat, lon, axes[2],
                    'Difference (days)',
                    vmin=-60, vmax=60, cmap='RdBu_r',
                    draw_gridlines=True, extend='both')

            plt.suptitle(f'{target_names[ch]} - Day of Year with Maximum Snow ({year})',
                        fontsize='xx-large', fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'doy_max_{year}_{target_names[ch]}.png',
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()

        print(f"  Creating overall scatter plot for {target_names[ch]}...")

        all_pred_doy = []
        all_targ_doy = []

        for year in unique_years:
            year_mask = years == year
            if not np.any(year_mask):
                continue

            year_preds = predictions[year_mask, ch]
            year_targs = targets[year_mask, ch]
            year_doys = day_of_year[year_mask]

            if len(year_doys) < 2:
                continue

            pred_max_idx = np.argmax(year_preds, axis=0)
            targ_max_idx = np.argmax(year_targs, axis=0)

            pred_doy_max = year_doys[pred_max_idx]
            targ_doy_max = year_doys[targ_max_idx]

            all_pred_doy.append(pred_doy_max.flatten())
            all_targ_doy.append(targ_doy_max.flatten())

        if all_pred_doy:
            all_pred_doy = np.concatenate(all_pred_doy)
            all_targ_doy = np.concatenate(all_targ_doy)

            fig, ax = plt.subplots(figsize=(8, 8))

            # Use hexbin for density visualization
            hexbin = ax.hexbin(all_targ_doy, all_pred_doy,
                              gridsize=50, cmap='Blues', mincnt=1,
                              alpha=0.8, edgecolors='none')
            plt.colorbar(hexbin, ax=ax, label='Count')

            ax.plot([1, 366], [1, 366], 'r--', linewidth=2, label='1:1 line')

            doy_diff_all = all_pred_doy - all_targ_doy
            doy_diff_all = np.where(doy_diff_all > 183, doy_diff_all - 365, doy_diff_all)
            doy_diff_all = np.where(doy_diff_all < -183, doy_diff_all + 365, doy_diff_all)
            mae_days = np.mean(np.abs(doy_diff_all))

            ax.set_xlabel('Target - Day of Year', fontsize=12)
            ax.set_ylabel('Prediction - Day of Year', fontsize=12)
            ax.set_title(f'{target_names[ch]} - Day of Maximum Snow\nMAE = {mae_days:.1f} days',
                        fontsize=14, fontweight='bold')
            ax.set_xlim(1, 366)
            ax.set_ylim(1, 366)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(output_dir / f'doy_max_scatter_{target_names[ch]}.png',
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()

            print(f"    MAE: {mae_days:.1f} days")

    print(f"  Saved plots to: {output_dir}")


def plot_regional_day_of_max_time_series(predictions, targets, times, output_dir, region_defs, target_names=None):
    """
    Plot time series of the day of year when maximum regional mean snow occurs.
    For each region, compute the spatial mean, then for each year find when this mean peaks.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of shape (n_samples, 2) with (year, time_idx)
        output_dir: Directory to save plots
        region_defs: Dictionary of region definitions with lat/lon bounds
        target_names: Names of target variables (default: ["Ch0", "Ch1", ...])
    """
    output_dir = Path(output_dir) / "regional_day_of_max_time_series"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = get_years(times)
    if years is None:
        print("  Warning: No time information available, skipping regional day of max plots")
        return

    unique_years = np.unique(years)
    lat, lon = get_lat_lon()
    if lat is None:
        print("  Warning: No lat/lon coordinates available, skipping regional day of max plots")
        return

    datetimes = get_datetimes_from_data(times)
    if datetimes is None:
        print("  Warning: Could not convert times to datetimes, skipping regional day of max plots")
        return

    def get_day_of_year(dt):
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.timetuple().tm_yday
    day_of_year = np.array([get_day_of_year(dt) for dt in datetimes])

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        print(f"  Processing {target_names[ch]}...")

        n_regions = len(region_defs)
        fig, axes = plt.subplots(n_regions, 1, figsize=(14, 4 * n_regions))
        if n_regions == 1:
            axes = [axes]

        for ax_idx, (region_name, bounds) in enumerate(region_defs.items()):
            print(f"    Processing region: {region_name}")

            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])

            if not np.any(mask):
                print(f"      Warning: No pixels in region {region_name}, skipping")
                continue

            pred_doy_max_years = []
            targ_doy_max_years = []
            year_labels = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                year_preds = predictions[year_mask, ch]  # (n_times, height, width)
                year_targs = targets[year_mask, ch]      # (n_times, height, width)
                year_doys = day_of_year[year_mask]        # (n_times,)

                if len(year_doys) < 2:
                    continue

                # Shape: (n_times,)
                pred_regional_mean = np.array([np.mean(year_preds[i][mask]) for i in range(len(year_preds))])
                targ_regional_mean = np.array([np.mean(year_targs[i][mask]) for i in range(len(year_targs))])

                pred_max_idx = np.argmax(pred_regional_mean)
                targ_max_idx = np.argmax(targ_regional_mean)

                pred_doy_max = year_doys[pred_max_idx]
                targ_doy_max = year_doys[targ_max_idx]

                pred_doy_max_years.append(pred_doy_max)
                targ_doy_max_years.append(targ_doy_max)
                year_labels.append(year)

            if len(year_labels) == 0:
                print(f"      Warning: No valid years for region {region_name}")
                continue

            pred_doy_max_years = np.array(pred_doy_max_years)
            targ_doy_max_years = np.array(targ_doy_max_years)
            year_labels = np.array(year_labels)

            ax = axes[ax_idx]
            ax.plot(year_labels, targ_doy_max_years, 'ko-', linewidth=2, markersize=6, label='Target', alpha=0.7)
            ax.plot(year_labels, pred_doy_max_years, 'bo--', linewidth=2, markersize=6, label='Prediction', alpha=0.7)

            doy_diff = pred_doy_max_years - targ_doy_max_years
            # Handle circular nature of DOY
            doy_diff = np.where(doy_diff > 183, doy_diff - 365, doy_diff)
            doy_diff = np.where(doy_diff < -183, doy_diff + 365, doy_diff)
            mae_days = np.mean(np.abs(doy_diff))
            bias_days = np.mean(doy_diff)

            ax.set_title(f'{region_name} - Day of Maximum Regional Mean\nMAE = {mae_days:.1f} days, Bias = {bias_days:.1f} days',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Day of Year', fontsize=11)
            ax.set_ylim(1, 366)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # Approx day of year for each month
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax2 = ax.twinx()
            ax2.set_ylim(1, 366)
            ax2.set_yticks(month_days)
            ax2.set_yticklabels(month_names, fontsize=9)
            ax2.set_ylabel('Month', fontsize=11)

        plt.suptitle(f'{target_names[ch]} - Regional Day of Maximum Snow Time Series',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'regional_doy_max_timeseries_{target_names[ch]}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved plots to: {output_dir}")


def plot_state_day_of_max_time_series(predictions, targets, times, output_dir, state_abbrevs, target_names=None):
    """
    Plot time series of the day of year when maximum state mean snow occurs.
    For each state, compute the spatial mean, then for each year find when this mean peaks.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of shape (n_samples, 2) with (year, time_idx)
        output_dir: Directory to save plots
        state_abbrevs: List of state abbreviations
        target_names: Names of target variables (default: ["Ch0", "Ch1", ...])
    """
    output_dir = Path(output_dir) / "state_day_of_max_time_series"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = get_years(times)
    if years is None:
        print("  Warning: No time information available, skipping state day of max plots")
        return

    unique_years = np.unique(years)
    lat, lon = get_lat_lon()
    if lat is None:
        print("  Warning: No lat/lon coordinates available, skipping state day of max plots")
        return

    datetimes = get_datetimes_from_data(times)
    if datetimes is None:
        print("  Warning: Could not convert times to datetimes, skipping state day of max plots")
        return

    def get_day_of_year(dt):
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.timetuple().tm_yday
    day_of_year = np.array([get_day_of_year(dt) for dt in datetimes])

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        print(f"  Processing {target_names[ch]}...")

        n_states = len(state_abbrevs)
        fig, axes = plt.subplots(n_states, 1, figsize=(14, 4 * n_states))
        if n_states == 1:
            axes = [axes]

        for ax_idx, state in enumerate(state_abbrevs):
            print(f"    Processing state: {state}")

            mask = get_state_mask(state, lat, lon)

            if mask is None or not np.any(mask):
                print(f"      Warning: No pixels in state {state}, skipping")
                continue

            pred_doy_max_years = []
            targ_doy_max_years = []
            year_labels = []

            for year in unique_years:
                year_mask = years == year
                if not np.any(year_mask):
                    continue

                year_preds = predictions[year_mask, ch]  # (n_times, height, width)
                year_targs = targets[year_mask, ch]      # (n_times, height, width)
                year_doys = day_of_year[year_mask]        # (n_times,)

                if len(year_doys) < 2:
                    continue

                # Shape: (n_times,)
                pred_state_mean = np.array([np.mean(year_preds[i][mask]) for i in range(len(year_preds))])
                targ_state_mean = np.array([np.mean(year_targs[i][mask]) for i in range(len(year_targs))])

                pred_max_idx = np.argmax(pred_state_mean)
                targ_max_idx = np.argmax(targ_state_mean)

                pred_doy_max = year_doys[pred_max_idx]
                targ_doy_max = year_doys[targ_max_idx]

                pred_doy_max_years.append(pred_doy_max)
                targ_doy_max_years.append(targ_doy_max)
                year_labels.append(year)

            if len(year_labels) == 0:
                print(f"      Warning: No valid years for state {state}")
                continue

            pred_doy_max_years = np.array(pred_doy_max_years)
            targ_doy_max_years = np.array(targ_doy_max_years)
            year_labels = np.array(year_labels)

            ax = axes[ax_idx]
            ax.plot(year_labels, targ_doy_max_years, 'ko-', linewidth=2, markersize=6, label='Target', alpha=0.7)
            ax.plot(year_labels, pred_doy_max_years, 'bo--', linewidth=2, markersize=6, label='Prediction', alpha=0.7)

            doy_diff = pred_doy_max_years - targ_doy_max_years
            # Handle circular nature of DOY
            doy_diff = np.where(doy_diff > 183, doy_diff - 365, doy_diff)
            doy_diff = np.where(doy_diff < -183, doy_diff + 365, doy_diff)
            mae_days = np.mean(np.abs(doy_diff))
            bias_days = np.mean(doy_diff)

            ax.set_title(f'{state} - Day of Maximum State Mean\nMAE = {mae_days:.1f} days, Bias = {bias_days:.1f} days',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Day of Year', fontsize=11)
            ax.set_ylim(1, 366)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # Approx day of year for each month
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax2 = ax.twinx()
            ax2.set_ylim(1, 366)
            ax2.set_yticks(month_days)
            ax2.set_yticklabels(month_names, fontsize=9)
            ax2.set_ylabel('Month', fontsize=11)

        plt.suptitle(f'{target_names[ch]} - State Day of Maximum Snow Time Series',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'state_doy_max_timeseries_{target_names[ch]}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved plots to: {output_dir}")


# ============================================================================
# EVALUATION PLOTTING FUNCTIONS
# ============================================================================

def plot_evaluation_time_series(evaluator, output_dir, target_names=None):
    """
    Plot time series of evaluation metrics over time.

    Args:
        evaluator: Evaluator object with collected predictions
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_metrics = evaluator.evaluate_by_time_series()

    if not time_metrics:
        print("No time series data available for plotting")
        return

    try:
        from config import TARGET_VARS
    except ImportError:
        TARGET_VARS = ['snow']

    if target_names is None:
        target_names = TARGET_VARS

    periods = sorted(time_metrics.keys())

    metric_names = ['rmse', 'mae', 'r2', 'correlation', 'bias']
    metric_labels = {'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²',
                     'correlation': 'Correlation', 'bias': 'Bias'}

    for var_idx, var_name in enumerate(TARGET_VARS):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        for metric_idx, metric in enumerate(metric_names):
            ax = axes[metric_idx]

            value_metric = []
            change_metric = []
            for period in periods:
                if var_name in time_metrics[period]:
                    value_metric.append(time_metrics[period][var_name].get(metric, np.nan))
                else:
                    value_metric.append(np.nan)

                change_var = f"{var_name}_change"
                if change_var in time_metrics[period]:
                    change_metric.append(time_metrics[period][change_var].get(metric, np.nan))
                else:
                    change_metric.append(np.nan)

            x = np.arange(len(periods))
            ax.plot(x, value_metric, 'o-', linewidth=2, markersize=6,
                   label='Values', color='blue', alpha=0.7)
            ax.plot(x, change_metric, 's--', linewidth=2, markersize=6,
                   label='Changes', color='red', alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(periods, rotation=45, ha='right')
            ax.set_xlabel('Time Period', fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_title(f'{metric_labels[metric]} Over Time', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].axis('off')

        plt.suptitle(f'{target_names[var_idx]} - Evaluation Metrics Time Series',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'time_series_metrics_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved time series plots to {output_dir}")


def plot_evaluation_spatial(evaluator, output_dir, target_names=None, grid_size=(5, 5)):
    """
    Plot spatial heatmaps of evaluation metrics.

    Args:
        evaluator: Evaluator object with collected predictions
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
        grid_size: Tuple of (n_rows, n_cols) for spatial grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_metrics = evaluator.evaluate_spatial(grid_size=grid_size)

    if not spatial_metrics:
        print("No spatial data available for plotting")
        return

    try:
        from config import TARGET_VARS
    except ImportError:
        TARGET_VARS = ['snow']

    if target_names is None:
        target_names = TARGET_VARS

    n_rows, n_cols = grid_size

    metrics_to_plot = ['rmse', 'mae', 'r2', 'bias']
    metric_labels = {'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²', 'bias': 'Bias'}

    for var_idx, var_name in enumerate(TARGET_VARS):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for metric_idx, metric in enumerate(metrics_to_plot):
            heatmap_values = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                for j in range(n_cols):
                    if (i, j) in spatial_metrics and var_name in spatial_metrics[(i, j)]:
                        heatmap_values[i, j] = spatial_metrics[(i, j)][var_name].get(metric, np.nan)
                    else:
                        heatmap_values[i, j] = np.nan

            ax = axes[metric_idx]
            hm_cmap, hm_norm, hm_levels = discretize_cmap(
                'RdYlGn_r' if metric != 'r2' else 'RdYlGn',
                np.nanmin(heatmap_values), np.nanmax(heatmap_values))
            im = ax.imshow(heatmap_values, cmap=hm_cmap, norm=hm_norm,
                          aspect='auto', interpolation='nearest')

            cbar = plt.colorbar(im, ax=ax, spacing='uniform')
            if hm_levels is not None:
                cbar.set_ticks(colorbar_ticks(hm_levels))
            cbar.set_label(metric_labels[metric], fontsize=10)

            for i in range(n_rows):
                for j in range(n_cols):
                    if not np.isnan(heatmap_values[i, j]):
                        text = ax.text(j, i, f'{heatmap_values[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)

            ax.set_xlabel('Column', fontsize=10)
            ax.set_ylabel('Row', fontsize=10)
            ax.set_title(f'{metric_labels[metric]} - Values', fontsize=11, fontweight='bold')
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))

        plt.suptitle(f'{target_names[var_idx]} - Spatial Distribution of Metrics (Values)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'spatial_metrics_values_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        change_var = f"{var_name}_change"
        for metric_idx, metric in enumerate(metrics_to_plot):
            heatmap_changes = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                for j in range(n_cols):
                    if (i, j) in spatial_metrics and change_var in spatial_metrics[(i, j)]:
                        heatmap_changes[i, j] = spatial_metrics[(i, j)][change_var].get(metric, np.nan)
                    else:
                        heatmap_changes[i, j] = np.nan

            ax = axes[metric_idx]
            hm_cmap, hm_norm, hm_levels = discretize_cmap(
                'RdYlGn_r' if metric != 'r2' else 'RdYlGn',
                np.nanmin(heatmap_changes), np.nanmax(heatmap_changes))
            im = ax.imshow(heatmap_changes, cmap=hm_cmap, norm=hm_norm,
                          aspect='auto', interpolation='nearest')

            cbar = plt.colorbar(im, ax=ax, spacing='uniform')
            if hm_levels is not None:
                cbar.set_ticks(colorbar_ticks(hm_levels))
            cbar.set_label(metric_labels[metric], fontsize=10)

            for i in range(n_rows):
                for j in range(n_cols):
                    if not np.isnan(heatmap_changes[i, j]):
                        text = ax.text(j, i, f'{heatmap_changes[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)

            ax.set_xlabel('Column', fontsize=10)
            ax.set_ylabel('Row', fontsize=10)
            ax.set_title(f'{metric_labels[metric]} - Changes', fontsize=11, fontweight='bold')
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))

        plt.suptitle(f'{target_names[var_idx]} - Spatial Distribution of Metrics (Changes)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'spatial_metrics_changes_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved spatial heatmap plots to {output_dir}")


def plot_evaluation_regions(evaluator, output_dir, target_names=None):
    """
    Plot regional comparison of evaluation metrics.

    Args:
        evaluator: Evaluator object with collected predictions and regions
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regional_metrics = evaluator.evaluate_by_region()

    if not regional_metrics:
        print("No regional data available for plotting (no regions defined)")
        return

    try:
        from config import TARGET_VARS
    except ImportError:
        TARGET_VARS = ['snow']

    if target_names is None:
        target_names = TARGET_VARS

    regions = list(regional_metrics.keys())
    metrics_to_plot = ['rmse', 'mae', 'r2', 'correlation']
    metric_labels = {'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²', 'correlation': 'Correlation'}

    for var_idx, var_name in enumerate(TARGET_VARS):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for metric_idx, metric in enumerate(metrics_to_plot):
            ax = axes[metric_idx]

            value_metrics = []
            change_metrics = []

            for region in regions:
                if var_name in regional_metrics[region]['values']:
                    value_metrics.append(regional_metrics[region]['values'][var_name].get(metric, np.nan))
                else:
                    value_metrics.append(np.nan)

                change_var = f"{var_name}_change"
                if change_var in regional_metrics[region]['changes']:
                    change_metrics.append(regional_metrics[region]['changes'][change_var].get(metric, np.nan))
                else:
                    change_metrics.append(np.nan)

            x = np.arange(len(regions))
            width = 0.35

            ax.bar(x - width/2, value_metrics, width, label='Values', color='blue', alpha=0.7)
            ax.bar(x + width/2, change_metrics, width, label='Changes', color='red', alpha=0.7)

            ax.set_xlabel('Region', fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_title(f'{metric_labels[metric]} by Region', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{target_names[var_idx]} - Regional Comparison of Metrics',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'regional_metrics_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved regional comparison plots to {output_dir}")


def plot_evaluation_states(evaluator, state_masks, output_dir, target_names=None):
    """
    Plot state-based comparison of evaluation metrics.

    Args:
        evaluator: Evaluator object with collected predictions
        state_masks: Dict of {state_name: mask_array}
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_metrics = evaluator.evaluate_by_states(state_masks)

    if not state_metrics:
        print("No state data available for plotting")
        return

    try:
        from config import TARGET_VARS
    except ImportError:
        TARGET_VARS = ['snow']

    if target_names is None:
        target_names = TARGET_VARS

    states = list(state_metrics.keys())
    metrics_to_plot = ['rmse', 'mae', 'r2', 'correlation', 'bias']
    metric_labels = {'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²',
                     'correlation': 'Correlation', 'bias': 'Bias'}

    for var_idx, var_name in enumerate(TARGET_VARS):
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()

        for metric_idx, metric in enumerate(metrics_to_plot):
            ax = axes[metric_idx]

            value_metrics = []
            change_metrics = []

            for state in states:
                if var_name in state_metrics[state]['values']:
                    value_metrics.append(state_metrics[state]['values'][var_name].get(metric, np.nan))
                else:
                    value_metrics.append(np.nan)

                change_var = f"{var_name}_change"
                if change_var in state_metrics[state]['changes']:
                    change_metrics.append(state_metrics[state]['changes'][change_var].get(metric, np.nan))
                else:
                    change_metrics.append(np.nan)

            x = np.arange(len(states))
            width = 0.35

            ax.bar(x - width/2, value_metrics, width, label='Values', color='blue', alpha=0.7)
            ax.bar(x + width/2, change_metrics, width, label='Changes', color='red', alpha=0.7)

            ax.set_xlabel('State', fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_title(f'{metric_labels[metric]} by State', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(states, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        axes[-1].axis('off')

        plt.suptitle(f'{target_names[var_idx]} - State Comparison of Metrics',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'state_metrics_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved state comparison plots to {output_dir}")


def plot_evaluation_scatter(evaluator, output_dir, target_names=None, n_samples=10000):
    """
    Plot scatter plots of predictions vs targets for values and changes.

    Args:
        evaluator: Evaluator object with collected predictions
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
        n_samples: Number of samples to plot (for performance)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not evaluator.predictions:
        print("No prediction data available for plotting")
        return

    try:
        from config import TARGET_VARS
    except ImportError:
        TARGET_VARS = ['snow']

    if target_names is None:
        target_names = TARGET_VARS

    all_preds = np.concatenate(evaluator.predictions, axis=0)
    all_targets = np.concatenate(evaluator.targets, axis=0)
    all_changes_pred = np.concatenate(evaluator.changes_pred, axis=0)
    all_changes_true = np.concatenate(evaluator.changes_true, axis=0)

    for var_idx, var_name in enumerate(TARGET_VARS):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        pred_flat = all_preds[:, var_idx].flatten()
        target_flat = all_targets[:, var_idx].flatten()

        if len(pred_flat) > n_samples:
            indices = np.random.choice(len(pred_flat), n_samples, replace=False)
            pred_flat = pred_flat[indices]
            target_flat = target_flat[indices]

        hb = ax.scatter(target_flat, pred_flat)

        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

        try:
            metrics = compute_metrics(pred_flat, target_flat)

            textstr = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}\nCorr: {metrics['correlation']:.4f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass

        ax.set_xlabel('Target Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title('Values: Predictions vs Targets', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=10)

        ax = axes[1]
        change_pred_flat = all_changes_pred[:, var_idx].flatten()
        change_true_flat = all_changes_true[:, var_idx].flatten()

        if len(change_pred_flat) > n_samples:
            indices = np.random.choice(len(change_pred_flat), n_samples, replace=False)
            change_pred_flat = change_pred_flat[indices]
            change_true_flat = change_true_flat[indices]

        hb = ax.scatter(change_true_flat, change_pred_flat)

        min_val = min(change_true_flat.min(), change_pred_flat.min())
        max_val = max(change_true_flat.max(), change_pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')

        try:
            metrics = compute_metrics(change_pred_flat, change_true_flat)

            textstr = f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}\nCorr: {metrics['correlation']:.4f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass

        ax.set_xlabel('Target Changes', fontsize=11)
        ax.set_ylabel('Predicted Changes', fontsize=11)
        ax.set_title('Changes: Predictions vs Targets', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=10)

        plt.suptitle(f'{target_names[var_idx]} - Scatter Plots',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'scatter_{var_name}.png',
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

    print(f"  Saved scatter plots to {output_dir}")


def plot_all_evaluation_results(evaluator, output_dir, target_names=None,
                                state_masks=None, grid_size=(5, 5)):
    """
    Generate all evaluation plots.

    Args:
        evaluator: Evaluator object with collected predictions
        output_dir: Directory to save plots
        target_names: Optional list of target variable display names
        state_masks: Optional dict of {state_name: mask_array}
        grid_size: Tuple of (n_rows, n_cols) for spatial grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 70)

    print("\n1. Generating time series plots...")
    plot_evaluation_time_series(evaluator, output_dir, target_names)

    print("\n2. Generating spatial heatmaps...")
    plot_evaluation_spatial(evaluator, output_dir, target_names, grid_size)

    if evaluator.region_definitions:
        print("\n3. Generating regional comparison plots...")
        plot_evaluation_regions(evaluator, output_dir, target_names)

    if state_masks:
        print("\n4. Generating state comparison plots...")
        plot_evaluation_states(evaluator, state_masks, output_dir, target_names)

    print("\n5. Generating scatter plots...")
    plot_evaluation_scatter(evaluator, output_dir, target_names)

    print("\n" + "=" * 70)
    print(f"ALL PLOTS SAVED TO: {output_dir}")
    print("=" * 70)


def plot_regional_snow_sum_time_series(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'Target',
    prediction_label_name: str = 'Prediction',
):
    """
    Create time series plot of total snow sum for regions.

    The sum is calculated as: sum(snow * 1e-6 * resolution^2) / sum(resolution^2)
    This gives the equivalent depth if all snow were flattened uniformly across the region.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of time information
        output_dir: Directory to save plots
        region_defs: Dictionary of region definitions with lat/lon bounds
        resolution_km: Grid resolution in kilometers
        target_names: Optional list of channel names
        target_label_name: Label for target line in plot
        prediction_label_name: Label for prediction line in plot
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    lat, lon = get_lat_lon()
    if lat is None:
        print("Warning: Could not load lat/lon data")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))
    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])
            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)
    dates = convert_to_plot_dates(dates)

    output_dir = Path(output_dir) / "regional_snow_sum"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating regional snow sum time series (resolution: {resolution_km} km)")

    for region_name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])

        if not np.any(mask):
            print(f"  Warning: No grid points found for region {region_name}")
            continue

        n_pixels = np.sum(mask)

        for ch_idx in range(n_channels):
            fig, ax = plt.subplots(figsize=(14, 6))

            pred_sum = []
            target_sum = []

            for t in range(len(predictions_sorted)):
                pred_region = predictions_sorted[t, ch_idx][mask]
                target_region = targets_sorted[t, ch_idx][mask]

                pred_val = np.sum(pred_region * 1e-6 * resolution_km * resolution_km)
                target_val = np.sum(target_region * 1e-6 * resolution_km * resolution_km)

                pred_sum.append(pred_val)
                target_sum.append(target_val)

            ax.plot(dates, target_sum, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
            ax.plot(dates, pred_sum, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)

            ax.set_title(f'Regional Snow Sum - {region_name} - {target_names[ch_idx]}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Total Snow Water (km^3)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if is_daily:
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            output_path = output_dir / f'snow_sum_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_whole_domain_snow_volume_time_series(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'Target',
    prediction_label_name: str = 'Prediction',
):
    """
    Create time series plot of total snow water volume (km³) for the whole domain.

    Volume = sum(snow_mm * 1e-6 km/mm * resolution_km²) over all grid cells.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of time information
        output_dir: Directory to save plots
        resolution_km: Grid resolution in kilometers
        target_names: Optional list of channel names
        target_label_name: Label for target line in plot
        prediction_label_name: Label for prediction line in plot
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot plot time series properly.")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    if has_season_idx:
        intra_year_indices = times[:, 2]
    else:
        intra_year_indices = times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))
    years_sorted = years[sort_idx]
    intra_sorted = intra_year_indices[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    dates_from_data = get_datetimes_from_data(times)
    is_daily = np.max(intra_sorted) > 20
    if dates_from_data is not None:
        dates = [dates_from_data[i] for i in sort_idx]
    else:
        dates = []
        for i in range(len(years_sorted)):
            y = int(years_sorted[i])
            idx = int(intra_sorted[i])
            if is_daily:
                d = datetime(y, 10, 1) + timedelta(days=idx)
            else:
                target_year = y + idx // 12
                target_month = (idx % 12) + 1
                d = datetime(target_year, target_month, 1)
            dates.append(d)
    dates = convert_to_plot_dates(dates)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_area_km2 = resolution_km * resolution_km
    print(f"\nCreating whole domain snow volume time series (resolution: {resolution_km} km, cell area: {cell_area_km2} km²)")

    unique_years = np.unique(years_sorted)

    for ch_idx in range(n_channels):
        # Volume per timestep (km³)
        pred_vol = np.sum(predictions_sorted[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2
        target_vol = np.sum(targets_sorted[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2

        # --- Full time series ---
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(dates, target_vol, 'k-', linewidth=1.5, label=target_label_name, alpha=0.8)
        ax.plot(dates, pred_vol, 'b--', linewidth=1.5, label=prediction_label_name, alpha=0.8)
        ax.set_title(f'Whole Domain Total Snow Water Volume - {target_names[ch_idx]}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Snow Water Volume (km³)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if is_daily:
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        output_path = output_dir / f'whole_domain_snow_volume_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")

        # --- Yearly mean ---
        pred_yearly_means = []
        target_yearly_means = []
        pred_yearly_stds = []
        target_yearly_stds = []

        for year in unique_years:
            year_mask = years_sorted == year
            pred_yearly_means.append(np.mean(pred_vol[year_mask]))
            target_yearly_means.append(np.mean(target_vol[year_mask]))
            pred_yearly_stds.append(np.std(pred_vol[year_mask]))
            target_yearly_stds.append(np.std(target_vol[year_mask]))

        pred_yearly_means = np.array(pred_yearly_means)
        target_yearly_means = np.array(target_yearly_means)
        pred_yearly_stds = np.array(pred_yearly_stds)
        target_yearly_stds = np.array(target_yearly_stds)

        fig, ax = plt.subplots(figsize=(max(10, len(unique_years) * 0.6), 6))
        x = np.arange(len(unique_years))
        width = 0.35
        ax.bar(x - width / 2, target_yearly_means, width,
               yerr=target_yearly_stds, label=target_label_name,
               color='steelblue', alpha=0.8, capsize=4)
        ax.bar(x + width / 2, pred_yearly_means, width,
               yerr=pred_yearly_stds, label=prediction_label_name,
               color='coral', alpha=0.8, capsize=4)
        ax.set_title(f'Whole Domain Yearly Mean Snow Water Volume - {target_names[ch_idx]}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Mean Snow Water Volume (km³)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(y)) for y in unique_years], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')
        plt.tight_layout()
        output_path = output_dir / f'whole_domain_snow_volume_yearly_mean_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_whole_domain_yearly_total_snow(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'Target',
    prediction_label_name: str = 'Prediction',
):
    """
    Create line plot of total annual snow water volume (km³) for the whole domain.

    For each year, sums the per-timestep snow water volume across all timesteps in that year.
    Volume per timestep = sum(snow_mm * 1e-6 km/mm * resolution_km²) over all grid cells.
    The area between the target and prediction lines is shaded to highlight differences.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of time information
        output_dir: Directory to save plots
        resolution_km: Grid resolution in kilometers
        target_names: Optional list of channel names
        target_label_name: Label for target line in plot
        prediction_label_name: Label for prediction line in plot
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot compute yearly totals.")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    intra_year_indices = times[:, 2] if has_season_idx else times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))
    years_sorted = years[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    cell_area_km2 = resolution_km * resolution_km
    unique_years = np.unique(years_sorted)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating whole domain yearly total snow figure (resolution: {resolution_km} km)")

    for ch_idx in range(n_channels):
        # Volume per timestep (km³)
        pred_vol = np.sum(predictions_sorted[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2
        target_vol = np.sum(targets_sorted[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2

        pred_yearly_totals = []
        target_yearly_totals = []
        for year in unique_years:
            year_mask = years_sorted == year
            pred_yearly_totals.append(np.sum(pred_vol[year_mask]))
            target_yearly_totals.append(np.sum(target_vol[year_mask]))

        pred_yearly_totals = np.array(pred_yearly_totals)
        target_yearly_totals = np.array(target_yearly_totals)

        fig, ax = plt.subplots(figsize=(max(10, len(unique_years) * 0.6), 6))
        x = np.arange(len(unique_years))

        ax.plot(x, target_yearly_totals, 'o-', color='steelblue', linewidth=2,
                markersize=5, label=target_label_name)
        ax.plot(x, pred_yearly_totals, 's--', color='coral', linewidth=2,
                markersize=5, label=prediction_label_name)
        ax.fill_between(x, target_yearly_totals, pred_yearly_totals,
                        alpha=0.25, color='purple', label='Difference')

        ax.set_title(f'Whole Domain Total Annual Snow Water Volume - {target_names[ch_idx]}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Total Annual Snow Water Volume (km³)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(y)) for y in unique_years], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        output_path = output_dir / f'whole_domain_yearly_total_snow_{target_names[ch_idx]}.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_regional_yearly_total_snow(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'Target',
    prediction_label_name: str = 'Prediction',
):
    """
    Create line plots of total annual snow water volume (km³) for each region.

    For each region and year, sums the per-timestep snow water volume across all timesteps
    in that year within the region. The area between target and prediction lines is shaded.

    Args:
        predictions: Array of shape (n_samples, n_channels, height, width)
        targets: Array of shape (n_samples, n_channels, height, width)
        times: Array of time information
        output_dir: Directory to save plots
        region_defs: Dictionary of region definitions with lat/lon bounds
        resolution_km: Grid resolution in kilometers
        target_names: Optional list of channel names
        target_label_name: Label for target line in plot
        prediction_label_name: Label for prediction line in plot
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    n_channels = predictions.shape[1]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot compute yearly totals.")
        return

    lat, lon = get_lat_lon()
    if lat is None:
        print("Warning: Could not load lat/lon data")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    intra_year_indices = times[:, 2] if has_season_idx else times[:, 1]

    sort_idx = np.lexsort((intra_year_indices, years))
    years_sorted = years[sort_idx]
    predictions_sorted = predictions[sort_idx]
    targets_sorted = targets[sort_idx]

    cell_area_km2 = resolution_km * resolution_km
    unique_years = np.unique(years_sorted)

    output_dir = Path(output_dir) / "regional_yearly_total_snow"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating regional yearly total snow figures (resolution: {resolution_km} km)")

    for region_name, bounds in region_defs.items():
        mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
               (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])

        if not np.any(mask):
            print(f"  Warning: No grid points found for region {region_name}")
            continue

        for ch_idx in range(n_channels):
            # Volume per timestep within the region (km³)
            pred_vol = np.sum(predictions_sorted[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2
            target_vol = np.sum(targets_sorted[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2

            pred_yearly_totals = []
            target_yearly_totals = []
            for year in unique_years:
                year_mask = years_sorted == year
                pred_yearly_totals.append(np.sum(pred_vol[year_mask]))
                target_yearly_totals.append(np.sum(target_vol[year_mask]))

            pred_yearly_totals = np.array(pred_yearly_totals)
            target_yearly_totals = np.array(target_yearly_totals)

            fig, ax = plt.subplots(figsize=(max(10, len(unique_years) * 0.6), 6))
            x = np.arange(len(unique_years))

            ax.plot(x, target_yearly_totals, 'o-', color='steelblue', linewidth=2,
                    markersize=5, label=target_label_name)
            ax.plot(x, pred_yearly_totals, 's--', color='coral', linewidth=2,
                    markersize=5, label=prediction_label_name)
            ax.fill_between(x, target_yearly_totals, pred_yearly_totals,
                            alpha=0.25, color='purple', label='Difference')

            ax.set_title(f'Total Annual Snow Water Volume - {region_name} - {target_names[ch_idx]}',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Total Annual Snow Water Volume (km³)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(y)) for y in unique_years], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            plt.tight_layout()
            output_path = output_dir / f'regional_yearly_total_snow_{region_name}_{target_names[ch_idx]}.png'
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")


def plot_annual_snow_volume_scatter_domain(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'WRF Target',
    prediction_label_name: str = 'DESUNet',
):
    """
    Figure 5: Annual Total Snow Water Volume — whole-domain scatter.

    Each point is one water year. x-axis = WRF target domain-integrated annual
    total snow water (km³); y-axis = DESUNet prediction. Includes a 1:1 line,
    r / MAE annotation, and points colored by year.

    Args:
        predictions: (n_samples, n_channels, height, width)
        targets:     (n_samples, n_channels, height, width)
        times:       (n_samples, …)  times[:, 0] = year
        output_dir:  directory to save figure
        resolution_km: grid cell side length in km
        target_names: optional list of channel names
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot compute yearly totals.")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    intra = times[:, 2] if has_season_idx else times[:, 1]
    sort_idx = np.lexsort((intra, years))
    years_s = years[sort_idx]
    pred_s = predictions[sort_idx]
    targ_s = targets[sort_idx]

    cell_area_km2 = resolution_km ** 2
    unique_years = np.unique(years_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IPCC style: one discrete colour class per water year, not a blended ramp.
    year_cmap, norm, _ = year_cmap_norm(unique_years)

    for ch_idx in range(predictions.shape[1]):
        pred_vol = np.sum(pred_s[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2
        targ_vol = np.sum(targ_s[:, ch_idx], axis=(1, 2)) * 1e-6 * cell_area_km2

        pred_ann, targ_ann = [], []
        for y in unique_years:
            ym = years_s == y
            pred_ann.append(np.sum(pred_vol[ym]))
            targ_ann.append(np.sum(targ_vol[ym]))
        pred_ann = np.array(pred_ann)
        targ_ann = np.array(targ_ann)

        r = np.corrcoef(targ_ann, pred_ann)[0, 1]
        mae = np.mean(np.abs(targ_ann - pred_ann))

        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(targ_ann, pred_ann,
                        norm=norm, s=70, zorder=3, edgecolors='k', linewidths=0.5)

        all_vals = np.concatenate([targ_ann, pred_ann])
        pad = (all_vals.max() - all_vals.min()) * 0.05
        lims = [all_vals.min() - pad, all_vals.max() + pad]
        ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='1:1 line')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                     edgecolor='#8B7355', linewidth=1.5)
        ax.text(0.05, 0.95,
                f'r = {r:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')

        ax.set_title(
            f'Annual Total Snow Water Volume\n'
            f'Whole Domain — {target_names[ch_idx]}',
            fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{target_label_name} (km³)', fontsize=11)
        ax.set_ylabel(f'{prediction_label_name} (km³)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        out = output_dir / f'annual_snow_volume_scatter_{target_names[ch_idx]}.png'
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out.name}")


def plot_annual_snow_volume_scatter_regions(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'WRF Target',
    prediction_label_name: str = 'DESUNet',
):
    """
    Figure 5 (regional): Annual Total Snow Water Volume — 6-panel scatter.

    One panel per region. Each point is one water year. x = WRF target
    domain-integrated annual total snow water (km³); y = DESUNet prediction.
    Points are colored by year. Includes 1:1 line and r / MAE annotation.

    Args:
        predictions: (n_samples, n_channels, height, width)
        targets:     (n_samples, n_channels, height, width)
        times:       (n_samples, …)  times[:, 0] = year
        output_dir:  directory to save figure
        region_defs: dict of region definitions with lat/lon bounds
        resolution_km: grid cell side length in km
        target_names: optional list of channel names
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot compute yearly totals.")
        return

    lat, lon = get_lat_lon()
    if lat is None:
        print("Warning: Could not load lat/lon data")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    intra = times[:, 2] if has_season_idx else times[:, 1]
    sort_idx = np.lexsort((intra, years))
    years_s = years[sort_idx]
    pred_s = predictions[sort_idx]
    targ_s = targets[sort_idx]

    cell_area_km2 = resolution_km ** 2
    unique_years = np.unique(years_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IPCC style: one discrete colour class per water year, not a blended ramp.
    year_cmap, norm, _ = year_cmap_norm(unique_years)
    n_regions = len(region_defs)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols

    for ch_idx in range(predictions.shape[1]):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 5 * n_rows),
                                 squeeze=False)

        for panel_idx, (region_name, bounds) in enumerate(region_defs.items()):
            row, col = divmod(panel_idx, n_cols)
            ax = axes[row][col]

            mask = ((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
                    (lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))
            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_vol = np.sum(pred_s[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2
            targ_vol = np.sum(targ_s[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2

            pred_ann, targ_ann = [], []
            for y in unique_years:
                ym = years_s == y
                pred_ann.append(np.sum(pred_vol[ym]))
                targ_ann.append(np.sum(targ_vol[ym]))
            pred_ann = np.array(pred_ann)
            targ_ann = np.array(targ_ann)

            r = np.corrcoef(targ_ann, pred_ann)[0, 1]
            mae = np.mean(np.abs(targ_ann - pred_ann))

            sc = ax.scatter(targ_ann, pred_ann, c=unique_years, cmap=year_cmap,
                            norm=norm, s=60, zorder=3, edgecolors='k', linewidths=0.5)

            all_vals = np.concatenate([targ_ann, pred_ann])
            pad = (all_vals.max() - all_vals.min()) * 0.05
            lims = [all_vals.min() - pad, all_vals.max() + pad]
            ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='1:1')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                         edgecolor='#8B7355', linewidth=1.2)
            ax.text(0.05, 0.95,
                    f'R = {r:.3f}\n',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')

            ax.set_title(panel_label(panel_idx, region_name), fontsize=11, fontweight='bold')
            ax.set_xlabel(f'{target_label_name} (km³)', fontsize=9)
            ax.set_ylabel(f'{prediction_label_name} (km³)', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)

        for panel_idx in range(n_regions, n_rows * n_cols):
            row, col = divmod(panel_idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f'Annual Total Snow Water Volume by Region\n',
            fontsize=13, fontweight='bold')

        plt.tight_layout()
        out = output_dir / f'fig5_annual_snow_volume_scatter_regions_{target_names[ch_idx]}.png'
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out.name}")


def plot_avg_snow_volume_scatter_regions(
    predictions: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
    region_defs: dict,
    resolution_km: float,
    target_names: list = None,
    target_label_name: str = 'WRF Target',
    prediction_label_name: str = 'DESUNet',
):
    """
    Figure 5 (variant): Yearly Average Snow Water Volume — 6-panel scatter by region.

    Same layout as plot_annual_snow_volume_scatter_regions but each point
    represents the **mean** (average over timesteps) rather than the sum of
    per-timestep snow water volumes within a water year.  One point per year,
    colored by year.  Includes 1:1 line and R annotation.

    Args:
        predictions: (n_samples, n_channels, height, width)
        targets:     (n_samples, n_channels, height, width)
        times:       (n_samples, …)  times[:, 0] = year
        output_dir:  directory to save figure
        region_defs: dict of region definitions with lat/lon bounds
        resolution_km: grid cell side length in km
        target_names: optional list of channel names
    """
    if target_names is None:
        target_names = [f"Channel {i}" for i in range(predictions.shape[1])]

    if times.ndim == 1:
        print("Warning: times array is 1D, cannot compute yearly averages.")
        return

    lat, lon = get_lat_lon()
    if lat is None:
        print("Warning: Could not load lat/lon data")
        return

    years = times[:, 0]
    has_season_idx = times.shape[1] >= 3
    intra = times[:, 2] if has_season_idx else times[:, 1]
    sort_idx = np.lexsort((intra, years))
    years_s = years[sort_idx]
    pred_s = predictions[sort_idx]
    targ_s = targets[sort_idx]

    cell_area_km2 = resolution_km ** 2
    unique_years = np.unique(years_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IPCC style: one discrete colour class per water year, not a blended ramp.
    year_cmap, norm, _ = year_cmap_norm(unique_years)
    n_regions = len(region_defs)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols

    for ch_idx in range(predictions.shape[1]):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for panel_idx, (region_name, bounds) in enumerate(region_defs.items()):
            row, col = divmod(panel_idx, n_cols)
            ax = axes[row][col]

            mask = ((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
                    (lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))
            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_vol = np.sum(pred_s[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2
            targ_vol = np.sum(targ_s[:, ch_idx][:, mask], axis=1) * 1e-6 * cell_area_km2

            # Average (not sum) over timesteps within each water year
            pred_ann = np.array([np.mean(pred_vol[years_s == y]) for y in unique_years])
            targ_ann = np.array([np.mean(targ_vol[years_s == y]) for y in unique_years])

            r = np.corrcoef(targ_ann, pred_ann)[0, 1]

            sc = ax.scatter(targ_ann, pred_ann, c=unique_years, cmap=year_cmap,
                            norm=norm, s=60, zorder=3, edgecolors='k', linewidths=0.5)

            all_vals = np.concatenate([targ_ann, pred_ann])
            pad = (all_vals.max() - all_vals.min()) * 0.05
            lims = [all_vals.min() - pad, all_vals.max() + pad]
            ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='1:1')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85,
                         edgecolor='#8B7355', linewidth=1.2)
            ax.text(0.05, 0.95,
                    f'R = {r:.3f}',
                    transform=ax.transAxes, fontsize='large',
                    verticalalignment='top', bbox=props, family='monospace')

            ax.set_title(panel_label(panel_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.set_xlabel(f'Target SWE [km³]', fontsize='large')
            ax.set_ylabel(f'Prediction SWE [km³]', fontsize='large')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)

        for panel_idx in range(n_regions, n_rows * n_cols):
            row, col = divmod(panel_idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f'Yearly Average Snow Water Volume by Region\n',
            fontsize='xx-large', fontweight='bold')

        plt.tight_layout()
        out = output_dir / f'real_fig5_avg_snow_volume_scatter_regions_{target_names[ch_idx]}.png'
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out.name}")


def plot_qq(predictions, targets, output_dir, region_defs=None, target_names=None,
            n_quantiles=500, filename_suffix=''):
    """
    Generate QQ plots comparing the distribution of ML predictions vs targets.

    Produces:
      - One QQ plot for the whole domain ("all")
      - One QQ plot per region (if region_defs and lat/lon are available)

    Args:
        predictions: np.ndarray of shape (N, C, H, W)
        targets:     np.ndarray of shape (N, C, H, W)
        output_dir:  Path-like; plots saved to output_dir/qq/
        region_defs: dict of {region_name: {lon_min, lon_max, lat_min, lat_max}} or None
        target_names: list of channel names (length C)
        n_quantiles: number of quantile points to compute
        filename_suffix: optional string appended to filenames
    """
    output_dir = Path(output_dir) / "qq"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    quantile_probs = np.linspace(0, 1, n_quantiles)

    def _make_qq(pred_flat, targ_flat, title, fpath):
        q_pred = np.quantile(pred_flat, quantile_probs)
        q_targ = np.quantile(targ_flat, quantile_probs)

        vmin = min(q_pred.min(), q_targ.min())
        vmax = max(q_pred.max(), q_targ.max())

        r = np.corrcoef(q_targ, q_pred)[0, 1]
        # MAE between the plotted quantile pairs (same quantities r is computed
        # on), i.e. the mean distributional offset of the prediction CDF.
        mae = np.mean(np.abs(q_pred - q_targ))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(q_targ, q_pred, s=25, alpha=0.7, color='steelblue',
                   label=f'Quantiles\nr   = {r:.3f}\nMAE = {mae:.3f}')
        ax.plot([vmin, vmax], [vmin, vmax], '--', color='gray', linewidth=1.0, alpha=0.5, label='1:1 line')
        ax.set_xlabel('Target quantiles', fontsize=12)
        ax.set_ylabel('Prediction quantiles', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")

    # --- Whole-domain QQ plots ---
    for ch in range(n_channels):
        pred_flat = predictions[:, ch].ravel()
        targ_flat = targets[:, ch].ravel()
        title = f'QQ Plot – All – {target_names[ch]}'
        suffix = f'_{filename_suffix}' if filename_suffix else ''
        fpath = output_dir / f'qq_all_{target_names[ch]}{suffix}.png'
        _make_qq(pred_flat, targ_flat, title, fpath)

    # --- Per-region QQ plots ---
    if region_defs is not None:
        lat, lon = get_lat_lon()
        if lat is None:
            print("  Warning: lat/lon not available, skipping per-region QQ plots")
            return

        for region_name, bounds in region_defs.items():
            mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) & \
                   (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
            if not np.any(mask):
                continue

            for ch in range(n_channels):
                # predictions shape: (N, C, H, W) → flatten spatial within mask
                pred_flat = predictions[:, ch, mask].ravel()
                targ_flat = targets[:, ch, mask].ravel()
                title = f'QQ Plot – {region_name} – {target_names[ch]}'
                suffix = f'_{filename_suffix}' if filename_suffix else ''
                fpath = output_dir / f'qq_{region_name}_{target_names[ch]}{suffix}.png'
                _make_qq(pred_flat, targ_flat, title, fpath)


def plot_qq_regions_combined(predictions, targets, output_dir, region_defs, target_names=None,
                              n_quantiles=500, filename_suffix=''):
    """
    Generate a single combined QQ-plot figure with all regions in separate panels.

    Args:
        predictions: np.ndarray of shape (N, C, H, W)
        targets:     np.ndarray of shape (N, C, H, W)
        output_dir:  Path-like; plots saved to output_dir/qq/
        region_defs: dict of {region_name: {lon_min, lon_max, lat_min, lat_max}}
        target_names: list of channel names (length C)
        n_quantiles: number of quantile points to compute
        filename_suffix: optional string appended to filenames
    """
    output_dir = Path(output_dir) / "qq"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    lat, lon = get_lat_lon()
    if lat is None:
        print("  Warning: lat/lon not available, skipping combined QQ plot")
        return

    quantile_probs = np.linspace(0, 1, n_quantiles)
    region_names = list(region_defs.keys())
    n_regions = len(region_names)

    region_masks = {}
    for region_name, bounds in region_defs.items():
        mask = (
            (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
            (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
        )
        region_masks[region_name] = mask

    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))

    for ch in range(n_channels):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax_idx, region_name in enumerate(region_names):
            ax = axes[ax_idx]
            mask = region_masks[region_name]

            if not np.any(mask):
                ax.set_visible(False)
                continue

            pred_flat = predictions[:, ch][:, mask].ravel()
            targ_flat = targets[:, ch][:, mask].ravel()

            q_pred = np.quantile(pred_flat, quantile_probs)
            q_targ = np.quantile(targ_flat, quantile_probs)

            vmin = min(q_pred.min(), q_targ.min())
            vmax = max(q_pred.max(), q_targ.max())

            r = np.corrcoef(q_targ, q_pred)[0, 1]
            # MAE between the plotted quantile pairs (same quantities r is
            # computed on), in mm of SWE.
            mae = np.mean(np.abs(q_pred - q_targ))

            ax.scatter(q_targ, q_pred, s=25, alpha=0.7, color='steelblue',
                       label=f'Quantiles\nr   = {r:.3f}\nMAE = {mae:.3f} mm')
            ax.plot([vmin, vmax], [vmin, vmax], '--', color='gray', linewidth=1.0,
                    alpha=0.5, label='1:1 line')
            ax.set_xlabel('Target SWE quantiles [mm]', fontsize=11)
            ax.set_ylabel('Prediction SWE quantiles [mm]', fontsize=11)
            ax.set_title(panel_label(ax_idx, region_name), fontsize='x-large', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        for ax_idx in range(n_regions, len(axes)):
            axes[ax_idx].set_visible(False)

        plt.suptitle(f'SWE Q-Q Plot by Regions', fontsize='xx-large', fontweight='bold')
        plt.tight_layout()
        suffix = f'_{filename_suffix}' if filename_suffix else ''
        fpath = output_dir / f'qq_all_regions_{target_names[ch]}{suffix}.png'
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")


def plot_regional_state_yearly_max_bar_combined(
        predictions, targets, times, output_dir,
        region_defs, state_abbrevs, target_names=None,
        diff_ylim=(-20, 20)):
    """
    Combined figure with regional (left) and state-level (right) mean yearly max bar charts.

    Layout per channel:
        ┌──────────────────┬──────────────────┐
        │  Regional bars   │   State bars     │  (top row, height ratio 3)
        ├──────────────────┼──────────────────┤
        │  Regional diff%  │   State diff%    │  (bottom row, height ratio 1)
        └──────────────────┴──────────────────┘

    The diff panel y-axis is fixed to diff_ylim (default -20% to +20%).
    Saved to output_dir/region_state_yearly_max_bar_combined_{target_name}.png
    """
    years = get_years(times)
    if years is None:
        return
    unique_years = np.unique(years)
    lat, lon = get_lat_lon()
    if lat is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        # ── Compute regional values ──────────────────────────────────────────
        region_names_list = list(region_defs.keys())
        reg_vals_pred, reg_vals_targ, valid_regions = [], [], []
        for name in region_names_list:
            bounds = region_defs[name]
            mask = ((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
                    (lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))
            if not np.any(mask):
                continue
            valid_regions.append(name)
            pred_maxs, targ_maxs = [], []
            for y in unique_years:
                ym = years == y
                if not np.any(ym):
                    continue
                pred_maxs.append(np.max(predictions[ym, ch][:, mask]))
                targ_maxs.append(np.max(targets[ym, ch][:, mask]))
            reg_vals_pred.append(np.mean(pred_maxs))
            reg_vals_targ.append(np.mean(targ_maxs))

        # ── Compute state values ─────────────────────────────────────────────
        state_vals_pred, state_vals_targ, valid_states = [], [], []
        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask):
                continue
            valid_states.append(state)
            pred_maxs, targ_maxs = [], []
            for y in unique_years:
                ym = years == y
                if not np.any(ym):
                    continue
                pred_maxs.append(np.max(predictions[ym, ch][:, mask]))
                targ_maxs.append(np.max(targets[ym, ch][:, mask]))
            state_vals_pred.append(np.mean(pred_maxs))
            state_vals_targ.append(np.mean(targ_maxs))

        if not valid_regions and not valid_states:
            continue

        # ── Build figure ─────────────────────────────────────────────────────
        n_reg = len(valid_regions)
        n_sta = len(valid_states)
        fig_w = max(14, n_reg * 1.2 + n_sta * 1.2)
        fig, axes = plt.subplots(
            2, 2,
            figsize=(fig_w, 8),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        ax_reg_bar, ax_sta_bar = axes[0, 0], axes[0, 1]
        ax_reg_diff, ax_sta_diff = axes[1, 0], axes[1, 1]

        width = 0.35

        # ── Regional bar (top-left) ──────────────────────────────────────────
        if valid_regions:
            x_reg = np.arange(n_reg)
            ax_reg_bar.bar(x_reg - width / 2, reg_vals_targ, width, color='dimgray', label='Target')
            ax_reg_bar.bar(x_reg + width / 2, reg_vals_pred, width, color='royalblue', label='Prediction')
            ax_reg_bar.set_ylabel('SWE [mm]', fontsize='x-large')
            ax_reg_bar.set_title(panel_label(0, 'SWE Mean Yearly Max by Region'),
                                 fontsize='xx-large', fontweight='bold')
            ax_reg_bar.set_xticks(x_reg)
            ax_reg_bar.set_xticklabels([])
            ax_reg_bar.legend(fontsize='x-large')
            ax_reg_bar.grid(axis='y', linestyle='--', alpha=0.5)

            reg_targ_arr = np.array(reg_vals_targ)
            reg_pred_arr = np.array(reg_vals_pred)
            diff_reg = np.zeros_like(reg_targ_arr)
            nz = reg_targ_arr != 0
            diff_reg[nz] = (reg_pred_arr[nz] - reg_targ_arr[nz]) / reg_targ_arr[nz] * 100.0

            ax_reg_diff.bar(x_reg, diff_reg,
                            color=['red' if d > 0 else 'royalblue' for d in diff_reg])
            ax_reg_diff.axhline(0, color='black', linewidth=0.8)
            ax_reg_diff.set_ylabel('Diff (%)', fontsize='x-large')
            ax_reg_diff.set_xticks(x_reg)
            ax_reg_diff.set_xticklabels(valid_regions, rotation=0)
            ax_reg_diff.set_ylim(diff_ylim)
            ax_reg_diff.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax_reg_bar.set_visible(False)
            ax_reg_diff.set_visible(False)

        # ── State bar (top-right) ────────────────────────────────────────────
        if valid_states:
            x_sta = np.arange(n_sta)
            ax_sta_bar.bar(x_sta - width / 2, state_vals_targ, width, color='dimgray', label='Target')
            ax_sta_bar.bar(x_sta + width / 2, state_vals_pred, width, color='royalblue', label='Prediction')
            ax_sta_bar.set_ylabel('SWE [mm]', fontsize='x-large')
            ax_sta_bar.set_title(panel_label(1, 'SWE Mean Yearly Max by State'),
                                 fontsize='xx-large', fontweight='bold')
            ax_sta_bar.set_xticks(x_sta)
            ax_sta_bar.set_xticklabels([])
            ax_sta_bar.legend(fontsize='x-large')
            ax_sta_bar.grid(axis='y', linestyle='--', alpha=0.5)

            sta_targ_arr = np.array(state_vals_targ)
            sta_pred_arr = np.array(state_vals_pred)
            diff_sta = np.zeros_like(sta_targ_arr)
            nz = sta_targ_arr != 0
            diff_sta[nz] = (sta_pred_arr[nz] - sta_targ_arr[nz]) / sta_targ_arr[nz] * 100.0

            ax_sta_diff.bar(x_sta, diff_sta,
                            color=['red' if d > 0 else 'royalblue' for d in diff_sta])
            ax_sta_diff.axhline(0, color='black', linewidth=0.8)
            ax_sta_diff.set_ylabel('Diff (%)', fontsize='x-large')
            ax_sta_diff.set_xticks(x_sta)
            ax_sta_diff.set_xticklabels(valid_states, rotation=0)
            ax_sta_diff.set_ylim(diff_ylim)
            ax_sta_diff.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax_sta_bar.set_visible(False)
            ax_sta_diff.set_visible(False)

        plt.tight_layout()
        fpath = output_dir / f'region_state_yearly_max_bar_combined_{target_names[ch]}.png'
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")


def plot_regional_state_yearly_sum_bar_combined(
        predictions, targets, times, output_dir,
        region_defs, state_abbrevs, resolution_km, target_names=None,
        diff_ylim=(-20, 20)):
    """
    Combined figure with regional (left) and state-level (right) mean yearly
    peak snow water VOLUME (km³) bar charts.

    Same layout as plot_regional_state_yearly_max_bar_combined, but the plotted
    quantity is the regional/state snow water volume instead of the single-pixel
    max SWE:
        volume(t) = sum over region of snow_mm * 1e-6 km/mm * resolution_km²  [km³]
    For each year the peak (max-over-time) of volume(t) is taken; those per-year
    peaks are averaged over all years.

    Layout per channel:
        ┌──────────────────┬──────────────────┐
        │  Regional bars   │   State bars     │  (top row, height ratio 3)
        ├──────────────────┼──────────────────┤
        │  Regional diff%  │   State diff%    │  (bottom row, height ratio 1)
        └──────────────────┴──────────────────┘

    The diff panel y-axis is fixed to diff_ylim (default -20% to +20%).
    Saved to output_dir/region_state_yearly_sum_bar_combined_{target_name}.png
    """
    years = get_years(times)
    if years is None:
        return
    unique_years = np.unique(years)
    lat, lon = get_lat_lon()
    if lat is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_area_km2 = resolution_km * resolution_km
    print(f"\nCreating combined regional + state yearly peak volume figure "
          f"(resolution: {resolution_km} km, cell area: {cell_area_km2} km²)")

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    for ch in range(n_channels):
        # ── Compute regional values ──────────────────────────────────────────
        region_names_list = list(region_defs.keys())
        reg_vals_pred, reg_vals_targ, valid_regions = [], [], []
        for name in region_names_list:
            bounds = region_defs[name]
            mask = ((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
                    (lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))
            if not np.any(mask):
                continue
            valid_regions.append(name)
            pred_peaks, targ_peaks = [], []
            for y in unique_years:
                ym = years == y
                if not np.any(ym):
                    continue
                # Per-timestep volume (km³), then the year's peak
                pred_vol = predictions[ym, ch][:, mask].sum(axis=1) * 1e-6 * cell_area_km2
                targ_vol = targets[ym, ch][:, mask].sum(axis=1) * 1e-6 * cell_area_km2
                pred_peaks.append(np.max(pred_vol))
                targ_peaks.append(np.max(targ_vol))
            reg_vals_pred.append(np.mean(pred_peaks))
            reg_vals_targ.append(np.mean(targ_peaks))

        # ── Compute state values ─────────────────────────────────────────────
        state_vals_pred, state_vals_targ, valid_states = [], [], []
        for state in state_abbrevs:
            mask = get_state_mask(state, lat, lon)
            if mask is None or not np.any(mask):
                continue
            valid_states.append(state)
            pred_peaks, targ_peaks = [], []
            for y in unique_years:
                ym = years == y
                if not np.any(ym):
                    continue
                pred_vol = predictions[ym, ch][:, mask].sum(axis=1) * 1e-6 * cell_area_km2
                targ_vol = targets[ym, ch][:, mask].sum(axis=1) * 1e-6 * cell_area_km2
                pred_peaks.append(np.max(pred_vol))
                targ_peaks.append(np.max(targ_vol))
            state_vals_pred.append(np.mean(pred_peaks))
            state_vals_targ.append(np.mean(targ_peaks))

        if not valid_regions and not valid_states:
            continue

        # ── Build figure ─────────────────────────────────────────────────────
        n_reg = len(valid_regions)
        n_sta = len(valid_states)
        fig_w = max(14, n_reg * 1.2 + n_sta * 1.2)
        fig, axes = plt.subplots(
            2, 2,
            figsize=(fig_w, 8),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        ax_reg_bar, ax_sta_bar = axes[0, 0], axes[0, 1]
        ax_reg_diff, ax_sta_diff = axes[1, 0], axes[1, 1]

        width = 0.35

        # ── Regional bar (top-left) ──────────────────────────────────────────
        if valid_regions:
            x_reg = np.arange(n_reg)
            ax_reg_bar.bar(x_reg - width / 2, reg_vals_targ, width, color='dimgray', label='Target')
            ax_reg_bar.bar(x_reg + width / 2, reg_vals_pred, width, color='royalblue', label='Prediction')
            ax_reg_bar.set_ylabel('SWE Volume [km³]', fontsize='x-large')
            ax_reg_bar.set_title(panel_label(0, 'SWE Mean Yearly Peak Volume by Region'),
                                 fontsize='xx-large', fontweight='bold')
            ax_reg_bar.set_xticks(x_reg)
            ax_reg_bar.set_xticklabels([])
            ax_reg_bar.legend(fontsize='x-large')
            ax_reg_bar.grid(axis='y', linestyle='--', alpha=0.5)

            reg_targ_arr = np.array(reg_vals_targ)
            reg_pred_arr = np.array(reg_vals_pred)
            diff_reg = np.zeros_like(reg_targ_arr)
            nz = reg_targ_arr != 0
            diff_reg[nz] = (reg_pred_arr[nz] - reg_targ_arr[nz]) / reg_targ_arr[nz] * 100.0

            ax_reg_diff.bar(x_reg, diff_reg,
                            color=['red' if d > 0 else 'royalblue' for d in diff_reg])
            ax_reg_diff.axhline(0, color='black', linewidth=0.8)
            ax_reg_diff.set_ylabel('Diff (%)', fontsize='x-large')
            ax_reg_diff.set_xticks(x_reg)
            ax_reg_diff.set_xticklabels(valid_regions, rotation=0)
            ax_reg_diff.set_ylim(diff_ylim)
            ax_reg_diff.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax_reg_bar.set_visible(False)
            ax_reg_diff.set_visible(False)

        # ── State bar (top-right) ────────────────────────────────────────────
        if valid_states:
            x_sta = np.arange(n_sta)
            ax_sta_bar.bar(x_sta - width / 2, state_vals_targ, width, color='dimgray', label='Target')
            ax_sta_bar.bar(x_sta + width / 2, state_vals_pred, width, color='royalblue', label='Prediction')
            ax_sta_bar.set_ylabel('SWE Volume [km³]', fontsize='x-large')
            ax_sta_bar.set_title(panel_label(1, 'SWE Mean Yearly Peak Volume by State'),
                                 fontsize='xx-large', fontweight='bold')
            ax_sta_bar.set_xticks(x_sta)
            ax_sta_bar.set_xticklabels([])
            ax_sta_bar.legend(fontsize='x-large')
            ax_sta_bar.grid(axis='y', linestyle='--', alpha=0.5)

            sta_targ_arr = np.array(state_vals_targ)
            sta_pred_arr = np.array(state_vals_pred)
            diff_sta = np.zeros_like(sta_targ_arr)
            nz = sta_targ_arr != 0
            diff_sta[nz] = (sta_pred_arr[nz] - sta_targ_arr[nz]) / sta_targ_arr[nz] * 100.0

            ax_sta_diff.bar(x_sta, diff_sta,
                            color=['red' if d > 0 else 'royalblue' for d in diff_sta])
            ax_sta_diff.axhline(0, color='black', linewidth=0.8)
            ax_sta_diff.set_ylabel('Diff (%)', fontsize='x-large')
            ax_sta_diff.set_xticks(x_sta)
            ax_sta_diff.set_xticklabels(valid_states, rotation=0)
            ax_sta_diff.set_ylim(diff_ylim)
            ax_sta_diff.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax_sta_bar.set_visible(False)
            ax_sta_diff.set_visible(False)

        plt.tight_layout()
        fpath = output_dir / f'region_state_yearly_sum_bar_combined_{target_names[ch]}.png'
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")


def _get_hgt():
    """Load terrain height (HGT) from static file. Returns 2-D array or None."""
    if STATIC_PATH and os.path.exists(STATIC_PATH):
        try:
            with nc.Dataset(str(STATIC_PATH), "r") as fin:
                hgt = _crop_highres(fin['HGT'][0, :, :])
                return np.array(hgt)
        except Exception as e:
            print(f"  Warning: could not load HGT from static file: {e}")
    return None


def plot_error_vs_terrain(predictions, targets, output_dir, times=None,
                           region_defs=None, target_names=None, n_sample_pts=50000,
                           snow_threshold=1.0, pct_threshold=50.0, filename_suffix=''):
    """
    Plot absolute and percentage prediction error vs terrain height (HGT).

    Produces per channel:
      Binned (all N×H×W samples):
        - Absolute error vs HGT: all pixels + snow-only, with binned mean ± std
        - Percentage error vs HGT: snow-only, with binned mean ± std
        - Per-region combined panels for both metrics

      Mean-of-yearly-max (one point per pixel, no binning, requires `times`):
        - Absolute error of mean yearly max per pixel vs HGT
        - Percentage error of mean yearly max per pixel vs HGT (snow-only)
        - Per-region combined panels for both metrics

    Args:
        predictions:      np.ndarray (N, C, H, W)
        targets:          np.ndarray (N, C, H, W)
        output_dir:       Path-like; plots saved to output_dir/error_vs_terrain/
        times:            np.ndarray (N, 2+) with times[:,0] = year; enables mean-yearly-max plots
        region_defs:      dict {name: {lon_min, lon_max, lat_min, lat_max}} or None
        target_names:     list of channel names (length C)
        n_sample_pts:     max scatter points for binned plots (random subsample)
        snow_threshold:   SWE threshold (mm) for absolute-error snow-only plots
        pct_threshold:    SWE threshold (mm) for percentage-error plots; should be
                          much higher than snow_threshold to avoid inflated percentages
                          from near-zero denominators (default 50 mm)
        filename_suffix:  optional string appended to filenames
    """
    output_dir = Path(output_dir) / "error_vs_terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    hgt = _get_hgt()
    if hgt is None:
        print("  Warning: HGT not available, skipping error vs terrain plots")
        return

    n_channels = predictions.shape[1]
    if target_names is None:
        target_names = [f"Ch{i}" for i in range(n_channels)]

    suffix = f'_{filename_suffix}' if filename_suffix else ''

    hgt_broadcast = np.broadcast_to(hgt, predictions[:, 0].shape)  # (N, H, W)

    # ------------------------------------------------------------------
    # Helper: binned panel (many N×H×W points → subsample + bin mean)
    # ------------------------------------------------------------------
    def _make_binned_panel(ax, hgt_flat, err_flat, title, ylabel, n_bins=40):
        rng = np.random.default_rng(42)
        n = len(hgt_flat)
        if n > n_sample_pts:
            idx = rng.choice(n, n_sample_pts, replace=False)
            xs, ys = hgt_flat[idx], err_flat[idx]
        else:
            xs, ys = hgt_flat, err_flat

        ax.scatter(xs, ys, s=3, alpha=0.15, color='steelblue', rasterized=True)
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--')

        hmin, hmax = np.percentile(hgt_flat, 1), np.percentile(hgt_flat, 99)
        bins = np.linspace(hmin, hmax, n_bins + 1)
        bin_idx = np.digitize(hgt_flat, bins) - 1
        bin_cx, bin_mean, bin_std = [], [], []
        for b in range(n_bins):
            m = bin_idx == b
            if m.sum() < 5:
                continue
            bin_cx.append(0.5 * (bins[b] + bins[b + 1]))
            bin_mean.append(err_flat[m].mean())
            bin_std.append(err_flat[m].std())
        if bin_cx:
            bin_cx = np.array(bin_cx)
            bin_mean = np.array(bin_mean)
            bin_std = np.array(bin_std)
            ax.plot(bin_cx, bin_mean, color='firebrick', linewidth=2, label='Bin mean')
            ax.fill_between(bin_cx, bin_mean - bin_std, bin_mean + bin_std,
                            color='firebrick', alpha=0.2, label='±1 std')

        bias = float(np.mean(err_flat))
        mae = float(np.mean(np.abs(err_flat)))
        ax.set_title(f'{title}\nbias={bias:.2f}  MAE={mae:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Terrain height [m]', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Helper: per-pixel panel (one point per spatial pixel, no binning)
    # ------------------------------------------------------------------
    def _make_pixel_panel(ax, hgt_px, err_px, title, ylabel):
        ax.scatter(hgt_px, err_px, s=6, alpha=0.4, color='steelblue', rasterized=True)
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
        bias = float(np.mean(err_px))
        mae = float(np.mean(np.abs(err_px)))
        ax.set_title(f'{title}\nbias={bias:.2f}  MAE={mae:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Terrain height [m]', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    def _save_single_binned(hgt_flat, err_flat, title, ylabel, fpath):
        fig, ax = plt.subplots(figsize=(7, 5))
        _make_binned_panel(ax, hgt_flat, err_flat, title, ylabel)
        plt.tight_layout()
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")

    def _save_single_pixel(hgt_px, err_px, title, ylabel, fpath):
        fig, ax = plt.subplots(figsize=(7, 5))
        _make_pixel_panel(ax, hgt_px, err_px, title, ylabel)
        plt.tight_layout()
        plt.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fpath.name}")

    # ------------------------------------------------------------------
    # Compute mean-of-yearly-max arrays if times are provided
    # Shape: (C, H, W) — one value per pixel per channel
    # ------------------------------------------------------------------
    pred_mean_max = None
    targ_mean_max = None
    if times is not None:
        years = get_years(times)
        if years is not None:
            unique_years = np.unique(years)
            pred_mmax_list = []
            targ_mmax_list = []
            for y in unique_years:
                ym = years == y
                if not np.any(ym):
                    continue
                pred_mmax_list.append(np.max(predictions[ym], axis=0))  # (C, H, W)
                targ_mmax_list.append(np.max(targets[ym], axis=0))
            pred_mean_max = np.mean(pred_mmax_list, axis=0)  # (C, H, W)
            targ_mean_max = np.mean(targ_mmax_list, axis=0)

    # ==================================================================
    # BINNED PLOTS (all N×H×W time-space samples)
    # ==================================================================
    for ch in range(n_channels):
        hgt_flat = hgt_broadcast.ravel()
        err_flat = (predictions[:, ch] - targets[:, ch]).ravel()
        targ_flat = targets[:, ch].ravel()
        snow_mask = targ_flat >= snow_threshold

        _save_single_binned(hgt_flat, err_flat,
                            f'Absolute Error vs Terrain – All – {target_names[ch]}',
                            'Error (pred − target) [mm]',
                            output_dir / f'error_vs_terrain_all_{target_names[ch]}{suffix}.png')

        if snow_mask.sum() > 10:
            _save_single_binned(hgt_flat[snow_mask], err_flat[snow_mask],
                                f'Absolute Error vs Terrain – Snow Only (≥{snow_threshold} mm) – {target_names[ch]}',
                                'Error (pred − target) [mm]',
                                output_dir / f'error_vs_terrain_snowonly_{target_names[ch]}{suffix}.png')

        pct_mask = targ_flat >= pct_threshold
        if pct_mask.sum() > 10:
            pct_err_flat = err_flat[pct_mask] / targ_flat[pct_mask] * 100.0
            _save_single_binned(hgt_flat[pct_mask], pct_err_flat,
                                f'Percentage Error vs Terrain',
                                'Error (%) = (pred − target) / target × 100',
                                output_dir / f'pct_error_vs_terrain_snowonly_{target_names[ch]}{suffix}.png')

    # ==================================================================
    # TEMPORAL-MEAN PIXEL SCATTER (one point per pixel, no binning)
    # mean over all N time steps → shape (C, H, W)
    # ==================================================================
    pred_tmean = predictions.mean(axis=0)  # (C, H, W)
    targ_tmean = targets.mean(axis=0)
    hgt_px = hgt.ravel()

    for ch in range(n_channels):
        err_px = (pred_tmean[ch] - targ_tmean[ch]).ravel()
        targ_px = targ_tmean[ch].ravel()
        snow_px = targ_px >= snow_threshold
        pct_px_mask = targ_px >= pct_threshold

        _save_single_pixel(hgt_px, err_px,
                           f'Abs Error of Temporal Mean vs Terrain – All – {target_names[ch]}',
                           'Error of temporal mean (pred − target) [mm]',
                           output_dir / f'error_vs_terrain_tmean_all_{target_names[ch]}{suffix}.png')

        if snow_px.sum() > 10:
            _save_single_pixel(hgt_px[snow_px], err_px[snow_px],
                               f'Abs Error of Temporal Mean vs Terrain – Snow Only – {target_names[ch]}',
                               'Error of temporal mean (pred − target) [mm]',
                               output_dir / f'error_vs_terrain_tmean_snowonly_{target_names[ch]}{suffix}.png')

        if pct_px_mask.sum() > 10:
            pct_px = err_px[pct_px_mask] / targ_px[pct_px_mask] * 100.0
            _save_single_pixel(hgt_px[pct_px_mask], pct_px,
                               f'Pct Error of Temporal Mean vs Terrain – Target',
                               'Error (%) of temporal mean',
                               output_dir / f'pct_error_vs_terrain_tmean_snowonly_{target_names[ch]}{suffix}.png')

    # ==================================================================
    # MEAN-OF-YEARLY-MAX PIXEL SCATTER (one point per pixel, no binning)
    # ==================================================================
    if pred_mean_max is not None:
        for ch in range(n_channels):
            err_px = (pred_mean_max[ch] - targ_mean_max[ch]).ravel()
            targ_px = targ_mean_max[ch].ravel()
            snow_px = targ_px >= snow_threshold
            pct_px_mask = targ_px >= pct_threshold

            _save_single_pixel(hgt_px, err_px,
                               f'Abs Error of Mean Yearly Max vs Terrain – All – {target_names[ch]}',
                               'Error of mean yearly max (pred − target) [mm]',
                               output_dir / f'error_vs_terrain_mean_yrmax_all_{target_names[ch]}{suffix}.png')

            if snow_px.sum() > 10:
                _save_single_pixel(hgt_px[snow_px], err_px[snow_px],
                                   f'Abs Error of Mean Yearly Max vs Terrain – Snow Only – {target_names[ch]}',
                                   'Error of mean yearly max (pred − target) [mm]',
                                   output_dir / f'error_vs_terrain_mean_yrmax_snowonly_{target_names[ch]}{suffix}.png')

            if pct_px_mask.sum() > 10:
                pct_px = err_px[pct_px_mask] / targ_px[pct_px_mask] * 100.0
                _save_single_pixel(hgt_px[pct_px_mask], pct_px,
                                   f'Pct Error of Mean Yearly Max vs Terrain – Target ≥{pct_threshold} mm – {target_names[ch]}',
                                   'Error (%) of mean yearly max',
                                   output_dir / f'pct_error_vs_terrain_mean_yrmax_snowonly_{target_names[ch]}{suffix}.png')

    # ==================================================================
    # PER-REGION COMBINED PANELS
    # ==================================================================
    if region_defs is not None:
        lat, lon = get_lat_lon()
        if lat is None:
            print("  Warning: lat/lon not available, skipping per-region error vs terrain plots")
            return

        region_names = list(region_defs.keys())
        n_regions = len(region_names)
        ncols = min(3, n_regions)
        nrows = int(np.ceil(n_regions / ncols))

        for ch in range(n_channels):
            # --- Binned regional panels ---
            fig_abs, axes_abs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes_abs = np.array(axes_abs).flatten()
            fig_pct, axes_pct = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes_pct = np.array(axes_pct).flatten()

            # --- Mean-yearly-max pixel regional panels ---
            if pred_mean_max is not None:
                fig_px_abs, axes_px_abs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
                axes_px_abs = np.array(axes_px_abs).flatten()
                fig_px_pct, axes_px_pct = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
                axes_px_pct = np.array(axes_px_pct).flatten()

            # --- Temporal-mean pixel regional panels ---
            fig_tm_abs, axes_tm_abs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes_tm_abs = np.array(axes_tm_abs).flatten()
            fig_tm_pct, axes_tm_pct = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes_tm_pct = np.array(axes_tm_pct).flatten()

            for ax_idx, region_name in enumerate(region_names):
                bounds = region_defs[region_name]
                spatial_mask = (
                    (lon >= bounds['lon_min']) & (lon <= bounds['lon_max']) &
                    (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
                )
                if not np.any(spatial_mask):
                    axes_abs[ax_idx].set_visible(False)
                    axes_pct[ax_idx].set_visible(False)
                    axes_tm_abs[ax_idx].set_visible(False)
                    axes_tm_pct[ax_idx].set_visible(False)
                    if pred_mean_max is not None:
                        axes_px_abs[ax_idx].set_visible(False)
                        axes_px_pct[ax_idx].set_visible(False)
                    continue

                hgt_flat = hgt_broadcast[:, spatial_mask].ravel()
                targ_flat = targets[:, ch][:, spatial_mask].ravel()
                err_flat = (predictions[:, ch][:, spatial_mask] - targets[:, ch][:, spatial_mask]).ravel()
                pct_mask_r = targ_flat >= pct_threshold

                panel_name = panel_label(ax_idx, region_name)
                _make_binned_panel(axes_abs[ax_idx], hgt_flat, err_flat,
                                   panel_name, 'Error (pred − target) [mm]')
                if pct_mask_r.sum() > 10:
                    pct_flat = err_flat[pct_mask_r] / targ_flat[pct_mask_r] * 100.0
                    _make_binned_panel(axes_pct[ax_idx], hgt_flat[pct_mask_r], pct_flat,
                                       panel_name, f'Error (%) target ≥{pct_threshold} mm')
                else:
                    axes_pct[ax_idx].set_visible(False)

                hgt_rpx = hgt[spatial_mask].ravel()
                err_tm = (pred_tmean[ch][spatial_mask] - targ_tmean[ch][spatial_mask]).ravel()
                targ_tm = targ_tmean[ch][spatial_mask].ravel()
                snow_tm = targ_tm >= snow_threshold
                pct_tm_mask = targ_tm >= pct_threshold

                _make_pixel_panel(axes_tm_abs[ax_idx], hgt_rpx, err_tm,
                                  panel_name, 'Error of temporal mean [mm]')
                if pct_tm_mask.sum() > 10:
                    pct_tm = err_tm[pct_tm_mask] / targ_tm[pct_tm_mask] * 100.0
                    _make_pixel_panel(axes_tm_pct[ax_idx], hgt_rpx[pct_tm_mask], pct_tm,
                                      panel_name, f'Pct error of temporal mean (%) target ≥{pct_threshold} mm')
                else:
                    axes_tm_pct[ax_idx].set_visible(False)

                if pred_mean_max is not None:
                    err_rpx = (pred_mean_max[ch][spatial_mask] - targ_mean_max[ch][spatial_mask]).ravel()
                    targ_rpx = targ_mean_max[ch][spatial_mask].ravel()
                    snow_rpx = targ_rpx >= snow_threshold
                    pct_rpx_mask = targ_rpx >= pct_threshold

                    _make_pixel_panel(axes_px_abs[ax_idx], hgt_rpx, err_rpx,
                                      panel_name, 'Error of mean yearly max [mm]')
                    if pct_rpx_mask.sum() > 10:
                        pct_rpx = err_rpx[pct_rpx_mask] / targ_rpx[pct_rpx_mask] * 100.0
                        _make_pixel_panel(axes_px_pct[ax_idx], hgt_rpx[pct_rpx_mask], pct_rpx,
                                          panel_name, f'Pct error of mean yearly max (%) target ≥{pct_threshold} mm')
                    else:
                        axes_px_pct[ax_idx].set_visible(False)

            for ax_idx in range(n_regions, len(axes_abs)):
                axes_abs[ax_idx].set_visible(False)
                axes_pct[ax_idx].set_visible(False)
                axes_tm_abs[ax_idx].set_visible(False)
                axes_tm_pct[ax_idx].set_visible(False)
                if pred_mean_max is not None:
                    axes_px_abs[ax_idx].set_visible(False)
                    axes_px_pct[ax_idx].set_visible(False)

            fig_abs.suptitle(f'Absolute Error vs Terrain by Region – {target_names[ch]}',
                             fontsize='xx-large', fontweight='bold')
            fig_abs.tight_layout()
            fpath = output_dir / f'error_vs_terrain_regions_{target_names[ch]}{suffix}.png'
            fig_abs.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig_abs)
            print(f"  Saved: {fpath.name}")

            fig_pct.suptitle(f'Percentage Error vs Terrain by Region (Snow Only) – {target_names[ch]}',
                             fontsize='xx-large', fontweight='bold')
            fig_pct.tight_layout()
            fpath = output_dir / f'pct_error_vs_terrain_regions_{target_names[ch]}{suffix}.png'
            fig_pct.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig_pct)
            print(f"  Saved: {fpath.name}")

            fig_tm_abs.suptitle(f'Abs Error of Temporal Mean vs Terrain by Region – {target_names[ch]}',
                                fontsize='xx-large', fontweight='bold')
            fig_tm_abs.tight_layout()
            fpath = output_dir / f'error_vs_terrain_tmean_regions_{target_names[ch]}{suffix}.png'
            fig_tm_abs.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig_tm_abs)
            print(f"  Saved: {fpath.name}")

            fig_tm_pct.suptitle(f'Pct Error of Temporal Mean vs Terrain by Region (Snow Only) – {target_names[ch]}',
                                fontsize='xx-large', fontweight='bold')
            fig_tm_pct.tight_layout()
            fpath = output_dir / f'pct_error_vs_terrain_tmean_regions_{target_names[ch]}{suffix}.png'
            fig_tm_pct.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close(fig_tm_pct)
            print(f"  Saved: {fpath.name}")

            if pred_mean_max is not None:
                fig_px_abs.suptitle(f'Abs Error of Mean Yearly Max vs Terrain by Region – {target_names[ch]}',
                                    fontsize='xx-large', fontweight='bold')
                fig_px_abs.tight_layout()
                fpath = output_dir / f'error_vs_terrain_mean_yrmax_regions_{target_names[ch]}{suffix}.png'
                fig_px_abs.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
                plt.close(fig_px_abs)
                print(f"  Saved: {fpath.name}")

                fig_px_pct.suptitle(f'Pct Error of Mean Yearly Max vs Terrain by Region (Snow Only) – {target_names[ch]}',
                                    fontsize='xx-large', fontweight='bold')
                fig_px_pct.tight_layout()
                fpath = output_dir / f'pct_error_vs_terrain_mean_yrmax_regions_{target_names[ch]}{suffix}.png'
                fig_px_pct.savefig(fpath, dpi=FIGURE_DPI, bbox_inches='tight')
                plt.close(fig_px_pct)
                print(f"  Saved: {fpath.name}")
