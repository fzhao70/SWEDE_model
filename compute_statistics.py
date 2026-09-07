"""
Standalone program to compute dataset statistics using streaming (online) algorithms.

This program efficiently computes:
- Static variables: mean, std
- Dynamic variables: mean, std
- Target variables: mean, std, min, max

Uses Welford's algorithm for numerically stable streaming mean/std computation.
Filters out invalid values (|value| > 1e29).

Usage:
    python compute_statistics.py [--sample-size N] [--output stats.json] [--workers N]
"""

import numpy as np
import argparse
from pathlib import Path
import netCDF4 as nc
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Manager
import time

from config import (
    DYNAMIC_PATH,
    TARGET_PATH,
    STATIC_PATH,
    ALL_STATIC_VARS,
    ALL_DYNAMIC_VARS_2D,
    ALL_DYNAMIC_VARS_MULTIDIM,
    ALL_DYNAMIC_3D_VARS,
    ALL_TARGET_VARS,
    MULTIDIM_VARS,
    DYNAMIC_FILE_TEMPLATE
)

TRAIN_YEARS = [i for i in range(1950, 2025)]

ALL_3D_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

ALL_DYNAMIC_VARS = ALL_DYNAMIC_VARS_2D + ALL_DYNAMIC_VARS_MULTIDIM

class WelfordAccumulator:
    """
    Welford's algorithm for streaming mean and variance computation.
    Numerically stable and memory efficient.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update_batch(self, values):
        """Update statistics with a batch of values (numpy array)."""
        valid_mask = np.isfinite(values) & (np.abs(values) <= 1e29)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return

        self.min_val = min(self.min_val, float(np.min(valid_values)))
        self.max_val = max(self.max_val, float(np.max(valid_values)))

        # Update mean and variance using Welford's algorithm
        for value in valid_values.ravel():
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def merge(self, other):
        """
        Merge another WelfordAccumulator into this one.
        Uses Chan's parallel variance algorithm.
        """
        if other.count == 0:
            return

        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.min_val = other.min_val
            self.max_val = other.max_val
            return

        total_count = self.count + other.count

        # Merge means using weighted average
        delta = other.mean - self.mean
        new_mean = self.mean + delta * other.count / total_count

        # Merge M2 (sum of squared differences) using Chan's algorithm
        new_m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total_count

        self.count = total_count
        self.mean = new_mean
        self.m2 = new_m2
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)

    def finalize(self):
        """Get final statistics."""
        if self.count < 2:
            return {
                'mean': float(self.mean),
                'std': 0.0,
                'min': float(self.min_val) if self.count > 0 else 0.0,
                'max': float(self.max_val) if self.count > 0 else 0.0,
                'count': self.count
            }

        variance = self.m2 / self.count
        std = np.sqrt(variance)

        return {
            'mean': float(self.mean),
            'std': float(std) + 1e-12,  # Add small epsilon to prevent division by zero
            'min': float(self.min_val),
            'max': float(self.max_val),
            'count': self.count
        }


def load_static_data():
    """Load static geoinformation from wrfinput file.

    For XLONG and XLAT, this function expands them into positional encodings:
    - XLONG -> cos_lon, sin_lon
    - XLAT -> cos_lat, sin_lat
    """
    static_path = Path(STATIC_PATH)
    print(f"Loading static data from {static_path}")

    static_data = {}
    with nc.Dataset(static_path, 'r') as ds:
        for var_name in ALL_STATIC_VARS:
            if var_name in ds.variables:
                var_data = ds.variables[var_name][:]

                if var_data.ndim > 2:
                    var_data = var_data.squeeze()
                    if var_data.ndim > 2:
                        var_data = var_data[0]

                var_data = np.nan_to_num(var_data, nan=0.0, posinf=1e10, neginf=-1e10)

                if var_name == 'XLONG':
                    lon_rad = np.deg2rad(var_data)
                    static_data['cos_lon'] = np.cos(lon_rad)
                    static_data['sin_lon'] = np.sin(lon_rad)
                    print(f"  Loaded {var_name}: {var_data.shape} -> cos_lon, sin_lon")
                elif var_name == 'XLAT':
                    lat_rad = np.deg2rad(var_data)
                    static_data['cos_lat'] = np.cos(lat_rad)
                    static_data['sin_lat'] = np.sin(lat_rad)
                    print(f"  Loaded {var_name}: {var_data.shape} -> cos_lat, sin_lat")
                else:
                    static_data[var_name] = var_data
                    print(f"  Loaded {var_name}: {var_data.shape}")

    return static_data


def build_time_index(years):
    """Build index of all time steps for given years."""
    dynamic_path = Path(DYNAMIC_PATH)
    time_index = []

    print(f"\nIndexing years: {years}")
    for year in years:
        # Find sample file to get number of time steps
        pattern = DYNAMIC_FILE_TEMPLATE.replace("{var_name}", "*").replace("{year}", str(year))
        year_files = list(dynamic_path.glob(pattern))
        if not year_files:
            print(f"  Warning: No files found for year {year}")
            continue

        with nc.Dataset(year_files[0], 'r') as ds:
            if 'time' in ds.dimensions:
                n_times = len(ds.dimensions['time'])
            elif 'Time' in ds.dimensions:
                n_times = len(ds.dimensions['Time'])
            else:
                first_var = list(ds.variables.keys())[-1]
                n_times = ds.variables[first_var].shape[0]

        for t in range(n_times):
            time_index.append((year, t))

        print(f"  Year {year}: {n_times} time steps")

    print(f"Total: {len(time_index)} samples")
    return time_index


def load_dynamic_data(year, time_idx):
    """Load dynamic variables for a specific time. Returns dict keyed by variable/component name."""
    dynamic_path = Path(DYNAMIC_PATH)
    dynamic_data = {}

    for var_name in ALL_DYNAMIC_VARS:
        filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=year)
        var_file = dynamic_path / filename

        try:
            with nc.Dataset(var_file, 'r') as ds:
                var_names = [v for v in ds.variables.keys()
                            if v not in ['time', 'Time', 'lat', 'day', 'lon', 'latitude', 'longitude', 'lat2d', 'lon2d']]
                if not var_names:
                    continue

                var_data = ds.variables[var_names[0]][time_idx]

                if var_name in MULTIDIM_VARS:
                    n_components = MULTIDIM_VARS[var_name]['n_components']
                    component_names = MULTIDIM_VARS[var_name]['component_names']
                    if var_data.ndim == 3 and var_data.shape[0] == n_components:
                        var_data = np.where(np.abs(var_data) > 1e29, np.nan, var_data)

                        for i, comp_name in enumerate(component_names):
                            dynamic_data[comp_name] = var_data[i]
                else:
                    if var_data.ndim > 2:
                        var_data = var_data.squeeze()

                    if var_data.ndim != 2:
                        continue

                    var_data = np.where(np.abs(var_data) > 1e29, np.nan, var_data)
                    dynamic_data[var_name] = var_data

        except Exception as e:
            continue

    return dynamic_data


def load_3d_data(year, time_idx):
    """Load 3D variables for specific levels. Returns dict keyed by 'varname_level_N'."""
    dynamic_path = Path(DYNAMIC_PATH)
    data_3d = {}

    for var_name in ALL_DYNAMIC_3D_VARS:
        filename = DYNAMIC_FILE_TEMPLATE.format(var_name=var_name, year=year)
        var_file = dynamic_path / filename

        try:
            with nc.Dataset(var_file, 'r') as ds:
                var_names = [v for v in ds.variables.keys()
                            if v not in ['time', 'Time', 'lat', 'day', 'lon', 'latitude', 'longitude', 'lat2d', 'lon2d']]
                if not var_names:
                    continue

                var_data = ds.variables[var_names[0]][time_idx]

                # var_data should have shape (level, lat, lon) or similar
                for level in ALL_3D_LEVELS:
                    if var_data.ndim >= 3 and level < var_data.shape[0]:
                        # Assume first dimension is vertical level
                        level_data = var_data[level]
                        if level_data.ndim > 2:
                            level_data = level_data.squeeze()

                        if level_data.ndim != 2:
                            continue

                        level_data = np.where(np.abs(level_data) > 1e29, np.nan, level_data)
                        data_3d[f"{var_name}_level_{level}"] = level_data

        except Exception as e:
            continue

    return data_3d


def load_target(year, time_idx):
    """Load target variables. Returns dict keyed by variable name."""
    target_path = Path(TARGET_PATH)
    target_data = {}

    for target_var in ALL_TARGET_VARS:
        filename = DYNAMIC_FILE_TEMPLATE.format(var_name=target_var, year=year)
        target_file = target_path / filename

        if not target_file.exists():
            continue

        try:
            with nc.Dataset(target_file, 'r') as ds:
                var_names = [v for v in ds.variables.keys()
                            if v not in ['time', 'Time', 'lat', 'lon', 'latitude', 'longitude', 'day', 'lat2d', 'lon2d']]
                if not var_names:
                    continue

                var_data = ds.variables[var_names[0]][time_idx]
                if var_data.ndim > 2:
                    var_data = var_data.squeeze()

                if var_data.ndim != 2:
                    continue

                var_data = np.where(np.abs(var_data) > 1e29, np.nan, var_data)
                target_data[target_var] = var_data

        except Exception as e:
            continue

    return target_data


def process_samples_worker(worker_id, indices, time_index, progress_dict):
    """
    Worker function - processes its assigned samples and returns accumulators.

    Args:
        worker_id: Worker ID for progress tracking
        indices: List of sample indices to process
        time_index: Time index mapping
        progress_dict: Shared dict for progress updates
    """
    dynamic_accumulators = {}
    dynamic_3d_accumulators = {}
    target_accumulators = {}

    n_processed = 0
    n_failed = 0

    for idx in indices:
        year, time_idx = time_index[idx]

        try:
            dynamic = load_dynamic_data(year, time_idx)
            for var_name, var_data in dynamic.items():
                if var_name not in dynamic_accumulators:
                    dynamic_accumulators[var_name] = WelfordAccumulator()
                dynamic_accumulators[var_name].update_batch(var_data)

            dynamic_3d = load_3d_data(year, time_idx)
            for var_name, var_data in dynamic_3d.items():
                if var_name not in dynamic_3d_accumulators:
                    dynamic_3d_accumulators[var_name] = WelfordAccumulator()
                dynamic_3d_accumulators[var_name].update_batch(var_data)

            target = load_target(year, time_idx)
            for var_name, var_data in target.items():
                if var_name not in target_accumulators:
                    target_accumulators[var_name] = WelfordAccumulator()
                target_accumulators[var_name].update_batch(var_data)

            n_processed += 1

            if n_processed % 10 == 0:
                progress_dict[worker_id] = n_processed

        except Exception as e:
            n_failed += 1
            continue

    progress_dict[worker_id] = n_processed

    return worker_id, dynamic_accumulators, dynamic_3d_accumulators, target_accumulators, n_processed, n_failed


def compute_statistics(sample_size=None, n_workers=None):
    """
    Compute statistics using parallel streaming algorithm.
    NOW COMPUTES ALL VARIABLES and stores as dict!

    Args:
        sample_size: Number of samples to use (None = use all)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        Dictionary containing all statistics keyed by variable name
    """
    print("=" * 70)
    print("Computing Dataset Statistics with Parallel Streaming")
    print("ALL VARIABLES - Dictionary Format")
    print("=" * 70)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers")

    static_data = load_static_data()

    time_index = build_time_index(TRAIN_YEARS)

    total_samples = len(time_index)
    if sample_size is None or sample_size > total_samples:
        sample_size = total_samples

    indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)

    print(f"\nProcessing {sample_size} samples...")
    print(f"  Static vars:     {len(ALL_STATIC_VARS)}")
    print(f"  Dynamic 2D vars: {len(ALL_DYNAMIC_VARS_2D)}")
    print(f"  Dynamic MD vars: {len(ALL_DYNAMIC_VARS_MULTIDIM)} (multi-dimensional)")
    print(f"  Dynamic 3D vars: {len(ALL_DYNAMIC_3D_VARS)} × {len(ALL_3D_LEVELS)} levels")
    print(f"  Target vars:     {len(ALL_TARGET_VARS)}")
    print(f"  Filtering values with |value| > 1e29\n")

    static_accumulators = {}

    print("  Computing static statistics...")
    for var_name, var_data in static_data.items():
        if var_name not in static_accumulators:
            static_accumulators[var_name] = WelfordAccumulator()
        static_accumulators[var_name].update_batch(var_data)
    print("  ✓ Static statistics computed\n")

    dynamic_accumulators = {}
    dynamic_3d_accumulators = {}
    target_accumulators = {}

    samples_per_worker = len(indices) // n_workers
    worker_indices = []

    for i in range(n_workers):
        start = i * samples_per_worker
        end = len(indices) if i == n_workers - 1 else (i + 1) * samples_per_worker
        worker_indices.append(indices[start:end])

    print(f"  Splitting {len(indices)} samples across {n_workers} workers:")
    for i, w_indices in enumerate(worker_indices):
        print(f"    Worker {i}: {len(w_indices)} samples")

    manager = Manager()
    progress_dict = manager.dict()
    for i in range(n_workers):
        progress_dict[i] = 0

    total_processed = 0
    total_failed = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for wid in range(n_workers):
                future = executor.submit(
                    process_samples_worker,
                    wid,
                    worker_indices[wid],
                    time_index,
                    progress_dict
                )
                futures[future] = wid

            print(f"\n  Worker Progress:")
            with tqdm(total=len(indices), desc="  Overall", position=0) as pbar:
                last_total = 0

                while futures:
                    current_total = sum(progress_dict.values())
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total

                    done = [f for f in futures if f.done()]
                    for future in done:
                        wid = futures.pop(future)
                        try:
                            w_id, dyn_accs, dyn_3d_accs, tgt_accs, n_proc, n_fail = future.result()

                            for var_name, acc in dyn_accs.items():
                                if var_name not in dynamic_accumulators:
                                    dynamic_accumulators[var_name] = WelfordAccumulator()
                                dynamic_accumulators[var_name].merge(acc)

                            for var_name, acc in dyn_3d_accs.items():
                                if var_name not in dynamic_3d_accumulators:
                                    dynamic_3d_accumulators[var_name] = WelfordAccumulator()
                                dynamic_3d_accumulators[var_name].merge(acc)

                            for var_name, acc in tgt_accs.items():
                                if var_name not in target_accumulators:
                                    target_accumulators[var_name] = WelfordAccumulator()
                                target_accumulators[var_name].merge(acc)

                            total_processed += n_proc
                            total_failed += n_fail

                            tqdm.write(f"  ✓ Worker {w_id}: {n_proc} processed, {n_fail} failed")

                        except Exception as e:
                            tqdm.write(f"  ✗ Worker {wid} error: {e}")

                    if futures:
                        time.sleep(0.5)

                final_total = sum(progress_dict.values())
                if final_total > last_total:
                    pbar.update(final_total - last_total)

            print(f"\n  ✓ Total: {total_processed} processed, {total_failed} failed")

    else:
        print("\n  Sequential processing...")
        w_id, dyn_accs, dyn_3d_accs, tgt_accs, n_proc, n_fail = process_samples_worker(
            0, indices, time_index, progress_dict
        )

        dynamic_accumulators = dyn_accs
        dynamic_3d_accumulators = dyn_3d_accs
        target_accumulators = tgt_accs

        print(f"  ✓ Processed: {n_proc}, failed: {n_fail}")

    print("\nFinalizing statistics...")

    results = {
        'static': {},
        'dynamic': {},
        'dynamic_3d': {},
        'target': {}
    }

    for var_name, acc in static_accumulators.items():
        stats = acc.finalize()
        results['static'][var_name] = {
            'mean': stats['mean'],
            'std': stats['std']
        }

    for var_name, acc in dynamic_accumulators.items():
        stats = acc.finalize()
        results['dynamic'][var_name] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max']
        }

    for var_name, acc in dynamic_3d_accumulators.items():
        stats = acc.finalize()
        results['dynamic_3d'][var_name] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max']
        }

    for var_name, acc in target_accumulators.items():
        stats = acc.finalize()
        results['target'][var_name] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max']
        }

    print("\n" + "=" * 70)
    print("Statistics Summary")
    print("=" * 70)

    print(f"\nStatic Variables ({len(results['static'])} vars):")
    for var_name, stats in sorted(results['static'].items()):
        print(f"  {var_name:15s}: mean={stats['mean']:12.4e}, std={stats['std']:12.4e}")

    print(f"\nDynamic Variables ({len(results['dynamic'])} vars/components):")
    for var_name, stats in sorted(results['dynamic'].items()):
        print(f"  {var_name:15s}: min={stats['min']:10.4f}, max={stats['max']:10.4f}, mean={stats['mean']:10.4f}, std={stats['std']:10.4f}")

    print(f"\nDynamic 3D Variables ({len(results['dynamic_3d'])} var-levels):")
    for var_name, stats in sorted(results['dynamic_3d'].items()):
        print(f"  {var_name:20s}: min={stats['min']:10.4f}, max={stats['max']:10.4f}, mean={stats['mean']:10.4f}, std={stats['std']:10.4f}")

    print(f"\nTarget Variables ({len(results['target'])} vars):")
    for var_name, stats in sorted(results['target'].items()):
        print(f"  {var_name:15s}: min={stats['min']:10.4f}, max={stats['max']:10.4f}, mean={stats['mean']:10.4f}, std={stats['std']:10.4f}")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute dataset statistics with parallel streaming'
    )
    parser.add_argument('--output', type=str, default='normalization_stats.json',
                       help='Output file (default: normalization_stats.json)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers (default: auto)')

    args = parser.parse_args()

    stats = compute_statistics(sample_size=None, n_workers=32)

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Statistics saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
