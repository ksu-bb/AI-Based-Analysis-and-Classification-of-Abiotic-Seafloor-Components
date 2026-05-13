from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, laplace

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


def robust_normalize_sonar_by_raster(sonar: np.ndarray, source_rasters: np.ndarray):
    """Robust z-score normalization independently for each source sonar raster.

    Parameters
    ----------
    sonar : array, shape (N, H, W)
    source_rasters : array, shape (N,)

    Returns
    -------
    sonar_norm : array, shape (N, H, W)
    stats : dict with median/IQR per raster
    """
    sonar = np.asarray(sonar, dtype=np.float32)
    source_rasters = np.asarray(source_rasters).astype(str)
    out = np.empty_like(sonar, dtype=np.float32)
    stats = {}
    for src in np.unique(source_rasters):
        idx = source_rasters == src
        vals = sonar[idx].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            med, iqr = 0.0, 1.0
        else:
            med = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr < 1e-6:
                iqr = float(np.std(vals) + 1e-6)
        out[idx] = (sonar[idx] - med) / (iqr + 1e-6)
        stats[src] = {"median": med, "iqr": iqr}
    return out, stats


def robust_normalize_single_raster(arr: np.ndarray, valid_mask: np.ndarray | None = None):
    vals = arr[np.isfinite(arr)] if valid_mask is None else arr[valid_mask & np.isfinite(arr)]
    if len(vals) == 0:
        return arr.astype(np.float32), {"median": 0.0, "iqr": 1.0}
    med = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = float(np.std(vals) + 1e-6)
    return ((arr - med) / (iqr + 1e-6)).astype(np.float32), {"median": med, "iqr": iqr}


def _nan_safe_flat_stats(arr: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    names = ["mean", "std", "iqr", "p05", "p25", "p50", "p75", "p95", "min", "max"]
    if len(flat) == 0:
        vals = [np.nan] * len(names)
    else:
        q05, q25, q50, q75, q95 = np.percentile(flat, [5, 25, 50, 75, 95])
        vals = [float(np.mean(flat)), float(np.std(flat)), float(q75 - q25), float(q05), float(q25), float(q50), float(q75), float(q95), float(np.min(flat)), float(np.max(flat))]
    return vals, [f"{prefix}_{n}" for n in names]


def _center_context_features(arr: np.ndarray, prefix: str, center_frac: float = 0.5) -> tuple[list[float], list[str]]:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    h0 = int(h * (1 - center_frac) / 2)
    h1 = int(h * (1 + center_frac) / 2)
    w0 = int(w * (1 - center_frac) / 2)
    w1 = int(w * (1 + center_frac) / 2)
    center = arr[h0:h1, w0:w1]
    mask = np.ones_like(arr, dtype=bool)
    mask[h0:h1, w0:w1] = False
    outer = arr[mask]
    full = arr.reshape(-1)
    def s(x, f):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        return np.nan if len(x) == 0 else float(f(x))
    center_mean = s(center, np.mean)
    full_mean = s(full, np.mean)
    outer_mean = s(outer, np.mean)
    center_std = s(center, np.std)
    full_std = s(full, np.std)
    outer_std = s(outer, np.std)
    vals = [
        center_mean, center_std, outer_mean, outer_std,
        center_mean - full_mean if np.isfinite(center_mean) and np.isfinite(full_mean) else np.nan,
        center_mean - outer_mean if np.isfinite(center_mean) and np.isfinite(outer_mean) else np.nan,
        center_std - full_std if np.isfinite(center_std) and np.isfinite(full_std) else np.nan,
        center_std - outer_std if np.isfinite(center_std) and np.isfinite(outer_std) else np.nan,
    ]
    names = [
        "center_mean", "center_std", "outer_mean", "outer_std",
        "center_minus_full_mean", "center_minus_outer_mean",
        "center_minus_full_std", "center_minus_outer_std",
    ]
    return vals, [f"{prefix}_{n}" for n in names]


def _multiscale_features(arr: np.ndarray, prefix: str, sizes=(16, 32, 64)) -> tuple[list[float], list[str]]:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    vals, names = [], []
    for s in sizes:
        ss = min(s, h, w)
        h0 = (h - ss) // 2
        w0 = (w - ss) // 2
        patch = arr[h0:h0+ss, w0:w0+ss]
        v, n = _nan_safe_flat_stats(patch, f"{prefix}_s{ss}")
        vals.extend(v)
        names.extend(n)
    return vals, names


def _bathymorph_features(depth: np.ndarray, slope: np.ndarray) -> tuple[list[float], list[str]]:
    vals, names = [], []
    depth = np.asarray(depth, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)
    # roughness as residual after local mean
    for arr, prefix in [(depth, "depth"), (slope, "slope")]:
        smooth = uniform_filter(arr, size=7, mode="nearest")
        residual = arr - smooth
        v, n = _nan_safe_flat_stats(residual, f"{prefix}_roughness_residual")
        vals.extend(v); names.extend(n)
        local_lap = laplace(arr)
        v, n = _nan_safe_flat_stats(local_lap, f"{prefix}_laplace")
        vals.extend(v); names.extend(n)
    # relief for depth
    flat = depth[np.isfinite(depth)]
    if len(flat) > 0:
        vals.append(float(np.percentile(flat, 95) - np.percentile(flat, 5)))
    else:
        vals.append(np.nan)
    names.append("depth_local_relief_p95_p05")
    # slope thresholds in degrees, if slope channel is in degrees
    for thr in [1, 2, 5, 10]:
        sf = slope[np.isfinite(slope)]
        vals.append(float(np.mean(sf > thr)) if len(sf) else np.nan)
        names.append(f"slope_frac_gt_{thr}")
    return vals, names


def _glcm_features(arr: np.ndarray, prefix: str = "sonar_norm", levels: int = 32, distances=(1,2,4), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)) -> tuple[list[float], list[str]]:
    names = [f"{prefix}_glcm_contrast", f"{prefix}_glcm_dissimilarity", f"{prefix}_glcm_entropy"]
    if not SKIMAGE_AVAILABLE:
        return [np.nan, np.nan, np.nan], names
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return [np.nan, np.nan, np.nan], names
    lo, hi = np.percentile(x[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [np.nan, np.nan, np.nan], names
    xq = np.clip((x - lo) / (hi - lo), 0, 1)
    xq = np.floor(xq * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(xq, distances=list(distances), angles=list(angles), levels=levels, symmetric=True, normed=True)
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    dissim = float(np.mean(graycoprops(glcm, "dissimilarity")))
    p = glcm.astype(np.float64)
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    return [contrast, dissim, entropy], names


def patch_features_one(sonar_raw: np.ndarray, sonar_norm: np.ndarray, depth: np.ndarray, slope: np.ndarray, valid_mask: np.ndarray | None = None, use_glcm: bool = True) -> tuple[list[float], list[str]]:
    vals, names = [], []
    for arr, prefix in [(sonar_raw, "sonar_raw"), (sonar_norm, "sonar_norm"), (depth, "depth"), (slope, "slope")]:
        v, n = _nan_safe_flat_stats(arr, prefix); vals.extend(v); names.extend(n)
        v, n = _center_context_features(arr, prefix); vals.extend(v); names.extend(n)
        v, n = _multiscale_features(arr, prefix); vals.extend(v); names.extend(n)
    if valid_mask is not None:
        v, n = _nan_safe_flat_stats(valid_mask.astype(np.float32), "valid_mask"); vals.extend(v); names.extend(n)
    v, n = _bathymorph_features(depth, slope); vals.extend(v); names.extend(n)
    if use_glcm:
        v, n = _glcm_features(sonar_norm); vals.extend(v); names.extend(n)
    return vals, names


def build_patch_feature_table(
    sonar_raw: np.ndarray,
    bathy: np.ndarray,
    valid_masks: np.ndarray,
    y: np.ndarray | None = None,
    stations: np.ndarray | None = None,
    source_rasters: np.ndarray | None = None,
    shift_xy_px: np.ndarray | None = None,
    valid_ratios: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    use_glcm: bool = True,
):
    sonar_norm, sonar_norm_stats = robust_normalize_sonar_by_raster(sonar_raw, source_rasters if source_rasters is not None else np.array(["unknown"]*len(sonar_raw)))
    rows = []
    feature_names = None
    n = len(sonar_raw)
    for i in range(n):
        vals, names = patch_features_one(sonar_raw[i], sonar_norm[i], bathy[i,0], bathy[i,1], valid_masks[i], use_glcm=use_glcm)
        if feature_names is None:
            feature_names = names
        row = dict(zip(feature_names, vals))
        if y is not None: row["target"] = float(y[i])
        if stations is not None: row["station"] = str(stations[i])
        if source_rasters is not None: row["source_raster"] = str(source_rasters[i])
        if shift_xy_px is not None:
            row["shift_x_px"] = int(shift_xy_px[i,0]); row["shift_y_px"] = int(shift_xy_px[i,1])
        if valid_ratios is not None: row["valid_ratio"] = float(valid_ratios[i])
        if coords is not None:
            row["x"] = float(coords[i,0]); row["y"] = float(coords[i,1])
        rows.append(row)
    return pd.DataFrame(rows), feature_names, sonar_norm_stats


def aggregate_feature_table(df: pd.DataFrame, group_cols: list[str], target_col: str = "target") -> pd.DataFrame:
    meta_cols = set(group_cols + [target_col, "source_raster", "station"])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
    agg = df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "min", "max"]) 
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    if target_col in df.columns:
        target = df.groupby(group_cols)[target_col].mean().reset_index()
        agg = agg.merge(target, on=group_cols, how="left")
    return agg


def clean_feature_columns(columns: list[str], exclude_patterns=None, extra_exclude=None) -> list[str]:
    exclude_patterns = exclude_patterns or ["valid_mask", "valid_ratio", "n_patches", "n_rasters", "shift_", "source_raster"]
    extra_exclude = set(extra_exclude or [])
    out = []
    for c in columns:
        if c in extra_exclude:
            continue
        if any(p in c for p in exclude_patterns):
            continue
        out.append(c)
    return out
