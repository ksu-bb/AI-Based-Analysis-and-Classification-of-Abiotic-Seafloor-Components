from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter


def read_station_file(path: str | Path) -> pd.DataFrame:
    """Read station table with columns Station, X, Y, Mz.

    Delimiter can be tab, comma, semicolon or whitespace.
    """
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ["station", "name", "id"]:
            rename[c] = "Station"
        elif cl == "x":
            rename[c] = "X"
        elif cl == "y":
            rename[c] = "Y"
        elif cl in ["mz", "mz_phi", "target"]:
            rename[c] = "Mz"
    df = df.rename(columns=rename)
    required = {"Station", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Station file {path} misses columns: {missing}")
    if "Mz" not in df.columns:
        df["Mz"] = np.nan
    return df[["Station", "X", "Y", "Mz"]].copy()


def read_raster_array(path: str | Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        mask = src.read_masks(1) > 0
    valid = mask & np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    return arr, valid, profile, transform, crs, nodata


def reproject_to_match(src_arr, src_valid, src_profile, dst_profile, resampling=Resampling.bilinear):
    dst_arr = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=resampling,
        src_nodata=src_profile.get("nodata"),
        dst_nodata=np.nan,
    )
    dst_valid_f = np.zeros((dst_profile["height"], dst_profile["width"]), dtype=np.float32)
    reproject(
        source=src_valid.astype(np.float32),
        destination=dst_valid_f,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    dst_valid = dst_valid_f > 0.5
    return dst_arr, dst_valid


def compute_slope_from_bathy(depth: np.ndarray, transform) -> np.ndarray:
    """Compute slope in degrees using raster pixel size."""
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)
    # Fill NaNs for gradient to avoid propagation, then restore NaN-like areas by caller mask.
    filled = np.asarray(depth, dtype=np.float32).copy()
    if np.isnan(filled).any():
        med = np.nanmedian(filled)
        if not np.isfinite(med):
            med = 0.0
        filled[~np.isfinite(filled)] = med
        filled = median_filter(filled, size=3)
    gy, gx = np.gradient(filled, pixel_y, pixel_x)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad).astype(np.float32)


def make_sonar_valid_mask(arr: np.ndarray, base_valid: np.ndarray, valid_min=57.0, valid_max=255.0, extra_nodata=(255.0,)):
    valid = base_valid & np.isfinite(arr)
    if valid_min is not None:
        valid &= arr >= valid_min
    if valid_max is not None:
        valid &= arr < valid_max
    for v in extra_nodata or []:
        valid &= arr != v
    return valid


def extract_window(arr: np.ndarray, row: int, col: int, patch_size: int):
    half = patch_size // 2
    r0, r1 = row - half, row - half + patch_size
    c0, c1 = col - half, col - half + patch_size
    if r0 < 0 or c0 < 0 or r1 > arr.shape[0] or c1 > arr.shape[1]:
        return None
    return arr[r0:r1, c0:c1]
