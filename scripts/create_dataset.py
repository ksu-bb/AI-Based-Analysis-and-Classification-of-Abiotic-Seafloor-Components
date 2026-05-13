#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from tqdm import tqdm

from seabed_rf.raster_utils import read_station_file, read_raster_array, compute_slope_from_bathy, make_sonar_valid_mask, extract_window


def parse_args():
    p = argparse.ArgumentParser(description="Create NPZ dataset of sonar+bathymetry patches around sampling stations.")
    p.add_argument("--sonar", nargs="+", required=True, help="One or more side-scan sonar GeoTIFF files.")
    p.add_argument("--bathy", required=True, help="Bathymetry GeoTIFF file on the same grid/CRS as sonar rasters.")
    p.add_argument("--stations", required=True, help="Station table with Station, X, Y, Mz columns.")
    p.add_argument("--output", required=True, help="Output NPZ path.")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--shift", type=int, default=None, help="Shift in pixels. Default: patch_size//4.")
    p.add_argument("--center-only", action="store_true", help="Use only centered patch, no shifted patches.")
    p.add_argument("--min-valid-ratio", type=float, default=0.80)
    p.add_argument("--sonar-valid-min", type=float, default=57.0)
    p.add_argument("--sonar-valid-max", type=float, default=255.0)
    p.add_argument("--sonar-extra-nodata", nargs="*", type=float, default=[255.0])
    return p.parse_args()


def local_fill(arr, valid):
    arr = arr.astype(np.float32).copy()
    vals = arr[valid & np.isfinite(arr)]
    fill = float(np.median(vals)) if len(vals) else 0.0
    arr[~valid] = fill
    arr[~np.isfinite(arr)] = fill
    return arr


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    stations_df = read_station_file(args.stations)
    patch_size = args.patch_size
    shift = args.shift if args.shift is not None else patch_size // 4
    offsets = [(0, 0)] if args.center_only else [
        (0, 0), (-shift, 0), (shift, 0), (0, -shift), (0, shift),
        (-shift, -shift), (-shift, shift), (shift, -shift), (shift, shift),
    ]

    bathy_arr, bathy_valid, bathy_profile, bathy_transform, bathy_crs, _ = read_raster_array(args.bathy)
    bathy_slope = compute_slope_from_bathy(bathy_arr, bathy_transform)

    sonar_patches = []
    bathy_patches = []
    valid_masks = []
    targets = []
    station_names = []
    coords = []
    source_rasters = []
    shift_xy_px = []
    valid_ratios = []
    scalar_features = []

    skipped = []

    for sonar_path in args.sonar:
        sonar_arr, sonar_base_valid, sonar_profile, sonar_transform, sonar_crs, _ = read_raster_array(sonar_path)
        if sonar_crs != bathy_crs:
            raise ValueError(f"CRS mismatch for {sonar_path}. Reproject rasters before running this script.")
        if sonar_arr.shape != bathy_arr.shape or sonar_transform != bathy_transform:
            raise ValueError(f"Grid mismatch for {sonar_path}. This dataset script expects sonar and bathy on the same grid.")
        sonar_valid = make_sonar_valid_mask(sonar_arr, sonar_base_valid, args.sonar_valid_min, args.sonar_valid_max, args.sonar_extra_nodata)
        combined_valid = sonar_valid & bathy_valid & np.isfinite(bathy_slope)

        print(f"Processing {sonar_path}; stations={len(stations_df)}")
        for _, st in tqdm(stations_df.iterrows(), total=len(stations_df)):
            station, x, y, target = str(st["Station"]), float(st["X"]), float(st["Y"]), float(st["Mz"])
            col_f, row_f = ~sonar_transform * (x, y)
            row0, col0 = int(round(row_f)), int(round(col_f))
            if row0 < 0 or col0 < 0 or row0 >= sonar_arr.shape[0] or col0 >= sonar_arr.shape[1]:
                skipped.append((station, os.path.basename(sonar_path), "outside_raster"))
                continue
            for dx, dy in offsets:
                row = row0 + dy
                col = col0 + dx
                mask_win = extract_window(combined_valid.astype(np.uint8), row, col, patch_size)
                if mask_win is None:
                    skipped.append((station, os.path.basename(sonar_path), "window_outside"))
                    continue
                mask_win = mask_win.astype(bool)
                vr = float(mask_win.mean())
                if vr < args.min_valid_ratio:
                    skipped.append((station, os.path.basename(sonar_path), "low_valid_ratio"))
                    continue
                sonar_win = extract_window(sonar_arr, row, col, patch_size)
                depth_win = extract_window(bathy_arr, row, col, patch_size)
                slope_win = extract_window(bathy_slope, row, col, patch_size)
                sonar_win = local_fill(sonar_win, mask_win)
                depth_win = local_fill(depth_win, mask_win & bathy_valid[row-patch_size//2:row-patch_size//2+patch_size, col-patch_size//2:col-patch_size//2+patch_size])
                slope_win = local_fill(slope_win, mask_win)
                cx, cy = xy(sonar_transform, row, col)

                sonar_patches.append(sonar_win)
                bathy_patches.append(np.stack([depth_win, slope_win], axis=0))
                valid_masks.append(mask_win.astype(np.float32))
                targets.append(target)
                station_names.append(station)
                coords.append([cx, cy])
                source_rasters.append(str(sonar_path))
                shift_xy_px.append([dx, dy])
                valid_ratios.append(vr)
                scalar_features.append([
                    float(np.mean(depth_win)),
                    float(np.std(depth_win)),
                    float(np.mean(slope_win)),
                    float(np.std(slope_win)),
                ])

    np.savez_compressed(
        output,
        sonar_patches=np.asarray(sonar_patches, dtype=np.float32),
        bathy_patches=np.asarray(bathy_patches, dtype=np.float32),
        valid_masks=np.asarray(valid_masks, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        scalar_features=np.asarray(scalar_features, dtype=np.float32),
        stations=np.asarray(station_names).astype(str),
        coords=np.asarray(coords, dtype=np.float32),
        source_rasters=np.asarray(source_rasters).astype(str),
        shift_xy_px=np.asarray(shift_xy_px, dtype=np.int32),
        valid_ratios=np.asarray(valid_ratios, dtype=np.float32),
    )
    meta = {
        "patch_size": patch_size,
        "offsets": offsets,
        "min_valid_ratio": args.min_valid_ratio,
        "n_patches": len(targets),
        "n_stations": int(len(np.unique(station_names))) if station_names else 0,
        "skipped_count": len(skipped),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset: {output}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
