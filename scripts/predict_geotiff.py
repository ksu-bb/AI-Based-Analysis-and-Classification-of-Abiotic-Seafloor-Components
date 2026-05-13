#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm

from seabed_rf.features import robust_normalize_single_raster, patch_features_one
from seabed_rf.raster_utils import (
    read_raster_array, reproject_to_match, compute_slope_from_bathy, make_sonar_valid_mask,
    extract_window, read_station_file,
)
from seabed_rf.modeling import regression_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Predict Mz_phi map from sonar and bathymetry GeoTIFF using trained RF patch-level model.")
    p.add_argument("--sonar", required=True, help="Side-scan sonar GeoTIFF.")
    p.add_argument("--bathy", required=True, help="Bathymetry GeoTIFF.")
    p.add_argument("--model", required=True, help="Path to RF_patch_level_weighted.joblib.")
    p.add_argument("--features", required=True, help="Path to RF_patch_level_weighted_feature_columns.json.")
    p.add_argument("--output", required=True, help="Output prediction GeoTIFF.")
    p.add_argument("--stations", default=None, help="Optional station table with Station, X, Y, Mz for point validation.")
    p.add_argument("--station-output", default=None, help="Optional CSV path for station predictions.")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--min-valid-ratio", type=float, default=0.80)
    p.add_argument("--nodata", type=float, default=-9999.0)
    p.add_argument("--sonar-valid-min", type=float, default=57.0)
    p.add_argument("--sonar-valid-max", type=float, default=255.0)
    p.add_argument("--sonar-extra-nodata", nargs="*", type=float, default=[255.0])
    p.add_argument("--no-glcm", action="store_true")
    p.add_argument("--preview", default=None, help="Optional PNG preview path.")
    p.add_argument("--fill-stride-blocks", action="store_true", help="Fill stride x stride block around each prediction.")
    return p.parse_args()


def align_inputs(sonar_tif, bathy_tif, args):
    sonar, sonar_base_valid, sonar_profile, sonar_transform, sonar_crs, _ = read_raster_array(sonar_tif)
    bathy, bathy_valid, bathy_profile, bathy_transform, bathy_crs, _ = read_raster_array(bathy_tif)
    if sonar_crs != bathy_crs or sonar.shape != bathy.shape or sonar_transform != bathy_transform:
        print("Bathymetry is reprojected/resampled to sonar grid...")
        dst_profile = sonar_profile.copy()
        dst_profile["transform"] = sonar_transform
        dst_profile["crs"] = sonar_crs
        dst_profile["height"], dst_profile["width"] = sonar.shape
        bathy, bathy_valid = reproject_to_match(bathy, bathy_valid, bathy_profile, dst_profile)
        bathy_transform = sonar_transform
    slope = compute_slope_from_bathy(bathy, sonar_transform)
    sonar_valid = make_sonar_valid_mask(sonar, sonar_base_valid, args.sonar_valid_min, args.sonar_valid_max, args.sonar_extra_nodata)
    global_valid = sonar_valid & bathy_valid & np.isfinite(slope)
    sonar_norm, norm_stats = robust_normalize_single_raster(sonar, sonar_valid)
    return sonar, sonar_norm, bathy.astype(np.float32), slope.astype(np.float32), global_valid, sonar_profile, sonar_transform, norm_stats


def make_feature_row(sonar, sonar_norm, bathy, slope, valid, row, col, patch_size, use_glcm):
    sw = extract_window(sonar, row, col, patch_size)
    snw = extract_window(sonar_norm, row, col, patch_size)
    dw = extract_window(bathy, row, col, patch_size)
    slw = extract_window(slope, row, col, patch_size)
    mw = extract_window(valid.astype(np.float32), row, col, patch_size)
    if sw is None or snw is None or dw is None or slw is None or mw is None:
        return None, "patch_window_outside_raster"
    if float(np.mean(mw > 0.5)) < 0.80:
        return None, "low_valid_ratio"
    vals, names = patch_features_one(sw, snw, dw, slw, mw, use_glcm=use_glcm)
    return dict(zip(names, vals)), "predicted"


def predict_map(args):
    model = joblib.load(args.model)
    model_feature_columns = json.loads(Path(args.features).read_text(encoding="utf-8"))
    sonar, sonar_norm, bathy, slope, valid, profile, transform, norm_stats = align_inputs(args.sonar, args.bathy, args)
    h, w = sonar.shape
    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.uint16)
    half = args.patch_size // 2
    rows = range(half, h - half, args.stride)
    cols = range(half, w - half, args.stride)
    candidate_count = len(list(rows)) * len(list(cols))
    print(f"Raster shape: {h} x {w}; candidate windows: {candidate_count}")
    rows = range(half, h - half, args.stride)
    cols = range(half, w - half, args.stride)
    skipped_center = 0
    skipped_lowvalid = 0
    predicted_windows = 0
    for r in tqdm(rows, desc="Rows"):
        batch_feats, batch_pos = [], []
        for c in cols:
            if not valid[r, c]:
                skipped_center += 1
                continue
            rowdict, status = make_feature_row(sonar, sonar_norm, bathy, slope, valid, r, c, args.patch_size, use_glcm=not args.no_glcm)
            if status != "predicted":
                if status == "low_valid_ratio": skipped_lowvalid += 1
                continue
            batch_feats.append([rowdict.get(f, np.nan) for f in model_feature_columns])
            batch_pos.append((r, c))
        if batch_feats:
            preds = model.predict(np.asarray(batch_feats, dtype=np.float32))
            for (r, c), p in zip(batch_pos, preds):
                if args.fill_stride_blocks:
                    r0, r1 = max(0, r - args.stride//2), min(h, r + args.stride//2)
                    c0, c1 = max(0, c - args.stride//2), min(w, c + args.stride//2)
                    block_valid = valid[r0:r1, c0:c1]
                    pred_sum[r0:r1, c0:c1][block_valid] += p
                    pred_count[r0:r1, c0:c1][block_valid] += 1
                else:
                    pred_sum[r, c] += p
                    pred_count[r, c] += 1
                predicted_windows += 1
    out = np.full((h, w), args.nodata, dtype=np.float32)
    ok = (pred_count > 0) & valid
    out[ok] = pred_sum[ok] / pred_count[ok]
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=args.nodata, compress="deflate")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            output.unlink()
        except PermissionError:
            output = output.with_name(output.stem + "_new" + output.suffix)
            print("Output is locked; writing to", output)
    with rasterio.open(output, "w", **out_profile) as dst:
        dst.write(out, 1)
    stats = {
        "output": str(output),
        "predicted_pixels": int(ok.sum()),
        "predicted_windows": int(predicted_windows),
        "skipped_center_invalid": int(skipped_center),
        "skipped_low_valid": int(skipped_lowvalid),
        "sonar_norm_stats": norm_stats,
    }
    output.with_suffix(".json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved", output)
    return output, out, valid, transform


def predict_stations(args, output_tif=None):
    if args.stations is None:
        return None
    model = joblib.load(args.model)
    model_feature_columns = json.loads(Path(args.features).read_text(encoding="utf-8"))
    sonar, sonar_norm, bathy, slope, valid, profile, transform, norm_stats = align_inputs(args.sonar, args.bathy, args)
    st = read_station_file(args.stations)
    rows = []
    for _, s in st.iterrows():
        station, x, ytrue = str(s["Station"]), float(s["X"]), float(s["Mz"])
        ycoord = float(s["Y"])
        try:
            r, c = rowcol(transform, x, ycoord)
        except Exception:
            rows.append({"Station": station, "X": x, "Y": ycoord, "y_true": ytrue, "status": "outside_raster_bounds"})
            continue
        if r < 0 or c < 0 or r >= sonar.shape[0] or c >= sonar.shape[1]:
            rows.append({"Station": station, "X": x, "Y": ycoord, "y_true": ytrue, "status": "outside_raster_index"})
            continue
        if not valid[r, c]:
            rows.append({"Station": station, "X": x, "Y": ycoord, "y_true": ytrue, "status": "center_invalid"})
            continue
        rowdict, status = make_feature_row(sonar, sonar_norm, bathy, slope, valid, r, c, args.patch_size, use_glcm=not args.no_glcm)
        rec = {"Station": station, "X": x, "Y": ycoord, "y_true": ytrue, "row": r, "col": c, "status": status}
        if status == "predicted":
            X = np.asarray([[rowdict.get(f, np.nan) for f in model_feature_columns]], dtype=np.float32)
            rec["y_pred"] = float(model.predict(X)[0])
            rec["error"] = rec["y_pred"] - ytrue if np.isfinite(ytrue) else np.nan
            rec["abs_error"] = abs(rec["error"]) if np.isfinite(rec["error"]) else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    out_csv = Path(args.station_output) if args.station_output else Path(args.output).with_name(Path(args.output).stem + "_stations.csv")
    df.to_csv(out_csv, index=False)
    pred = df[df["status"] == "predicted"].copy()
    if len(pred) and pred["y_true"].notna().all():
        print("Station metrics:", regression_metrics(pred["y_true"], pred["y_pred"]))
    print("Saved station predictions", out_csv)
    return df


def preview(output_tif, preview_png, nodata):
    if preview_png is None:
        return
    with rasterio.open(output_tif) as src:
        arr = src.read(1).astype(np.float32)
        b = src.bounds
    arr = np.where(arr == nodata, np.nan, arr)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(arr, extent=[b.left,b.right,b.bottom,b.top], cmap="viridis")
    plt.colorbar(im, label="Predicted Mz_phi")
    plt.title(Path(output_tif).name)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(preview_png, dpi=200)
    plt.close()
    print("Saved preview", preview_png)


def main():
    args = parse_args()
    out_tif, _, _, _ = predict_map(args)
    predict_stations(args, out_tif)
    preview(out_tif, args.preview, args.nodata)

if __name__ == "__main__":
    main()
