#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance

from seabed_rf.features import build_patch_feature_table, aggregate_feature_table, clean_feature_columns
from seabed_rf.modeling import make_rf_pipeline, nested_param_grid_for_selected, regression_metrics, make_regression_bins


def parse_args():
    p = argparse.ArgumentParser(description="Train final RandomForest models for Mz_phi prediction from NPZ patch dataset.")
    p.add_argument("--dataset", required=True, help="NPZ dataset produced by create_dataset.py.")
    p.add_argument("--out-dir", default="Dataset/rf_final_training_outputs", help="Output directory.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--target-bins", type=int, default=5)
    p.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV before repeated CV.")
    p.add_argument("--tune-iter", type=int, default=50)
    p.add_argument("--tune-cv-splits", type=int, default=5)
    p.add_argument("--no-glcm", action="store_true", help="Disable GLCM features.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_scatter(df, experiment, out_png):
    m = regression_metrics(df["y_true"], df["y_pred"])
    plt.figure(figsize=(5,5))
    plt.scatter(df["y_true"], df["y_pred"], s=45, edgecolors="black", alpha=0.85)
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True Mz_phi")
    plt.ylabel("Predicted Mz_phi")
    plt.title(f"{experiment}\nMAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, R²={m['r2']:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def evaluate_repeated_cv(experiment, table, feature_cols, group_col, args, out_dir, sample_weight_col=None):
    X = table[feature_cols].values
    y = table["target"].values
    units = table[group_col].astype(str).values if group_col in table.columns else table.index.astype(str).values
    y_bins = make_regression_bins(y, args.target_bins)

    pipe = make_rf_pipeline(random_state=args.seed, feature_selection=True)
    best_params = None
    if args.tune:
        cv = StratifiedKFold(n_splits=min(args.tune_cv_splits, np.bincount(y_bins).min()), shuffle=True, random_state=args.seed)
        search = RandomizedSearchCV(
            pipe,
            nested_param_grid_for_selected(),
            n_iter=args.tune_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
            verbose=0,
        )
        fit_kwargs = {}
        if sample_weight_col and sample_weight_col in table.columns:
            # For sklearn Pipeline, pass sample weights to the final model only.
            fit_kwargs["model__sample_weight"] = table[sample_weight_col].values
        search.fit(X, y, **fit_kwargs)
        pipe = search.best_estimator_
        best_params = search.best_params_
        print(experiment, "best inner MAE:", -search.best_score_)
        print(experiment, "best params:", best_params)

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed)
    rows, preds = [], []
    for split_id, (tr, te) in enumerate(rskf.split(X, y_bins), 1):
        model = clone(pipe)
        try:
            model.set_params(model__random_state=args.seed + split_id)
        except Exception:
            pass
        fit_kwargs = {}
        if sample_weight_col and sample_weight_col in table.columns:
            fit_kwargs["model__sample_weight"] = table.iloc[tr][sample_weight_col].values
        model.fit(X[tr], y[tr], **fit_kwargs)
        yp = model.predict(X[te])
        m = regression_metrics(y[te], yp)
        rows.append({"experiment": experiment, "split_id": split_id, "repeat": (split_id-1)//args.n_splits+1, "fold": (split_id-1)%args.n_splits+1, **m, "n_test": len(te)})
        preds.append(pd.DataFrame({"experiment": experiment, "split_id": split_id, group_col: units[te], "y_true": y[te], "y_pred": yp}))

    cv_metrics = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)

    # Aggregated station-level OOF. For patch and station-raster levels, group_col is row id/raster id;
    # if station exists in table, map predictions to station for fair station-level metric.
    if "station" in table.columns and group_col != "station":
        unit_to_station = table[[group_col, "station"]].drop_duplicates().set_index(group_col)["station"].astype(str)
        pred_df["station"] = pred_df[group_col].map(unit_to_station)
    elif group_col == "station":
        pred_df["station"] = pred_df[group_col]
    station_oof = pred_df.groupby("station").agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), y_pred_std=("y_pred", "std"), n_predictions=("y_pred", "size")).reset_index()
    station_oof["experiment"] = experiment
    station_metrics = regression_metrics(station_oof["y_true"], station_oof["y_pred"])

    # Fit final model on all rows.
    final_model = clone(pipe)
    fit_kwargs = {}
    if sample_weight_col and sample_weight_col in table.columns:
        fit_kwargs["model__sample_weight"] = table[sample_weight_col].values
    final_model.fit(X, y, **fit_kwargs)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / f"{experiment}.joblib")
    (model_dir / f"{experiment}_feature_columns.json").write_text(json.dumps(feature_cols, indent=2, ensure_ascii=False), encoding="utf-8")
    if best_params is not None:
        (model_dir / f"{experiment}_best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")

    save_scatter(station_oof, experiment, out_dir / "figures" / f"{experiment}_station_oof_scatter.png")
    return cv_metrics, pred_df, station_oof, station_metrics


def main():
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    sonar = data["sonar_patches"].astype(np.float32)
    bathy = data["bathy_patches"].astype(np.float32)
    valid_masks = data["valid_masks"].astype(np.float32) if "valid_masks" in data else np.ones_like(sonar, dtype=np.float32)
    y = data["targets"].astype(np.float32)
    stations = data["stations"].astype(str)
    source_rasters = data["source_rasters"].astype(str) if "source_rasters" in data else np.array(["unknown"] * len(y))
    shift_xy_px = data["shift_xy_px"].astype(np.int32) if "shift_xy_px" in data else np.zeros((len(y), 2), dtype=np.int32)
    valid_ratios = data["valid_ratios"].astype(np.float32) if "valid_ratios" in data else valid_masks.reshape(len(y), -1).mean(axis=1)
    coords = data["coords"].astype(np.float32) if "coords" in data else np.full((len(y), 2), np.nan, dtype=np.float32)

    patch_df, patch_feature_names, sonar_norm_stats = build_patch_feature_table(
        sonar, bathy, valid_masks, y, stations, source_rasters, shift_xy_px, valid_ratios, coords,
        use_glcm=not args.no_glcm,
    )
    patch_df["patch_id"] = [f"patch_{i}" for i in range(len(patch_df))]
    # Equalize station contribution in patch-level model.
    counts = patch_df.groupby("station")["patch_id"].transform("count")
    patch_df["sample_weight"] = 1.0 / counts

    station_raster_df = aggregate_feature_table(patch_df, ["station", "source_raster"])
    station_raster_df["station_raster_id"] = station_raster_df["station"] + "__" + station_raster_df["source_raster"].astype(str)
    sr_counts = station_raster_df.groupby("station")["station_raster_id"].transform("count")
    station_raster_df["sample_weight"] = 1.0 / sr_counts

    station_df = aggregate_feature_table(patch_df, ["station"])

    # Save feature tables.
    patch_df.to_csv(out_dir / "patch_level_features.csv", index=False)
    station_raster_df.to_csv(out_dir / "station_raster_level_features.csv", index=False)
    station_df.to_csv(out_dir / "station_level_features.csv", index=False)
    (out_dir / "sonar_normalization_stats.json").write_text(json.dumps(sonar_norm_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    experiments = [
        ("RF_patch_level_weighted", patch_df, clean_feature_columns(list(patch_df.select_dtypes(include=[np.number]).columns), extra_exclude=["target", "x", "y", "sample_weight"]), "patch_id", "sample_weight"),
        ("RF_station_raster_level", station_raster_df, clean_feature_columns(list(station_raster_df.select_dtypes(include=[np.number]).columns), extra_exclude=["target", "sample_weight"]), "station_raster_id", "sample_weight"),
        ("RF_station_level", station_df, clean_feature_columns(list(station_df.select_dtypes(include=[np.number]).columns), extra_exclude=["target"]), "station", None),
    ]

    all_cv, all_pred, all_station_oof, summary_rows = [], [], [], []
    for exp, table, feature_cols, group_col, weight_col in experiments:
        print("\n" + "="*80)
        print(exp, "rows=", len(table), "features=", len(feature_cols))
        cv, pred, st_oof, st_metrics = evaluate_repeated_cv(exp, table, feature_cols, group_col, args, out_dir, weight_col)
        all_cv.append(cv); all_pred.append(pred); all_station_oof.append(st_oof)
        summary_rows.append({"experiment": exp, **st_metrics, "n_stations": st_oof["station"].nunique(), "n_features": len(feature_cols)})

    cv_metrics = pd.concat(all_cv, ignore_index=True)
    repeated_oof_preds = pd.concat(all_pred, ignore_index=True)
    station_oof = pd.concat(all_station_oof, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("mae")

    cv_metrics.to_csv(out_dir / "cv_metrics_repeated.csv", index=False)
    repeated_oof_preds.to_csv(out_dir / "repeated_oof_predictions_raw.csv", index=False)
    station_oof.to_csv(out_dir / "repeated_oof_predictions_by_station.csv", index=False)
    summary.to_csv(out_dir / "FINAL_SUMMARY.csv", index=False)

    # Inference config.
    config = {
        "patch_size": int(sonar.shape[1]),
        "use_glcm": not args.no_glcm,
        "model_for_geotiff_inference": "RF_patch_level_weighted",
        "target": "Mz_phi",
        "npz_required_keys": ["sonar_patches", "bathy_patches", "targets", "stations"],
        "notes": "For GeoTIFF map inference use RF_patch_level_weighted because it accepts one local patch feature vector.",
    }
    (out_dir / "inference_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print("Saved outputs to", out_dir)

if __name__ == "__main__":
    main()
