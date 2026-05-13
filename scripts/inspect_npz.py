#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Inspect NPZ patch dataset and plot random patches.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-png", default=None)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    print("keys:", list(data.keys()))
    sonar = data["sonar_patches"].astype(np.float32)
    bathy = data["bathy_patches"].astype(np.float32)
    y = data["targets"].astype(np.float32)
    stations = data["stations"].astype(str)
    masks = data["valid_masks"].astype(np.float32) if "valid_masks" in data else np.ones_like(sonar)
    rasters = data["source_rasters"].astype(str) if "source_rasters" in data else np.array(["unknown"]*len(y))
    shifts = data["shift_xy_px"].astype(int) if "shift_xy_px" in data else np.zeros((len(y),2), dtype=int)
    valid_ratios = data["valid_ratios"].astype(float) if "valid_ratios" in data else masks.reshape(len(y), -1).mean(axis=1)
    print("sonar:", sonar.shape, sonar.dtype)
    print("bathy:", bathy.shape, bathy.dtype)
    print("targets:", y.shape, y.dtype)
    print("stations:", len(np.unique(stations)), "unique")
    print("rasters:", len(np.unique(rasters)), "unique")
    summary = pd.DataFrame({"station":stations, "target":y, "source_raster":rasters, "valid_ratio":valid_ratios}).groupby("station").agg(n_patches=("target","size"), n_rasters=("source_raster","nunique"), target=("target","mean"), valid_ratio_mean=("valid_ratio","mean")).reset_index()
    print(summary.describe(include="all").to_string())
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(np.unique(stations), size=min(args.n, len(np.unique(stations))), replace=False)
    idxs = []
    for st in chosen:
        ii = np.where(stations == st)[0]
        idxs.append(rng.choice(ii))
    fig, axes = plt.subplots(len(idxs), 4, figsize=(16, 3.4*len(idxs)), squeeze=False)
    for r, idx in enumerate(idxs):
        panels = [(sonar[idx], "sonar", "gray"), (bathy[idx,0], "depth", "viridis"), (bathy[idx,1], "slope", "magma"), (masks[idx], "valid mask", "gray")]
        title = f"idx={idx} | station={stations[idx]} | y={y[idx]:.2f}\nraster={Path(rasters[idx]).name}\nshift={tuple(shifts[idx])} | valid={valid_ratios[idx]:.2f}"
        for c,(arr,name,cmap) in enumerate(panels):
            ax = axes[r,c]
            if name != "valid mask":
                vmin,vmax = np.percentile(arr[np.isfinite(arr)], [2,98])
            else:
                vmin,vmax = 0,1
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title((name + "\n" + title) if c==0 else name, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if args.out_png:
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_png, dpi=200)
        print("Saved", args.out_png)
    else:
        plt.show()

if __name__ == "__main__":
    main()
