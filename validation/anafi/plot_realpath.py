#!/usr/bin/env python3
"""
plot_realpath.py — Overlay real Anafi GPS track on AFSACPSO simulation terrain map
=====================================================================================
Runs NATIVELY on macOS (does NOT need Docker or Olympe).

Given:
  - A GPS track CSV from stream_gps.py  (lat, lon, alt columns)
  - An origin GPS coordinate (where map [0,0] is in real world)
  - The DEM terrain grid (loaded from the .mat file or a GeoTIFF)

It produces a publication-quality figure showing:
  [left]  Top-down 2D view:  terrain contour + real GPS path
  [right] 3D view:           terrain surface + real GPS path in 3D

Usage:
    python plot_realpath.py \\
        --gps_csv  validation/anafi/logs/gps_track_20260223_120000.csv \\
        --origin_lat 21.005000 \\
        --origin_lon 105.845000 \\
        --terrain_mat shared/environment/ChrismasTerrain2.mat \\
        --output outputs/figures/paths/real_anafi_path.png

Dependencies (all available on macOS via pip):
    pip install numpy scipy matplotlib pandas pyproj
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

matplotlib.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size":      9,
})

# ── Earth constants ──────────────────────────────────────────────────────────
EARTH_R = 6_371_000  # meters


def gps_to_local_xy(lat: np.ndarray, lon: np.ndarray,
                     origin_lat: float, origin_lon: float):
    """
    Convert GPS arrays (decimal degrees) to local Cartesian (x, y) in metres
    relative to (origin_lat, origin_lon).
    Simple flat-Earth approximation — accurate to < 0.1 m for paths < 5 km.
    """
    lat_r = np.radians(lat)
    origin_lat_r = np.radians(origin_lat)
    origin_lon_r = np.radians(origin_lon)
    lon_r = np.radians(lon)

    y = EARTH_R * (lat_r - origin_lat_r)           # North (+y)
    x = EARTH_R * np.cos(origin_lat_r) * (lon_r - origin_lon_r)  # East (+x)
    return x, y


def load_terrain(mat_path: str):
    """
    Load terrain grid from:
      - GeoTIFF (.tif/.tiff) via rasterio
      - MATLAB .mat file via scipy.io.loadmat
    Falls back to flat grid if file not found.
    Returns: tx (1D), ty (1D), grid (2D array of elevations)
    """
    if not os.path.exists(mat_path):
        print(f"[WARN] Terrain file not found: {mat_path}")
        print("[WARN] Using synthetic flat terrain.")
        size = 400
        return np.linspace(0, 400, size), np.linspace(0, 400, size), np.zeros((size, size))

    ext = os.path.splitext(mat_path)[1].lower()

    # ── GeoTIFF ──────────────────────────────────────────────────────────────
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(mat_path) as src:
                grid = src.read(1).astype(float)
                nodata = src.nodata
                # Mask nodata
                if nodata is not None:
                    grid[grid == nodata] = np.nan
                grid[grid < -500] = np.nan   # safety clamp for bad nodata
                # Replace NaN with 0 for plotting
                grid = np.where(np.isnan(grid), 0, grid)
                h, w = grid.shape
                # Return pixel-space axis (0..w, 0..h) scaled by map_scale
                tx = np.linspace(0, w, w)
                ty = np.linspace(0, h, h)
                valid = grid[grid > 0]
                print(f"[OK] GeoTIFF terrain loaded: {h}×{w} grid, "
                      f"elev range {valid.min() if len(valid) else 0:.0f}–{valid.max() if len(valid) else 0:.0f} m")
                return tx, ty, grid
        except Exception as e:
            print(f"[WARN] rasterio failed ({e}). Using flat terrain.")
            size = 400
            return np.linspace(0, 400, size), np.linspace(0, 400, size), np.zeros((size, size))


    # ── MATLAB .mat ───────────────────────────────────────────────────────────
    try:
        mat = loadmat(mat_path)
        grid = tx = ty = None
        for gkey in ["terrainGrid", "Z", "elevation", "DEM"]:
            if gkey in mat:
                grid = mat[gkey].astype(float); break
        for xkey in ["terrainX", "X", "x"]:
            if xkey in mat:
                tx = mat[xkey].flatten().astype(float); break
        for ykey in ["terrainY", "Y", "y"]:
            if ykey in mat:
                ty = mat[ykey].flatten().astype(float); break
        if grid is not None and tx is not None and ty is not None:
            print(f"[OK] MAT terrain loaded: {grid.shape[0]}×{grid.shape[1]} grid")
            return tx, ty, grid
        raise ValueError("Required variables not found in .mat file")
    except Exception as e:
        print(f"[WARN] Could not parse terrain file ({e}). Using flat terrain.")
        size = 400
        return np.linspace(0, 400, size), np.linspace(0, 400, size), np.zeros((size, size))



def plot_path_on_map(gps_csv: str, origin_lat: float, origin_lon: float,
                     terrain_mat: str, output: str, map_scale: float = 1.0):
    """
    Main plotting function.
    """
    # ── Load GPS track ────────────────────────────────────────────────────────
    df = pd.read_csv(gps_csv)
    required = {"latitude", "longitude", "altitude_m"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] CSV missing columns: {missing}")
        sys.exit(1)

    lats = df["latitude"].values
    lons = df["longitude"].values
    alts = df["altitude_m"].values
    t    = df["t_sec"].values if "t_sec" in df.columns else np.arange(len(df))

    print(f"[INFO] GPS track: {len(df)} points, "
          f"duration={t[-1]:.1f}s, "
          f"alt range {alts.min():.1f}–{alts.max():.1f} m AGL")

    # Convert GPS → local XY in map units
    x_m, y_m = gps_to_local_xy(lats, lons, origin_lat, origin_lon)
    x_map = x_m / map_scale
    y_map = y_m / map_scale
    z_map = alts / map_scale

    # ── Load terrain ──────────────────────────────────────────────────────────
    tx, ty, tgrid = load_terrain(terrain_mat)
    TX, TY = np.meshgrid(tx, ty)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")

    cmap_terrain = plt.cm.terrain
    cmap_path    = plt.cm.plasma

    # ── LEFT: 2D top-down ─────────────────────────────────────────────────────
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set_facecolor("#0d1117")

    terrain_img = ax2d.contourf(TX, TY, tgrid, levels=40, cmap="terrain", alpha=0.75)
    ax2d.contour(TX, TY, tgrid, levels=15, colors="white", linewidths=0.3, alpha=0.4)

    # Colour the path by time
    norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
    for i in range(len(x_map) - 1):
        c = cmap_path(norm(t[i]))
        ax2d.plot(x_map[i:i+2], y_map[i:i+2], color=c, linewidth=2.5, solid_capstyle="round")

    # Start / end markers
    ax2d.scatter(x_map[0],  y_map[0],  s=120, c="lime",  zorder=10,
                 marker="o", edgecolors="white", linewidths=1.2, label="Start (GPS fix)")
    ax2d.scatter(x_map[-1], y_map[-1], s=120, c="red",   zorder=10,
                 marker="X", edgecolors="white", linewidths=1.2, label="End")

    sm = plt.cm.ScalarMappable(cmap=cmap_path, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2d, shrink=0.6, pad=0.02)
    cbar.set_label("Time (s)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax2d.set_xlabel("X (map units)", color="white")
    ax2d.set_ylabel("Y (map units)", color="white")
    ax2d.set_title("Real GPS Track — Top-down View", color="white", fontweight="bold")
    ax2d.tick_params(colors="white")
    for spine in ax2d.spines.values():
        spine.set_edgecolor("#444")
    ax2d.legend(loc="upper left", fontsize=8,
                 facecolor="#1c1c2e", edgecolor="#555", labelcolor="white")

    # ── RIGHT: 3D ─────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d.set_facecolor("#0d1117")

    # Down-sample terrain for 3D (performance)
    step = max(1, tgrid.shape[0] // 60)
    ax3d.plot_surface(TX[::step, ::step], TY[::step, ::step], tgrid[::step, ::step],
                      cmap="terrain", alpha=0.55, linewidth=0, antialiased=True)

    # GPS path in 3D coloured by altitude
    norm_alt = mcolors.Normalize(vmin=z_map.min(), vmax=z_map.max())
    for i in range(len(x_map) - 1):
        c = cmap_path(norm_alt(z_map[i]))
        ax3d.plot(x_map[i:i+2], y_map[i:i+2], z_map[i:i+2],
                  color=c, linewidth=2.5)

    ax3d.scatter(x_map[0],  y_map[0],  z_map[0],  s=80, c="lime", zorder=10)
    ax3d.scatter(x_map[-1], y_map[-1], z_map[-1], s=80, c="red",  zorder=10, marker="X")

    ax3d.set_xlabel("X", color="white", labelpad=8)
    ax3d.set_ylabel("Y", color="white", labelpad=8)
    ax3d.set_zlabel("Z (m)", color="white", labelpad=8)
    ax3d.set_title("Real GPS Track — 3D View", color="white", fontweight="bold")
    ax3d.tick_params(colors="white", labelsize=7)

    # Stats annotation
    total_dist = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2 + np.diff(alts)**2))
    stats_text = (f"Points : {len(df)}\n"
                  f"Duration: {t[-1]:.1f} s\n"
                  f"3-D dist: {total_dist:.1f} m\n"
                  f"Alt range: {alts.min():.1f}–{alts.max():.1f} m")
    fig.text(0.51, 0.02, stats_text, color="white", fontsize=8,
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1c1c2e", edgecolor="#555", alpha=0.9))

    plt.suptitle("Parrot Anafi USA — Real Flight Path on AFSACPSO Terrain Map",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[OK] Figure saved to: {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot real Anafi GPS track on simulation terrain map")
    parser.add_argument("--gps_csv",     required=True,
                        help="Path to GPS CSV from stream_gps.py")
    parser.add_argument("--origin_lat",  type=float, required=True,
                        help="Latitude of map origin (map coordinate [0,0])")
    parser.add_argument("--origin_lon",  type=float, required=True,
                        help="Longitude of map origin (map coordinate [0,0])")
    parser.add_argument("--terrain_mat", default="shared/environment/ChrismasTerrain2.mat",
                        help="Path to MATLAB terrain .mat file")
    parser.add_argument("--map_scale",   type=float, default=1.0,
                        help="Metres per map unit (default: 1.0)")
    parser.add_argument("--output",      default="outputs/figures/paths/real_anafi_path.png",
                        help="Output figure path")
    args = parser.parse_args()

    plot_path_on_map(
        gps_csv=args.gps_csv,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        terrain_mat=args.terrain_mat,
        output=args.output,
        map_scale=args.map_scale,
    )


if __name__ == "__main__":
    main()
