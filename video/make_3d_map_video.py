"""Video of the 3D maps (lab style, short title): the sonar's 3D occupancy map built on the 3D SLAM poses next to the
3D lidar map, rotating, coloured by height above the local ground.

    python video/make_3d_map_video.py <grid3d npz> --out <mp4> [--seconds 24] [--title ...]
Input: scripts/grid3d_eval.py output (sonar_occ: occupied voxel centres, lidar: lidar map points, lidar_observed:
the part of the lidar map inside the sonar's coverage, sensor_xyz: the sonar's path), all in the lidar map frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))          # this project's cosys_video first
import cosys_video as cv  # noqa: E402

TITLE = "RTISSLAM-3D in a Forest:\\n3D Occupancy Map From an Imaging Sonar"


def ground(M, cell=0.5):
    ij = np.floor(M[:, :2] / cell).astype(int); o = ij.min(0); ij -= o
    G = np.full(tuple(ij.max(0) + 1), np.inf)
    np.minimum.at(G, (ij[:, 0], ij[:, 1]), M[:, 2])
    def gz(P):
        i = np.clip(np.floor(P[:, 0] / cell).astype(int) - o[0], 0, G.shape[0] - 1)
        j = np.clip(np.floor(P[:, 1] / cell).astype(int) - o[1], 0, G.shape[1] - 1)
        z = G[i, j]; z[~np.isfinite(z)] = np.nanmedian(G[np.isfinite(G)])
        return z
    return gz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--note", default="Sonar: DMAS5-CF over the frontal hemisphere, CFAR on the sphere, ground mirrors removed; "
                    "0.1 m occupancy with beam-shaped evidence and free-space carving.")
    a = ap.parse_args()
    import matplotlib.pyplot as plt
    z = np.load(a.grid)
    S, L, obs, path = z["sonar_occ"], z["lidar"], z["lidar_observed"], z["sensor_xyz"]
    gz = ground(L)
    L = L[obs]
    L = L[(L[:, 2] - gz(L)) < 3.0]                                    # the sonar sees nothing above ~2 m: compare like with like
    hS, hL = S[:, 2] - gz(S), L[:, 2] - gz(L)
    sS = 0.6
    if "logodds" in z.files:
        lo_ = z["logodds"].astype(float)
        sS = 0.25 + 0.9 * np.clip((lo_ - lo_.min()) / (np.percentile(lo_, 95) - lo_.min() + 1e-9), 0, 1)
    rng = np.random.default_rng(0)
    L = L[rng.permutation(len(L))[:60000]]; hL = L[:, 2] - gz(L)
    c0 = np.r_[L[:, :2], S[:, :2]].mean(0)
    span = np.ptp(np.r_[L[:, :2], S[:, :2]], axis=0).max() / 2
    fps, n = 10, int(a.seconds * 10)
    fig = cv.new_fig()
    fig.text(0.5, 0.93, a.title.replace("\\n", "\n"), ha="center", va="center", fontsize=26, linespacing=1.2)
    axs = [fig.add_axes([0.0, 0.14, 0.5, 0.72], projection="3d"), fig.add_axes([0.5, 0.14, 0.5, 0.72], projection="3d")]
    for ax, P, h, t in ((axs[0], S, hS, "Sonar (eRTIS) 3D map"), (axs[1], L, hL, "Lidar 3D map (reference)")):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.clip(h, -0.3, 2.5), cmap="viridis", s=sS if ax is axs[0] else 0.15,
                   vmin=-0.3, vmax=2.5, depthshade=False, rasterized=True)
        ax.plot(path[:, 0], path[:, 1], path[:, 2] + 0.05, c=cv.MAGENTA, lw=1.3)
        ax.set_xlim(c0[0] - span, c0[0] + span); ax.set_ylim(c0[1] - span, c0[1] + span)
        zl = np.percentile(np.r_[S[:, 2], L[:, 2]], [1, 99]); ax.set_zlim(zl[0] - 0.5, zl[0] - 0.5 + 2 * span * 0.3)
        ax.set_box_aspect((1, 1, 0.3), zoom=1.3); ax.set_axis_off()
        ax.set_title(t, fontsize=18, fontweight="bold", color=cv.MAGENTA if ax is axs[0] else "k", y=0.93)
    fig.text(0.5, 0.135, a.note + "\nLidar (reference only): points up to 3 m above the ground. Colour: height above the ground "
             "(0 to 2.5 m). Magenta: the sonar's path.",
             ha="center", va="center", fontsize=12, color="0.4", linespacing=1.4)
    cv.logos(fig)
    wr = cv.writer(a.out, fps=fps)
    cv.hold(wr, cv.slide_title(a.title.replace("\\n", "\n")), 3, fps=fps, fade_in=False)
    for i in range(n):
        az = -60 + 360 * i / n
        el = 28 + 8 * np.sin(2 * np.pi * i / n)
        for ax in axs:
            ax.view_init(elev=el, azim=az)
        img = cv.grab(fig)
        wr.append_data(cv.fade(img, min(1.0, (i + 1) / 6, (n - i) / 6)))
    cv.hold(wr, cv.slide_end(), 3, fps=fps, fade_out=False)
    wr.close()
    print(a.out)


if __name__ == "__main__":
    main()
