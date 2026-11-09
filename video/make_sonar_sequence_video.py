"""A sequence of the raw sonar data of the forest drive, at the real pulse rate: what the eRTIS sees and what the
front-end extracts from it, next to the GoPro view and the simultaneous lidar scan (reference only).

    python video/make_sonar_sequence_video.py <work dir> --out <mp4> [--k0 3300 --n 200] [--test]

Row 1, the acoustic image of every capture (DMAS5-CF over the frontal hemisphere, 475 ranges x 2293 directions):
front view (maximum over range), top view (maximum over elevation) and side view (maximum over azimuth), each
normalized to its maximum, 40 dB range, +40 log r spreading compensation in the top and side views (fans in 3 deg bins).
Row 2, the detections of the capture (CFAR on the sphere, ground mirrors removed) over the simultaneous lidar scan,
top and side view, coloured by the height above the sonar's own ground plane, and the local map of the acoustic
odometry (the detections of the last 20 captures, placed with the odometry, in the current sensor frame).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cosys_video as cv  # noqa: E402

TITLE = "RTISSLAM-3D in a Forest:\nWhat the 3D Imaging Sonar Sees"
FPS = 15
DR = 40.0                    # display range (dB)
H_MAX = 2.5


def height_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("height", ["#6cc4ad", "#2f9fb0", "#2a78d6", "#26338a", "#1a1040"])


class Projector:
    """Maximum projections of a [R, D] volume: front view (orthographic disk, maximum over range), and fans of range x
    azimuth (maximum over elevation) and range x elevation (maximum over azimuth) in angular bins."""

    def __init__(self, U, rr, bin_deg=3.0):
        self.U, self.rr = U, rr
        self.comp = (40 * np.log10(np.maximum(rr, 0.3)))[:, None]            # two-way spreading, power
        az = np.degrees(np.arctan2(U[:, 1], U[:, 0])); el = np.degrees(np.arcsin(np.clip(U[:, 2], -1, 1)))
        self.edges = np.arange(-90, 90 + bin_deg / 2, bin_deg)
        self.b_az = np.clip(np.digitize(az, self.edges) - 1, 0, len(self.edges) - 2)
        self.b_el = np.clip(np.digitize(el, self.edges) - 1, 0, len(self.edges) - 2)
        from matplotlib.tri import Triangulation
        self.tri = Triangulation(-U[:, 1], U[:, 2])                          # front view, left of the robot on the left
        th = np.radians(self.edges); re = np.r_[rr - (rr[1] - rr[0]) / 2, rr[-1] + (rr[1] - rr[0]) / 2]
        self.X = re[:, None] * np.cos(th)[None]; self.Y = re[:, None] * np.sin(th)[None]   # fan cell corners

    def fan(self, E, which):
        b = self.b_az if which == "az" else self.b_el
        n = len(self.edges) - 1
        out = np.full((E.shape[0], n), -np.inf, np.float32)
        for j in range(n):
            m = b == j
            if m.any():
                out[:, j] = E[:, m].max(1)
        for j in range(n):                                                     # empty bins at the edges: nearest neighbour
            if not np.isfinite(out[0, j]):
                ok = np.where(np.isfinite(out[0]))[0]; out[:, j] = out[:, ok[np.argmin(np.abs(ok - j))]]
        return out - out.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k0", type=int, default=3300, help="first capture")
    ap.add_argument("--n", type=int, default=200, help="number of captures")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--recording", default="Forest, sequence 1")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    import h5py
    import matplotlib.pyplot as plt
    from rtis3d.poses import _mount
    work = Path(a.work)
    st = json.loads((work / "pipeline.json").read_text())
    vol = work / "vol_dmas5cf"
    V = np.load(vol / "volumes.npy", mmap_mode="r")
    U, rr, t_ac = np.load(vol / "directions.npy"), np.load(vol / "ranges.npy"), np.load(vol / "t.npy")
    off = json.loads((Path(st["site"]) / "cache" / f"{st['name']}_clock_sync.json").read_text())["offset_s"]
    D = np.load(work / "detections.npz")
    O = np.load(work / "ao3d.npz")
    T_sb, T_el = _mount(st["h5"])
    s_odo = np.r_[0, np.cumsum(np.linalg.norm(np.diff(O["P_ao"][:, :2, 3], axis=0), axis=1))]
    cap_i = {int(c): i for i, c in enumerate(O["caps"])}
    rate = 1.0 / np.median(np.diff(t_ac))
    ks = list(range(a.k0, min(a.k0 + a.n, len(t_ac))))
    if a.test:
        ks = ks[::max(1, len(ks) // 4)][:4]
    pj = Projector(U, rr)
    cm_h = height_cmap()
    cm_e = plt.get_cmap("magma").copy(); cm_e.set_bad("black")

    fig = cv.new_fig()
    fig.text(0.5, 0.935, a.title, ha="center", va="center", fontsize=28, linespacing=1.2)
    labels = ["Recording", "Capture", "Time", "Distance", "Detections"]
    vals = {"Recording": "Forest, seq. 1"}
    stv = {}
    for i, lab in enumerate(labels):
        fig.text(0.02, 0.80 - 0.036 * i, f"{lab}:", fontsize=15, fontweight="bold")
        stv[lab] = fig.text(0.105, 0.80 - 0.036 * i, "", fontsize=15)
    fig.text(0.02, 0.80 - 0.036 * len(labels), f"Real time, {rate:.1f} pulses/s", fontsize=13, color="0.35")
    ctx = fig.add_axes([0.02, 0.20, 0.19, 0.34]); ctx.axis("off"); ctx.set_title("Robot camera (illustration)", fontsize=14)
    # panels: one row for the acoustic image, one for the front-end output
    x0, pw, ph, gap = 0.245, 0.2, 0.29, 0.035
    rows = (0.54, 0.185)
    axs = [[fig.add_axes([x0 + c * (pw + gap), rows[r], pw, ph]) for c in range(3)] for r in range(2)]
    titles = [["Image: front view", "Image: top view", "Image: side view"],
              ["Detections: top view", "Detections: side view", "Odometry submap"]]
    for r in range(2):
        for c in range(3):
            ax = axs[r][c]; ax.set_title(titles[r][c], fontsize=15, fontweight="bold", pad=5,
                                         color=cv.MAGENTA if r == 1 and c == 2 else "k")
            ax.tick_params(labelsize=10)
    # row 1 artists
    a0 = axs[0][0]
    tp = a0.tripcolor(pj.tri, np.zeros(len(U)), shading="gouraud", cmap=cm_e, vmin=-DR, vmax=0)
    th = np.linspace(0, 2 * math.pi, 361)
    for ang in (30, 60):
        r_ = math.sin(math.radians(ang)); a0.plot(r_ * np.cos(th), r_ * np.sin(th), c="white", lw=0.5, alpha=0.5)
    a0.plot([-1, 1], [0, 0], c="white", lw=0.5, alpha=0.5); a0.plot([0, 0], [-1, 1], c="white", lw=0.5, alpha=0.5)
    a0.plot(np.cos(th), np.sin(th), c="0.3", lw=0.8)
    for x_, y_, t_, ha, va in ((-1.04, 0, "left", "right", "center"), (1.04, 0, "right", "left", "center"),
                               (0, 1.04, "up", "center", "bottom"), (0, -1.04, "down", "center", "top")):
        a0.text(x_, y_, t_, ha=ha, va=va, fontsize=11, color="0.3")
    a0.set_xlim(-1.3, 1.3); a0.set_ylim(-1.15, 1.15); a0.set_aspect("equal"); a0.axis("off")
    nf = len(pj.edges) - 1
    im_top = axs[0][1].pcolormesh(pj.X, pj.Y, np.zeros((len(rr), nf)), cmap=cm_e, vmin=-DR, vmax=0, shading="flat")
    im_side = axs[0][2].pcolormesh(pj.X, pj.Y, np.zeros((len(rr), nf)), cmap=cm_e, vmin=-DR, vmax=0, shading="flat")
    for ax, yl, lim in ((axs[0][1], "y (m)", (-4, 4)), (axs[0][2], "z (m)", (-1.5, 3))):
        ax.set_xlabel("x (m)", fontsize=12); ax.set_ylabel(yl, fontsize=12)
        ax.set_xlim(0, 8); ax.set_ylim(*lim); ax.set_aspect("equal"); ax.set_facecolor("white")
    cax = fig.add_axes([0.95, rows[0] + 0.03, 0.008, ph - 0.06])
    cb = fig.colorbar(tp, cax=cax); cb.set_label("Normalized acoustic energy (dB)", fontsize=11); cb.ax.tick_params(labelsize=10)
    # row 2 artists
    sc, li = [], []
    for c, (lim_a, lim_b, lb) in enumerate((((0, 8), (-4, 4), "y (m)"), ((0, 8), (-1.5, 3), "z (m)"), ((-4, 8), (-6, 6), "y (m)"))):
        ax = axs[1][c]
        li.append(ax.scatter([], [], s=1.0, c="#a9a7a1", lw=0, zorder=1))
        sc.append(ax.scatter([], [], s=14 if c < 2 else 5, c=[], cmap=cm_h, vmin=0, vmax=H_MAX, lw=0.3 if c < 2 else 0,
                             edgecolors="white", zorder=3))
        ax.set_xlim(*lim_a); ax.set_ylim(*lim_b); ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=12); ax.set_ylabel(lb, fontsize=12)
        ax.plot([0], [0], marker=">" if c != 1 else "s", ms=8, c="k", zorder=5)
    gp_line, = axs[1][1].plot([], [], c="k", lw=1.0, ls="--", zorder=4)
    cax = fig.add_axes([0.95, rows[1] + 0.03, 0.008, ph - 0.06])
    cb = fig.colorbar(sc[0], cax=cax); cb.set_label("Height above ground (m)", fontsize=11); cb.ax.tick_params(labelsize=10)
    fig.text(0.545, 0.045, "Image: DMAS5-CF, maximum projections, normalized per image, 40 dB range, +40 log r in the fans.\n"
             "Detections: CFAR on the sphere, ground mirrors removed. Grey: simultaneous lidar scan (reference only).\n"
             "Dashed: ground plane fitted to the detections. Submap: last 20 captures, placed with the acoustic odometry.",
             ha="center", fontsize=11, color="0.4", linespacing=1.4)
    cv.logos(fig)

    from chirppy import CHIRPDataset
    from rtisreal.video_frames import VideoFrames
    ds = CHIRPDataset(st["h5"]); ds.__enter__()
    vf = VideoFrames(ds.get_video(0), float(t_ac[ks[0]] + off) - 1, width=960)
    img0 = vf.at(float(t_ac[ks[0]] + off))
    gimg = ctx.imshow(img0)
    f5 = h5py.File(st["h5"], "r")
    lid = f5[f"sensors/{st['sensors']['lidar']}/data"]; tl = lid["timestamps"][:]
    wr = cv.writer(a.out, fps=FPS)
    t_start = t_ac[ks[0]]
    n_frames_total = 0
    for n, k in enumerate(ks):
        Ek = np.asarray(V[k], np.float32)
        E = Ek - Ek.max()
        Ec = Ek + pj.comp; Ec = Ec - Ec.max()
        tp.set_array(np.clip(E[2:].max(0), -DR, 0))                         # drop the first range cells (direct path)
        im_top.set_array(np.clip(pj.fan(Ec, "az"), -DR, 0).ravel())
        im_side.set_array(np.clip(pj.fan(Ec, "el"), -DR, 0).ravel())
        p = D["pts"][D["start"][k]:D["start"][k + 1]]
        hgt = float(D["height"][k]) if np.isfinite(D["height"][k]) else 0.32
        n_, d_ = D["plane_n"][k], None
        hz = p[:, 2] + hgt if not np.all(np.isfinite(n_)) else p @ n_ + hgt * n_[2]
        j = int(np.argmin(np.abs(tl - (t_ac[k] + off))))
        L = np.asarray(lid["points"][j][:int(lid["point_counts"][j]), :3], float) @ T_el[:3, :3].T + T_el[:3, 3]
        L = L[(L[:, 0] > -0.5) & (np.linalg.norm(L, axis=1) < 9)]
        for c, (i0, i1) in enumerate(((0, 1), (0, 2))):
            o = np.argsort(hz)
            sc[c].set_offsets(np.c_[p[o, i0], p[o, i1]]); sc[c].set_array(np.clip(hz[o], 0, H_MAX))
            Lv = L[(L[:, 2] + hgt) > 0.15] if c == 0 else L[np.abs(L[:, 1]) < 3]
            li[c].set_offsets(np.c_[Lv[:, i0], Lv[:, i1]])
        xs = np.array([0, 8.0])
        if np.all(np.isfinite(n_)) and abs(n_[2]) > 0.3:
            gp_line.set_data(xs, -(n_[0] * xs + hgt * n_[2]) / n_[2])
        # odometry submap: last 20 captures in the current sensor frame
        i = cap_i.get(k)
        if i is not None:
            Q, Hh = [], []
            Tk = O["P_ao"][i] @ T_sb
            for ii in range(max(0, i - 20), i + 1):
                kk = int(O["caps"][ii])
                pp = D["pts"][D["start"][kk]:D["start"][kk + 1]]
                hk = float(D["height"][kk]) if np.isfinite(D["height"][kk]) else 0.32
                Tw = O["P_ao"][ii] @ T_sb
                Rr = np.linalg.inv(Tk) @ Tw
                Q.append(pp @ Rr[:3, :3].T + Rr[:3, 3]); Hh.append(pp[:, 2] + hk)
            Q = np.vstack(Q); Hh = np.concatenate(Hh); o = np.argsort(Hh)
            sc[2].set_offsets(Q[o][:, :2]); sc[2].set_array(np.clip(Hh[o], 0, H_MAX))
            stv["Distance"].set_text(f"{s_odo[i]:.1f} m")
        gimg.set_data(vf.at(float(t_ac[k] + off)))
        stv["Recording"].set_text(vals["Recording"])
        stv["Capture"].set_text(f"{k} of {len(t_ac)}")
        stv["Time"].set_text(f"{t_ac[k] - t_ac[0]:.1f} s")
        stv["Detections"].set_text(f"{len(p)}")
        frame = cv.grab(fig)
        # real time: the capture stays on screen until the next pulse
        t_next = t_ac[ks[n + 1]] if n + 1 < len(ks) else t_ac[k] + 1 / rate
        n_until = int(round((t_next - t_start) * FPS)) if not a.test else n_frames_total + 3
        n_until = max(n_until, n_frames_total + 1)                           # at least one frame per capture
        while n_frames_total < n_until:
            nf = max(1, int(round(0.6 * FPS)))
            fa = min(1.0, (n_frames_total + 1) / nf)
            wr.append_data(cv.fade(frame, fa)); n_frames_total += 1
        if n % 20 == 0:
            print(f"capture {k} ({n + 1}/{len(ks)})", flush=True)
    for i in range(int(0.6 * FPS)):                                           # fade out
        wr.append_data(cv.fade(frame, 1 - (i + 1) / (0.6 * FPS)))
    wr.close(); vf.close(); f5.close(); ds.__exit__(None, None, None)
    print(a.out)


if __name__ == "__main__":
    main()
