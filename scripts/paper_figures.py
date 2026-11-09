"""Figures and numbers of the RTISSLAM-3D paper (paper/figures/*.pdf, paper/numbers.json), from one pipeline run
(scripts/pipeline3d.py) and its ablations (scripts/ablations3d.py).

    python scripts/paper_figures.py --work <data>/<site>/threed/<name>
        [--only imaging,odometry,place,trajectory,map,build] [--capture 2400]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))
import rtis3d  # noqa: E402,F401
from rtis3d.metrics import ate, fit4, rpe  # noqa: E402

OUT = ROOT / "paper" / "figures"
W2 = 7.16                                    # IEEE double-column figure width (in)

# One colour per entity, the same in every figure.
C_REF = "#1a1a1a"                            # lidar reference
C_SLAM = "#2a78d6"                           # RTISSLAM-3D
C_ODO = "#eb6834"                            # 3D acoustic odometry (no loops)
C_WHEEL = "#9a9893"                          # wheels + gyroscope
C_FREE = "#1baf7a"                           # sonar also estimating the vertical motion
C_LOOP = "#b10091"                           # loop closures
C_GROUND = "#d9d6cf"                         # ground voxels / points
C_LIDAR = "#c4c2bc"                          # lidar points behind sonar data
MAG = (177 / 255, 0, 145 / 255)
H_MAX = 2.5                                  # height above ground of the top of the height colour scale (m)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.titleweight": "normal", "axes.titlepad": 4, "axes.linewidth": 0.6,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.linewidth": 1.1, "lines.solid_capstyle": "round", "pdf.fonttype": 42, "savefig.dpi": 400,
        "image.interpolation": "nearest"})
    return plt


def ptitle(ax, letter, text, **kw):
    """Panel title with its bold letter: (a)  text."""
    ax.set_title(rf"$\bf{{({letter})}}$  {text}", loc="left", **kw)


def height_cmap():
    """Sequential height scale: teal (ground level) to dark blue (2.5 m)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("height", ["#6cc4ad", "#2f9fb0", "#2a78d6", "#26338a", "#1a1040"])


def scalebar(ax, length, label, loc=(0.06, 0.06), color="k", lw=1.6):
    """Scale bar in data units at axes fraction loc (equal-aspect axes)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0); y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], c=color, lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(x + length / 2, y + 0.022 * (y1 - y0), label, ha="center", va="bottom", fontsize=6.5, color=color, zorder=20)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fitted(X, G):
    """Positions of the poses X [n, 4, 4] after the 4-DoF fit onto G [n, 4, 4]."""
    R, t = fit4(X[:, :3, 3], G[:, :3, 3])
    return X[:, :3, 3] @ R.T + t, R, t


def load(work):
    st = json.loads((work / "pipeline.json").read_text())
    Z = np.load(work / "slam3d.npz")
    O = np.load(work / "ao3d.npz")
    return st, Z, O


# ----------------------------------------------------------------------------------------------------------------------
def fig_imaging(work, st, k, plt, num):
    """One capture: camera, the acoustic image (maximum over range, azimuth x elevation) with the CFAR detections,
    and the 3D detections (kept / ground mirrors) over the simultaneous lidar scan, top and side view."""
    import h5py
    import torch
    from matplotlib.tri import Triangulation
    from rtis3d.detections import Detector
    from rtis3d.laplacian import sphere_cfar
    from rtis3d.poses import _mount
    vol = work / "vol_dmas5cf"
    V = np.load(vol / "volumes.npy", mmap_mode="r")
    U, rr, t_ac = np.load(vol / "directions.npy"), np.load(vol / "ranges.npy"), np.load(vol / "t.npy")
    off = json.loads((Path(st["site"]) / "cache" / f"{st['name']}_clock_sync.json").read_text())["offset_s"]
    det = Detector(U, rr)
    Vk = np.asarray(V[k], np.float32)
    P = torch.as_tensor(10 ** (Vk / 10), device="cuda")
    Q = sphere_cfar(P, det.H, gain=det.gain)
    Q[~torch.as_tensor(det.keep_r, device="cuda")] = 0
    ri, di = torch.nonzero(Q > Q.max() * det.peak, as_tuple=True)
    pts = det.pos[ri, di].cpu().numpy(); w = Q[ri, di].cpu().numpy()
    ri, di = ri.cpu().numpy(), di.cpu().numpy()
    n, d = det.ground_plane(pts, w)
    mirror = pts @ n - d < -det.below
    T_sb, T_el = _mount(st["h5"])
    with h5py.File(st["h5"], "r") as f:
        li = f[f"sensors/{st['sensors']['lidar']}/data"]
        tl = li["timestamps"][:]
        j = int(np.argmin(np.abs(tl - (t_ac[k] + off))))
        L = np.asarray(li["points"][j][:int(li["point_counts"][j]), :3], float)
    L = L @ T_el[:3, :3].T + T_el[:3, 3]
    L = L[(L[:, 0] > 0) & (np.linalg.norm(L, axis=1) < 9)]

    from chirppy import CHIRPDataset
    from rtisreal.video_frames import VideoFrames
    with CHIRPDataset(st["h5"]) as ds:
        vf = VideoFrames(ds.get_video(0), float(t_ac[k] + off) - 1, width=1280)
        img = vf.at(float(t_ac[k] + off)); vf.close()
    hz = pts @ n - d                                                        # signed height above the fitted plane
    keep = ~mirror

    fig = plt.figure(figsize=(W2, 3.9))
    # (a) camera, cropped vertically to the aspect of its panel (keeps the full width)
    pa = [0.065, 0.56, 0.50, 0.385]
    H_, W_ = img.shape[:2]; ch = int(W_ * (pa[3] * 3.9) / (pa[2] * W2)); y0 = (H_ - ch) // 2
    a0 = fig.add_axes(pa); a0.imshow(img[max(0, y0):y0 + ch]); bare(a0)
    ptitle(a0, "a", "Robot camera (illustration)")
    # (b) acoustic image: maximum over range per direction, the frontal hemisphere seen from behind the sensor
    #     (orthographic: left of the robot on the left)
    E = Vk[det.keep_r].max(0); E = E - E.max()
    a1 = fig.add_axes([0.62, 0.56, 0.29, 0.385])
    tri = Triangulation(-U[:, 1], U[:, 2])
    pc = a1.tripcolor(tri, np.clip(E, -40, 0), shading="gouraud", cmap="magma", vmin=-40, vmax=0, rasterized=True)
    th = np.linspace(0, 2 * math.pi, 361)
    for ang in (30, 60):                                                    # cones of 30 and 60 deg off the axis
        r_ = math.sin(math.radians(ang)); a1.plot(r_ * np.cos(th), r_ * np.sin(th), c="white", lw=0.4, alpha=0.5)
        a1.text(r_ * math.cos(math.radians(50)), r_ * math.sin(math.radians(50)), f"{ang}°", color="white", fontsize=5.5,
                alpha=0.8, ha="left", va="bottom")
    a1.plot([-1, 1], [0, 0], c="white", lw=0.4, alpha=0.5); a1.plot([0, 0], [-1, 1], c="white", lw=0.4, alpha=0.5)
    a1.plot(np.cos(th), np.sin(th), c="#52514e", lw=0.6)
    kd, md = di[keep], di[mirror]
    a1.scatter(-U[kd, 1], U[kd, 2], s=5, facecolors="none", edgecolors="#ffffff", linewidths=0.5, label="Detection")
    a1.scatter(-U[md, 1], U[md, 2], s=6, marker="x", c="#7fd4ff", linewidths=0.6, label="Ground mirror")
    for x_, y_, t_, ha, va in ((-1.04, 0, "left", "right", "center"), (1.04, 0, "right", "left", "center"),
                               (0, 1.04, "up", "center", "bottom"), (0, -1.04, "down", "center", "top")):
        a1.text(x_, y_, t_, ha=ha, va=va, fontsize=6.5, color="#52514e")
    a1.set_xlim(-1.25, 1.25); a1.set_ylim(-1.12, 1.12); a1.set_aspect("equal"); bare(a1)
    ptitle(a1, "b", "Acoustic image, max. over range (front view)")
    ca = fig.add_axes([0.935, 0.59, 0.012, 0.33]); cb = fig.colorbar(pc, cax=ca); cb.set_label("dB", labelpad=1)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2)
    lg = a1.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.04), fontsize=6, handletextpad=0.2, borderpad=0.4,
                   frameon=True, framealpha=1.0, facecolor="#1a1040", edgecolor="none")
    for t_ in lg.get_texts():
        t_.set_color("white")
    # (c) top view and (d) side view over the lidar scan
    cm = height_cmap()
    a3 = fig.add_axes([0.065, 0.155, 0.50, 0.32]); a2 = fig.add_axes([0.66, 0.155, 0.25, 0.32])
    a3.set_anchor("W"); a2.set_anchor("W")
    up = (L @ n - d) > 0.2                                                 # simultaneous lidar scan: above-ground returns darker
    for ax, (i0, i1) in ((a2, (0, 1)), (a3, (0, 2))):
        ax.scatter(L[~up, i0], L[~up, i1], s=0.2, c="#d6d4ce", lw=0, rasterized=True, zorder=1)
        ax.scatter(L[up, i0], L[up, i1], s=1.0, c="#8f8d87", lw=0, rasterized=True, zorder=1)
        o = np.argsort(hz[keep]) if ax is a2 else slice(None)
        sc = ax.scatter(pts[keep, i0][o], pts[keep, i1][o], s=9, c=np.clip(hz[keep][o], 0, H_MAX), cmap=cm, vmin=0,
                        vmax=H_MAX, lw=0.3, edgecolors="white", rasterized=True, zorder=3)
        ax.scatter(pts[mirror, i0], pts[mirror, i1], s=8, c="#e34948", marker="x", linewidths=0.6, rasterized=True, zorder=2)
        ax.scatter([0], [0], s=40, marker=">" if ax is a2 else "s", c=C_REF, zorder=5)
    xs = np.array([0, 9.0])
    a3.plot(xs, (d - n[0] * xs) / n[2], c=C_REF, lw=0.8, ls=(0, (4, 2)), zorder=4)
    a3.text(8.9, (d - n[0] * 8.9) / n[2] - 0.15, "fitted ground plane", ha="right", va="top", fontsize=6.5, color=C_REF,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    a2.set_aspect("equal"); a2.set_xlim(0, 9); a2.set_ylim(-4.5, 4.5); a2.set_xlabel("$x$ (m)"); a2.set_ylabel("$y$ (m)")
    ptitle(a2, "d", "Top view")
    a3.set_aspect("equal"); a3.set_xlim(0, 9); a3.set_ylim(-1.0, 2.6); a3.set_xlabel("$x$ (m)"); a3.set_ylabel("$z$ (m)")
    ptitle(a3, "c", "Side view, sensor frame")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=4, mfc=cm(0.55), mec="white", mew=0.3, label="Sonar detection"),
          Line2D([], [], ls="", marker="x", ms=4, mec="#e34948", mew=0.8, label="Removed ground mirror"),
          Line2D([], [], ls="", marker="o", ms=2.5, mfc="#8f8d87", mec="none", label="Lidar scan, same instant")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=2.0)
    cb2 = fig.colorbar(sc, cax=fig.add_axes([0.935, 0.155, 0.012, 0.32])); cb2.set_label("Height above ground (m)", labelpad=2)
    cb2.outline.set_linewidth(0.4); cb2.ax.tick_params(width=0.5, length=2)
    fig.savefig(OUT / "imaging.pdf"); plt.close(fig)
    num["imaging_example"] = {"capture": int(k), "detections": int(len(pts)), "mirrors": int(mirror.sum()),
                              "sensor_height_m": float(d / n[2] * -1) if n[2] else None}


# ----------------------------------------------------------------------------------------------------------------------
def fig_odometry(work, st, plt, num):
    from rtis3d.odometry3d import load_attitude
    from rtis3d.poses import lidar_reference
    O = np.load(work / "ao3d.npz")
    A = load_attitude(st["h5"])[0]
    L = lidar_reference(Path(st["site"]) / "lidar3d" / f"{st['name']}_kiss.npz", O["t"], A)
    var = {"Wheels + gyroscope": (O["P_prior"], C_WHEEL)}
    free = work / "ablation" / "ao3d_freevert.npz"
    if free.exists():
        var["Sonar, free vertical"] = (np.load(free)["P_ao"], C_FREE)
    var["3D acoustic odometry (ours)"] = (O["P_ao"], C_ODO)
    segs = [2, 5, 10, 20, 40]
    fig = plt.figure(figsize=(W2, 2.45))
    ax = [fig.add_axes([0.065, 0.165, 0.25, 0.63]), fig.add_axes([0.395, 0.165, 0.25, 0.63]),
          fig.add_axes([0.69, 0.03, 0.30, 0.77])]
    num["odometry"] = {}
    for name, (P, c) in var.items():
        r = [rpe(P, L, s) for s in segs]
        z = 3 if "ours" in name else 2
        for x in r:                                                       # RMSEs add in squares: horizontal part exactly
            x["xy_rmse_m"] = math.sqrt(max(x["trans_rmse_m"] ** 2 - x["z_rmse_m"] ** 2, 0.0))
        ax[0].plot(segs, [x["xy_rmse_m"] for x in r], "o-", ms=3.5, c=c, mec="white", mew=0.5, zorder=z, label=name)
        ax[1].plot(segs, [x["z_rmse_m"] for x in r], "o-", ms=3.5, c=c, mec="white", mew=0.5, zorder=z)
        a_, e = ate(P, L)
        if name != "Sonar, free vertical":
            Qp = fitted(P, L)[0]
            ax[2].plot(Qp[:, 0], Qp[:, 1], c=c, lw=0.8, zorder=z)
        num["odometry"][name.title()] = {"rpe": {f"{s}m": x for s, x in zip(segs, r)}, "dead_reckoning": a_}
    ax[2].plot(L[:, 0, 3], L[:, 1, 3], c=C_REF, lw=0.8, zorder=4)
    ax[2].plot(L[0, 0, 3], L[0, 1, 3], "o", ms=4, c=C_REF, zorder=5); ax[2].text(L[0, 0, 3] + 1.5, L[0, 1, 3] - 3.5, "start", fontsize=6.5, ha="left", va="top")
    for a_, lab in ((ax[0], "Horizontal RPE (m)"), (ax[1], "Height RPE (m)")):
        a_.set_xlabel("Segment length (m)"); a_.set_ylabel(lab); a_.set_xticks(segs); a_.set_xlim(0, 42); a_.set_ylim(0, None)
    ptitle(ax[0], "a", "Horizontal drift"); ptitle(ax[1], "b", "Height drift")
    ax[2].set_aspect("equal"); ptitle(ax[2], "c", "Dead reckoning, 4-DoF fit"); bare(ax[2]); ax[2].set_anchor("N")
    ax[2].autoscale_view(); scalebar(ax[2], 10, "10 m", loc=(0.05, 0.05))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], c=c, marker="o", ms=3.5, mec="white", mew=0.5, label=nm) for nm, (_, c) in var.items()]
    hs.append(Line2D([], [], c=C_REF, label="Lidar reference"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.055, 1.01), ncol=4, columnspacing=1.6)
    fig.savefig(OUT / "odometry.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def fig_place(work, st, plt, num):
    """Descriptor example, registration example (before / after), and the separation of true / false candidates."""
    from matplotlib.colors import LinearSegmentedColormap
    from rtis3d.place3d import build_keyframes, descriptor_match, register4
    from rtis3d.poses import _mount
    Z = np.load(work / "slam3d.npz")
    O = np.load(work / "ao3d.npz")
    D = np.load(work / "detections.npz")
    T_sb, _ = _mount(st["h5"])
    sel, s, KF = build_keyframes(O["P_ao"], O["caps"], D["pts"], D["w"], D["start"], T_sb)
    links = Z["links"]
    le = Z["link_err"]
    i = int(np.argmin(np.abs(le[:, 0] - np.median(le[:, 0]))))           # a typical (median-error) loop closure
    q, c = int(links[i, 0]), int(links[i, 1])
    kq, ka = KF[q], KF[c]
    sim, sh = descriptor_match(ka.desc, kq.desc)
    yaw0 = sh * 2 * math.pi / 60
    T, info = register4(kq.pts, kq.w, ka.pts, yaw0)
    fig = plt.figure(figsize=(W2, 2.45))
    cm_w = LinearSegmentedColormap.from_list("wm", ["#ffffff", "#f3c6e4", "#d9479f", C_LOOP, "#3a0030"])
    axs = [fig.add_axes([0.01 + 0.155 * j, 0.22, 0.14, 0.58], projection="polar") for j in range(2)]
    vmax = max(np.percentile(k_.desc[0][k_.desc[0] > 0], 75) for k_ in (kq, ka))
    for ax, k_, t in zip(axs, (kq, ka), ("Query", "Matched keyframe")):
        Dd = k_.desc[0]
        th = np.linspace(-math.pi, math.pi, Dd.shape[1] + 1); r = np.linspace(1, 8, Dd.shape[0] + 1)
        ax.grid(False)
        pm = ax.pcolormesh(th, r, Dd, cmap=cm_w, vmin=0, vmax=vmax, shading="flat", rasterized=True, edgecolors="none")
        for rr_ in (2, 4, 6):
            ax.plot(np.linspace(0, 2 * math.pi, 200), np.full(200, rr_), c="#bdbbb5", lw=0.35, zorder=0)
        ax.set_ylim(0, 8); ax.set_yticks([]); ax.set_xticks([])
        ax.spines["polar"].set_linewidth(0.5); ax.spines["polar"].set_color("#52514e")
        ax.annotate("", xy=(0, 8.9), xytext=(0, 6.9), xycoords="data", arrowprops=dict(arrowstyle="-|>", color="#52514e",
                    lw=0.6, mutation_scale=6), annotation_clip=False)
        ax.set_title(t, fontsize=7.5, pad=3)
    fig.text(0.155, 0.15, f"similarity {sim:.2f}; rings 1 to 8 m; arrow: heading", fontsize=6.5, ha="center",
             color="#52514e")
    cb = fig.colorbar(pm, cax=fig.add_axes([0.08, 0.075, 0.15, 0.022]), orientation="horizontal")
    cb.set_ticks([]); cb.outline.set_linewidth(0.4)
    fig.text(0.075, 0.087, "low", fontsize=6, ha="right", va="center", color="#52514e")
    fig.text(0.235, 0.087, "high", fontsize=6, ha="left", va="center", color="#52514e")
    fig.text(0.01, 0.97, r"$\bf{(a)}$  Polar descriptors", fontsize=8, va="top")
    # (b) the two submaps after the 4-DoF registration, above-ground points (what the test counts)
    ga, gq = ka.pts[:, 2] >= 0.25, kq.pts[:, 2] >= 0.25
    Q1 = kq.pts @ T[:3, :3].T + T[:3, 3]
    lim = np.percentile(np.abs(np.r_[ka.pts[ga, :2].ravel(), Q1[gq, :2].ravel()]), 99.5) + 0.5
    a2 = fig.add_axes([0.355, 0.13, 0.27, 0.72])
    a2.scatter(ka.pts[ga, 0], ka.pts[ga, 1], s=7, c=C_WHEEL, lw=0, rasterized=True, label="Matched keyframe")
    a2.scatter(Q1[gq, 0], Q1[gq, 1], s=3, c=C_LOOP, lw=0, rasterized=True, label="Query, registered")
    a2.set_aspect("equal"); a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim); a2.set_xlabel("$x$ (m)"); a2.set_ylabel("$y$ (m)")
    ptitle(a2, "b", f"Registered submaps, {100 * info['inliers_above']:.0f} % inliers")
    a2.legend(loc="lower left", fontsize=6.5, markerscale=1.5, handletextpad=0.2, borderaxespad=0.2)
    # (c) inlier fraction of all registration candidates when the spatial gate is off
    a3 = fig.add_axes([0.745, 0.17, 0.24, 0.68])
    pe = work / "ablation" / "place.npz"
    if pe.exists():
        Pz = np.load(pe); R = Pz["rows"]; cols = list(Pz["cols"])
        I = lambda n_: R[:, cols.index(n_)]  # noqa: E731
        good = (I("err_xy") < 0.5) & (I("err_yaw") < 5)
        bins = np.linspace(0, 1, 41)
        a3.axvspan(0.85, 1.0, color="#f6e3f0", lw=0, zorder=0)
        a3.hist(I("inliers_above")[~good], bins=bins, color=C_WHEEL, lw=0, label="Wrong", zorder=2)
        a3.hist(I("inliers_above")[good], bins=bins, color=C_LOOP, lw=0, alpha=0.9, label="Correct", zorder=3)
        a3.axvline(0.85, c=C_REF, ls=(0, (3, 2)), lw=0.7, zorder=4)
        a3.text(0.835, 450, "accept", ha="right", va="top", fontsize=6.5, color=C_REF)
        a3.set_yscale("log"); a3.set_ylim(0.8, 600)
        a3.set_xlabel("Inlier fraction above ground"); a3.set_ylabel("Candidates")
        a3.legend(loc="upper left", fontsize=6.5, handlelength=1.0)
        a3.set_xlim(0, 1)
        num["place_hist"] = {"wrong_above_0.85": int((I("inliers_above")[~good] >= 0.85).sum()),
                             "correct_above_0.85": int((I("inliers_above")[good] >= 0.85).sum())}
    ptitle(a3, "c", "All candidates, no spatial gate")
    fig.savefig(OUT / "place.pdf"); plt.close(fig)
    num["place_example"] = {"query_kf": q, "anchor_kf": c, "similarity": sim, "inliers_above": info["inliers_above"],
                            "link_err_m": float(le[i, 0])}


# ----------------------------------------------------------------------------------------------------------------------
def fig_trajectory(work, st, plt, num):
    Z = np.load(work / "slam3d.npz")
    sel, K, P, L = Z["kf_index"], Z["kf_poses"], Z["odom"], Z["lidar"]
    G = L[sel]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G[:, :2, 3], axis=0), axis=1))]
    Qo = fitted(P[sel], G)[0]; Qs = fitted(K, G)[0]
    fig = plt.figure(figsize=(W2, 3.05))
    a0 = fig.add_axes([0.0, 0.02, 0.42, 0.88])
    a0.plot(Qo[:, 0], Qo[:, 1], c=C_ODO, lw=0.8, zorder=2)
    a0.plot(G[:, 0, 3], G[:, 1, 3], c=C_REF, lw=1.6, zorder=3)
    a0.plot(Qs[:, 0], Qs[:, 1], c=C_SLAM, lw=0.8, zorder=4)
    lq = np.unique(Z["links"][:, 0].astype(int))
    a0.scatter(Qs[lq, 0], Qs[lq, 1], s=4, facecolors="none", edgecolors=C_LOOP, linewidths=0.4, zorder=5)
    a0.plot(G[0, 0, 3], G[0, 1, 3], "o", ms=4.5, c=C_REF, zorder=6)
    a0.annotate("start", (G[0, 0, 3], G[0, 1, 3]), xytext=(6, -6), textcoords="offset points", ha="left", va="top", fontsize=6.5)
    a0.set_aspect("equal"); bare(a0); a0.autoscale_view(); scalebar(a0, 10, "10 m", loc=(0.70, 0.93))
    ptitle(a0, "a", f"Trajectories, {len(Z['links'])} loop closures"); a0.set_anchor("NW")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], c=C_REF, lw=1.6, label="Lidar reference"), Line2D([], [], c=C_SLAM, label="RTISSLAM-3D (ours)"),
          Line2D([], [], c=C_ODO, label="3D acoustic odometry, no loops"), Line2D([], [], ls="", marker="o", ms=3.5, mfc="none", mec=C_LOOP, mew=0.7, label="Loop closure (query keyframe)")]
    a0.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.5)
    # (b) height, (c) horizontal error, (d) height error: shared distance axis
    a1 = fig.add_axes([0.53, 0.68, 0.45, 0.24]); a2 = fig.add_axes([0.53, 0.395, 0.45, 0.21], sharex=a1)
    a3 = fig.add_axes([0.53, 0.12, 0.45, 0.21], sharex=a1)
    a1.plot(s, Qo[:, 2], c=C_ODO, lw=0.8); a1.plot(s, G[:, 2, 3], c=C_REF, lw=1.6); a1.plot(s, Qs[:, 2], c=C_SLAM, lw=0.8)
    a1.set_ylabel("Height (m)"); ptitle(a1, "b", "Height along the drive")
    eo, es = np.linalg.norm(Qo[:, :2] - G[:, :2, 3], axis=1), np.linalg.norm(Qs[:, :2] - G[:, :2, 3], axis=1)
    zo, zs = np.abs(Qo[:, 2] - G[:, 2, 3]), np.abs(Qs[:, 2] - G[:, 2, 3])
    for ax, (o_, s_), lab in ((a2, (eo, es), "Error (m)"), (a3, (zo, zs), "Error (m)")):
        ax.plot(s, o_, c=C_ODO, lw=0.8); ax.plot(s, s_, c=C_SLAM, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(0.01, 10); ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels(["0.01", "0.1", "1", "10"])
        ax.set_ylabel(lab)
        rm = float(np.sqrt(np.mean(s_ ** 2)))
        ax.axhline(rm, c=C_SLAM, lw=0.6, ls=(0, (3, 2)))
        ax.text(s[-1] * 0.995, 0.012, f"RTISSLAM-3D RMSE {rm:.2f} m", ha="right", va="bottom", fontsize=6.5, color="#0b0b0b",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    plt.setp(a1.get_xticklabels(), visible=False); plt.setp(a2.get_xticklabels(), visible=False)
    a3.set_xlabel("Distance driven (m)"); a1.set_xlim(0, s[-1])
    ptitle(a2, "c", "Horizontal error against the lidar reference"); ptitle(a3, "d", "Height error", pad=2)
    fig.savefig(OUT / "trajectory.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def _site_top(ax, P, gz, cm, size, ground_col=C_GROUND, ground=True):
    """Top view of a point map: ground grey, above-ground points coloured by height, highest on top."""
    h = P[:, 2] - gz(P)
    g = h < 0.25
    if ground:
        ax.scatter(P[g, 0], P[g, 1], s=size, c=ground_col, lw=0, rasterized=True, zorder=1)
    o = np.argsort(h[~g])
    A = P[~g][o]
    return ax.scatter(A[:, 0], A[:, 1], s=size, c=np.clip(h[~g][o], 0, H_MAX), cmap=cm, vmin=0, vmax=H_MAX, lw=0,
                      rasterized=True, zorder=2)


def fig_map(work, st, plt, num):
    """Top views of the whole sonar map and lidar map (same scale and colour); the most revisited part of the site in
    perspective (sonar) and as a top-view overlay of the sonar map on the lidar map."""
    from render3d import View, look_at
    from make_drive_through_video import ground
    from scipy.spatial import cKDTree
    plt = style()
    occ = np.load(work / "occ3d.npz")
    S, L0, path = occ["sonar_occ"].astype(float), occ["lidar"].astype(float), occ["sensor_xyz"].astype(float)
    gz = ground(L0)
    keep3 = (L0[:, 2] - gz(L0)) < 3.0
    L = L0[keep3]
    Lo = L0[keep3 & occ["lidar_observed"].astype(bool)]
    cm = height_cmap()
    fig = plt.figure(figsize=(W2, 3.4))
    xl = (path[:, 0].min() - 4, path[:, 0].max() + 4); yl = (path[:, 1].min() - 4, path[:, 1].max() + 4)
    axs = []
    for j, (M, t, sz) in enumerate(((S, "Sonar map, RTISSLAM-3D poses", 0.35), (Lo, "Lidar map, within 6 m of the path", 0.08))):
        ax = fig.add_axes([0.0 + 0.33 * j, 0.02, 0.32, 0.88]); axs.append(ax); ax.set_anchor("N")
        sc = _site_top(ax, M, gz, cm, sz)
        ax.plot(path[:, 0], path[:, 1], c=C_REF, lw=0.4, alpha=0.6, zorder=3)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); bare(ax)
        ptitle(ax, "ab"[j], t)
    scalebar(axs[0], 10, "10 m", loc=(0.70, 0.04))
    cnt = np.array([len(x) for x in cKDTree(path[:, :2]).query_ball_point(path[::20, :2], 6.0)])
    c0 = path[::20][int(np.argmax(cnt))]                                  # the most revisited part of the site
    hb = 7.0
    for ax in axs:
        ax.add_patch(plt.Rectangle((c0[0] - hb, c0[1] - hb), 2 * hb, 2 * hb, fill=False, lw=0.7, ec=C_REF, ls=(0, (3, 2)), zorder=6))
    axs[0].text(c0[0] - hb, c0[1] + hb + 0.6, "(c, d)", fontsize=6.5, va="bottom")
    # (c) sonar map in perspective
    Vw = View(1300, 820, fov_deg=55)
    eye = c0 + np.array([-8.0, -10.0, 6.5]); Rc = look_at(eye, c0 + np.array([1.0, 1.5, -0.3]))
    gcol = np.array([int(C_GROUND[i:i + 2], 16) for i in (1, 3, 5)]) * 0.9
    def col(P_):
        h = np.clip((P_[:, 2] - gz(P_)) / H_MAX, 0, 1)
        C = np.asarray(cm(h))[:, :3] * 255
        C[h * H_MAX < 0.25] = gcol
        return C.astype(np.uint8)
    lo = occ["logodds"].astype(float)
    scl = 0.7 + 0.6 * np.clip((lo - lo.min()) / (np.percentile(lo, 95) - lo.min() + 1e-9), 0, 1)
    mS = (np.abs(S[:, 0] - c0[0]) < 11) & (np.abs(S[:, 1] - c0[1]) < 11)
    im = Vw.points(S[mS], col(S[mS]), 0.055, eye, Rc, rmax=5, scale=scl[mS], fog=60.0)
    nz = np.where((im < 250).any(2)); im = im[max(0, nz[0].min() - 10):nz[0].max() + 10, max(0, nz[1].min() - 10):nz[1].max() + 10]
    a2 = fig.add_axes([0.665, 0.50, 0.33, 0.40]); a2.imshow(im); bare(a2); a2.set_anchor("N")
    ptitle(a2, "c", "Sonar map, perspective")
    # (d) sonar over lidar, top view of the box
    a3 = fig.add_axes([0.665, 0.02, 0.33, 0.40]); a3.set_anchor("NW")
    box = lambda P_: (np.abs(P_[:, 0] - c0[0]) < hb) & (np.abs(P_[:, 1] - c0[1]) < hb)  # noqa: E731
    Lb = L[box(L)]; Lb = Lb[(Lb[:, 2] - gz(Lb)) > 0.25]
    a3.scatter(Lb[:, 0], Lb[:, 1], s=0.15, c=C_LIDAR, lw=0, rasterized=True, zorder=1)
    _site_top(a3, S[box(S)], gz, cm, 2.0, ground=False)
    a3.plot(path[:, 0], path[:, 1], c=C_REF, lw=0.4, alpha=0.6, zorder=3)
    a3.set_xlim(c0[0] - hb, c0[0] + hb); a3.set_ylim(c0[1] - hb, c0[1] + hb); a3.set_aspect("equal"); bare(a3)
    a3.add_patch(plt.Rectangle((c0[0] - hb, c0[1] - hb), 2 * hb, 2 * hb, fill=False, lw=0.5, ec="#52514e", zorder=6))
    scalebar(a3, 2, "2 m", loc=(0.05, 0.05))
    ptitle(a3, "d", "Close-up over the lidar (grey)")
    cax = fig.add_axes([0.05, 0.075, 0.12, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("Height above ground (m)", fontsize=6.5, labelpad=1)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.5, length=2, labelsize=6)
    fig.savefig(OUT / "map.pdf"); plt.close(fig)


def fig_build(work, st, plt, num):
    """The map and trajectory as RTISSLAM-3D builds them: three moments, top view, the same scale (as in the video)."""
    from rtis3d.place3d import build_keyframes, rotz
    from rtis3d.poses import _mount
    plt = style()
    Z = np.load(work / "slam3d.npz")
    O = np.load(work / "ao3d.npz")
    D = np.load(work / "detections.npz")
    Hh = np.load(work / "slam3d_hist.npy", mmap_mode="r")
    T_sb, _ = _mount(st["h5"])
    sel, s, KF = build_keyframes(O["P_ao"], O["caps"], D["pts"], D["w"], D["start"], T_sb)
    sel_s, K, P, L = Z["kf_index"], Z["kf_poses"], Z["odom"], Z["lidar"]
    G = L[sel_s]
    _, R, t = fitted(K, G)                                                 # one display frame: the final fit
    disp = lambda X: X @ R.T + t  # noqa: E731
    links = Z["links"][:, :2].astype(int)
    n_kf = len(K)
    moments = [int(0.2 * n_kf), int(0.55 * n_kf), n_kf - 1]
    cm = height_cmap()
    fig = plt.figure(figsize=(W2, 2.85))
    allp = disp(K[:, :3, 3])
    xl = (allp[:, 0].min() - 3, allp[:, 0].max() + 3); yl = (allp[:, 1].min() - 3, allp[:, 1].max() + 3)
    w_ = 0.325
    for j, m in enumerate(moments):
        E = np.asarray(Hh[m][:m + 1], float)                               # the estimate after keyframe m
        ax = fig.add_axes([0.005 + j * (w_ + 0.007), 0.10, w_, 0.81])
        pts, hh = [], []
        for i in range(0, m + 1, 2):
            yaw = math.atan2(E[i, 1, 0], E[i, 0, 0])
            p = KF[i].pts; p = p[p[:, 2] >= 0.25]
            pts.append(p @ rotz(yaw).T + E[i, :, 3]); hh.append(p[:, 2])
        A = disp(np.vstack(pts)); h = np.clip(np.concatenate(hh), 0, H_MAX)
        ax.scatter(A[:, 0], A[:, 1], s=0.35, c=h, cmap=cm, vmin=0, vmax=H_MAX, lw=0, alpha=0.6, rasterized=True, zorder=1)
        Ep = disp(E[:, :, 3])
        ax.plot(G[:m + 1, 0, 3], G[:m + 1, 1, 3], c=C_REF, lw=1.6, alpha=0.3, zorder=2)
        ax.plot(Ep[:, 0], Ep[:, 1], c=C_SLAM, lw=0.9, zorder=4)
        lk = links[(links[:, 0] <= m) & (links[:, 1] <= m)]
        lq = np.unique(lk[:, 0])
        ax.scatter(Ep[lq, 0], Ep[lq, 1], s=5, facecolors="none", edgecolors=C_LOOP, linewidths=0.45, zorder=5)
        Rm = R @ E[m, :, :3]; hd = Rm[:2, 0] / (np.linalg.norm(Rm[:2, 0]) + 1e-9)
        ax.annotate("", xy=Ep[m, :2] + 3.5 * hd, xytext=Ep[m, :2], arrowprops=dict(arrowstyle="-|>", color=C_REF, lw=1.0,
                    mutation_scale=8), zorder=6)
        ax.plot(*Ep[m, :2], "o", ms=4, c=C_REF, zorder=6)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); bare(ax)
        ptitle(ax, "abc"[j], f"After {s[sel[m]]:.0f} m, {len(lq)} loop closures")
        if j == 0:
            scalebar(ax, 10, "10 m", loc=(0.68, 0.04))
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], c=C_SLAM, label="RTISSLAM-3D"), 
          Line2D([], [], c=C_REF, lw=1.6, alpha=0.3, label="Lidar reference"),
          Line2D([], [], ls="", marker="o", ms=3.5, mfc="none", mec=C_LOOP, mew=0.7, label="Loop closure"),
          Line2D([], [], ls="", marker="o", ms=3, mfc=cm(0.5), mec="none", label="Sonar map, above ground"),
          Line2D([], [], ls="", marker=r"$\rightarrow$", ms=7, mfc=C_REF, mec=C_REF, mew=0.3, label="Current pose")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, columnspacing=1.3, fontsize=6.5)
    fig.savefig(OUT / "build.pdf"); plt.close(fig)


def tables(work, st, num):
    """paper/generated/tab_ablation.tex (main text), tab_thresholds.tex (appendix) and runtime.tex."""
    gen = ROOT / "paper" / "generated"
    gen.mkdir(exist_ok=True)
    A = json.loads((work / "ablation" / "summary.json").read_text())
    rows_top = [("ref", "\\textbf{RTISSLAM-3D (ours)}"), ("noloops", "No loop closures"), ("prior_loops", "Wheel odometry"),
                ("prior_noloops", "Wheel odometry, no loops"), ("freevert", "Sonar vertical"), ("heightbands", "Four height bands")]
    rows_bot = [("inl075", "Inliers $\\geq$ 0.75"), ("inl080", "Inliers $\\geq$ 0.80"), ("ref", "\\textbf{Inliers $\\geq$ 0.85}"),
                ("inl090", "Inliers $\\geq$ 0.90"), ("inl095", "Inliers $\\geq$ 0.95"), ("gate3", "Gate \\SI{3}{\\meter}"),
                ("gate8", "Gate \\SI{8}{\\meter}"), ("gate12", "Gate \\SI{12}{\\meter}"), ("rmse015", "RMSE $\\leq$ \\SI{0.15}{\\meter}")]
    def row(k, lab):
        v = A[k]
        cells = [str(v["links"]), f"{v['ate_xy_m']:.2f}", f"{v['z_rmse_m']:.2f}", f"{v['xy_max_m']:.2f}",
                 str(v["over_0.5m"]) if v["links"] else "--"]
        if lab.startswith("\\textbf"):                                       # the default configuration
            cells = ["\\textbf{" + c + "}" for c in cells]
        return lab + " & " + " & ".join(cells) + " \\\\\n"
    def table(label, caption, head, rows):
        return ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
                "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
                + head + " & Loops & $xy$ & $z$ & Max & Bad \\\\\n\\midrule\n"
                + "".join(row(k, l) for k, l in rows if k in A) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (gen / "tab_ablation.tex").write_text(table(
        "tab:slam", "Ablation of RTISSLAM-3D Against the Lidar Reference: Horizontal ATE, Height RMSE and Maximum Horizontal "
        "Error After a 4-DoF Fit (m), and Loop Closures With an Error Above \\SI{0.5}{\\meter} (Bad).", "Variant", rows_top))
    (gen / "tab_thresholds.tex").write_text(table(
        "tab:thresholds", "Sensitivity of RTISSLAM-3D to the Loop Closure Test (All Else as the Default, in Bold; Errors as in "
        "Table~\\ref{tab:slam}).", "Test", rows_bot))
    sec = st.get("seconds", {})
    n_cap = int(np.load(work / "ao3d.npz")["caps"].shape[0])
    R = json.loads((work / "slam3d.json").read_text())
    dur = float(np.ptp(np.load(work / "ao3d.npz")["t"]))
    txt = (f"On our workstation, the 3D imaging of all 6090 captures took \\SI{{{sec.get('volumes', 0):.0f}}}{{\\second}} on the GPU "
           f"(\\SI{{{1000 * sec.get('volumes', 0) / 6090:.0f}}}{{\\milli\\second}} per capture, against a pulse interval of \\SI{{230}}{{\\milli\\second}}), "
           f"the detection \\SI{{{sec.get('detections', 0):.0f}}}{{\\second}}, the 3D acoustic odometry of {n_cap} captures "
           f"\\SI{{{sec.get('odometry', 0):.0f}}}{{\\second}} and the SLAM back-end with the place recognition "
           f"\\SI{{{R['seconds']:.0f}}}{{\\second}} (both on a single CPU thread, in Python), and the 3D map \\SI{{{sec.get('map', 0):.0f}}}{{\\second}}. "
           f"For a drive of \\SI{{{dur / 60:.1f}}}{{\\minute}}, the whole chain thus runs about "
           f"{dur / (sec.get('volumes', 0) + sec.get('detections', 0) + sec.get('odometry', 0) + R['seconds'] + sec.get('map', 0)):.1f} times faster than real time.")
    (gen / "runtime.tex").write_text(txt + "\n")
    num["runtime_s"] = sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="imaging,odometry,place,trajectory,map,build")
    ap.add_argument("--capture", type=int, default=3400)
    a = ap.parse_args()
    work = Path(a.work)
    OUT.mkdir(parents=True, exist_ok=True)
    plt = style()
    st = json.loads((work / "pipeline.json").read_text())
    nf = ROOT / "paper" / "numbers.json"
    num = json.loads(nf.read_text()) if nf.exists() else {}
    only = a.only.split(",")
    if "imaging" in only:
        fig_imaging(work, st, a.capture, plt, num)
    if "odometry" in only:
        fig_odometry(work, st, plt, num)
    if "place" in only:
        fig_place(work, st, plt, num)
    if "trajectory" in only:
        fig_trajectory(work, st, plt, num)
    if "map" in only:
        fig_map(work, st, plt, num)
    if "build" in only:
        fig_build(work, st, plt, num)
    if not (work / "slam3d.json").exists():
        nf.write_text(json.dumps(num, indent=1, default=float)); return
    r = json.loads((work / "slam3d.json").read_text())
    num["slam3d"] = r.get("slam3d"); num["odometry_kf"] = r.get("odometry"); num["links"] = r["links"]; num["link_err"] = r.get("link_err")
    num["map"] = json.loads((work / "occ3d.json").read_text()).get("thr_2")
    if (work / "ablation" / "summary.json").exists():
        num["ablation"] = json.loads((work / "ablation" / "summary.json").read_text())
        tables(work, st, num)
    nf.write_text(json.dumps(num, indent=1, default=float))
    print("figures in", OUT)


if __name__ == "__main__":
    main()
