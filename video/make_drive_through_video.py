"""Drive-through video of the 3D maps: a camera following the robot's path through the sonar 3D map and the lidar 3D
map, side by side, with the GoPro view as an inset (lab style).

    python video/make_drive_through_video.py <grid3d npz> <drive npz> <dataset.h5> --out <mp4>
        [--step 0.5] [--fps 15] [--jobs 6] [--test]
Inputs: scripts/grid3d_eval.py output (sonar_occ voxels, lidar map points, sensor_xyz: the sonar's path, one per drive
node, all in the lidar map frame) and the drive (robot-clock times of the nodes, for the GoPro).
Renderer: a pinhole camera 3.5 m behind and 1.8 m above the robot, looking along the smoothed heading, 8 deg down;
points projected and drawn far-to-near as square splats whose size follows the distance; colour = height above the
local ground (viridis, 0..2.5 m).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "video"))
import cosys_video as cv  # noqa: E402

TITLE = "RTISSLAM-3D in a Forest:\nDriving Through the Sonar and Lidar 3D Maps"
PW, PH = 940, 600            # panel size (pixels)
FOV = 75.0                   # horizontal field of view (deg)


def ground(M, cell=0.5):
    ij = np.floor(M[:, :2] / cell).astype(int); o = ij.min(0); ij -= o
    G = np.full(tuple(ij.max(0) + 1), np.inf)
    np.minimum.at(G, (ij[:, 0], ij[:, 1]), M[:, 2])
    med = np.median(G[np.isfinite(G)])
    def gz(P):
        i = np.clip(np.floor(P[:, 0] / cell).astype(int) - o[0], 0, G.shape[0] - 1)
        j = np.clip(np.floor(P[:, 1] / cell).astype(int) - o[1], 0, G.shape[1] - 1)
        z = G[i, j].copy(); z[~np.isfinite(z)] = med
        return z
    return gz


def min_jerk(y, fps, max_acc_deg_s2):
    """Minimum-jerk fit of the heading track y [rad] (one sample per video frame): argmin ||x - y||^2 + lam ||D3 x||^2,
    lam the smallest (bisection) for which the angular acceleration stays within max_acc_deg_s2 everywhere. The
    rate and the acceleration both change continuously (no sudden start or stop of a turn).
    -> (x, lam, peak rate deg/s, peak acceleration deg/s^2)."""
    from scipy.sparse import diags, eye as speye
    from scipy.sparse.linalg import spsolve
    n = len(y)
    if n < 8:
        return y, 0.0, 0.0, 0.0
    D = diags([-1, 3, -3, 1], [0, 1, 2, 3], shape=(n - 3, n)) * fps ** 3          # jerk, rad/s^3
    DtD = (D.T @ D).tocsc()
    I = speye(n, format="csc")

    def fit(lam):
        x = spsolve(I + lam * DtD, y)
        return x, np.degrees(np.abs(np.diff(x, 2)).max() * fps ** 2)
    lo, hi = -8.0, 6.0                                                            # log10 lam
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fit(10 ** mid)[1] > max_acc_deg_s2:
            lo = mid
        else:
            hi = mid
    x, acc = fit(10 ** hi)
    return x, 10 ** hi, float(np.degrees(np.abs(np.diff(x)).max() * fps)), float(acc)


def smooth_rig(path, s_path, s_list, fps, pos_sigma=1.5, look=3.0, max_acc_deg_s2=40.0):
    """Steady camera rig: position = Gaussian-smoothed path (sigma pos_sigma m of arc length), heading = direction of the
    smoothed path over +-look m, then fitted with a minimum-jerk track whose angular acceleration stays within
    max_acc_deg_s2 (deg/s^2 of video time; min_jerk).
    -> list of (eye [3], Rcw [3, 3], node index)."""
    from scipy.ndimage import gaussian_filter1d
    ds = np.median(np.diff(s_path)) if len(s_path) > 1 else 0.1
    sm = np.stack([gaussian_filter1d(path[:, i], pos_sigma / max(ds, 1e-3), mode="nearest") for i in range(3)], 1)
    pos = np.stack([np.interp(s_list, s_path, sm[:, i]) for i in range(3)], 1)
    a = np.stack([np.interp(np.clip(s_list - look, 0, s_path[-1]), s_path, sm[:, i]) for i in range(2)], 1)
    b = np.stack([np.interp(np.clip(s_list + look, 0, s_path[-1]), s_path, sm[:, i]) for i in range(2)], 1)
    yaw = np.unwrap(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0]))
    yaw, lam, rate, acc = min_jerk(yaw, fps, max_acc_deg_s2)
    print(f"camera heading: min-jerk lam {lam:.3g}, peak turn rate {rate:.1f} deg/s, peak acceleration {acc:.1f} deg/s^2", flush=True)
    out = []
    tilt = math.radians(10)
    for p, y, s in zip(pos, yaw, s_list):
        f = np.array([math.cos(y), math.sin(y), 0.0])
        eye = p + np.array([0, 0, 1.8]) - 3.5 * f
        fwd = np.array([math.cos(y) * math.cos(tilt), math.sin(y) * math.cos(tilt), -math.sin(tilt)])
        left = np.array([-math.sin(y), math.cos(y), 0.0])
        out.append((eye, np.stack([fwd, left, np.cross(fwd, left)]), int(np.clip(np.searchsorted(s_path, s), 0, len(path) - 1))))
    return out


def camera(path, s_path, s):
    """Camera pose at arc length s: position behind / above the robot, looking along the smoothed heading."""
    i = int(np.clip(np.searchsorted(s_path, s), 1, len(path) - 1))
    p = path[i]
    j0, j1 = np.searchsorted(s_path, s - 2.0), np.searchsorted(s_path, s + 2.0)
    j0, j1 = max(0, j0), min(len(path) - 1, j1)
    d = path[j1, :2] - path[j0, :2]
    yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-3 else 0.0
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    eye = p + np.array([0, 0, 1.8]) - 3.5 * f
    tilt = math.radians(8)
    fwd = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), -math.sin(tilt)])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(fwd, left)
    return eye, np.stack([fwd, left, up]), i


def render(P, C, size, eye, Rcw, bg=255, rmax=4, fog=30.0, path=None, scale=None):
    """Points P [n, 3] with colours C [n, 3] (uint8) -> image [PH, PW, 3]: solid square splats of half-size
    ~ size * f / depth (at most rmax px), drawn far to near, colours faded towards white with depth (fog, m);
    path: [m, 3] points of the road ahead, drawn on top in magenta."""
    img = np.full((PH, PW, 3), bg, np.uint8)
    q = (P - eye) @ Rcw.T                                 # x forward, y left, z up (camera)
    m = (q[:, 0] > 0.5) & (q[:, 0] < 40)
    q, C = q[m], C[m].astype(np.float32)
    sc = scale[m] if scale is not None else 1.0
    f = PW / 2 / math.tan(math.radians(FOV / 2))
    u = np.round(PW / 2 - f * q[:, 1] / q[:, 0]).astype(int)
    v = np.round(PH / 2 - f * q[:, 2] / q[:, 0]).astype(int)
    r = np.clip(np.round(size * sc * f / q[:, 0]).astype(int), 0, rmax)
    ok = (u > -rmax) & (u < PW + rmax) & (v > -rmax) & (v < PH + rmax)
    u, v, r, C, dep = u[ok], v[ok], r[ok], C[ok], q[ok, 0]
    a = np.exp(-dep / fog)[:, None]
    C = (a * C + (1 - a) * bg).astype(np.uint8)
    order = np.argsort(-dep)
    u, v, r, C = u[order], v[order], r[order], C[order]
    for du in range(-rmax, rmax + 1):
        for dv in range(-rmax, rmax + 1):
            sel = r >= max(abs(du), abs(dv))
            uu, vv = u[sel] + du, v[sel] + dv
            k = (uu >= 0) & (uu < PW) & (vv >= 0) & (vv < PH)
            img[vv[k], uu[k]] = C[sel][k]
    if path is not None and len(path) > 1:
        qp = (path - eye) @ Rcw.T
        mp = qp[:, 0] > 0.5
        up = np.round(PW / 2 - f * qp[mp, 1] / qp[mp, 0]).astype(int)
        vp = np.round(PH / 2 - f * qp[mp, 2] / qp[mp, 0]).astype(int)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                k = (up + du >= 0) & (up + du < PW) & (vp + dv >= 0) & (vp + dv < PH)
                img[vp[k] + dv, up[k] + du] = (177, 0, 145)
    return img


def frames_job(args):
    grid, drive, h5, s_all, i_lo, i_hi, out, fps, min_hits, title, lidar_all = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chirppy import CHIRPDataset
    import rtis3d  # noqa: F401
    from rtisreal.video_frames import VideoFrames
    z = np.load(grid)
    S, L, path = z["sonar_occ"].astype(float), z["lidar"].astype(float), z["sensor_xyz"].astype(float)
    if "logodds" in z.files:                           # evidence map (scripts/build_occ3d.py): size / opacity by log-odds
        lo_ = z["logodds"].astype(float)
        ev = np.clip((lo_ - lo_.min()) / (np.percentile(lo_, 95) - lo_.min() + 1e-9), 0, 1)
        S_scale = 0.6 + 0.8 * ev
        s_size = float(z["res"]) * 0.55 if "res" in z.files else 0.06
    else:
        S = S[z["hits"] >= min_hits]                   # intensity threshold: voxels detected in >= min_hits captures
        ev, S_scale, s_size = np.ones(len(S)), None, 0.10
    gz = ground(L)
    keep = (L[:, 2] - gz(L)) < 3.0                     # the band the sonar can see (it sees nothing above ~2 m)
    if not lidar_all and "lidar_observed" in z.files:  # as the map figure: within 6 m in front of a sensor pose
        keep &= z["lidar_observed"].astype(bool)
    L = L[keep]
    cmap = plt.get_cmap("viridis")
    col = lambda P: (np.asarray(cmap(np.clip((P[:, 2] - gz(P) + 0.3) / 2.8, 0, 1)))[:, :3] * 255).astype(np.uint8)  # noqa: E731
    CS, CL = col(S), col(L)
    CS = (ev[:, None] * CS + (1 - ev[:, None]) * (0.35 * CS + 0.65 * 255)).astype(np.uint8)   # weak evidence: faint
    # the ground band (< 0.25 m above the local ground) lighter, so trunks and vegetation stand out
    for P_, C_ in ((S, CS), (L, CL)):
        gnd = (P_[:, 2] - gz(P_)) < 0.25
        C_[gnd] = (0.45 * C_[gnd] + 0.55 * 255).astype(np.uint8)
    s_path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    d = np.load(drive)
    t_nodes = d["t"]
    fig = cv.new_fig()
    fig.text(0.5, 0.935, title, ha="center", va="center", fontsize=26, linespacing=1.2)
    ax = [fig.add_axes([0.015, 0.25, 0.48, 0.58]), fig.add_axes([0.505, 0.25, 0.48, 0.58])]
    ims = []
    for a_, t in zip(ax, ("Sonar (eRTIS) 3D map", "Lidar 3D map (reference)")):
        a_.set_axis_off(); ims.append(a_.imshow(np.full((PH, PW, 3), 255, np.uint8)))
        a_.set_title(t, fontsize=18, fontweight="bold", color=cv.MAGENTA if "Sonar" in t else "k", pad=6)
    gax = fig.add_axes([0.41, 0.025, 0.18, 0.18]); gax.set_axis_off()
    gim = gax.imshow(np.zeros((540, 960, 3), np.uint8))
    stx = fig.text(0.5, 0.215, "", ha="center", fontsize=14)
    fig.text(0.5, 0.012, "", ha="center")
    fig.text(0.205, 0.165, "Sonar: 3D occupancy (0.1 m, beam-shaped evidence, free space carved);\nsize and opacity follow the evidence.\nLidar (reference): the 3D lidar SLAM map up to 3 m, within 6 m of the path.\n"
             "Colour: height above the local ground, 0 to 2.5 m.", ha="center", va="center", fontsize=11, color="0.4", linespacing=1.4)
    fig.text(0.795, 0.165, "Camera: 3.5 m behind and 1.8 m above the robot.\nMagenta: the road ahead. Inset: the robot's GoPro.",
             ha="center", va="center", fontsize=11, color="0.4", linespacing=1.4)
    cv.logos(fig)
    wr = cv.writer(out, fps=fps)
    s_all = np.asarray(s_all)
    rig = smooth_rig(path, s_path, s_all, fps)[i_lo:i_hi]           # the rig of the whole path: no seams between chunks
    s_list = s_all[i_lo:i_hi]
    with CHIRPDataset(h5) as ds:
        i0 = rig[0][2]
        gopro = VideoFrames(ds.get_video(0), float(t_nodes[min(i0, len(t_nodes) - 1)]), width=960)
        for s, (eye, Rcw, i) in zip(s_list, rig):
            ahead = path[i:np.searchsorted(s_path, s + 15.0)] - np.array([0, 0, 0.30])   # road ahead, on the ground
            ims[0].set_data(render(S, CS, s_size, eye, Rcw, path=ahead, scale=S_scale))
            ims[1].set_data(render(L, CL, 0.035, eye, Rcw, path=ahead))
            gim.set_data(gopro.at(float(t_nodes[min(i, len(t_nodes) - 1)])))
            stx.set_text(f"Distance driven: {s:.0f} m of {s_path[-1]:.0f} m")
            wr.append_data(cv.grab(fig))
        gopro.close()
    wr.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("drive")
    ap.add_argument("h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.4)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--min-hits", type=int, default=6, help="sonar voxels detected in at least this many captures")
    ap.add_argument("--lidar-all", action="store_true", help="show the whole lidar map, not only the part within 6 m in "
                    "front of the sensor path (the volume the map is scored in)")
    a = ap.parse_args()
    z = np.load(a.grid)
    path = z["sensor_xyz"]
    L = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
    s_all = np.arange(2.0, L - 1.0, a.step)
    if a.test:
        s_all = s_all[:400]
    bounds = np.linspace(0 if not a.test else 380, len(s_all), (1 if a.test else a.jobs) + 1).astype(int)
    out = Path(a.out)
    parts = out.parent / (out.stem + "_parts"); parts.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    with ProcessPoolExecutor(len(bounds) - 1, mp_context=mp.get_context("spawn")) as ex:
        files = list(ex.map(frames_job, [(a.grid, a.drive, a.h5, s_all.tolist(), int(bounds[i]), int(bounds[i + 1]),
                                          parts / f"p{i:02d}.mp4", a.fps, a.min_hits, a.title.replace("\\n", "\n"), a.lidar_all) for i in range(len(bounds) - 1)]))
    wr = cv.writer(parts / "title.mp4", fps=a.fps); cv.hold(wr, cv.slide_title(a.title.replace("\\n", "\n")), 3, fps=a.fps, fade_in=False); wr.close()
    wr = cv.writer(parts / "end.mp4", fps=a.fps); cv.hold(wr, cv.slide_end(), 3, fps=a.fps, fade_out=False); wr.close()
    import imageio_ffmpeg
    lst = parts / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in [parts / "title.mp4", *files, parts / "end.mp4"]))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
