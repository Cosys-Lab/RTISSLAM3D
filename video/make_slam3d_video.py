"""Video of the full 3D SLAM being built: the 6-DoF keyframe poses appear one by one, the sonar map grows from them,
loop closures are found, verified and drawn, and the whole estimate snaps into place when they are added.

    python video/make_slam3d_video.py <slam3d npz> <dataset.h5> <detections.npz> --out <mp4>
        [--jobs 6] [--test]
Input: scripts/run_slam3d.py output (<prefix>.npz and <prefix>_hist.npy, the keyframe estimate after every keyframe).
Panels: overview (orbiting camera over the whole site: map, SLAM path, odometry-only path, lidar path for reference,
6-DoF pose triads, loop links in magenta), chase view (behind the robot: a triad at every keyframe, the newest
map points in black, loop links), the GoPro, live numbers, and the height profile. Map points: each keyframe's own detections
(the captures since the previous keyframe, 0.1 m voxel means), carried by the keyframe's current pose estimate, so the
map deforms with the graph. Colour: height above the keyframe's base (on the ground), 0 to 2.5 m.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cosys_video as cv  # noqa: E402

TITLE = "RTISSLAM-3D in a Forest:\nFull 6-DoF Sonar SLAM, Step by Step"
FPK = 2                     # video frames per keyframe (1 m of travel)
FPS = 15
FINALE = 120                # frames of the finished map, orbiting
OV = (1120, 640)            # overview panel (px)
CH = (760, 440)             # chase panel


def prepare(slam, h5, dets, cache):
    """Keyframe slices, per-frame blended estimates, events -> cache npz."""
    import rtis3d  # noqa: F401
    from rtis3d.place3d import voxel_down
    from rtis3d.poses import _mount
    Z = np.load(slam)
    H = np.load(str(slam)[:-4] + "_hist.npy")
    sel, P, links = Z["kf_index"], Z["odom"], Z["links"]
    Lg = Z["lidar"] if "lidar" in Z.files else np.full_like(P, np.nan)          # no reference: nothing drawn
    caps = Z["caps"]
    D = np.load(dets)
    pts, w, st = D["pts"], D["w"], D["start"]
    T_sb, _ = _mount(h5)
    n = len(sel)
    owner = np.clip(np.searchsorted(sel, np.arange(len(P)), side="right") - 1, 0, n - 1)
    XS, WS, KS = [], [], []
    for k in range(n):
        idx = np.where(owner == k)[0]
        Ti = np.linalg.inv(P[sel[k]])
        Q, W = [], []
        for i in idx:
            c = caps[i]
            p = pts[st[c]:st[c + 1]] @ T_sb[:3, :3].T + T_sb[:3, 3]
            M = Ti @ P[i]
            Q.append(p @ M[:3, :3].T + M[:3, 3]); W.append(w[st[c]:st[c + 1]])
        if not Q:
            continue
        Q, W = voxel_down(np.vstack(Q), np.concatenate(W), 0.1)
        XS.append(Q); WS.append(W); KS.append(np.full(len(Q), k))
    X, W, K = np.vstack(XS).astype(np.float32), np.concatenate(WS), np.concatenate(KS)
    # height above the keyframe's base: z in the keyframe's level frame (roll / pitch of the odometry = IMU)
    Rk = P[sel][:, :3, :3]
    hgt = np.einsum("nj,nj->n", np.einsum("nij,nj->ni", Rk[K], X), np.tile([0, 0, 1.0], (len(X), 1)))
    keep = (W >= np.percentile(W, 30)) & (hgt > -0.5) & (hgt < 3.0)
    X, W, K, hgt = X[keep], W[keep], K[keep], hgt[keep]
    # blended display estimate per frame: corrections glide in over ~0.5 s
    nf = n * FPK
    Dsp = np.full((nf, n, 3, 4), np.nan, np.float32)
    cur = H[0].copy()
    for f in range(nf):
        m = f // FPK
        tgt = H[m]
        new = np.isnan(cur[:, 0, 0]) & ~np.isnan(tgt[:, 0, 0])
        cur[new] = tgt[new]
        ok = ~np.isnan(tgt[:, 0, 0])
        cur[ok] = cur[ok] + 0.3 * (tgt[ok] - cur[ok])
        Dsp[f] = cur
    np.savez(cache, X=X, W=W, K=K, hgt=hgt, Dsp=Dsp, final=H[-1], sel=sel, odom=P[sel], lidar=Lg[sel], links=links,
             t=Z["t"][sel])
    return nf + FINALE


def frames_job(args):
    cache, h5, f0, f1, out, nf_total, title = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chirppy import CHIRPDataset
    import rtis3d  # noqa: F401
    from rtisreal.video_frames import VideoFrames
    from render3d import View, look_at
    from make_drive_through_video import smooth_rig
    C = np.load(cache)
    X, K, hgt, Dsp, final = C["X"], C["K"], C["hgt"], C["Dsp"], C["final"]
    odom, lid, links, t_kf = C["odom"], C["lidar"], C["links"], C["t"]
    n = len(C["sel"])
    nb = n * FPK
    cmap = plt.get_cmap("viridis")
    col = (np.asarray(cmap(np.clip(hgt / 2.5, 0, 1)))[:, :3] * 255).astype(np.uint8)
    gnd = hgt < 0.25
    col[gnd] = (0.4 * col[gnd] + 0.6 * 255).astype(np.uint8)
    s_kf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(final[:, :2, 3], axis=0), axis=1))]
    lq = links[:, 0].astype(int) if len(links) else np.zeros(0, int)
    lc = links[:, 1].astype(int) if len(links) else np.zeros(0, int)
    has_lid = bool(np.isfinite(lid[:, 0, 3]).any())
    ref = lid if has_lid else final
    centre = np.r_[ref[:, :2, 3].mean(0), ref[:, 2, 3].mean()]
    # chase camera: the drive-through rig (smoothed path, minimum-jerk heading, angular acceleration <= 40 deg/s^2)
    # computed once on the final path for every frame; the camera then follows the displayed robot pose
    s_frames = np.interp(np.arange(nb) / FPK, np.arange(n), s_kf)
    rig = smooth_rig(final[:, :3, 3], s_kf, s_frames, FPS)
    Vo, Vc = View(*OV, fov_deg=40), View(*CH, fov_deg=70)
    fig = cv.new_fig()
    fig.text(0.5, 0.935, title, ha="center", va="center", fontsize=26, linespacing=1.2)
    axo = fig.add_axes([0.005, 0.24, 0.585, 0.6]); axo.set_axis_off(); imo = axo.imshow(np.full((OV[1], OV[0], 3), 255, np.uint8))
    axo.set_title("Overview: map, 6-DoF poses and loop closures", fontsize=16, fontweight="bold", color=cv.MAGENTA, pad=4)
    axc = fig.add_axes([0.6, 0.43, 0.395, 0.41]); axc.set_axis_off(); imc = axc.imshow(np.full((CH[1], CH[0], 3), 255, np.uint8))
    axc.set_title("Behind the robot: a pose triad every 1 m", fontsize=16, fontweight="bold", pad=4)
    axg = fig.add_axes([0.605, 0.155, 0.2, 0.25]); axg.set_axis_off(); gim = axg.imshow(np.zeros((540, 960, 3), np.uint8))
    stats = fig.text(0.815, 0.40, "", ha="left", va="top", fontsize=12.5, linespacing=1.55, family="monospace")
    axh = fig.add_axes([0.235, 0.065, 0.33, 0.14])
    if has_lid:
        axh.plot(s_kf, lid[:, 2, 3], c="k", lw=0.9, label="lidar (reference)")
    (lh_s,) = axh.plot([], [], c="tab:blue", lw=1.2, label="RTISSLAM-3D")
    (lh_o,) = axh.plot([], [], c="tab:orange", lw=0.9, label="odometry only")
    axh.set_xlim(0, s_kf[-1]); lo_ = np.nanmin([np.nanmin(lid[:, 2, 3]) if has_lid else np.inf, final[:, 2, 3].min()]) - 1.0   # odometry may run off
    hi_ = np.nanmax([np.nanmax(lid[:, 2, 3]) if has_lid else -np.inf, final[:, 2, 3].max()]) + 1.0
    axh.set_ylim(lo_, hi_); axh.set_ylabel("height (m)", fontsize=10); axh.set_xlabel("distance driven (m)", fontsize=10)
    axh.tick_params(labelsize=9); axh.legend(fontsize=8, loc="upper left", ncol=3, frameon=False)
    fig.text(0.31, 0.225, "Blue: RTISSLAM-3D (eRTIS + IMU + wheels).  Orange: odometry only.  Grey: 3D lidar SLAM, reference only.\n"
             "Triads (6-DoF poses): x red, y green, z blue.  Magenta lines: verified 3D loop closures.  Black: the newest map points.",
             ha="center", va="center", fontsize=10.5, color="0.35", linespacing=1.4)
    cv.logos(fig)
    wr = cv.writer(out, fps=FPS)
    with CHIRPDataset(h5) as ds:
        m0 = min(f0 // FPK, n - 1)
        try:
            gopro = VideoFrames(ds.get_video(0), float(t_kf[m0]), width=960)
        except Exception:                                   # no GoPro in this recording
            gopro = None
        for f in range(f0, f1):
            fin = f >= nb
            m = n - 1 if fin else f // FPK
            E = final if fin else Dsp[f]
            ok = np.arange(n) <= m
            T = E[:m + 1]
            # map points carried by their keyframe's current pose
            vis = K <= m
            Kv = K[vis]
            R, tt = E[Kv, :, :3], E[Kv, :, 3]
            Pm = np.einsum("nij,nj->ni", R, X[vis]) + tt
            Cm = col[vis].copy()
            cur = (Kv > m - 4) & ~fin
            Cm[cur] = (25, 25, 25)
            # overview: orbit (half a turn over the build, then a full turn in the finale)
            az = math.radians(-60 + 180 * min(f, nb) / nb + (360 * (f - nb) / FINALE if fin else 0))
            el = math.radians(38)
            dist = 128.0
            eye = centre + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            Rc = look_at(eye, centre)
            img = Vo.points(Pm, Cm, 0.09, eye, Rc, rmax=2)
            Vo.polyline(img, lid[:m + 1, :3, 3], (150, 150, 150), 1, eye, Rc)
            Vo.polyline(img, odom[:m + 1, :3, 3], (255, 140, 20), 2, eye, Rc)
            Vo.polyline(img, T[:, :3, 3], (30, 100, 220), 3, eye, Rc)
            act = lq <= m
            if act.any():
                age = m - lq[act]
                Vo.lines(img, E[lc[act], :3, 3], E[lq[act], :3, 3], (205, 90, 185), 1, eye, Rc)
                fresh = age < 8
                if fresh.any() and not fin:
                    Vo.lines(img, E[lc[act][fresh], :3, 3] + [0, 0, 0.05], E[lq[act][fresh], :3, 3] + [0, 0, 0.05], (177, 0, 145), 5, eye, Rc)
            Vo.triads(img, T[::10], 1.3, 2, eye, Rc)
            Vo.triads(img, T[-1:], 3.5, 4, eye, Rc)
            imo.set_data(img)
            # chase view
            p = T[-1, :, 3]
            Rcc = rig[min(f, nb - 1)][1]
            fw = np.r_[Rcc[0, :2], 0.0]; fw /= np.linalg.norm(fw)
            eye_c = p - 5.0 * fw + np.array([0, 0, 2.4])
            near = np.linalg.norm(Pm[:, :2] - p[:2], axis=1) < 25
            imgc = Vc.points(Pm[near], Cm[near], 0.06, eye_c, Rcc, rmax=4, fog=35.0)
            nk = np.linalg.norm(T[:, :2, 3] - p[:2], axis=1) < 25
            Vc.polyline(imgc, T[:, :3, 3], (30, 100, 220), 2, eye_c, Rcc)
            Vc.triads(imgc, T[nk], 0.6, 2, eye_c, Rcc)
            if act.any():
                a_, b_ = E[lc[act], :3, 3], E[lq[act], :3, 3]
                kk = (np.linalg.norm(a_[:, :2] - p[:2], axis=1) < 25)
                Vc.lines(imgc, a_[kk], b_[kk], (177, 0, 145), 3, eye_c, Rcc)
            imc.set_data(imgc)
            if gopro is not None:
                gim.set_data(gopro.at(float(t_kf[m])))
            nl = int(act.sum())
            txt = f"distance  {s_kf[m]:6.0f} m\nkeyframes {m + 1:6d}\nloops     {nl:6d}"
            if has_lid:
                e_s = np.linalg.norm(T[-1, :2, 3] - lid[m, :2, 3])
                e_o = np.linalg.norm(odom[m, :2, 3] - lid[m, :2, 3])
                txt += f"\n\nposition error vs lidar\n RTISSLAM-3D {e_s:5.2f} m\n odometry    {e_o:5.2f} m"
            if not fin and act.any() and (m - lq[act]).min() < 6:
                txt += "\n\nLOOP CLOSED"
            stats.set_text(txt)
            lh_s.set_data(s_kf[:m + 1], T[:, 2, 3]); lh_o.set_data(s_kf[:m + 1], odom[:m + 1, 2, 3])
            wr.append_data(cv.grab(fig))
        if gopro is not None:
            gopro.close()
    wr.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slam")
    ap.add_argument("h5")
    ap.add_argument("dets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--test", action="store_true", help="a few frames around the first loop closures and the finale")
    a = ap.parse_args()
    out = Path(a.out)
    parts = out.parent / (out.stem + "_parts"); parts.mkdir(parents=True, exist_ok=True)
    cache = parts / "cache.npz"
    nf = prepare(a.slam, a.h5, a.dets, cache)
    if a.test:
        rng = [(int(nf * 0.62), int(nf * 0.62) + 3)]
    else:
        b = np.linspace(0, nf, a.jobs + 1).astype(int)
        rng = list(zip(b[:-1], b[1:]))
    import multiprocessing as mp
    with ProcessPoolExecutor(len(rng), mp_context=mp.get_context("spawn")) as ex:
        files = list(ex.map(frames_job, [(str(cache), a.h5, int(r0), int(r1), parts / f"p{i:02d}.mp4", nf, a.title.replace("\\n", "\n")) for i, (r0, r1) in enumerate(rng)]))
    if a.test:
        print(files)
        return
    wr = cv.writer(parts / "title.mp4", fps=FPS); cv.hold(wr, cv.slide_title(a.title.replace("\\n", "\n")), 3, fps=FPS, fade_in=False); wr.close()
    wr = cv.writer(parts / "intro.mp4", fps=FPS)
    cv.hold(wr, cv.slide_text("Full 3D SLAM with an eRTIS imaging sonar", [
        "eRTIS: 3D imaging of the frontal hemisphere (2293 directions), 4.3 Hz; CFAR on the sphere, ground mirrors removed.",
        "3D acoustic odometry: every capture registered to a local 3D submap (sonar translation, gyro rotation).",
        "Every 1 m a keyframe: a 6-DoF pose in a GTSAM Pose3 graph (iSAM2) with IMU attitude factors.",
        "3D place recognition: height-banded polar descriptors, candidates near the current estimate, verified by 4-DoF submap registration.",
        "Lidar: 3D lidar SLAM, used only as the reference."]), 8, fps=FPS); wr.close()
    wr = cv.writer(parts / "end.mp4", fps=FPS); cv.hold(wr, cv.slide_end(), 3, fps=FPS, fade_out=False); wr.close()
    import imageio_ffmpeg
    lst = parts / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in [parts / "title.mp4", parts / "intro.mp4", *files, parts / "end.mp4"]))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    main()
