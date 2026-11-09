"""Full 3D sonar SLAM (rtis3d/slam3d.py) on a recording, scored against the 3D lidar SLAM (KISS-SLAM).

    python scripts/run_slam3d.py <dataset.h5> <detections.npz> <ao3d npz> --offset 409980.68
        [--kiss <kiss.npz>] [--kf 1.0] [--submap 4] [--gate 5] [--min-inliers 0.85] [--out prefix]
Input odometry: scripts/run_ao3d.py output (3D acoustic odometry, sonar translation and gyro rotation; no lidar).
--kiss: the 3D lidar SLAM, gravity-aligned (rtis3d.poses.lidar_reference), for scoring only; without it only the poses.
Scores (keyframes; 4-DoF fit x, y, z, yaw since gravity is known, and 6-DoF fit): 3D / horizontal ATE, z RMSE, roll /
pitch RMSE, and each loop link's error against the lidar relative pose; the odometry alone for comparison.
Writes <out>.json, <out>.png and <out>.npz: poses (base, gravity frame, one per capture), caps, t (lidar clock),
kf_poses, kf_caps, links; <out>_hist.npy: the keyframe estimate after every keyframe [n, n, 3, 4] -- the npz is a valid --slam input of scripts/build_occ3d.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtis3d  # noqa: E402,F401
from rtis3d.place3d import BANDS, DescParams, build_keyframes, level_frame  # noqa: E402
from rtis3d.poses import _mount  # noqa: E402
from rtis3d.slam3d import Slam3D  # noqa: E402
from run_pose3 import fit4, fit6  # noqa: E402


def score(P, L):
    """P, L [N, 4, 4] (estimate, lidar truth; both gravity frame) -> dict."""
    R, t = fit4(P[:, :3, 3], L[:, :3, 3])
    e = P[:, :3, 3] @ R.T + t - L[:, :3, 3]
    R6, t6 = fit6(P[:, :3, 3], L[:, :3, 3])
    e6 = P[:, :3, 3] @ R6.T + t6 - L[:, :3, 3]
    rp = lambda T: (np.arctan2(T[:, 2, 1], T[:, 2, 2]), -np.arcsin(np.clip(T[:, 2, 0], -1, 1)))  # noqa: E731
    (r1, p1), (r2, p2) = rp(P), rp(L)
    return {"ate3d_m": float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), "ate_xy_m": float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1)))),
            "z_rmse_m": float(np.sqrt(np.mean(e[:, 2] ** 2))), "xy_max_m": float(np.linalg.norm(e[:, :2], axis=1).max()),
            "roll_rmse_deg": float(np.degrees(np.sqrt(np.mean((r1 - r2) ** 2)))), "pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean((p1 - p2) ** 2)))),
            "fit6_ate3d_m": float(np.sqrt(np.mean(np.sum(e6 ** 2, 1)))), "fit6_z_rmse_m": float(np.sqrt(np.mean(e6[:, 2] ** 2))),
            "fit6_tilt_deg": float(np.degrees(np.arccos(np.clip(R6[2, 2], -1, 1))))}, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("dets")
    ap.add_argument("ao", help="scripts/run_ao3d.py output (or eval_ao3d.py output)")
    ap.add_argument("--kiss", default=None, help="3D lidar SLAM (KISS-SLAM npz), for scoring only")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--kf", type=float, default=1.0)
    ap.add_argument("--submap", type=float, default=4.0)
    ap.add_argument("--gate", type=float, default=5.0)
    ap.add_argument("--sep", type=float, default=20.0)
    ap.add_argument("--min-inliers", type=float, default=0.85)
    ap.add_argument("--max-rmse", type=float, default=0.12)
    ap.add_argument("--sig-att", type=float, default=1.5, help="IMU attitude factor sigma (deg)")
    ap.add_argument("--sig-tz", type=float, nargs=2, default=None, help="body vertical odometry sigma: a + b * ds (m)")
    ap.add_argument("--no-loops", action="store_true")
    ap.add_argument("--odom-key", default="P_ao", help="P_ao (3D acoustic odometry) or P_prior (wheel + gyro only): ablation")
    ap.add_argument("--height-bands", action="store_true", help="descriptor with four height bands: ablation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = np.load(a.dets)
    pts, w, st, t_ac = D["pts"], D["w"], D["start"], D["t"] + a.offset
    Z = np.load(a.ao)
    P, caps = Z[a.odom_key], Z["caps"]
    Lg = None
    if a.kiss:
        from rtis3d.odometry3d import load_attitude
        from rtis3d.poses import lidar_reference
        Lg = lidar_reference(a.kiss, t_ac[caps], load_attitude(a.h5)[0])
    T_sb, _ = _mount(a.h5)
    t0 = time.time()
    sel, s, KF = build_keyframes(P, caps, pts, w, st, T_sb, a.kf, a.submap,
                                   DescParams(bands=BANDS) if a.height_bands else DescParams())
    print(f"{len(KF)} keyframes, {time.time() - t0:.0f} s", flush=True)
    S = Slam3D(gate_m=a.gate if not a.no_loops else -1.0, sep_m=a.sep, min_inliers=a.min_inliers, max_rmse=a.max_rmse,
               sig_att_deg=a.sig_att, sig_trans_z=tuple(a.sig_tz) if a.sig_tz else None)
    for n, j in enumerate(sel):
        if n == 0:
            S.add(KF[n], np.eye(4), P[j], 0.0)
        else:
            S.add(KF[n], np.linalg.inv(P[sel[n - 1]]) @ P[j], P[j], s[j] - s[sel[n - 1]])
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(KF)}, links {len(S.links)}, registrations {S.tried}, {time.time() - t0:.0f} s", flush=True)
    K = S.poses()
    # per-capture poses: each capture hangs off the keyframe before it (odometry in between)
    owner = np.clip(np.searchsorted(sel, np.arange(len(P)), side="right") - 1, 0, len(sel) - 1)
    Pc = np.einsum("nij,njk->nik", K[owner] @ np.linalg.inv(P[sel[owner]]), P)
    res = {"keyframes": len(KF), "kf_m": a.kf, "seconds": time.time() - t0, "links": len(S.links), "registrations": S.tried,
           "settings": {"gate_m": a.gate, "sep_m": a.sep, "min_inliers": a.min_inliers, "max_rmse": a.max_rmse, "sig_att_deg": a.sig_att,
                        "loops": not a.no_loops, "odom_key": a.odom_key, "height_bands": a.height_bands}}
    out = Path(a.out) if a.out else Path(a.ao).with_name("slam3d")
    H = np.full((len(S.hist), len(S.hist), 3, 4), np.nan, np.float32)
    for n, h in enumerate(S.hist):
        H[n, :len(h)] = h
    np.save(out.with_name(out.name + "_hist.npy"), H)                   # the estimate after every keyframe (videos)
    extra = {}
    if Lg is None:                                                       # no reference: poses only
        out.with_suffix(".json").write_text(json.dumps(res, indent=1))
        np.savez_compressed(out.with_suffix(".npz"), poses=Pc, caps=caps, t=t_ac[caps], kf_poses=K, kf_caps=caps[sel], kf_index=sel,
                            links=np.asarray(S.links).reshape(-1, 8), odom=P)
        print(json.dumps(res, indent=1))
        return
    G = Lg[sel]
    res["slam3d"], e_s = score(K, G)
    res["odometry"], e_o = score(P[sel], G)
    # link check against the lidar
    le = []
    for q, c, inl, rm, x, y, z, yaw in S.links:
        Lt = np.linalg.inv(level_frame(G[c])) @ level_frame(G[q])
        le.append((math.hypot(x - Lt[0, 3], y - Lt[1, 3]), abs(z - Lt[2, 3]),
                   abs(math.degrees((yaw - math.atan2(Lt[1, 0], Lt[0, 0]) + math.pi) % (2 * math.pi) - math.pi))))
    le = np.asarray(le).reshape(-1, 3)
    if len(le):
        res["link_err"] = {"xy_median_m": float(np.median(le[:, 0])), "xy_p95_m": float(np.percentile(le[:, 0], 95)),
                           "over_0.5m": int((le[:, 0] > 0.5).sum()), "over_1m": int((le[:, 0] > 1.0).sum()),
                           "z_median_m": float(np.median(le[:, 1])), "yaw_median_deg": float(np.median(le[:, 2]))}
    for k in ("slam3d", "odometry"):
        print(f"{k:9s} " + ", ".join(f"{kk} {v:.3f}" for kk, v in res[k].items()), flush=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("slam3d", "odometry")}, indent=1))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), poses=Pc, caps=caps, t=t_ac[caps], kf_poses=K, kf_caps=caps[sel], kf_index=sel,
                        links=np.asarray(S.links).reshape(-1, 8), link_err=le, lidar=Lg, odom=P)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for P_, c, lab in ((P[sel], "tab:orange", "3D acoustic odometry"), (K, "tab:blue", "RTISSLAM-3D")):
        R, t = fit4(P_[:, :3, 3], G[:, :3, 3]); Q = P_[:, :3, 3] @ R.T + t
        ax[0].plot(Q[:, 0], Q[:, 1], c=c, lw=0.9, label=lab)
        ax[1].plot(s[sel], Q[:, 2], c=c, lw=0.9, label=lab)
        if P_ is K:
            for q, c_ in np.asarray(S.links)[:, :2].astype(int) if len(S.links) else []:
                ax[0].plot(Q[[q, c_], 0], Q[[q, c_], 1], c="tab:green", lw=0.4, alpha=0.5)
    ax[0].plot(G[:, 0, 3], G[:, 1, 3], "k", lw=0.8, label="3D lidar SLAM (reference)")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].set_title(f"trajectory, {len(S.links)} loop links (green)")
    ax[1].plot(s[sel], G[:, 2, 3], "k", lw=0.8, label="lidar"); ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("z (m)")
    ax[1].legend(fontsize=8); ax[1].set_title("height (4-DoF fit)")
    ax[2].plot(s[sel], np.linalg.norm(e_o[:, :2], axis=1), c="tab:orange", lw=0.8, label="odometry")
    ax[2].plot(s[sel], np.linalg.norm(e_s[:, :2], axis=1), c="tab:blue", lw=0.8, label="RTISSLAM-3D")
    ax[2].plot(s[sel], np.abs(e_s[:, 2]), c="tab:blue", lw=0.6, ls="--", label="RTISSLAM-3D, |z|")
    ax[2].set_xlabel("distance (m)"); ax[2].set_ylabel("error (m)"); ax[2].legend(fontsize=8); ax[2].set_title("error vs lidar")
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
