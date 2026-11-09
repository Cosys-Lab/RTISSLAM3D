"""3D acoustic odometry (rtis3d/acoustic_odom3d.py) against the 3D lidar: per-step relative motion and dead reckoning,
with and without the sonar (the prior alone = wheel distance + IMU rotation).

    python scripts/eval_ao3d.py <dataset.h5> <detections.npz> <kiss.npz> --offset 409980.68
        [--every 1] [--n 0] [--out prefix]
Steps between consecutive used captures (--every). Priors: rotation from the attitude filter (gyro), translation
(ds, 0, 0) in the body with ds from the wheel speed integrated between the captures; prior sigmas: rotation 0.3 deg per
axis, translation along x 0.01 + 5 % of ds, lateral / vertical 0.02 m.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.odometry3d import run_odometry  # noqa: E402
from rtis3d.poses import lidar_reference  # noqa: E402


def log_rot(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("dets")
    ap.add_argument("kiss")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--submap", type=int, default=20)
    ap.add_argument("--sig-lat", type=float, default=0.05, help="prior sigma, lateral and vertical motion per step (m)")
    ap.add_argument("--sig-rot", type=float, default=0.05, help="prior sigma, rotation per axis per step (deg)")
    ap.add_argument("--sig-vert", type=float, default=0.005, help="prior sigma, vertical motion per step (m)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tk = np.load(a.kiss)["t"]
    od = run_odometry(a.h5, a.dets, a.offset, a.sig_lat, a.sig_rot, a.submap, a.every, (tk[0], tk[-1]), a.n,
                      log=lambda s: print(s, flush=True), sig_vert=a.sig_vert)
    caps, P_ao, P_pr = od["caps"], od["P_ao"], od["P_prior"]
    Lg = lidar_reference(a.kiss, od["t"], od["attitude"])
    res = {"steps": len(caps) - 1, "sonar_used": od["sonar_used"]}
    for name, P in (("ao", P_ao), ("prior", P_pr)):
        E = np.linalg.inv(np.linalg.inv(Lg[:-1]) @ Lg[1:]) @ (np.linalg.inv(P[:-1]) @ P[1:])
        e = np.c_[E[:, :3, 3], np.degrees([log_rot(R) for R in E[:, :3, :3]])]
        res[f"{name}_step_rmse"] = {k: float(np.sqrt(np.mean(e[:, i] ** 2))) for i, k in enumerate(("x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"))}
    # dead reckoning vs lidar (gravity frame, 6-DoF and 4-DoF fits)
    for name, P in (("ao", P_ao), ("prior", P_pr)):
        A_, B_ = P[:, :3, 3], Lg[:, :3, 3]
        ma, mb = A_.mean(0), B_.mean(0)
        U, _, Vt = np.linalg.svd((A_ - ma).T @ (B_ - mb))
        R6 = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        e6 = (A_ - ma) @ R6.T + mb - B_
        Hh = (A_[:, :2] - ma[:2]).T @ (B_[:, :2] - mb[:2]); yaw = math.atan2(Hh[0, 1] - Hh[1, 0], Hh[0, 0] + Hh[1, 1])
        R4 = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        e4 = (A_ - ma) @ R4.T + mb - B_
        res[f"{name}_dead_reckoning"] = {"fit4_xy_m": float(np.sqrt(np.mean(np.sum(e4[:, :2] ** 2, 1)))), "fit4_z_m": float(np.sqrt(np.mean(e4[:, 2] ** 2))),
                                         "fit6_z_m": float(np.sqrt(np.mean(e6[:, 2] ** 2))), "fit6_tilt_deg": float(np.degrees(np.arccos(np.clip(R6[2, 2], -1, 1))))}
    # relative pose error over segments of 5 m and 20 m of travel (lidar steps are noisy, segments are not)
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Lg[:, :2, 3], axis=0), axis=1))]
    for name, P in (("ao", np.asarray(P_ao)), ("prior", np.asarray(P_pr))):
        for seg in (5.0, 20.0):
            et, ez, er = [], [], []
            for i0 in range(0, len(sl), 5):
                i1 = np.searchsorted(sl, sl[i0] + seg)
                if i1 >= len(sl):
                    break
                Lr = np.linalg.inv(Lg[i0]) @ Lg[i1]; Pr = np.linalg.inv(P[i0]) @ P[i1]
                E = np.linalg.inv(Lr) @ Pr
                et.append(np.linalg.norm(E[:3, 3])); ez.append(abs(E[2, 3])); er.append(np.degrees(np.linalg.norm(log_rot(E[:3, :3]))))
            res[f"{name}_rpe_{seg:g}m"] = {"trans_rmse_m": float(np.sqrt(np.mean(np.square(et)))), "z_rmse_m": float(np.sqrt(np.mean(np.square(ez)))),
                                          "rot_rmse_deg": float(np.sqrt(np.mean(np.square(er))))}
    print(json.dumps(res, indent=1))
    out = Path(a.out) if a.out else Path(a.dets).with_name("ao3d_eval")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), P_ao=np.asarray(P_ao), P_prior=np.asarray(P_pr), lidar=Lg, caps=caps)


if __name__ == "__main__":
    main()
