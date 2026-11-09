"""Full 3D SLAM v1: the RTISSLAM 2D result (pose graph + links) lifted to 6-DoF with the IMU attitude filter and the
wheel odometry in a GTSAM Pose3 graph (rtis3d/pose3_slam.py); scored against the 3D lidar SLAM (KISS-SLAM).

    python scripts/run_pose3.py <dataset.h5> <drive.npz> <rtisslam run dir> <kiss.npz> [--out prefix]

Variants: odometry only (wheel + gyro + attitude, no links), lifted 2D (x, y, yaw from RTISSLAM, z integrated, no 3D
optimization), Pose3 graph (links + attitude). Scores after a 4-DoF fit (x, y, z, yaw; gravity is known): 3D ATE, z
RMSE, horizontal ATE; roll / pitch RMSE against the lidar in the gravity frame.
Writes <out>.json, <out>.npz (the 6-DoF poses of the Pose3 graph, base_link, gravity frame), <out>.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.attitude import Attitude, rot_rp  # noqa: E402
from rtis3d.pose3_slam import build_and_solve, decoupled, rotz  # noqa: E402


def fit4(A, B):
    """x, y, z, yaw fit of points A onto B ([N, 3]) -> (Rz, t)."""
    ma, mb = A.mean(0), B.mean(0)
    H = (A[:, :2] - ma[:2]).T @ (B[:, :2] - mb[:2])
    yaw = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = rotz(yaw)
    return R, mb - R @ ma


def fit6(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mb - R @ ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("drive")
    ap.add_argument("run")
    ap.add_argument("kiss")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    f = h5py.File(a.h5, "r")
    im = f["sensors/imu_01/data"]
    w = f["sensors/odometry_01/data"]
    A = Attitude(im["timestamps"][:], im["angular_velocity"][:], im["acceleration"][:], w["timestamps"][:], w["linear_velocity"][:, 0])
    R_imu = A.complementary(a.tau, motion_correction=True)
    d = np.load(a.drive)
    t, odo = d["t"], d["odom"]
    tr = np.load(Path(a.run) / "trace.npz")
    P2d, links = tr["final"], tr["links"]
    idx = np.clip(np.searchsorted(A.t, t), 0, len(A.t) - 1)
    R_att = R_imu[idx]
    dxy = np.diff(odo[:, :2], axis=0)
    hd = 0.5 * (odo[1:, 2] + odo[:-1, 2])
    ds = np.r_[0, dxy[:, 0] * np.cos(hd) + dxy[:, 1] * np.sin(hd)]
    # lidar truth at the node times, base_link, gravity frame
    z = np.load(a.kiss)
    Tb = z["T"] @ np.linalg.inv(z["lidar_to_base"])
    Tb = np.linalg.inv(Tb[0]) @ Tb
    G = np.eye(4); G[:3, :3] = rot_rp(A.r0, A.p0)
    Tg = G @ Tb
    il = np.clip(np.searchsorted(z["t"], t), 0, len(z["t"]) - 1)
    Tl = Tg[il]
    out = {}

    def score(name, P):
        R, tt = fit4(P[:, :3, 3], Tl[:, :3, 3])
        Q = P[:, :3, 3] @ R.T + tt
        e = Q - Tl[:, :3, 3]
        R6, t6 = fit6(P[:, :3, 3], Tl[:, :3, 3])
        e6 = P[:, :3, 3] @ R6.T + t6 - Tl[:, :3, 3]
        rp = lambda T: (np.arctan2(T[:, 2, 1], T[:, 2, 2]), -np.arcsin(np.clip(T[:, 2, 0], -1, 1)))  # noqa: E731
        r1, p1 = rp(P); r2, p2 = rp(Tl)
        out[name] = {"ate3d_m": float(np.sqrt(np.mean(np.sum(e ** 2, 1)))), "ate_xy_m": float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, 1)))),
                     "z_rmse_m": float(np.sqrt(np.mean(e[:, 2] ** 2))), "z_max_m": float(np.abs(e[:, 2]).max()),
                     "roll_rmse_deg": float(np.degrees(np.sqrt(np.mean((r1 - r2) ** 2)))),
                     "pitch_rmse_deg": float(np.degrees(np.sqrt(np.mean((p1 - p2) ** 2)))),
                     "fit6_ate3d_m": float(np.sqrt(np.mean(np.sum(e6 ** 2, 1)))), "fit6_z_rmse_m": float(np.sqrt(np.mean(e6[:, 2] ** 2))),
                     "fit6_tilt_deg": float(np.degrees(np.arccos(np.clip(R6[2, 2], -1, 1))))}
        print(f"{name:24s} " + ", ".join(f"{k} {v:.3f}" for k, v in out[name].items()), flush=True)
        return Q
    # lifted 2D: x, y, yaw from RTISSLAM, roll / pitch from the filter, z integrated
    rp_only = np.array([rotz(-math.atan2(R[1, 0], R[0, 0])) @ R for R in R_att])
    zi = np.cumsum(np.r_[0, [ds[i] * -rp_only[i][2, 0] for i in range(1, len(ds))]])
    P_l = np.tile(np.eye(4), (len(t), 1, 1))
    for i in range(len(t)):
        P_l[i, :3, :3] = rotz(P2d[i, 2]) @ rp_only[i]; P_l[i, :3, 3] = (P2d[i, 0], P2d[i, 1], zi[i])
    score("lifted_2d", P_l)
    P_d = decoupled(R_att, ds, P2d, links)
    score("decoupled", P_d)
    P_g, info = build_and_solve(R_att, ds, P2d, links, use_links=True, init=P_d)
    Q = score("pose3_graph", P_g)
    best = "pose3_graph" if out["pose3_graph"]["fit6_ate3d_m"] <= out["decoupled"]["fit6_ate3d_m"] else "decoupled"
    out["best"] = best
    if best == "decoupled":
        P_g = P_d
        Q = P_d[:, :3, 3] @ fit4(P_d[:, :3, 3], Tl[:, :3, 3])[0].T + fit4(P_d[:, :3, 3], Tl[:, :3, 3])[1]
    out["graph"] = info
    o = Path(a.out) if a.out else Path(a.run).with_name(Path(a.run).name + "_pose3")
    o.with_suffix(".json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(o.with_suffix(".npz"), poses=P_g, t=t, lidar=Tl, aligned_xyz=Q)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Tl[:, :2, 3], axis=0), axis=1))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(s, Tl[:, 2, 3], "k", lw=1, label="3D lidar SLAM")
    ax[0].plot(s, Q[:, 2], c="tab:blue", lw=0.8, label="Pose3 graph (sonar + wheel + IMU)")
    R, tt = fit4(P_l[:, :3, 3], Tl[:, :3, 3]); ax[0].plot(s, (P_l[:, :3, 3] @ R.T + tt)[:, 2], c="tab:red", lw=0.6, alpha=0.7, label="lifted 2D (z integrated)")
    R6, t6 = fit6(P_g[:, :3, 3], Tl[:, :3, 3]); ax[0].plot(s, (P_g[:, :3, 3] @ R6.T + t6)[:, 2], c="tab:green", lw=0.8,
                                                         label="Pose3, 6-DoF fit (tilt allowed)")
    ax[0].set_xlabel("distance (m)"); ax[0].set_ylabel("z (m)"); ax[0].legend(fontsize=8); ax[0].set_title("height")
    rp = lambda T: (np.degrees(np.arctan2(T[:, 2, 1], T[:, 2, 2])), np.degrees(-np.arcsin(np.clip(T[:, 2, 0], -1, 1))))  # noqa: E731
    (r1, p1), (r2, p2) = rp(P_g), rp(Tl)
    ax[1].plot(s, r2, "k", lw=0.8, label="lidar"); ax[1].plot(s, r1, c="tab:blue", lw=0.6, label="Pose3"); ax[1].set_title("roll (deg)"); ax[1].legend(fontsize=8)
    ax[2].plot(s, p2, "k", lw=0.8, label="lidar"); ax[2].plot(s, p1, c="tab:blue", lw=0.6, label="Pose3"); ax[2].set_title("pitch (deg)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(o.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
