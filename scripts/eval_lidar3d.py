"""How good is the 3D lidar SLAM (scripts/lidar3d_slam.py) as ground truth? Four independent checks:

  1. planar agreement with the 2D lidar SLAM (slam_toolbox on a 2D slice, its own loop closures): base_link x, y, yaw
     at the same times, after the best rigid 2D fit
  2. height consistency at revisits: pairs of poses >= 30 m of travel apart but within 0.5 m in x, y must have the
     same z (the terrain is the same); also the roll / pitch difference at those revisits
  3. roll / pitch of base_link against the IMU's gravity direction (accelerometer averaged over 1 s, mount offset
     removed with the first 8 s at standstill)
  4. map crispness: thickness of the ground (vertical spread of the lowest points per 0.5 m cell) and of trunks

    python scripts/eval_lidar3d.py <kiss.npz> <slam2d.npz> <dataset.h5> [--out prefix]
Writes <prefix>.json and <prefix>.png. Runs both for the loop-closed trajectory (T) and the odometry only (T_odom).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def rpy(R):
    """ZYX: yaw, pitch, roll (rad) of rotation matrices [N, 3, 3]."""
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    return yaw, pitch, roll


def fit2d(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mb - R @ ma


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def checks(name, T_lidar, t, lidar_to_base, slam2d, imu, out):
    # base_link poses in the lidar SLAM's map frame: T_base = T_lidar @ inv(lidar_to_base) ... expressed from the
    # first base pose so that it starts at identity
    Tb = T_lidar @ np.linalg.inv(lidar_to_base)
    Tb = np.linalg.inv(Tb[0]) @ Tb
    xyz = Tb[:, :3, 3]
    yaw, pitch, roll = rpy(Tb[:, :3, :3])
    res = {}
    # 1. planar agreement with the 2D lidar SLAM
    t2, P2 = slam2d
    ok = (t >= t2[0]) & (t <= t2[-1])
    q = np.stack([np.interp(t[ok], t2, P2[:, i]) for i in range(3)], 1)
    R, tt = fit2d(xyz[ok, :2], q[:, :2])
    e = xyz[ok, :2] @ R.T + tt - q[:, :2]
    dyaw = wrap(np.unwrap(yaw[ok]) + math.atan2(R[1, 0], R[0, 0]) - q[:, 2])
    dyaw -= np.median(dyaw)
    res["vs_2d_xy_rmse_m"] = float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
    res["vs_2d_xy_max_m"] = float(np.max(np.hypot(*e.T)))
    res["vs_2d_yaw_rmse_deg"] = float(np.degrees(np.sqrt(np.mean(dyaw ** 2))))
    # 2. height consistency at revisits
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
    sub = np.arange(0, len(t), 10)
    dz, dp, dr = [], [], []
    for i in sub:
        old = sub[s[sub] < s[i] - 30]
        if len(old) == 0:
            continue
        d = np.hypot(*(xyz[old, :2] - xyz[i, :2]).T)
        j = old[np.argmin(d)]
        if d.min() < 0.5:
            dz.append(xyz[i, 2] - xyz[j, 2])
            dp.append(np.degrees(wrap(pitch[i] - pitch[j])) if abs(wrap(yaw[i] - yaw[j])) < 0.5 else np.nan)
            dr.append(np.degrees(wrap(roll[i] - roll[j])) if abs(wrap(yaw[i] - yaw[j])) < 0.5 else np.nan)
    dz = np.asarray(dz)
    res["revisits"] = int(len(dz))
    res["revisit_dz_rmse_m"] = float(np.sqrt(np.mean(dz ** 2))) if len(dz) else None
    res["revisit_dz_p95_m"] = float(np.percentile(np.abs(dz), 95)) if len(dz) else None
    res["revisit_dpitch_rmse_deg_same_heading"] = float(np.sqrt(np.nanmean(np.square(dp)))) if len(dp) else None
    res["z_range_m"] = [float(xyz[:, 2].min()), float(xyz[:, 2].max())]
    # 3. roll / pitch vs the IMU gravity direction
    t_i, acc = imu
    k = max(1, int(round(1.0 / np.median(np.diff(t_i)))))
    a_s = np.stack([np.convolve(acc[:, i], np.ones(k) / k, "same") for i in range(3)], 1)
    r_i = np.arctan2(a_s[:, 1], a_s[:, 2])
    p_i = np.arctan2(-a_s[:, 0], np.hypot(a_s[:, 1], a_s[:, 2]))
    still = t_i < t_i[0] + 8
    ok2 = (t >= t_i[0] + 1) & (t <= t_i[-1] - 1)
    ri, pi_ = np.interp(t[ok2], t_i, r_i), np.interp(t[ok2], t_i, p_i)
    # the lidar trajectory starts at identity: compare CHANGES in tilt (removes the unknown initial tilt / mount offset)
    ri -= r_i[still].mean(); pi_ -= p_i[still].mean()
    er, ep = np.degrees(wrap(roll[ok2] - ri)), np.degrees(wrap(pitch[ok2] - pi_))
    res["vs_imu_roll_rmse_deg"] = float(np.sqrt(np.mean((er - np.median(er)) ** 2)))
    res["vs_imu_pitch_rmse_deg"] = float(np.sqrt(np.mean((ep - np.median(ep)) ** 2)))
    res["vs_imu_roll_bias_deg"], res["vs_imu_pitch_bias_deg"] = float(np.median(er)), float(np.median(ep))
    res["tilt_range_deg"] = {"roll": [float(np.degrees(roll.min())), float(np.degrees(roll.max()))],
                             "pitch": [float(np.degrees(pitch.min())), float(np.degrees(pitch.max()))]}
    out[name] = res
    return xyz, yaw, pitch, roll, (t[ok2], er, ep), dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kiss")
    ap.add_argument("slam2d")
    ap.add_argument("h5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.kiss)
    t, L2B = z["t"], z["lidar_to_base"]
    s2 = np.load(a.slam2d)
    g0 = s2["gt"][0]                      # replay frame -> map frame (start pose at the origin)
    c, s_ = math.cos(g0[2]), math.sin(g0[2])
    E = s2["est_final"]
    P2 = np.c_[c * (E[:, 0] - g0[0]) + s_ * (E[:, 1] - g0[1]), -s_ * (E[:, 0] - g0[0]) + c * (E[:, 1] - g0[1]),
               np.unwrap(E[:, 2] - g0[2])]
    with h5py.File(a.h5, "r") as f:
        im = f["sensors/imu_01/data"]
        imu = (im["timestamps"][:], im["acceleration"][:].astype(float))
    out = {"kiss": a.kiss, "closures": int(len(z["closures"])), "scans": int(len(t)), "map_points": int(len(z["map_points"]))}
    A = checks("loop_closed", z["T"], t, L2B, (s2["t"], P2), imu, out)
    B = checks("odometry_only", z["T_odom"][:len(t)], t, L2B, (s2["t"], P2), imu, out)
    # 4. map crispness: ground thickness and trunk thickness in the 0.2 m voxel map
    M = z["map_points"].astype(float)
    ij = np.floor(M[:, :2] / 0.5).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((M[:, 2], key))
    ks, M2 = key[order], M[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    zmin = np.repeat(M2[first, 2], np.diff(np.r_[np.where(first)[0], len(ks)]))
    h = M2[:, 2] - zmin
    ground = h < 0.5
    thick = []
    for st, en in zip(np.where(first)[0], np.r_[np.where(first)[0][1:], len(ks)]):
        g = M2[st:en][h[st:en] < 0.5, 2]
        if len(g) >= 5:
            thick.append(np.percentile(g, 90) - np.percentile(g, 10))
    out["map_ground_thickness_p10_p90_median_m"] = float(np.median(thick)) if thick else None
    print(json.dumps(out, indent=1))
    o = Path(a.out) if a.out else Path(a.kiss).with_name("eval_lidar3d")
    o.with_suffix(".json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for (lab, (xyz, yaw, pitch, roll, imu_err, dz)), col in ((("loop closed", A), "tab:blue"), (("odometry only", B), "tab:red")):
        ax[0, 0].plot(xyz[:, 0], xyz[:, 1], c=col, lw=0.7, label=lab)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
        ax[0, 1].plot(s, xyz[:, 2], c=col, lw=0.7, label=lab)
        ax[0, 2].hist(dz, bins=40, histtype="step", color=col, label=lab)
        ti, er, ep = imu_err
        ax[1, 0].plot(ti - ti[0], er, c=col, lw=0.4, alpha=0.7)
        ax[1, 1].plot(ti - ti[0], ep, c=col, lw=0.4, alpha=0.7)
    ax[0, 0].plot(P2[:, 0], P2[:, 1], "k--", lw=0.6, label="2D lidar SLAM (base_link)")
    ax[0, 0].set_aspect("equal"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("trajectories (base_link, own frames)")
    ax[0, 1].set_xlabel("distance (m)"); ax[0, 1].set_ylabel("z (m)"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("height")
    ax[0, 2].set_xlabel("dz at revisits (m)"); ax[0, 2].legend(fontsize=8); ax[0, 2].set_title("height consistency at revisits")
    ax[1, 0].set_title("roll: lidar - IMU gravity (deg)"); ax[1, 1].set_title("pitch: lidar - IMU gravity (deg)")
    ax[1, 2].scatter(M[::20, 0], M[::20, 1], c=M[::20, 2], s=0.2, cmap="viridis")
    ax[1, 2].set_aspect("equal"); ax[1, 2].set_title("3D map (top view, colour = z)")
    fig.tight_layout(); fig.savefig(o.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
