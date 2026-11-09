"""Trajectory metrics against a gravity-aligned reference (both [N, 4, 4], base poses, same times).

    ate(P, L)          -> dict: 4-DoF fit (x, y, z, yaw; gravity known): 3D / horizontal ATE, z RMSE, max horizontal
                          error, roll / pitch RMSE; 6-DoF fit: ATE, z RMSE and the tilt of the fit
    rpe(P, L, seg_m)   -> dict: relative pose error over segments of seg_m of travel (translation, z, rotation RMSE)
"""
from __future__ import annotations

import math

import numpy as np


def rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fit4(A, B):
    ma, mb = A.mean(0), B.mean(0)
    H = (A[:, :2] - ma[:2]).T @ (B[:, :2] - mb[:2])
    R = rotz(math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    return R, mb - R @ ma


def fit6(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mb - R @ ma


def log_rot(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def roll_pitch(T):
    return np.arctan2(T[:, 2, 1], T[:, 2, 2]), -np.arcsin(np.clip(T[:, 2, 0], -1, 1))


def ate(P, L):
    """-> (dict, e [N, 3]: per-pose position error after the 4-DoF fit)."""
    R, t = fit4(P[:, :3, 3], L[:, :3, 3])
    e = P[:, :3, 3] @ R.T + t - L[:, :3, 3]
    R6, t6 = fit6(P[:, :3, 3], L[:, :3, 3])
    e6 = P[:, :3, 3] @ R6.T + t6 - L[:, :3, 3]
    (r1, p1), (r2, p2) = roll_pitch(P), roll_pitch(L)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))  # noqa: E731
    return {"ate3d_m": rms(np.linalg.norm(e, axis=1)), "ate_xy_m": rms(np.linalg.norm(e[:, :2], axis=1)),
            "z_rmse_m": rms(e[:, 2]), "xy_max_m": float(np.linalg.norm(e[:, :2], axis=1).max()),
            "roll_rmse_deg": float(np.degrees(rms(r1 - r2))), "pitch_rmse_deg": float(np.degrees(rms(p1 - p2))),
            "fit6_ate3d_m": rms(np.linalg.norm(e6, axis=1)), "fit6_z_rmse_m": rms(e6[:, 2]),
            "fit6_tilt_deg": float(np.degrees(np.arccos(np.clip(R6[2, 2], -1, 1))))}, e


def rpe(P, L, seg_m, stride=5):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L[:, :2, 3], axis=0), axis=1))]
    et, ez, er = [], [], []
    for i0 in range(0, len(s), stride):
        i1 = np.searchsorted(s, s[i0] + seg_m)
        if i1 >= len(s):
            break
        E = np.linalg.inv(np.linalg.inv(L[i0]) @ L[i1]) @ (np.linalg.inv(P[i0]) @ P[i1])
        et.append(np.linalg.norm(E[:3, 3])); ez.append(abs(E[2, 3])); er.append(np.degrees(np.linalg.norm(log_rot(E[:3, :3]))))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")  # noqa: E731
    return {"trans_rmse_m": rms(et), "z_rmse_m": rms(ez), "rot_rmse_deg": rms(er), "segments": len(et)}
