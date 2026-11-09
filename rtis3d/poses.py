"""Sensor poses for mapping: the eRTIS pose of each SLAM node (or each lidar-timed capture), in the lidar map frame.

    caps, Ts = slam_sensor_poses(slam_npz, drive_npz, kiss_npz, h5)     # SLAM poses moved onto the lidar map (6-DoF fit)
    caps, Ts = lidar_sensor_poses(kiss_npz, h5, t_captures)               # the lidar's own poses (upper bound)
"""
from __future__ import annotations

import h5py
import numpy as np


def _mount(h5):
    """-> (T_sb: eRTIS -> base, T_el: lidar -> eRTIS (identity without a lidar)), from the CHIRP calibrations."""
    from .dataset import sensor_names
    S = sensor_names(h5)
    with h5py.File(h5, "r") as f:
        T_bo = f[f"sensors/{S['wheel']}/calibration/to_origin"][:]
        T_eo = f[f"sensors/{S['sonar']}/calibration/to_origin"][:]
        T_lo = f[f"sensors/{S['lidar']}/calibration/to_origin"][:] if S["lidar"] else T_eo
    return np.linalg.inv(T_bo) @ T_eo, np.linalg.inv(T_eo) @ T_lo


def slam_sensor_poses(slam_npz, drive_npz, kiss_npz, h5):
    """-> (capture indices [N], eRTIS poses [N, 4, 4]) of the SLAM nodes, in the lidar map frame (6-DoF fit of the SLAM
    node positions onto the lidar's base positions at the same times: for scoring and side-by-side display).
    slam_npz: poses [N, 4, 4] (base) and t [N] (lidar clock); capture indices from its `caps`, else from drive_npz `k`."""
    sl, kz = np.load(slam_npz), np.load(kiss_npz)
    caps = sl["caps"] if "caps" in sl.files else np.load(drive_npz)["k"]     # slam3d output carries its capture indices
    T_be, _ = _mount(h5)
    Tl = kz["T"] @ np.linalg.inv(kz["lidar_to_base"])
    il = np.clip(np.searchsorted(kz["t"], sl["t"]), 0, len(kz["t"]) - 1)
    A, B = sl["poses"][:, :3, 3], Tl[il][:, :3, 3]
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    F = np.eye(4); F[:3, :3] = R; F[:3, 3] = mb - R @ ma
    return np.asarray(caps), F @ sl["poses"] @ T_be


def lidar_sensor_poses(kiss_npz, h5, t_captures):
    kz = np.load(kiss_npz)
    _, T_el = _mount(h5)
    t = kz["t"]
    i = np.clip(np.searchsorted(t, t_captures), 1, len(t) - 1)
    i = np.where(np.abs(t[i - 1] - t_captures) < np.abs(t[i] - t_captures), i - 1, i)
    return kz["T"][i] @ np.linalg.inv(T_el)


def lidar_reference(kiss_npz, t, A):
    """Ground truth for scoring: the lidar SLAM's base poses at times t [N] (nearest scan), relative to the first one,
    in a gravity frame: started with the IMU standstill attitude (A: rtis3d.attitude.Attitude, as the odometry), then
    gravity-aligned with the whole drive's accelerometer (rtis3d.attitude.gravity_align; KISS-SLAM knows no gravity).
    -> [N, 4, 4]."""
    from .attitude import gravity_align, rot_rp
    z = np.load(kiss_npz)
    Tb = z["T"] @ np.linalg.inv(z["lidar_to_base"])
    il = np.clip(np.searchsorted(z["t"], t), 0, len(z["t"]) - 1)
    G = np.eye(4); G[:3, :3] = rot_rp(A.r0, A.p0)
    Tw = G @ np.linalg.inv(Tb[il[0]]) @ Tb
    Rg = np.eye(4); Rg[:3, :3] = gravity_align(A, z["t"], Tw[:, :3, :3])
    return Rg @ Tw[il]
