"""3D acoustic odometry over a whole recording (no lidar): IMU attitude filter, wheel distance, and the eRTIS detection
sets registered capture by capture (rtis3d/acoustic_odom3d.py).

    od = run_odometry(h5, dets_npz, offset)      # dict: P_ao, P_prior [N, 4, 4] (base, gravity frame), caps [N], t [N]

Prior per step: rotation from the attitude filter (gyro, motion-corrected gravity, tau 10 s), translation (ds, 0, 0) in
the body with ds the wheel distance between the captures; sigmas: along x 0.01 m + 5 % of ds, lateral sig_lat,
vertical sig_vert, rotation sig_rot per axis. Defaults (0.05 m, 0.005 m, 0.05 deg): the sonar gives the horizontal
translation, the gyro the rotation, and the vertical motion stays that of a wheeled robot (along its body x axis, so
height = pitch x distance; forest seq1: height 0.12 m instead of 0.23 m with a free vertical, horizontal 0.28 / 0.32 m) (the sonar's own rotation drifts: 2.4 -> 17 deg per 20 m when the rotation prior is loosened, forest seq1).
Captures: every `every`-th from 9 s after the IMU start (gyro bias from the standstill) to the end of IMU / wheel data,
optionally limited to t_range (e.g. the span of a reference).
"""
from __future__ import annotations

import time

import h5py
import numpy as np

from .acoustic_odom3d import AcousticOdometry3D
from .attitude import Attitude


def load_attitude(h5):
    from .dataset import sensor_names
    S = sensor_names(h5)
    with h5py.File(h5, "r") as f:
        im = f[f"sensors/{S['imu']}/data"]; wh = f[f"sensors/{S['wheel']}/data"]
        return Attitude(im["timestamps"][:], im["angular_velocity"][:], im["acceleration"][:], wh["timestamps"][:],
                        wh["linear_velocity"][:, 0]), wh["timestamps"][:], wh["linear_velocity"][:, 0]


def ground_mount_pitch(D):
    """Pitch (deg) of the body x axis below the ground, as the median tilt of the sonar's fitted ground planes
    (sonar and IMU share the sensor plate): the wheels drive along the ground, not along a tilted body x."""
    n = D["plane_n"]
    ok = np.isfinite(n).all(1) & (np.linalg.norm(n, axis=1) > 0.5)
    return float(-np.median(np.degrees(np.arctan2(n[ok, 0], n[ok, 2]))))


def run_odometry(h5, dets, offset, sig_lat=0.05, sig_rot=0.05, submap=20, every=1, t_range=None, n=0, log=print, sig_vert=0.005,
                 mount_pitch=0.0):
    from .poses import _mount
    A, t_w, v_w = load_attitude(h5)
    R_imu = A.complementary(10.0, True)
    s_w = np.r_[0, np.cumsum(0.5 * (v_w[1:] + v_w[:-1]) * np.diff(t_w))]
    T_sb, _ = _mount(h5)
    D = np.load(dets) if isinstance(dets, (str, bytes)) or hasattr(dets, "__fspath__") else dets
    pts, wts, st, t_ac = D["pts"], D["w"], D["start"], D["t"] + offset
    if mount_pitch == "ground":
        mount_pitch = ground_mount_pitch(D)
        log and log(f"  mount pitch from the sonar ground plane: {mount_pitch:.2f} deg")
    u = np.array([np.cos(np.radians(mount_pitch)), 0.0, np.sin(np.radians(mount_pitch))])   # wheel travel in body
    lo, hi = max(A.t[0] + 9, t_w[0]), min(A.t[-1], t_w[-1])
    if t_range is not None:
        lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    caps = np.arange(0, len(t_ac), every)
    caps = caps[(t_ac[caps] > lo) & (t_ac[caps] < hi)]
    if n:
        caps = caps[:n]
    Ra = R_imu[np.clip(np.searchsorted(A.t, t_ac[caps]), 0, len(A.t) - 1)]
    sw = np.interp(t_ac[caps], t_w, s_w)
    ao = AcousticOdometry3D(submap_frames=submap, T_sb=T_sb)
    ao.pose = np.eye(4); ao.pose[:3, :3] = Ra[0]                     # start in the gravity frame (IMU attitude)
    P_ao, P_pr = [ao.pose.copy()], [ao.pose.copy()]
    n_used, t0 = 0, time.time()
    for j in range(1, len(caps)):
        k = caps[j]
        ds = sw[j] - sw[j - 1]
        T_prior = np.eye(4); T_prior[:3, :3] = Ra[j - 1].T @ Ra[j]; T_prior[:3, 3] = ds * u
        Sig = np.diag([(0.01 + 0.05 * abs(ds)) ** 2, sig_lat ** 2, (sig_vert if sig_vert is not None else sig_lat) ** 2]
                      + [np.radians(sig_rot) ** 2] * 3)
        _, _, info = ao.step(pts[st[k]:st[k + 1]], wts[st[k]:st[k + 1]], T_prior, Sig)
        n_used += info["used_sonar"]
        P_ao.append(ao.pose.copy()); P_pr.append(P_pr[-1] @ T_prior)
        if log and j % 1000 == 0:
            log(f"  {j}/{len(caps)}, sonar used {n_used / j:.0%}, {time.time() - t0:.0f} s")
    return {"P_ao": np.asarray(P_ao), "P_prior": np.asarray(P_pr), "caps": caps, "t": t_ac[caps],
            "sonar_used": n_used / max(1, len(caps) - 1), "seconds": time.time() - t0, "attitude": A}
