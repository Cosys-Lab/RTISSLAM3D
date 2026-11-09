# Vendored into RTISSLAM-3D from RTISSLAM Real (rtisreal), commit 5c0dc2c.
"""CHIRP datasets (Cosys-Lab wjansen/chirp, HDF5, read with chirppy) -> RTISSLAM drives.

A real recording becomes a drive in the format of scripts/gen_drive.py (drives/<name>_<tag>.npz: odom, t, k,
<imaging>_db, ranges, azimuths, ...), so scripts/run_slam.py, fit_evidence.py and the plots run on it unchanged.
The eRTIS is the pulse clock: one drive pulse per eRTIS capture (gated by travel, as the simulated drives), with the
wheel odometry of base_link interpolated at the capture time. "gt" is NaN here; the lidar SLAM reference is added
later (it is evaluation only, as in simulation).

Frames: CHIRP's origin frame and every sensor's to_origin are X forward, Y left, Z up. The odometry streams carry
base_link's to_origin, so the eRTIS mount in base_link is T_base^-1 T_ertis. RTISSLAM assumes the sensor faces
forward, level, at (mount_x, mount_y) in base_link (rtisslam/grid.py, acoustic_odometry.py): the drive stores the
measured mount, and a yawed mount is reported (the energyscape would have to be rotated).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def find_ertis(ds) -> int:
    """Index of the 32-channel acoustic sensor (eRTIS); HiRIS has 1024."""
    found = [s.attrs.get("sensor_index", None) for s in ds.get_sensors_by_type("acoustic_sensor")
             if int(s.attrs.get("num_channels", 0)) == 32]
    if not found:
        raise ValueError("no 32-channel acoustic sensor (eRTIS) in this dataset")
    return int(found[0])


def find_odometry(ds, source: str = "wheel") -> int:
    for s in ds.get_sensors_by_type("odometry"):
        if str(s.attrs.get("source", "")) == source:
            return int(s.attrs.get("sensor_index", 0))
    raise ValueError(f"no odometry with source '{source}'")


def yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


@dataclass
class Mount:
    x: float
    y: float
    z: float
    yaw: float      # rad
    pitch: float
    roll: float

    @classmethod
    def from_transform(cls, T):
        R = T[:3, :3]
        return cls(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), math.atan2(R[1, 0], R[0, 0]),
                   -math.asin(max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[2, 1], R[2, 2]))

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.pitch, self.roll])


def ertis_mount(ertis, odom) -> Mount:
    """eRTIS array pose in base_link from the two to_origin transforms."""
    Te, Tb = ertis.get_transform_to_origin(), odom.get_transform_to_origin()
    if Te is None or Tb is None:
        raise ValueError("missing to_origin calibration (eRTIS or odometry)")
    return Mount.from_transform(np.linalg.inv(np.asarray(Tb)) @ np.asarray(Te))


def odometry_track(odom) -> tuple[np.ndarray, np.ndarray]:
    """(t [K], pose [K, 3] x, y, unwrapped yaw) of base_link in the odom frame; ROS stamps when present (the time
    the robot made the estimate), else host receipt."""
    d = odom.data_group
    t = np.array(d["ros_timestamps"] if "ros_timestamps" in d else d["timestamps"], float)
    p = np.array(d["position"], float)
    th = np.unwrap(np.array(d["yaw"], float))
    o = np.argsort(t, kind="stable")
    t, p, th = t[o], p[o], th[o]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], np.c_[p[keep, :2], th[keep]]


def interp_pose(t_track, pose, t) -> np.ndarray:
    """Linear interpolation of x, y, unwrapped yaw at times t (inside the track)."""
    return np.stack([np.interp(t, t_track, pose[:, i]) for i in range(3)], -1)


def gate(odom: np.ndarray, gate_m: float = 0.10, gate_rad: float = math.radians(6)) -> np.ndarray:
    """Indices kept by the travel gate of the simulated drives (>= gate_m moved or >= gate_rad turned since the
    last kept pulse)."""
    keep, last = [], None
    for i, o in enumerate(odom):
        if last is not None and np.hypot(*(o[:2] - last[:2])) < gate_m and abs(math.remainder(o[2] - last[2], 2 * math.pi)) < gate_rad:
            continue
        keep.append(i)
        last = o
    return np.asarray(keep, int)


def gyro_wheel_track(ds, wheel_source: str = "wheel", imu_index: int = 1, still_s: float = 8.0):
    """Proprioceptive odometry: translation from the wheel odometry, heading from the IMU gyro (yaw rate in the robot
    frame, bias = its mean over the first `still_s` seconds, when the robot stands still). Forest recording: heading
    error vs lidar median 11.5 deg, max 25.5 deg over 23 min, against 93 / 151 deg for the robot's EKF odometry.
    -> (t [K], pose [K, 3]) of base_link, like odometry_track (unwrapped yaw)."""
    od = ds.get_odometry(find_odometry(ds, wheel_source))
    t_w, p_w = odometry_track(od)
    imu = ds.get_imu(imu_index)
    t_i = np.asarray(imu.data_group["timestamps"], float)
    R = np.asarray(imu.get_transform_to_origin())[:3, :3] if imu.get_transform_to_origin() is not None else np.eye(3)
    wz = (np.asarray(imu.data_group["angular_velocity"], float) @ R.T)[:, 2]
    bias = float(wz[t_i < t_i[0] + still_s].mean())
    yaw_i = np.r_[0, np.cumsum(0.5 * (wz[1:] + wz[:-1] - 2 * bias) * np.diff(t_i))]
    yaw = np.interp(t_w, t_i, yaw_i) + p_w[0, 2]
    # wheel displacement, signed along the wheel heading (forward / backward), re-oriented by the gyro heading
    d = np.diff(p_w[:, :2], axis=0)
    hw = 0.5 * (p_w[1:, 2] + p_w[:-1, 2])
    ds_ = d[:, 0] * np.cos(hw) + d[:, 1] * np.sin(hw)
    hg = 0.5 * (yaw[1:] + yaw[:-1])
    xy = np.vstack([p_w[0, :2], p_w[0, :2] + np.cumsum(np.c_[ds_ * np.cos(hg), ds_ * np.sin(hg)], axis=0)])
    return t_w, np.c_[xy, yaw], bias
