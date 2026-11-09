"""Which sensors of a CHIRP recording the 3D pipeline uses (h5py only, nothing else needed).

    S = sensor_names(h5)    # dict: sonar, sonar_index, transmitter, imu, wheel, lidar (or None), video (or None)

sonar: the file's `reference_sensor` (else the first acoustic_sensor_*); transmitter: acoustic_transmitter_* with the
same index; imu: the imu_* with the most samples; wheel: the odometry_* whose description / source says wheel or
encoder (else the last odometry_*); lidar, video: the first lidar_* / video_*. Environment variables RTIS3D_SONAR,
RTIS3D_IMU, RTIS3D_WHEEL, RTIS3D_LIDAR override the choice.
"""
from __future__ import annotations

import os

import h5py


def sensor_names(h5):
    with h5py.File(h5, "r") as f:
        g = f["sensors"]
        names = list(g.keys())
        by = lambda p: [n for n in names if n.startswith(p)]  # noqa: E731
        sonar = str(f.attrs.get("reference_sensor", "")) or (by("acoustic_sensor")[0] if by("acoustic_sensor") else None)
        if sonar not in names:
            sonar = by("acoustic_sensor")[0]
        idx = int(g[sonar].attrs.get("sensor_index", int(sonar.rsplit("_", 1)[-1])))
        tx = [n for n in by("acoustic_transmitter") if int(g[n].attrs.get("sensor_index", -1)) == idx] or by("acoustic_transmitter")
        imus = by("imu")
        imu = max(imus, key=lambda n: g[n]["data/timestamps"].shape[0]) if imus else None
        odo = by("odometry")
        wheel = None
        for n in odo:
            txt = (str(g[n].attrs.get("description", "")) + " " + str(g[n].attrs.get("source", ""))).lower()
            if "wheel" in txt or "encoder" in txt:
                wheel = n
        wheel = wheel or (odo[-1] if odo else None)
        out = {"sonar": sonar, "sonar_index": idx, "transmitter": tx[0] if tx else None, "imu": imu, "wheel": wheel,
               "lidar": by("lidar")[0] if by("lidar") else None, "video": by("video")[0] if by("video") else None}
    for k, e in (("sonar", "RTIS3D_SONAR"), ("imu", "RTIS3D_IMU"), ("wheel", "RTIS3D_WHEEL"), ("lidar", "RTIS3D_LIDAR")):
        if os.environ.get(e):
            out[k] = os.environ[e]
    return out
