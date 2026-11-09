# Vendored into RTISSLAM-3D from RTISSLAM Real (scripts), commit 5c0dc2c.
"""Inventory of a CHIRP dataset (Cosys-Lab wjansen/chirp, HDF5): the first thing to run on a new recording.

    python scripts/chirp_inventory.py <dataset.h5> [--json out.json]

Per sensor: type, sync mode, captures, time span and rate, the mount (to_origin: position and yaw/pitch/roll in the
origin frame, X forward, Y left, Z up), and the type-specific facts that matter for RTISSLAM: acoustic sample rate,
samples per capture (-> maximum range), array aperture and disabled microphones; the emitted call (duration, band);
odometry source and path length; lidar points per scan. Reads only metadata and a few small arrays.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

C = 343.0


def ypr(T) -> tuple:
    R = np.asarray(T)[:3, :3]
    return (math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(-math.asin(max(-1, min(1, R[2, 0])))),
            math.degrees(math.atan2(R[2, 1], R[2, 2])))


def attrs(o) -> dict:
    out = {}
    for k, v in o.attrs.items():
        v = v.decode() if isinstance(v, bytes) else v
        out[k] = v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, np.generic) else v)
    return out


def call_band(sig: np.ndarray, fs: float) -> tuple:
    """-20 dB band of the emitted call (Hz)."""
    S = np.abs(np.fft.rfft(sig, 8 * len(sig))) ** 2
    f = np.fft.rfftfreq(8 * len(sig), 1 / fs)
    on = np.where(S > S.max() * 0.01)[0]
    return float(f[on[0]]), float(f[on[-1]])


def sensor_summary(name: str, g: h5py.Group) -> dict:
    a = attrs(g)
    s = {"name": name, "type": a.get("sensor_type", "?"), "sync_mode": a.get("sync_mode", "?")}
    d = g.get("data")
    cal = g.get("calibration")
    if cal is not None and "to_origin" in cal:
        T = np.array(cal["to_origin"])
        s["to_origin_xyz_m"] = np.round(T[:3, 3], 4).tolist()
        s["to_origin_yaw_pitch_roll_deg"] = np.round(ypr(T), 2).tolist()
    if d is not None and "timestamps" in d:
        t = np.array(d["timestamps"])
        s["n"] = int(t.size)
        if t.size:
            s["t_start"], s["t_end"], s["duration_s"] = float(t[0]), float(t[-1]), float(t[-1] - t[0])
            if t.size > 1:
                dt = np.diff(t)
                s["rate_hz_median"] = float(1 / np.median(dt)) if np.median(dt) > 0 else None
                s["dt_max_s"] = float(dt.max())
                s["non_monotonic"] = int((dt <= 0).sum())
    t = s["type"]
    if t == "acoustic_sensor" and d is not None and "signals" in d:
        fs = float(a.get("sample_rate_hz", 450000))
        sh = d["signals"].shape
        s.update(sample_rate_hz=fs, signals_shape=list(sh), dtype=str(d["signals"].dtype),
                 max_range_m=round(sh[1] / fs * C / 2, 2), compression=d["signals"].compression)
        if cal is not None and "receiver_relative_translations" in cal:
            p = np.array(cal["receiver_relative_translations"])
            s["mics"] = int(len(p))
            s["mics_disabled"] = np.where(np.isnan(p).any(1))[0].tolist()
            s["aperture_xyz_mm"] = np.round((np.nanmax(p, 0) - np.nanmin(p, 0)) * 1e3, 1).tolist()
        x = np.array(d["signals"][0])
        s["capture0_rms_per_mic"] = np.round(np.sqrt(np.mean(x.astype(float) ** 2, 0)), 5).tolist()
    elif t == "acoustic_transmitter" and d is not None and "signals" in d:
        fs = float(a.get("sample_rate_hz", 450000))
        sig = np.array(d["signals"])
        lo, hi = call_band(sig, fs)
        s.update(sample_rate_hz=fs, call_samples=int(sig.size), call_ms=round(sig.size / fs * 1e3, 3),
                 call_band_khz=[round(lo / 1e3, 1), round(hi / 1e3, 1)], call_attrs=attrs(d["signals"]))
    elif t == "odometry" and d is not None and "position" in d:
        p = np.array(d["position"])
        s.update(source=a.get("source"), ros_topic=a.get("ros_topic"),
                 path_length_m=round(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()), 2),
                 extent_xy_m=np.round(np.ptp(p[:, :2], 0), 2).tolist(),
                 yaw_turned_deg=round(float(np.degrees(np.abs(np.diff(np.unwrap(np.array(d["yaw"])))).sum())), 1),
                 has_ros_timestamps="ros_timestamps" in d)
    elif t in ("lidar", "radar") and d is not None and "point_counts" in d:
        c = np.array(d["point_counts"])
        s.update(max_points=a.get("max_points"), points_median=int(np.median(c)) if c.size else 0,
                 points_shape=list(d["points"].shape))
    if d is not None and "sync_to_reference_index" in d:
        s["sync_to_reference"] = True
    s["attrs"] = {k: v for k, v in a.items() if k not in ("sensor_type", "sync_mode")}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    p = Path(a.h5)
    with h5py.File(p, "r") as f:
        top = attrs(f)
        sensors = [sensor_summary(n, g) for n, g in f["sensors"].items()] if "sensors" in f else []
        other = [k for k in f.keys() if k != "sensors"]
    rep = {"file": str(p), "size_gb": round(p.stat().st_size / 1e9, 3), "attrs": top, "other_groups": other,
           "sensors": sensors}
    print(f"{p.name}: {rep['size_gb']} GB, CHIRP {top.get('format_version')}, {top.get('n_reference_captures')} "
          f"reference captures of {top.get('reference_sensor')}; {top.get('description', '')}")
    for s in sensors:
        line = f"  {s['name']:<24} {s['type']:<20} {s['sync_mode']:<17} n={s.get('n', '-'):<7}"
        if s.get("rate_hz_median"):
            line += f" {s['rate_hz_median']:7.2f} Hz {s.get('duration_s', 0):8.1f} s"
        if "to_origin_xyz_m" in s:
            line += f"  at {s['to_origin_xyz_m']} ypr {s['to_origin_yaw_pitch_roll_deg']}"
        print(line)
        for k in ("signals_shape", "max_range_m", "mics_disabled", "aperture_xyz_mm", "call_ms", "call_band_khz",
                  "source", "path_length_m", "extent_xy_m", "yaw_turned_deg", "points_median", "non_monotonic", "dt_max_s"):
            if k in s and s[k] not in (None, [], 0):
                print(f"      {k}: {s[k]}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
