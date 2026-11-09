"""3D acoustic odometry over a recording (rtis3d/odometry3d.py): no lidar needed. The input of scripts/run_slam3d.py.

    python scripts/run_ao3d.py <dataset.h5> <detections.npz> --offset 409980.68 --out <prefix>
        [--sig-lat 0.05] [--sig-rot 0.05] [--kiss <kiss.npz>]
--kiss limits the captures to the lidar's time span (so that every pose can be scored); without it, the whole drive.
Writes <prefix>.npz: P_ao, P_prior (base poses, gravity frame), caps, t; <prefix>.json: settings and sonar use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.odometry3d import run_odometry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("dets")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--sig-lat", type=float, default=0.05)
    ap.add_argument("--sig-rot", type=float, default=0.05)
    ap.add_argument("--sig-vert", type=float, default=0.005, help="vertical prior sigma (m per step)")
    ap.add_argument("--submap", type=int, default=20)
    ap.add_argument("--mount-pitch", default="0", help="deg, or 'ground': median tilt of the sonar ground planes")
    ap.add_argument("--kiss", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = None
    if a.kiss:
        t = np.load(a.kiss)["t"]; tr = (t[0], t[-1])
    od = run_odometry(a.h5, a.dets, a.offset, a.sig_lat, a.sig_rot, a.submap, t_range=tr, log=lambda s: print(s, flush=True),
                      sig_vert=a.sig_vert, mount_pitch=a.mount_pitch if a.mount_pitch == "ground" else float(a.mount_pitch))
    out = Path(a.out)
    np.savez_compressed(out.with_suffix(".npz"), P_ao=od["P_ao"], P_prior=od["P_prior"], caps=od["caps"], t=od["t"])
    res = {"captures": int(len(od["caps"])), "sonar_used": od["sonar_used"], "seconds": od["seconds"], "sig_lat": a.sig_lat,
           "sig_rot_deg": a.sig_rot, "sig_vert": a.sig_vert, "submap": a.submap, "mount_pitch": a.mount_pitch, "limited_to_lidar_span": bool(a.kiss)}
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
