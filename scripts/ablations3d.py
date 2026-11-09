"""Ablations and threshold sensitivity of RTISSLAM-3D on one pipeline run (needs the lidar reference):
variants of scripts/run_slam3d.py, run in parallel, summarized in <work>/ablation/summary.json (read by the report).

    python scripts/ablations3d.py --work <data>/<site>/threed/<name> [--jobs 14]
        [--summary-only]
Variants: the full system; no loop closures; wheel + gyro odometry instead of the 3D acoustic odometry (with and
without loops); the sonar also estimating the vertical motion; a descriptor with four height bands; the acceptance threshold on the registration inliers
(0.75 ... 0.95, default 0.85), the position gate (3, 5, 8, 12 m) and the registration RMSE limit (0.12 / 0.15 m).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "ref": ("RTISSLAM-3D (full)", []),
    "noloops": ("no loop closures", ["--no-loops"]),
    "prior_loops": ("wheel + gyro odometry, with loops", ["--odom-key", "P_prior"]),
    "prior_noloops": ("wheel + gyro odometry, no loops", ["--odom-key", "P_prior", "--no-loops"]),
    "heightbands": ("descriptor with four height bands", ["--height-bands"]),
    "inl075": ("inliers >= 0.75", ["--min-inliers", "0.75"]),
    "inl080": ("inliers >= 0.80", ["--min-inliers", "0.8"]),
    "inl090": ("inliers >= 0.90", ["--min-inliers", "0.9"]),
    "inl095": ("inliers >= 0.95", ["--min-inliers", "0.95"]),
    "gate3": ("gate 3 m", ["--gate", "3"]),
    "gate8": ("gate 8 m", ["--gate", "8"]),
    "gate12": ("gate 12 m", ["--gate", "12"]),
    "rmse015": ("registration RMSE <= 0.15 m", ["--max-rmse", "0.15"]),
    "freevert": ("sonar also estimates the vertical motion", [], "ao3d_freevert.npz"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    st = json.loads((work / "pipeline.json").read_text())
    site = Path(st["site"])
    out = work / "ablation"
    out.mkdir(exist_ok=True)
    offset = json.loads((site / "cache" / f"{st['name']}_clock_sync.json").read_text())["offset_s"]
    kiss = site / "lidar3d" / f"{st['name']}_kiss.npz"
    if not a.summary_only:
        env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        procs = []
        for k, (_, extra, *ao) in VARIANTS.items():
            if (out / f"{k}.json").exists():
                continue
            ao_f = out / ao[0] if ao else work / "ao3d.npz"
            if ao and not ao_f.exists():                          # the free-vertical odometry first
                subprocess.run([sys.executable, str(ROOT / "scripts" / "run_ao3d.py"), st["h5"], str(work / "detections.npz"), "--offset",
                                str(offset), "--kiss", str(kiss), "--sig-vert", "0.05", "--out", str(ao_f)[:-4]], check=True,
                               stdout=subprocess.DEVNULL)
            cmd = [sys.executable, str(ROOT / "scripts" / "run_slam3d.py"), st["h5"], str(work / "detections.npz"), str(ao_f),
                   "--offset", str(offset), "--kiss", str(kiss), "--out", str(out / k), *extra]
            while len([p for p in procs if p.poll() is None]) >= a.jobs:
                procs[0].wait() if procs[0].poll() is None else procs.pop(0)
            procs.append(subprocess.Popen(cmd, stdout=open(out / f"{k}.log", "w"), stderr=subprocess.STDOUT, env=env))
        for p in procs:
            p.wait()
    summ = {}
    for k, (label, *_) in VARIANTS.items():
        f = out / f"{k}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        s, le = r["slam3d"], r.get("link_err", {})
        summ[k] = {"label": label, "links": r["links"], "ate_xy_m": s["ate_xy_m"], "z_rmse_m": s["z_rmse_m"], "xy_max_m": s["xy_max_m"],
                   "roll_rmse_deg": s["roll_rmse_deg"], "pitch_rmse_deg": s["pitch_rmse_deg"], "tilt_deg": s["fit6_tilt_deg"],
                   "over_0.5m": le.get("over_0.5m", 0), "link_median_m": le.get("xy_median_m")}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for k, v in summ.items():
        print(f"{v['label']:38s} loops {v['links']:4d}  xy {v['ate_xy_m']:.3f}  z {v['z_rmse_m']:.3f}  max {v['xy_max_m']:.2f}  >0.5 m {v['over_0.5m']}")


if __name__ == "__main__":
    main()
