"""RTISSLAM-3D, the whole pipeline for one CHIRP recording, in one command:

    python scripts/pipeline3d.py <data>/<site>/raw/<file>.h5 --name <name>
        [--stages all|inventory,preproc,...] [--force stage,...] [--no-lidar] [--no-video] [--offset s] [--jobs 6]

Stages (each skipped when its output exists; --force re-runs it; outputs under <data>/<site>/):
  inventory    sensors, spans, rates (2D project's chirp_inventory.py)                 inventory/<name>.json
  preproc      direct path, static background and interference projector fit (~25 s)   <name>_preproc.npz
  energyscapes 2D DMAS5-CF images of every capture, for the clock offset (~4 min)       cache/<name>_dmas5cf.npz
  clock        eRTIS clock offset: sonar yaw rate vs odometry yaw rate (~1 min)         cache/<name>_clock_sync.json
  lidar3d      3D lidar SLAM reference, KISS-SLAM (RTIS3D_LIDAR_PYTHON; ~8 min; optional)   lidar3d/<name>_kiss.npz
  volumes      3D DMAS5-CF volumes over the frontal hemisphere (~8 min, GPU)            threed/<name>/vol_dmas5cf/
  detections   CFAR on the sphere, ground plane, mirrors removed (~2 min)               threed/<name>/detections.npz
  odometry     3D acoustic odometry (sonar translation, gyro rotation; ~4 min)          threed/<name>/ao3d.npz
  slam         RTISSLAM-3D: keyframes, Pose3 graph, 3D place recognition (~4 min)       threed/<name>/slam3d.npz
  map          3D log-odds occupancy on the SLAM poses (~30 s)                          threed/<name>/occ3d.npz
  videos       SLAM build video; with lidar and GoPro also drive-through and 3D map    threed/<name>/video/
  report       engineering report (LaTeX -> PDF) with all numbers and figures          report/<name>/main.pdf

The coarse clock offset is the difference of the eRTIS and odometry start times (the forest eRTIS clock was 4.7 days
off); clock_sync.py then searches +-30 s around it. Check its two halves agree (printed). Parameters are the ones
tuned on the forest recording and FROZEN: do not re-tune them on a recording you evaluate.
Without a lidar (or --no-lidar) everything runs, but there are no scores, and only the SLAM build video is made.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rtis3d  # noqa: E402
from rtis3d.dataset import sensor_names  # noqa: E402

STAGES = ["inventory", "preproc", "energyscapes", "clock", "lidar3d", "volumes", "detections", "odometry", "slam", "map",
          "videos", "report"]
PY = sys.executable
PY_LIDAR = os.environ.get("RTIS3D_LIDAR_PYTHON", sys.executable)        # a Python with kiss-icp / kiss-slam
REAL = ROOT                                   # the front-end stages live in this repository too


def run(cmd, log, cwd=ROOT):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = Path(log).read_text().splitlines()[-15:]
        raise SystemExit(f"stage failed ({r.returncode}), log {log}:\n" + "\n".join(tail))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--name", required=True)
    ap.add_argument("--site-dir", default=None, help="default: the folder above raw/")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--force", default="")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--offset", type=float, default=None, help="eRTIS clock offset (s), skips the clock stage")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--recording", default=None, help="label for videos / report, e.g. 'Forest, sequence 1'")
    a = ap.parse_args()
    h5 = Path(a.h5).resolve()
    site = Path(a.site_dir) if a.site_dir else (h5.parent.parent if h5.parent.name == "raw" else h5.parent)
    n = a.name
    work = site / "threed" / n
    logs = work / "logs"
    for d in (site / "inventory", site / "cache", site / "lidar3d", work, logs, work / "video"):
        d.mkdir(parents=True, exist_ok=True)
    S = sensor_names(h5)
    use_lidar = bool(S["lidar"]) and not a.no_lidar
    stages = STAGES if a.stages == "all" else a.stages.split(",")
    force = set(a.force.split(",")) if a.force else set()
    state_f = work / "pipeline.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    state.update({"h5": str(h5), "name": n, "site": str(site), "sensors": S, "lidar": use_lidar})
    P = {"inventory": site / "inventory" / f"{n}.json", "preproc": site / f"{n}_preproc.npz",
         "energyscapes": site / "cache" / f"{n}_dmas5cf.npz", "clock": site / "cache" / f"{n}_clock_sync.json",
         "lidar3d": site / "lidar3d" / f"{n}_kiss.npz", "volumes": work / "vol_dmas5cf" / "meta.json",
         "detections": work / "detections.npz", "odometry": work / "ao3d.npz", "slam": work / "slam3d.npz",
         "map": work / "occ3d.npz", "videos": work / "video" / "slam3d_build.mp4",
         "report": ROOT / "report" / n / "main.pdf"}
    label = a.recording or n.replace("_", " ")
    print(f"RTISSLAM-3D pipeline: {h5.name} -> {work}\n  sensors {S}\n  lidar reference: {use_lidar}", flush=True)

    def offset():
        if a.offset is not None:
            return a.offset
        return json.loads(P["clock"].read_text())["offset_s"]

    for st in STAGES:
        if st not in stages:
            continue
        if st == "lidar3d" and not use_lidar:
            continue
        if st == "videos" and a.no_video:
            continue
        if st == "clock" and a.offset is not None:
            continue
        if P[st].exists() and st not in force:
            print(f"[{st}] done ({P[st]})", flush=True)
            continue
        print(f"[{st}]", flush=True)
        log = logs / f"{st}.log"
        if st == "inventory":
            dt = run([PY, REAL / "scripts" / "chirp_inventory.py", h5, "--json", P[st]], log, cwd=REAL)
        elif st == "preproc":
            code = ("import sys; sys.path.insert(0, %r); import rtisreal; from chirppy import CHIRPDataset; "
                    "from rtisreal.rtis_frontend import RealPreprocessor; ds = CHIRPDataset(%r); ds.__enter__(); "
                    "RealPreprocessor.fit(ds.get_acoustic_sensor(%d), ds.get_acoustic_transmitter(%d)).save(%r)"
                    % (str(REAL), str(h5), S["sonar_index"], S["sonar_index"], str(P[st])))
            dt = run([PY, "-c", code], log)
        elif st == "energyscapes":
            dt = run([PY, REAL / "scripts" / "real_energyscapes.py", h5, P["preproc"], P[st], "--jobs", a.jobs], log, cwd=REAL)
        elif st == "clock":
            inv = json.loads(P["inventory"].read_text())
            t0s = {s["name"]: s.get("t_start") for s in inv["sensors"]}
            coarse = round(t0s[S["wheel"]] - t0s[S["sonar"]], 1)
            dt = run([PY, REAL / "scripts" / "clock_sync.py", h5, P["energyscapes"], "--coarse", coarse, "--search", 30,
                      "--out", str(P[st])[:-5]], log, cwd=REAL)
            c = json.loads(P[st].read_text())
            print(f"  offset {c['offset_s']:.3f} s (coarse {coarse}), corr {c['corr']:.2f}, halves "
                  f"{c['offset_first_half']:.3f} / {c['offset_second_half']:.3f}", flush=True)
            if abs(c["offset_first_half"] - c["offset_second_half"]) > 0.15:
                print("  WARNING: the halves disagree by more than 0.15 s: check cache/<name>_clock_sync.png", flush=True)
        elif st == "lidar3d":
            dt = run([PY_LIDAR, ROOT / "scripts" / "lidar3d_slam.py", h5, P[st], "--lidar", S["lidar"]], log)
        elif st == "volumes":
            dt = run([PY, ROOT / "scripts" / "real_volumes.py", h5, P["preproc"], work / "vol_dmas5cf", "--jobs", a.jobs], log)
        elif st == "detections":
            dt = run([PY, ROOT / "scripts" / "extract_detections.py", work / "vol_dmas5cf", "--out", P[st]], log)
        elif st == "odometry":
            cmd = [PY, ROOT / "scripts" / "run_ao3d.py", h5, P["detections"], "--offset", offset(), "--out", str(P[st])[:-4]]
            if use_lidar:
                cmd += ["--kiss", P["lidar3d"]]                     # same span as the reference, so every pose is scored
            dt = run(cmd, log)
        elif st == "slam":
            cmd = [PY, ROOT / "scripts" / "run_slam3d.py", h5, P["detections"], P["odometry"], "--offset", offset(),
                   "--out", str(P[st])[:-4]]
            if use_lidar:
                cmd += ["--kiss", P["lidar3d"]]
            dt = run(cmd, log)
        elif st == "map":
            cmd = [PY, ROOT / "scripts" / "build_occ3d.py", P["detections"], h5, "--slam", P["slam"], "--out", str(P[st])[:-4]]
            if use_lidar:
                cmd += ["--kiss", P["lidar3d"]]
            dt = run(cmd, log)
        elif st == "videos":
            vd = work / "video"
            dt = run([PY, ROOT / "video" / "make_slam3d_video.py", P["slam"], h5, P["detections"], "--out", vd / "slam3d_build.mp4",
                      "--jobs", a.jobs, "--title", f"RTISSLAM-3D: {label}\\nFull 6-DoF Sonar SLAM, Step by Step"], logs / "video_slam3d.log")
            if use_lidar and S["video"]:
                dt += run([PY, ROOT / "video" / "make_drive_through_video.py", P["map"], P["slam"], h5, "--out",
                           vd / "drive_through.mp4", "--jobs", a.jobs,
                           "--title", f"RTISSLAM-3D: {label}\\nDriving Through the Sonar and Lidar 3D Maps"], logs / "video_drive.log")
            if use_lidar:
                dt += run([PY, ROOT / "video" / "make_3d_map_video.py", P["map"], "--out", vd / "map3d.mp4",
                           "--title", f"RTISSLAM-3D: {label}\\n3D Map From Full 3D Sonar SLAM"], logs / "video_map.log")
            import imageio_ffmpeg
            res = ROOT / "results" / n
            res.mkdir(parents=True, exist_ok=True)
            for v in sorted(vd.glob("*.mp4")):                    # 1080p H.264 copies in results/<name>/
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(v), "-c:v", "libx264",
                                "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(res / f"{n}_{v.stem}.mp4")], check=True)
        elif st == "report":
            dt = run([PY, ROOT / "scripts" / "make_report3d.py", "--name", n, "--work", work, "--recording", label], log)
        state.setdefault("seconds", {})[st] = round(dt, 1)
        state_f.write_text(json.dumps(state, indent=1, default=str))
        print(f"  {dt:.0f} s", flush=True)
    state_f.write_text(json.dumps(state, indent=1, default=str))
    if (work / "slam3d.json").exists():
        r = json.loads((work / "slam3d.json").read_text())
        if "slam3d" in r:
            s = r["slam3d"]
            print(f"RTISSLAM-3D: horizontal ATE {s['ate_xy_m']:.2f} m, height {s['z_rmse_m']:.2f} m, roll/pitch "
                  f"{s['roll_rmse_deg']:.1f}/{s['pitch_rmse_deg']:.1f} deg, {r['links']} loops", flush=True)
        else:
            print(f"RTISSLAM-3D: {r['keyframes']} keyframes, {r['links']} loops (no reference)", flush=True)


if __name__ == "__main__":
    main()
