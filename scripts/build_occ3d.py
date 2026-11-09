"""3D log-odds occupancy (rtis3d/occupancy3d.py: beam-shaped evidence + free-space carving) from the detection sets on
SLAM (or lidar) poses, scored against the lidar map per height band, saved for the videos.

    python scripts/build_occ3d.py <detections.npz> <dataset.h5> --slam <slam3d npz>
        [--kiss <kiss.npz>] [--grid <grid3d npz>] [--res 0.1] [--thr 2.0] [--l-occ 4] [--out prefix]
--kiss: score against the 3D lidar map (map_points; observed = within 6 m and in front of a sensor pose); the SLAM poses
are moved onto the lidar map by a 6-DoF fit of the trajectories. --grid: take the lidar map and its observed mask from
a scripts/grid3d_eval.py output instead (as in the first forest evaluation). Neither: map only, in the SLAM frame.
Output npz (keys as scripts/grid3d_eval.py, for the video scripts): sonar_occ (centres of voxels with L >= thr),
logodds (their L), lidar, lidar_observed, sensor_xyz; plus <prefix>.json with the scores for several thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtis3d  # noqa: E402,F401
from rtis3d.occupancy3d import Occupancy3D  # noqa: E402
from rtis3d.poses import slam_sensor_poses  # noqa: E402
from grid3d_eval import BANDS, ground_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets")
    ap.add_argument("h5")
    ap.add_argument("--slam", required=True)
    ap.add_argument("--kiss", default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--drive", default=None, help="drive npz (only for SLAM npz files without capture indices)")
    ap.add_argument("--vol", required=True, help="volume directory (output of real_volumes.py)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--l-occ", type=float, default=4.0)
    ap.add_argument("--l-free", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from scipy.spatial import cKDTree
    D = np.load(a.dets)
    pts, w, st = D["pts"], D["w"], D["start"]
    U = np.load(Path(a.vol) / "directions.npy")
    if a.kiss:
        caps, Ts = slam_sensor_poses(a.slam, a.drive, a.kiss, a.h5)
    else:                                                    # no reference: the SLAM frame itself
        from rtis3d.poses import _mount
        sl = np.load(a.slam)
        caps, Ts = sl["caps"], sl["poses"] @ _mount(a.h5)[0]
    S = Ts[:, :3, 3]
    lo = np.r_[S[:, :2].min(0) - 7, S[:, 2].min() - 1.5]
    hi = np.r_[S[:, :2].max(0) + 7, S[:, 2].max() + 3.5]
    G = Occupancy3D(lo, hi, res=a.res, l_occ=a.l_occ, l_free=a.l_free)
    treeU = cKDTree(U)
    t0 = time.time()
    for j, k in enumerate(caps):
        p, ww = pts[st[k]:st[k + 1]], w[st[k]:st[k + 1]]
        di = treeU.query(p / np.linalg.norm(p, axis=1, keepdims=True))[1] if len(p) else None
        G.add(Ts[j], p, ww, U, dir_index=di)
        if (j + 1) % 1000 == 0:
            print(f"  {j + 1}/{len(caps)}, {time.time() - t0:.0f} s", flush=True)
    out = Path(a.out) if a.out else Path(a.dets).with_name("occ3d")
    if not (a.grid or a.kiss):
        m = G.L >= a.thr
        C = G.centres(m)
        np.savez_compressed(out.with_suffix(".npz"), sonar_occ=C.astype(np.float32), logodds=G.L[m].cpu().numpy(),
                            hits=np.full(len(C), 1000), sensor_xyz=S.astype(np.float32), res=a.res)
        out.with_suffix(".json").write_text(json.dumps({"captures": int(len(caps)), "res_m": a.res, "occupied_voxels": int(len(C)),
                                                        "seconds": time.time() - t0}, indent=1))
        print(out.with_suffix(".npz"), len(C), "voxels (no reference)")
        return
    if a.grid:
        g = np.load(a.grid)
        L_lid, obs = g["lidar"].astype(float), g["lidar_observed"]
    else:
        L_lid = np.load(a.kiss)["map_points"].astype(float)
        sens, fwd = Ts[::5, :3, 3], Ts[::5, :3, 0]
        obs = np.zeros(len(L_lid), bool)
        for i0, nb in enumerate(cKDTree(sens).query_ball_point(L_lid, 6.0)):
            if nb:
                obs[i0] = np.any(np.sum((L_lid[i0] - sens[nb]) * fwd[nb], axis=1) > 0)
    gz = ground_map(L_lid)
    hl = L_lid[:, 2] - gz(L_lid)
    tL = cKDTree(L_lid)
    res = {"captures": int(len(caps)), "res_m": a.res, "seconds": time.time() - t0, "voxels": int(np.prod(G.n))}
    Lg = G.L
    best = None
    for thr in (0.6, 1.0, 1.5, 2.0, 3.0):
        C = G.centres(Lg >= thr)
        if len(C) == 0:
            continue
        hs = C[:, 2] - gz(C)
        dS, _ = tL.query(C, distance_upper_bound=5)
        tS = cKDTree(C)
        r = {"occupied_voxels": int(len(C)), "precision": float(np.mean(dS <= a.tol))}
        dLo, _ = tS.query(L_lid[obs], distance_upper_bound=5)
        r["recall_observed"] = float(np.mean(dLo <= a.tol))
        for b, (blo, bhi) in BANDS.items():
            ms, ml = (hs >= blo) & (hs < bhi), obs & (hl >= blo) & (hl < bhi)
            dl, _ = tS.query(L_lid[ml], distance_upper_bound=5)
            r[b] = {"voxels": int(ms.sum()), "precision": float(np.mean(dS[ms] <= a.tol)) if ms.any() else None,
                    "recall": float(np.mean(dl <= a.tol)) if ml.any() else None}
        res[f"thr_{thr:g}"] = r
        print(f"thr {thr}: {len(C)} voxels, precision {r['precision']:.3f}, recall {r['recall_observed']:.3f}; "
              + "; ".join(f"{b} {v['precision'] if v['precision'] is None else round(v['precision'], 3)}/{v['recall'] if v['recall'] is None else round(v['recall'], 3)}"
                          for b, v in r.items() if isinstance(v, dict)), flush=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    m = Lg >= a.thr
    C = G.centres(m)
    np.savez_compressed(out.with_suffix(".npz"), sonar_occ=C.astype(np.float32), logodds=Lg[m].cpu().numpy(),
                        hits=np.full(len(C), 1000), lidar=L_lid.astype(np.float32), lidar_observed=obs, sensor_xyz=S.astype(np.float32),
                        res=a.res)
    print(out.with_suffix(".npz"), len(C), "voxels")


if __name__ == "__main__":
    main()
