"""3D sonar occupancy on the 6-DoF lidar poses (the upper bound of the 3D sonar map: no pose error), against the 3D
lidar map (KISS-SLAM), per height band.

  sensor poses   eRTIS pose at each capture = lidar SLAM pose at (eRTIS time + clock offset) x static calibration
  detections     CFAR on the sphere (heat kernel, --sigma deg, el scale 1.7, --gain) on the DMAS5-CF volume, cells within
                 --peak-db of the capture's strongest CFAR output, r >= --rmin
  ground gate    optional: detections more than --below m under the local ground of the lidar map are dropped (the
                 ground-multipath mirrors); --gate none | lidar
  occupancy      voxel hit counts (--voxel m); occupied = voxels with >= --min-hits detections and hit ratio >= --min-ratio
                 of the times the voxel was in view (simple "seen / hit" model; view = inside the sonar's hemisphere and
                 range of a capture)
  reference      lidar map voxels (map_points, 0.2 m); height above the local lidar ground (lowest point per 0.5 m cell
                 and its neighbours)
Scores per band (ground < 0.25 m, low 0.25..2 m, high > 2 m above the local ground) and overall: precision (sonar
occupied voxel within --tol of a lidar point), recall (lidar voxels within the sonar's observed volume within --tol of a
sonar voxel), voxel counts.

    python scripts/grid3d_eval.py <vol dir> <dataset.h5> <kiss.npz> --offset 409980.68 [--every 2] [--out prefix]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.laplacian import HeatKernel, sphere_cfar  # noqa: E402

BANDS = {"ground": (-1e9, 0.25), "low_0.25_2m": (0.25, 2.0), "high_2m+": (2.0, 1e9)}


def ground_map(M, cell=0.5):
    ij = np.floor(M[:, :2] / cell).astype(int)
    o = ij.min(0) - 1
    ij -= o
    G = np.full(tuple(ij.max(0) + 2), np.inf)
    np.minimum.at(G, (ij[:, 0], ij[:, 1]), M[:, 2])
    Gn = G.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            Gn[1:-1, 1:-1] = np.minimum(Gn[1:-1, 1:-1], G[1 + dx:G.shape[0] - 1 + dx, 1 + dy:G.shape[1] - 1 + dy])

    def gz(P):
        i = np.floor(P[:, 0] / cell).astype(int) - o[0]
        j = np.floor(P[:, 1] / cell).astype(int) - o[1]
        ok = (i >= 0) & (i < Gn.shape[0]) & (j >= 0) & (j < Gn.shape[1])
        z = np.full(len(P), np.nan)
        z[ok] = Gn[i[ok], j[ok]]
        z[~np.isfinite(z)] = np.nan
        return z
    return gz


def track_ground(Ts, V, caps, pos, H, a):
    """Sonar-only ground for the gate: the robot drives on the ground, so the ground under each sensor position is its
    z minus the sensor height h. h: given, or from the sonar's own near-field ground echoes (median of z_sensor - z of the
    strongest CFAR detection below -30 deg elevation within 1.5 m, in the gravity frame of the pose). The ground at any
    (x, y): inverse-distance weighted mean over the 8 nearest track points (local terrain)."""
    import torch
    from scipy.spatial import cKDTree
    from rtis3d.laplacian import sphere_cfar
    S = np.array([T[:3, 3] for T in Ts])
    if a.sensor_height is not None:
        h = a.sensor_height
    else:
        U = pos[-1] / np.linalg.norm(pos[-1], axis=1, keepdims=True)
        el = np.arcsin(np.clip(U[:, 2], -1, 1))
        r = np.linalg.norm(pos[:, 0], axis=-1)
        hs = []
        for j in range(0, len(caps), 10):
            q = sphere_cfar(torch.as_tensor(10 ** (np.asarray(V[caps[j]], np.float32) / 10), device="cuda"), H, gain=a.gain).cpu().numpy()
            q[:, el > np.radians(-30)] = 0
            q[r > 1.5] = 0
            if q.max() <= 0:
                continue
            ri, di = np.unravel_index(np.argmax(q), q.shape)
            Xw = pos[ri, di] @ Ts[j][:3, :3].T + Ts[j][:3, 3]
            hs.append(Ts[j][2, 3] - Xw[2])
        h = float(np.median(hs))
    tree = cKDTree(S[:, :2])

    def g(P):
        d, i = tree.query(P[:, :2], k=8)
        w = 1.0 / np.maximum(d, 0.2) ** 2
        return np.sum(w * (S[i, 2] - h), 1) / w.sum(1)
    return g, h


def sensor_poses(kiss, t_query, T_el):
    """eRTIS poses [N, 4, 4] in the lidar map frame at t_query (nearest lidar pose; <= 0.05 s at 10 Hz)."""
    z = np.load(kiss)
    t, T = z["t"], z["T"]
    i = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
    i = np.where(np.abs(t[i - 1] - t_query) < np.abs(t[i] - t_query), i - 1, i)
    T_le = np.linalg.inv(T_el)                                   # eRTIS -> lidar
    return T[i] @ T_le, np.abs(t[i] - t_query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("h5")
    ap.add_argument("kiss")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=12.0)
    ap.add_argument("--gain", type=float, default=5.0)
    ap.add_argument("--peak-db", type=float, default=25.0)
    ap.add_argument("--rmin", type=float, default=0.5)
    ap.add_argument("--rmax", type=float, default=6.0)
    ap.add_argument("--gate", default="lidar", choices=["none", "lidar", "track"])
    ap.add_argument("--sensor-height", type=float, default=None, help="track gate: sensor height above the ground (m); "
                    "default: from the sonar's own near-field ground echoes")
    ap.add_argument("--below", type=float, default=0.2)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--min-ratio", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--slam", default=None, help="SLAM poses instead of lidar poses: run_pose3 npz (base_link, gravity "
                    "frame, one per drive node); needs --drive. The sonar map is built on them; for scoring it is moved "
                    "onto the lidar map with the 6-DoF fit of the SLAM trajectory to the lidar trajectory")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import h5py
    import torch
    from scipy.spatial import cKDTree
    vol = Path(a.vol)
    V = np.load(vol / "volumes.npy", mmap_mode="r")
    U, rr, t_ac = np.load(vol / "directions.npy"), np.load(vol / "ranges.npy"), np.load(vol / "t.npy") + a.offset
    with h5py.File(a.h5, "r") as f:
        T_el = np.linalg.inv(f["sensors/acoustic_sensor_01/calibration/to_origin"][:]) @ f["sensors/lidar_00/calibration/to_origin"][:]
    kz = np.load(a.kiss)
    M = kz["map_points"].astype(float)
    if a.slam:
        # SLAM poses (base_link) at the drive's nodes; eRTIS = base @ (base -> eRTIS); everything moved into the lidar
        # map frame by the 6-DoF fit of the SLAM node positions onto the lidar node positions
        sl = np.load(a.slam)
        dr = np.load(a.drive)
        caps = dr["k"][::max(1, a.every // 2)]
        node = np.searchsorted(dr["k"], caps)
        with h5py.File(a.h5, "r") as f:
            T_be = np.linalg.inv([g for k_, g in f["sensors"].items() if k_.startswith("odometry")][0]["calibration/to_origin"][:]) \
                @ f["sensors/acoustic_sensor_01/calibration/to_origin"][:]
        Pb = sl["poses"][node]
        L2B = kz["lidar_to_base"]
        Tl_all = kz["T"] @ np.linalg.inv(L2B)                       # lidar SLAM base poses, lidar map frame
        il = np.clip(np.searchsorted(kz["t"], sl["t"]), 0, len(kz["t"]) - 1)
        A_, B_ = sl["poses"][:, :3, 3], Tl_all[il][:, :3, 3]
        ma, mb = A_.mean(0), B_.mean(0)
        U_, _, Vt = np.linalg.svd((A_ - ma).T @ (B_ - mb))
        R6 = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]) @ U_.T
        F = np.eye(4); F[:3, :3] = R6; F[:3, 3] = mb - R6 @ ma
        Ts = F @ Pb @ T_be
        dt = np.zeros(len(caps))
    else:
        caps = np.arange(0, len(t_ac), a.every)
        caps = caps[(t_ac[caps] > kz["t"][0]) & (t_ac[caps] < kz["t"][-1])]
        Ts, dt = sensor_poses(a.kiss, t_ac[caps], T_el)
    gz = ground_map(M)
    H = HeatKernel(U, sigma_deg=a.sigma, el_scale=1.7)
    keep_r = (rr >= a.rmin) & (rr <= a.rmax)
    pos = (rr[:, None, None] * U[None]).astype(np.float32)
    gate_fn = gz
    if a.gate == "track":
        gate_fn, h_used = track_ground(Ts, V, caps, pos, H, a)
    pts, n_det, n_gated = [], 0, 0
    for j, k in enumerate(caps):
        q = sphere_cfar(torch.as_tensor(10 ** (np.asarray(V[k], np.float32) / 10), device="cuda"), H, gain=a.gain).cpu().numpy()
        q[~keep_r] = 0
        ri, di = np.nonzero(q > q.max() * 10 ** (-a.peak_db / 10))
        X = pos[ri, di] @ Ts[j][:3, :3].T + Ts[j][:3, 3]
        n_det += len(X)
        if a.gate in ("lidar", "track") and len(X):
            g = gate_fn(X)
            bad = X[:, 2] < g - a.below
            n_gated += int(np.sum(bad))
            X = X[~bad]
        pts.append(X)
    P = np.vstack(pts)
    # sonar voxel hits
    vs = a.voxel
    key = np.floor(P / vs).astype(np.int64)
    uk, cnt = np.unique(key, axis=0, return_counts=True)
    occ = uk[cnt >= a.min_hits]
    Socc = (occ + 0.5) * vs
    # observed volume: voxels within rmax of some sensor position (sub-sampled) and in front of it (x > 0 in sensor frame)
    lid = M
    hl = lid[:, 2] - gz(lid)
    sens = np.array([T[:3, 3] for T in Ts[::5]])
    fwd = np.array([T[:3, 0] for T in Ts[::5]])
    tree_s = cKDTree(sens)
    obs = np.zeros(len(lid), bool)
    for i0, nb in enumerate(tree_s.query_ball_point(lid, a.rmax)):
        if nb:
            dvec = lid[i0] - sens[nb]
            obs[i0] = np.any(np.sum(dvec * fwd[nb], axis=1) > 0)
    tL, tS = cKDTree(lid), cKDTree(Socc) if len(Socc) else None
    hs = Socc[:, 2] - gz(Socc)
    res = {"captures": int(len(caps)), "pose_dt_max_s": float(dt.max()), "detections": int(n_det),
           "sensor_height_m": float(h_used) if a.gate == "track" else None,
           "gated_below_ground": int(n_gated), "sonar_occupied_voxels": int(len(Socc)), "args": vars(a)}
    dS, _ = tL.query(Socc, distance_upper_bound=5)
    dLo, _ = tS.query(lid[obs], distance_upper_bound=5)
    res["overall"] = {"precision": float(np.mean(dS <= a.tol)), "recall_observed": float(np.mean(dLo <= a.tol)),
                      "lidar_voxels_observed": int(obs.sum())}
    for b, (lo, hi) in BANDS.items():
        ms = (hs >= lo) & (hs < hi)
        ml = obs & (hl >= lo) & (hl < hi)
        dl, _ = tS.query(lid[ml], distance_upper_bound=5) if ml.any() else (np.zeros(0), None)
        res[b] = {"sonar_voxels": int(ms.sum()), "precision": float(np.mean(dS[ms] <= a.tol)) if ms.any() else None,
                  "lidar_voxels_observed": int(ml.sum()), "recall": float(np.mean(dl <= a.tol)) if ml.any() else None}
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1))
    out = Path(a.out) if a.out else vol.parent / "grid3d_gt"
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), sonar_occ=Socc.astype(np.float32), hits=cnt[cnt >= a.min_hits],
                        lidar=lid.astype(np.float32), lidar_observed=obs, sensor_xyz=np.array([T[:3, 3] for T in Ts], np.float32))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    m = obs
    for i_, (b, (lo, hi)) in enumerate(BANDS.items()):
        ml, ms = m & (hl >= lo) & (hl < hi), (hs >= lo) & (hs < hi)
        ax[i_].scatter(lid[ml, 0], lid[ml, 1], s=0.2, c="0.7")
        ax[i_].scatter(Socc[ms, 0], Socc[ms, 1], s=0.4, c="m")
        ax[i_].set_aspect("equal"); ax[i_].set_title(f"{b}: lidar (grey) vs sonar (magenta); prec {res[b]['precision']}, rec {res[b]['recall']}", fontsize=9)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
