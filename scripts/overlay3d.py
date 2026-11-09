"""3D sonar detections against the 3D lidar, per capture, in the eRTIS frame (no SLAM involved):

  detections   CFAR on the sphere (rtis3d.laplacian.sphere_cfar) on the DMAS5-CF volume, then cells within --peak-db of
               the capture's strongest CFAR output, positions r * u (u: direction, r: cell centre)
  truth        the lidar scan closest in time (eRTIS clock + offset), moved into the eRTIS frame (static calibration),
               0.3..9.5 m and inside the sonar's frontal hemisphere
  scores       precision: fraction of detections within --tol (3D) of a lidar point; below-ground fraction: detections
               more than 0.2 m below the local lidar ground (mirror images of the ground multipath); detections per
               capture. Lidar recall is not scored (the lidar sees many surfaces a sonar cannot: oblique, fine twigs).
Variants: RW-CFAR (background = mean over all directions per range), and heat-kernel CFAR with several angular scales
and gains. A figure with the sonar detections and the lidar points for a few captures (az x sin(el), equal area).

    python scripts/overlay3d.py <vol dir> <dataset.h5> --offset 409980.68 [--n 200] [--out prefix]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.laplacian import HeatKernel, sphere_cfar  # noqa: E402
from rtis3d.sphere import az_el  # noqa: E402


def lidar_in_sensor(f, k_l, T_el):
    P, C = f["sensors/lidar_00/data/points"], f["sensors/lidar_00/data/point_counts"]
    p = np.asarray(P[k_l][:int(C[k_l]), :3], float)
    p = p[np.isfinite(p).all(1)]
    return p @ T_el[:3, :3].T + T_el[:3, 3]


def local_ground(pts, cell=0.5):
    """lowest z per xy cell (and its 8 neighbours) -> function (x, y) -> ground z (nan where unknown)."""
    ij = np.floor(pts[:, :2] / cell).astype(int)
    o = ij.min(0) - 1
    ij -= o
    G = np.full(tuple(ij.max(0) + 2), np.inf)
    np.minimum.at(G, (ij[:, 0], ij[:, 1]), pts[:, 2])
    Gn = G.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            Gn[1:-1, 1:-1] = np.minimum(Gn[1:-1, 1:-1], G[1 + dx:G.shape[0] - 1 + dx, 1 + dy:G.shape[1] - 1 + dy])

    def gz(x, y):
        i = np.floor(x / cell).astype(int) - o[0]
        j = np.floor(y / cell).astype(int) - o[1]
        ok = (i >= 0) & (i < Gn.shape[0]) & (j >= 0) & (j < Gn.shape[1])
        z = np.full(len(x), np.nan)
        z[ok] = Gn[i[ok], j[ok]]
        z[~np.isfinite(z)] = np.nan
        return z
    return gz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("h5")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--peak-db", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import h5py
    import torch
    from scipy.spatial import cKDTree
    vol = Path(a.vol)
    V = np.load(vol / "volumes.npy", mmap_mode="r")
    U, rr, t_ac = np.load(vol / "directions.npy"), np.load(vol / "ranges.npy"), np.load(vol / "t.npy") + a.offset
    f = h5py.File(a.h5, "r")
    T_e = f["sensors/acoustic_sensor_01/calibration/to_origin"][:]
    T_l = f["sensors/lidar_00/calibration/to_origin"][:]
    T_el = np.linalg.inv(T_e) @ T_l
    t_l = f["sensors/lidar_00/data/timestamps"][:]
    variants = {"rwcfar": None}
    for sig in (6.0, 12.0, 25.0):
        variants[f"heat{sig:g}"] = HeatKernel(U, sigma_deg=sig, el_scale=1.7)
    gains = (2.0, 3.0, 5.0)
    caps = np.linspace(200, len(t_ac) - 200, a.n).astype(int)
    pos = (rr[:, None, None] * U[None]).astype(np.float32)                   # [R, D, 3]
    dev = "cuda"
    stats = {f"{v}_g{g:g}": {"det": 0, "hit": 0, "below": 0, "n": 0} for v in variants for g in gains}
    examples = {}
    for ci, k in enumerate(caps):
        kl = int(np.argmin(np.abs(t_l - t_ac[k])))
        L = lidar_in_sensor(f, kl, T_el)
        rL = np.linalg.norm(L, axis=1)
        L = L[(rL > 0.3) & (rL < 9.5) & (L[:, 0] > 0)]
        tree = cKDTree(L)
        gz = local_ground(L)
        P = torch.as_tensor(10 ** (np.asarray(V[k], np.float32) / 10), device=dev)
        for vname, H in variants.items():
            if H is None:
                B = P.mean(1, keepdim=True).expand_as(P)
                n = 9
                g = torch.exp(-0.5 * (torch.arange(-n, n + 1, device=dev, dtype=P.dtype) / 3.0) ** 2); g = (g / g.sum()).view(1, 1, -1)
                B = torch.nn.functional.conv1d(torch.nn.functional.pad(B.T.unsqueeze(1), (n, n), mode="replicate"), g).squeeze(1).T
            for gain in gains:
                if H is None:
                    Q = torch.clamp(P - gain * B, min=0)
                else:
                    Q = sphere_cfar(P, H, gain=gain, range_sigma=3.0)
                q = Q.cpu().numpy()
                thr = q.max() * 10 ** (-a.peak_db / 10)
                ri, di = np.nonzero(q > max(thr, 0))
                X = pos[ri, di]
                st = stats[f"{vname}_g{gain:g}"]
                st["n"] += 1
                if len(X) == 0:
                    continue
                d, _ = tree.query(X, distance_upper_bound=5.0)
                z0 = gz(X[:, 0], X[:, 1])
                st["det"] += len(X); st["hit"] += int((d <= a.tol).sum())
                st["below"] += int(np.nansum(X[:, 2] < z0 - 0.2))
                if ci in (10, 60, 120, 170) and gain == 3.0 and vname in ("rwcfar", "heat12"):
                    examples[(ci, vname)] = (X, q[ri, di], L)
        if (ci + 1) % 50 == 0:
            print(f"  {ci + 1}/{len(caps)}", flush=True)
    res = {}
    for kk, st in stats.items():
        res[kk] = {"precision": st["hit"] / max(1, st["det"]), "below_ground_frac": st["below"] / max(1, st["det"]),
                   "detections_per_capture": st["det"] / max(1, st["n"])}
    print(json.dumps(res, indent=1))
    out = Path(a.out) if a.out else vol.parent / "overlay3d"
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({c for c, _ in examples})
    fig, axs = plt.subplots(len(keys), 2, figsize=(13, 3.2 * len(keys)), squeeze=False)
    for i, c in enumerate(keys):
        for j, v in enumerate(("rwcfar", "heat12")):
            if (c, v) not in examples:
                continue
            X, w, L = examples[(c, v)]
            ax = axs[i, j]
            azL, elL = az_el(L / np.linalg.norm(L, axis=1, keepdims=True))
            ax.scatter(np.degrees(azL), np.sin(elL), s=0.3, c="0.75", label="lidar points (0.3-9.5 m)")
            azS, elS = az_el(X / np.linalg.norm(X, axis=1, keepdims=True))
            sc = ax.scatter(np.degrees(azS), np.sin(elS), s=4, c=np.linalg.norm(X, axis=1), cmap="plasma", vmin=0, vmax=9)
            ax.set_xlim(90, -90); ax.set_ylim(-1, 1)
            ax.set_xlabel("azimuth (deg, + left)"); ax.set_ylabel("sin(elevation)")
            ax.set_title(f"capture {caps[c]}: sonar {v} gain 3 (colour = range m) on lidar", fontsize=9)
            plt.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=100)


if __name__ == "__main__":
    main()
