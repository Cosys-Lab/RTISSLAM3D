"""All captures' 3D detection sets (rtis3d/detections.py) from the cached volumes, in the sensor frame.

    python scripts/extract_detections.py <vol dir> [--out <dets.npz>]
Writes pts [M, 3], w [M], start [N + 1] (capture k: rows start[k]:start[k+1]), height [N] (sensor height above the
fitted ground plane, nan without one), plane_n [N, 3], n_mirror [N], n_raw [N], t [N] (eRTIS clock).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.detections import Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vol = Path(a.vol)
    V = np.load(vol / "volumes.npy", mmap_mode="r")
    det = Detector(np.load(vol / "directions.npy"), np.load(vol / "ranges.npy"))
    P, W, start, H, Nn, nm, nr = [], [], [0], [], [], [], []
    t0 = time.time()
    for k in range(len(V)):
        p, w, info = det(V[k])
        P.append(p); W.append(w); start.append(start[-1] + len(p))
        H.append(info["height"] if info["height"] is not None else np.nan)
        Nn.append(info["plane"][0] if info["plane"] else [np.nan] * 3)
        nm.append(info["n_mirror"]); nr.append(info["n_raw"])
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(V)}, {time.time() - t0:.0f} s", flush=True)
    out = Path(a.out) if a.out else vol.parent / "detections.npz"
    np.savez_compressed(out, pts=np.vstack(P), w=np.concatenate(W), start=np.asarray(start), height=np.asarray(H),
                        plane_n=np.asarray(Nn, np.float32), n_mirror=np.asarray(nm), n_raw=np.asarray(nr),
                        t=np.load(vol / "t.npy"))
    H = np.asarray(H)
    print(f"{out}: {start[-1]} points, {np.mean(np.asarray(nr)):.0f} raw / {start[-1] / len(V):.0f} kept per capture, "
          f"ground plane in {np.mean(np.isfinite(H)):.0%} of the captures, height median {np.nanmedian(H):.3f} m, "
          f"mirrors removed {np.sum(nm) / max(1, np.sum(nr)):.1%}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
