"""All eRTIS captures of a CHIRP recording -> 3D energy volumes (range x direction over the frontal hemisphere), with
the approved real-data chain (rtisreal RealPreprocessor -> RTIS CUDA DMAS5-CF -> zero-phase 5 kHz envelope -> 2 cm
cells to 9.7 m), cached once as a memory-mapped array.

    python scripts/real_volumes.py <dataset.h5> <preproc.npz> <out dir> [--spacing 3] [--jobs 6]
Writes <out dir>/volumes.npy [N, R, D] float16 dB (no CFAR), directions.npy [D, 3], ranges.npy [R], t.npy [N]
(eRTIS clock, no offset), meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401

N_R, LP_HZ = 475, 5e3


def work(args):
    h5, pre_path, out, method, k0, k1 = args
    import rtis3d  # noqa: F401
    from chirppy import CHIRPDataset
    from rtisreal.rtis_frontend import RealPreprocessor, RtisCudaImager, RtisSetup
    from sondec.das import RangeGrid
    U = np.load(Path(out) / "directions.npy")
    V = np.load(Path(out) / "volumes.npy", mmap_mode="r+")
    pre = RealPreprocessor.load(pre_path)
    from rtis3d.dataset import sensor_names
    S_IDX = sensor_names(h5)["sonar_index"]
    with CHIRPDataset(h5) as ds:
        ac, tx = ds.get_acoustic_sensor(S_IDX), ds.get_acoustic_transmitter(S_IDX)
        im = RtisCudaImager(RtisSetup.from_chirp(ac, tx), grid=RangeGrid(n_r=N_R), method=method, directions=U,
                            lp_hz=LP_HZ, lp_shape="filtfilt")
        for k in range(k0, k1):
            V[k] = im.from_signals(pre(np.asarray(ac.signals[k]).T)).astype(np.float16)
        im.close()
    V.flush()
    return k1 - k0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("preproc")
    ap.add_argument("out")
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--method", default="dmas5cf")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    import h5py
    from rtis3d.dataset import sensor_names
    from rtis3d.sphere import frontal_hemisphere
    from sondec.das import RangeGrid
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.h5, "r") as f:
        t = f[f"sensors/{sensor_names(a.h5)['sonar']}/data/timestamps"][:]
    n = len(t) if not a.max else min(a.max, len(t))
    U = frontal_hemisphere(a.spacing)
    np.save(out / "directions.npy", U)
    np.save(out / "ranges.npy", RangeGrid(n_r=N_R).r("cpu").numpy())
    np.save(out / "t.npy", t[:n])
    np.lib.format.open_memmap(out / "volumes.npy", mode="w+", dtype=np.float16, shape=(n, N_R, len(U))).flush()
    edges = np.linspace(0, n, a.jobs * 4 + 1).astype(int)            # small chunks: even load
    t0 = time.time()
    import multiprocessing as mp
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        done = 0
        for c in ex.map(work, [(a.h5, a.preproc, str(out), a.method, int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]):
            done += c
            print(f"  {done}/{n} captures, {time.time() - t0:.0f} s", flush=True)
    (out / "meta.json").write_text(json.dumps({"method": a.method, "lp_hz": LP_HZ, "spacing_deg": a.spacing,
                                               "directions": int(len(U)), "captures": int(n), "h5": a.h5,
                                               "preproc": a.preproc, "seconds": time.time() - t0}, indent=1))
    print(f"{out}: {n} x {N_R} x {len(U)}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
