# Vendored into RTISSLAM-3D from RTISSLAM Real (scripts), commit 5c0dc2c.
"""All eRTIS captures of a CHIRP recording -> cached energyscapes with the approved chain, plus a scene-change signal
for the clock synchronization (scripts/clock_sync.py), in one parallel pass.

    python scripts/real_energyscapes.py <dataset.h5> <preproc.npz> <out.npz> [--method dmas5cf] [--jobs 6]

Chain (docs/rendering_real_ertis.md): RealPreprocessor (direct-path alignment, static background subtraction,
interference nulling) -> RTIS CUDA DMAS-n(-CF) -> zero-phase 5 kHz envelope -> mean power per 2 cm range cell, 0.2 to
9.7 m (RangeGrid(n_r=475)), 181 azimuths. Stored WITHOUT the RW-CFAR and smoothing (RTISSLAM applies its own post).

Output: es [N, 475, 181] float16 dB, t [N] (the eRTIS timestamps as recorded, no clock correction), ranges, azimuths,
change [N] (scene change against the previous capture: 1 - correlation of the preprocessed signals beyond 0.8 m;
NaN for the first), shift [N] (direct-path alignment applied, samples), method, lp_hz.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtisreal  # noqa: E402,F401  (puts the RTISSLAM core on sys.path)

LP_HZ, N_R, CHANGE_FROM = 5e3, 475, 2000          # change signal from sample 2000 (~0.76 m)


def work(args):
    h5, pre_path, method, k0, k1 = args
    from chirppy import CHIRPDataset
    from rtisreal.rtis_frontend import RealPreprocessor, RtisCudaImager, RtisSetup, align_direct_path
    from sondec.das import RangeGrid
    pre = RealPreprocessor.load(pre_path)
    out_es, out_ch, out_sh = [], [], []
    with CHIRPDataset(h5) as ds:
        ac, tx = ds.get_acoustic_sensor(1), ds.get_acoustic_transmitter(1)
        im = RtisCudaImager(RtisSetup.from_chirp(ac, tx), grid=RangeGrid(n_r=N_R), method=method, lp_hz=LP_HZ,
                            lp_shape="filtfilt")
        prev = None
        for k in range(max(k0 - 1, 0), k1):
            raw = np.asarray(ac.signals[k]).T
            _, sh = align_direct_path(raw, pre.chirp, pre.dp_ref)
            x = pre(raw)
            v = x[:, CHANGE_FROM:].ravel().astype(np.float64)
            ch = np.nan if prev is None else 1.0 - float(np.dot(v, prev) / (np.linalg.norm(v) * np.linalg.norm(prev) + 1e-30))
            prev = v
            if k < k0:
                continue
            out_es.append(im.from_signals(x).astype(np.float16))
            out_ch.append(ch)
            out_sh.append(sh)
        im.close()
    return k0, np.stack(out_es), np.array(out_ch), np.array(out_sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("preproc")
    ap.add_argument("out")
    ap.add_argument("--method", default="dmas5cf")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    import h5py
    with h5py.File(a.h5, "r") as f:
        t = f["sensors/acoustic_sensor_01/data/timestamps"][:]
    n = len(t)
    edges = np.linspace(0, n, a.jobs + 1).astype(int)
    t0 = time.time()
    import multiprocessing as mp
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        res = sorted(ex.map(work, [(a.h5, a.preproc, a.method, int(edges[i]), int(edges[i + 1])) for i in range(a.jobs)]),
                     key=lambda r: r[0])
    es = np.concatenate([r[1] for r in res])
    change = np.concatenate([r[2] for r in res])
    shift = np.concatenate([r[3] for r in res])
    from sondec.das import RangeGrid
    np.savez(a.out, es=es, t=t, change=change, shift=shift, ranges=RangeGrid(n_r=N_R).r("cpu").numpy(),
             azimuths=np.radians(np.linspace(-90, 90, 181)).astype(np.float32), method=a.method, lp_hz=LP_HZ)
    print(f"{a.out}: {es.shape}, {time.time() - t0:.0f} s; aligned captures {np.mean(shift != 0):.1%}")


if __name__ == "__main__":
    main()
