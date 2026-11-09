"""3D imaging with the real eRTIS geometry, on simulated point targets (no interference, exact truth):

  PSF        a point target at (r, az, el): -3 dB extent of the main lobe in azimuth and elevation, the strongest
             sidelobe outside 20 deg, peak position error; for DAS, DMAS3-CF and DMAS5-CF
  multipath  the same target above a reflecting ground (image method): direct, transmit-bounce, receive-bounce and
             double-bounce paths; where the mirror images appear in the 3D image

Geometry from the CHIRP recording: microphone positions (array frame), emitter offset (transmitter to_origin minus
array to_origin), the emitted call. Ground: sensor height h above a flat ground (z = -h in the sensor frame),
reflection coefficient rho per bounce.

    python scripts/sim_psf3d.py <dataset.h5> [--h 0.28] [--rho 0.5] [--out prefix]
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
from rtis3d.sphere import angle_between, az_el, frontal_hemisphere  # noqa: E402
from rtisreal.rtis_frontend import RtisCudaImager, RtisSetup  # noqa: E402
from sondec.das import RangeGrid  # noqa: E402
from sondec.sensor import C_SOUND  # noqa: E402


def unit(az, el):
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def synth(setup, emitter, paths, N, rng, noise=1e-4):
    """paths: list of (tx point, rx point, amplitude): echo from tx-leg |emitter - tx| + rx-leg |rx - mic|."""
    fs = setup.fs_adc
    f = np.fft.rfftfreq(N, 1 / fs)
    C = np.fft.rfft(setup.chirp, N)
    X = np.zeros((len(setup.positions), len(f)), complex)
    for ptx, prx, amp in paths:
        dtx = np.linalg.norm(ptx - emitter)
        drx = np.linalg.norm(setup.positions - prx[None], axis=1)
        tau = (dtx + drx) / C_SOUND
        X += amp / (dtx * drx[:, None]) * C[None] * np.exp(-2j * np.pi * f[None] * tau[:, None])
    x = np.fft.irfft(X, N)
    return (x + noise * rng.standard_normal(x.shape)).astype(np.float32)


def lobe_stats(P, U, u_true, peak_db_drop=3.0, side_deg=20.0):
    """P [D] linear power at the target's range; -> dict."""
    i = int(np.argmax(P))
    db = 10 * np.log10(P / P[i] + 1e-30)
    ang = np.degrees(angle_between(U, U[i][None]))
    az, el = az_el(U)
    main = (db > -peak_db_drop) & (ang < side_deg)
    daz = np.degrees(np.abs((az[main] - az[i] + np.pi) % (2 * np.pi) - np.pi))
    dl = np.degrees(np.abs(el[main] - el[i]))
    return {"peak_error_deg": float(np.degrees(angle_between(U[i], u_true))),
            "width_az_deg": float(2 * daz.max()) if main.any() else None,
            "width_el_deg": float(2 * dl.max()) if main.any() else None,
            "max_sidelobe_db": float(db[ang > side_deg].max()) if (ang > side_deg).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--h", type=float, default=0.28, help="sensor height above the ground (m)")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from chirppy import CHIRPDataset
    with CHIRPDataset(a.h5) as ds:
        ac, tx = ds.get_acoustic_sensor(1), ds.get_acoustic_transmitter(1)
        setup = RtisSetup.from_chirp(ac, tx)
        emitter = np.asarray(tx.get_transform_to_origin())[:3, 3] - np.asarray(ac.get_transform_to_origin())[:3, 3]
    U = frontal_hemisphere(3.0)
    grid = RangeGrid(n_r=475)
    rr = grid.r("cpu").numpy()
    rng = np.random.default_rng(0)
    res = {"directions": int(len(U)), "emitter_m": emitter.tolist(), "sensor_height_m": a.h, "rho": a.rho, "psf": {}, "multipath": {}}
    ims = {m: RtisCudaImager(setup, grid=grid, method=m, directions=U, lp_hz=5e3, lp_shape="filtfilt")
           for m in ("das", "dmas3cf", "dmas5cf")}
    targets = [(3.0, 0, 0), (3.0, 30, 0), (3.0, 0, 20), (3.0, -40, -15), (5.0, 15, 35), (5.0, -60, 10)]
    for r, azd, eld in targets:
        u = unit(math.radians(azd), math.radians(eld))
        x = synth(setup, emitter, [(r * u, r * u, 1.0)], setup.n_adc, rng)
        key = f"r{r:g}_az{azd}_el{eld}"
        res["psf"][key] = {}
        for m, im in ims.items():
            es = im.from_signals(x.T if x.shape[0] != 32 else x)
            ri = int(np.argmin(np.abs(rr - r)))
            P = 10 ** (es[max(0, ri - 2):ri + 3].max(0) / 10)
            res["psf"][key][m] = lobe_stats(P, U, u)
    # ground multipath: target 0.4 m above the ground, 4 m ahead
    h = a.h
    for r, azd, z_above in [(4.0, 10, 0.4), (4.0, -25, 1.0)]:
        zt = z_above - h                                         # target height in the sensor frame
        horiz = math.sqrt(r * r - zt * zt)
        T = np.array([horiz * math.cos(math.radians(azd)), horiz * math.sin(math.radians(azd)), zt])
        Tm = T * np.array([1, 1, 1]); Tm[2] = -2 * h - zt        # mirror below the ground z = -h
        paths = [(T, T, 1.0), (Tm, T, a.rho), (T, Tm, a.rho), (Tm, Tm, a.rho ** 2)]
        x = synth(setup, emitter, paths, setup.n_adc, rng)
        es = ims["dmas5cf"].from_signals(x)
        P = 10 ** (es / 10)
        # strongest echoes in the volume: peaks above -25 dB, merged within 6 deg / 10 cm
        flat = np.argsort(P.ravel())[::-1]
        peaks = []
        for fi in flat[:4000]:
            ri, di = np.unravel_index(fi, P.shape)
            if 10 * np.log10(P[ri, di] / P.max()) < -25:
                break
            if all(abs(rr[ri] - rr[pr]) > 0.1 or math.degrees(angle_between(U[di], U[pd])) > 6 for pr, pd in peaks):
                peaks.append((ri, di))
        az_, el_ = az_el(U)
        res["multipath"][f"r{r:g}_az{azd}_z{z_above:g}"] = {
            "true_el_deg": math.degrees(math.asin(zt / r)), "mirror_el_deg": math.degrees(math.asin(Tm[2] / np.linalg.norm(Tm))),
            "peaks": [{"range_m": float(rr[ri]), "az_deg": float(np.degrees(az_[di])), "el_deg": float(np.degrees(el_[di])),
                       "rel_db": float(10 * np.log10(P[ri, di] / P.max()))} for ri, di in peaks[:6]]}
    for im in ims.values():
        im.close()
    print(json.dumps(res, indent=1))
    out = Path(a.out) if a.out else Path("results/sim_psf3d")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
