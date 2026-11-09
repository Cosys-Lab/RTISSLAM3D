# Vendored into RTISSLAM-3D from RTISSLAM Real (scripts), commit 5c0dc2c.
"""Clock offset of the eRTIS against the robot (odometry / lidar clock), from motion, with two signals:

  rotation (default, signed): the robot's yaw rate seen by the sonar, i.e. the azimuth shift that best maps one capture's
      energyscape onto the next (sub-degree, parabolic peak; RW-CFAR + smoothing first) divided by the capture
      interval, against the yaw rate of the odometry poses (differentiated: chirppy notes that the velocity fields are
      rolling averages stamped late). Turning is frequent and signed, so the correlation peak is sharp.
  change: the scene change between consecutive captures (scripts/real_energyscapes.py) against |v| + |w|; weak when
      the robot drives at a nearly constant speed (forest recording: r = 0.30, peak +-0.4 s).
The offset maximizes the correlation.

    python scripts/clock_sync.py <dataset.h5> <cache.npz> [--coarse 409981] [--search 30] [--out prefix]

Coarse step: search +-search s around the coarse offset (0.05 s steps); fine step: +-0.5 s around the best (5 ms steps,
the eRTIS fires every ~0.23 s, so the resolution comes from the many motion transitions, not the pulse rate).
Motion = |v| / median moving |v| + |w| / median moving |w| (translation and rotation both change the scene).
Writes <out>.json (offset, correlation, curve) and <out>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtisreal  # noqa: E402,F401
from rtisreal.chirp_io import find_odometry, odometry_track  # noqa: E402


def motion_signal(t_od, p_od, smooth_s=0.25):
    dt = np.gradient(t_od)
    v = np.hypot(np.gradient(p_od[:, 0]), np.gradient(p_od[:, 1])) / dt
    w = np.abs(np.gradient(p_od[:, 2])) / dt
    k = max(1, int(round(smooth_s / np.median(dt))))
    ker = np.ones(k) / k
    v, w = np.convolve(v, ker, "same"), np.convolve(w, ker, "same")
    mv = np.median(v[v > 0.05]) if np.any(v > 0.05) else 1.0
    mw = np.median(w[w > 0.05]) if np.any(w > 0.05) else 1.0
    return v / mv + w / mw


def sonar_yaw_rate(es: np.ndarray, t: np.ndarray, az: np.ndarray, r0: int = 25, r1: int = 400, max_shift: int = 12):
    """[N, R, D] dB energyscapes -> yaw rate (rad/s) between consecutive captures (NaN for the first): the azimuth shift
    s maximizing corr(es_k[:, j], es_{k-1}[:, j + s]) over the range cells r0..r1 (robot turning left: the scene moves
    to the right, the shift is positive)."""
    import torch
    from scipy.ndimage import gaussian_filter
    from rtisslam.imaging_post import POST
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    daz = float(az[1] - az[0])
    out = np.full(len(es), np.nan)
    prev = None
    S = np.arange(-max_shift, max_shift + 1)
    for k in range(len(es)):
        P = gaussian_filter(10 ** (POST["ams2s"](es[k].astype(np.float32)) / 10), (1.5, 1.5))[r0:r1]
        x = torch.as_tensor(np.sqrt(P), dtype=torch.float32, device=dev)
        x = (x - x.mean()) / (x.std() + 1e-12)
        if prev is not None:
            D = x.shape[1]
            c = []
            for sh in S:
                a_ = x[:, max(0, -sh):D - max(0, sh)]
                b_ = prev[:, max(0, sh):D - max(0, -sh)]
                c.append(float((a_ * b_).mean()))
            c = np.array(c)
            i = int(np.argmax(c))
            d = 0.0
            if 0 < i < len(c) - 1:
                den = c[i - 1] - 2 * c[i] + c[i + 1]
                d = 0.5 * (c[i - 1] - c[i + 1]) / den if den < 0 else 0.0
            out[k] = (S[i] + d) * daz / (t[k] - t[k - 1])
        prev = x
    return out


def corr_at(off, t_ac, ch, t_od, m):
    tt = t_ac + off
    ok = (tt > t_od[0]) & (tt < t_od[-1]) & np.isfinite(ch)
    a, b = ch[ok], np.interp(tt[ok], t_od, m)
    return float(np.corrcoef(a, b)[0, 1]) if ok.sum() > 100 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("cache")
    ap.add_argument("--coarse", type=float, default=409981.0)
    ap.add_argument("--search", type=float, default=30.0)
    ap.add_argument("--odom-source", default="ekf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--signal", default="rotation", choices=["rotation", "change"])
    a = ap.parse_args()
    from chirppy import CHIRPDataset
    z = np.load(a.cache)
    t_ac = z["t"]
    with CHIRPDataset(a.h5) as ds:
        t_od, p_od = odometry_track(ds.get_odometry(find_odometry(ds, a.odom_source)))
    if a.signal == "change":
        ch = np.log(np.clip(z["change"].astype(float), 1e-4, None))   # heavy-tailed: compare its log
        m = motion_signal(t_od, p_od)
    else:
        cache = Path(a.cache).with_name(Path(a.cache).stem + "_yawrate.npy")
        if cache.exists():
            ch = np.load(cache)
        else:
            ch = sonar_yaw_rate(z["es"], t_ac, z["azimuths"])
            np.save(cache, ch)
        ch = np.clip(ch, -1.5, 1.5)                                   # rad/s: drop the odd failed match
        m = np.convolve(np.gradient(p_od[:, 2]) / np.gradient(t_od), np.ones(5) / 5, "same")
    offs = a.coarse + np.arange(-a.search, a.search + 1e-9, 0.05)
    c = np.array([corr_at(o, t_ac, ch, t_od, m) for o in offs])
    best = offs[np.nanargmax(c)]
    fine = best + np.arange(-0.5, 0.5 + 1e-9, 0.005)
    cf = np.array([corr_at(o, t_ac, ch, t_od, m) for o in fine])
    off = float(fine[np.nanargmax(cf)])
    # uncertainty: half width of the fine peak at 99 % of its maximum
    near = fine[cf >= 0.99 * np.nanmax(cf)]
    # stability: offsets of the two halves of the recording
    half = []
    mid = t_ac[len(t_ac) // 2]
    for sel in (t_ac < mid, t_ac >= mid):
        cc = np.array([corr_at(o, t_ac[sel], ch[sel], t_od, m) for o in fine])
        half.append(float(fine[np.nanargmax(cc)]))
    summ = {"offset_s": off, "corr": float(np.nanmax(cf)), "coarse_given": a.coarse, "coarse_best": float(best),
            "peak_width_99_s": [float(near.min()), float(near.max())], "offset_first_half": half[0],
            "offset_second_half": half[1], "odometry": a.odom_source, "signal": a.signal}
    print(json.dumps(summ, indent=1))
    out = Path(a.out) if a.out else Path(a.cache).with_name("clock_sync")
    out.with_suffix(".json").write_text(json.dumps({**summ, "curve": {"offset": offs.tolist(), "corr": c.tolist()}}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    ax[0].plot(offs - a.coarse, c); ax[0].axvline(off - a.coarse, color="m", lw=0.8)
    ax[0].set_xlabel(f"offset - {a.coarse:.0f} (s)"); ax[0].set_ylabel("correlation"); ax[0].set_title("coarse search")
    ax[1].plot(fine - a.coarse, cf); ax[1].axvline(off - a.coarse, color="m", lw=0.8)
    ax[1].set_xlabel(f"offset - {a.coarse:.0f} (s)"); ax[1].set_title(f"fine: {off - a.coarse:+.3f} s, r = {np.nanmax(cf):.3f}")
    sel = slice(len(t_ac) // 3, len(t_ac) // 3 + 400)
    tt = t_ac[sel] + off
    ax[2].plot(tt - tt[0], (ch[sel] - np.nanmean(ch[sel])) / np.nanstd(ch[sel]), lw=0.8,
               label="sonar yaw rate" if a.signal == "rotation" else "sonar scene change (log)")
    mm = np.interp(tt, t_od, m)
    ax[2].plot(tt - tt[0], (mm - mm.mean()) / mm.std(), lw=0.8, label="odometry yaw rate" if a.signal == "rotation" else "odometry motion")
    ax[2].set_xlabel("s"); ax[2].legend(fontsize=8); ax[2].set_title("aligned, 400 captures")
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
