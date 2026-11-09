# Vendored into RTISSLAM-3D from RTISSLAM core (rtisslam), commit 09706bf, BSD-3-Clause.
"""Post-processing of 2D energyscapes before RTISSLAM.

RW-CFAR (Range-Wise CFAR; Jan's name, 2026-10-12; in code the keys are still "ams*" for azimuthal mean
subtraction), from the RTIS pipeline: per range cell, the mean
energy over all directions is subtracted and the result is clipped at zero,
    P+(r, psi) = max(P(r, psi) - mean_psi' P(r, psi'), 0),
i.e. a cell-averaging CFAR threshold (gain alpha, 1 above; the RTIS pipeline typically uses ~1.5) whose reference cells are all directions at the same range. The
sidelobes of a strong reflector spread over all directions at its range and raise exactly that mean.
Energyscapes are in dB; zeros map to FLOOR_DB.
"""
from __future__ import annotations

import numpy as np

FLOOR_DB = -300.0


def ams(es_db: np.ndarray, gain: float = 1.0, smooth_cells: float = 0.0) -> np.ndarray:
    """gain alpha: P+ = max(P - alpha T, 0), T(r) = mean_psi P(r, psi) (the RTIS pipeline typically uses alpha ~ 1.5).
    smooth_cells > 0: the threshold map is smoothed with a Gaussian of that many range cells first (a 2D smoothing of
    the threshold map, which is constant over psi, acts along range only)."""
    P = np.power(10.0, np.asarray(es_db, np.float64) / 10)
    T = P.mean(axis=1, keepdims=True)
    if smooth_cells > 0:
        from scipy.ndimage import gaussian_filter1d
        T = gaussian_filter1d(T, smooth_cells, axis=0, mode="nearest")
    Pp = np.maximum(P - gain * T, 0.0)
    out = np.full(P.shape, FLOOR_DB, np.float32)
    nz = Pp > 0
    out[nz] = 10 * np.log10(Pp[nz])
    return out


def cfar2d(es_db: np.ndarray, gain: float = 1.5, sigma_r: float = 3.0, sigma_psi: float = 5.0) -> np.ndarray:
    """2D cell-averaging CFAR (range:direction smoothing, as sometimes used in the RTIS pipeline): the reference map is
    the energyscape smoothed with a 2D Gaussian (sigma_r range cells, sigma_psi directions);
    P+ = max(P - alpha (G * P), 0)."""
    from scipy.ndimage import gaussian_filter
    P = np.power(10.0, np.asarray(es_db, np.float64) / 10)
    Pp = np.maximum(P - gain * gaussian_filter(P, (sigma_r, sigma_psi), mode="nearest"), 0.0)
    out = np.full(P.shape, FLOOR_DB, np.float32)
    nz = Pp > 0
    out[nz] = 10 * np.log10(Pp[nz])
    return out


POST = {"none": lambda x: np.asarray(x, np.float32), "ams": ams, "ams15": lambda x: ams(x, 1.5),
        "ams2": lambda x: ams(x, 2.0), "ams25": lambda x: ams(x, 2.5),
        "ams15s": lambda x: ams(x, 1.5, 3.0), "ams2s": lambda x: ams(x, 2.0, 3.0),
        "cfar2d15": lambda x: cfar2d(x, 1.5), "cfar2d2": lambda x: cfar2d(x, 2.0)}
