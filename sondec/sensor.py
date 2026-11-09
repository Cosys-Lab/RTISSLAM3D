# Vendored into RTISSLAM-3D from RTISSLAM core (sondec), commit 09706bf, BSD-3-Clause.
"""The simulated eRTIS sensor: one broadband emitter and 32 MEMS microphones in a random planar array.

eRTIS (Kerstens, Laurijssen, Steckel, ICRA 2019; "eRTIS in the wild", 2025):
- emitter: a broadband capacitive transducer (SensComp 7000 series) sending a logarithmic FM sweep
  between 20 and 80 kHz (here: a 3 ms HYPERBOLIC sweep 80 -> 20 kHz, the user's choice);
- receivers: 32 digital (PDM) MEMS microphones on a known, irregular layout (Poisson-disc sampling,
  which suppresses grating lobes in wideband beamforming);
- image formation: PDM demodulation, matched filter, time-domain delay-and-sum beamforming per
  direction of interest, envelope detection -> the "energyscape" (range x direction).

Choices made here (documented in README, not taken from the papers):
- array: 32 microphones, Poisson-disc sampled (dart throwing with the largest feasible minimum
  distance) inside an ellipse of 7 cm (horizontal, y) x 5 cm (vertical, z), in the sensor's y-z plane;
- emitter 4 cm above the array centre, a baffled piston (effective radius EMITTER_A) facing +x;
- microphones omnidirectional over the front half-space (PCB baffle), nothing from behind;
- the TRUE array differs from the NOMINAL one used by the beamformer: position errors (0.15 mm rms),
  gain errors (0.8 dB rms) and phase errors (5 deg rms) per microphone. The deconvolver's forward model
  has to learn these from data (it is never given them);
- PDM is not simulated (the signals are what the demodulator would output); white sensor noise.

Frames: the sensor frame is the "head" frame of kinematics.py: x forward, y left, z up, origin at the
array centre. Directions u = (cos el cos az, cos el sin az, sin el).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FS = 450_000.0          # Hz, eRTIS sampling rate after PDM demodulation
C_SOUND = 343.0         # m/s
F_LO, F_HI = 20e3, 80e3
CALL_DURATION = 3e-3    # s
N_MICS = 32
APERTURE = (0.07, 0.05)  # m, full ellipse axes (y, z)
EMITTER_POS = (0.0, 0.0, 0.04)   # m, sensor frame
EMITTER_A = 0.003       # m, effective piston radius, emitter v1 (wide beam over the frontal hemisphere)
# Realistic emitter (v2, user request 2026-10-06: "model the spectrum of the emitter plus its directivity;
# it is not uniform in frequency and direction, which makes the PSFs differ and hard to calibrate"):
EMITTER_V2 = {
    # directivity: NOT a piston (user, 2026-10-06: the SensComp 7000 falls off much less to the sides than
    # the piston equations imply, probably diffraction). Empirical: Gaussian main lobe with width ~ 1/f plus
    # a smooth diffraction floor, no nulls (emitter_directivity_v2). Replace with a measured pattern if available.
    "sigma50_deg": 25.0,         # main-lobe std at 50 kHz (power pattern); scales as 50 kHz / f
    "floor_db": -13.0,           # diffraction floor relative to on-axis, x ((1 + cos th) / 2)^2
    "axis_azel_deg": (2.0, -4.0),  # mounting misalignment of the emitter axis (unknown to the nominal model)
    "peak_hz": 45e3, "width_oct": 1.2,   # transmit response: -12 dB at +-width_oct octaves from the peak
    "ripple_db": 1.5, "ripple_hz": 9e3,  # standing-wave ripple of an electrostatic membrane
}
ARRAY_FILE = Path(__file__).resolve().parents[1] / "data" / "array.json"


def chirp() -> np.ndarray:
    """Hyperbolic FM downsweep F_HI -> F_LO over CALL_DURATION (user's choice, 2026-10-06): instantaneous
    frequency f(t) = 1 / (1/F_HI + (1/F_LO - 1/F_HI) t / T), i.e. linear period; Hann-tapered 5 % at both
    ends."""
    N = int(round(CALL_DURATION * FS))
    t = np.arange(N) / FS
    a, b = 1 / F_HI, (1 / F_LO - 1 / F_HI) / CALL_DURATION
    phase = 2 * math.pi / b * np.log1p(b * t / a)
    x = np.sin(phase)
    rn = int(0.05 * N)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(rn) / rn)
    x[:rn] *= ramp; x[-rn:] *= ramp[::-1]
    return x


def absorption_db_per_m(f: np.ndarray, temp_c: float = 20.0, rh: float = 50.0, p_kpa: float = 101.325) -> np.ndarray:
    """ISO 9613-1 atmospheric absorption (dB/m)."""
    T, T0, T01, p0 = temp_c + 273.15, 293.15, 273.16, 101.325
    psat = 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = rh * psat * (p0 / p_kpa)
    frO = (p_kpa / p0) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (p_kpa / p0) * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    cl = 1.84e-11 * (p0 / p_kpa) * (T / T0) ** 0.5
    tO = 0.01275 * np.exp(-2239.1 / T) * (frO / (frO ** 2 + f ** 2))
    tN = 0.1068 * np.exp(-3352.0 / T) * (frN / (frN ** 2 + f ** 2))
    return 8.686 * f ** 2 * (cl + (T / T0) ** -2.5 * (tO + tN))


def piston_directivity(cos_theta: np.ndarray, f: np.ndarray, a: float = EMITTER_A) -> np.ndarray:
    """Baffled piston, amplitude 2 J1(ka sin th) / (ka sin th); zero behind (cos_theta <= 0).
    cos_theta [...], f [F] -> [..., F]."""
    from scipy.special import j1
    s = np.sqrt(np.clip(1 - cos_theta ** 2, 0, 1))[..., None]
    x = 2 * math.pi * f / C_SOUND * a * s
    d = np.where(x < 1e-6, 1.0, 2 * j1(np.maximum(x, 1e-6)) / np.maximum(x, 1e-6))
    return d * (cos_theta[..., None] > 0)


def emitter_response_db(f: np.ndarray, em: dict = EMITTER_V2, wavy_db: float = 0.0) -> np.ndarray:
    """Transmit (on-axis) response in dB: band-pass on log f plus ripple (v2). wavy_db > 0 (v3, user request
    2026-10-06: "non-flat spectra, wavy with peaks of up to +-10 dB"): plus a fixed, smooth random
    waviness (three sinusoids in f, periods 7-16 kHz, random phases, seed 3) scaled to peaks of +-wavy_db."""
    lg = np.log2(np.maximum(f, 1.0) / em["peak_hz"])
    out = -12.0 * (lg / em["width_oct"]) ** 2 + em["ripple_db"] * np.sin(2 * math.pi * f / em["ripple_hz"])
    if wavy_db > 0:
        rng = np.random.default_rng(3)
        per = rng.uniform(7e3, 16e3, 3); ph = rng.uniform(0, 2 * math.pi, 3); amp = rng.uniform(0.5, 1.0, 3)
        fg = np.linspace(F_LO, F_HI, 2001)
        w = lambda x: sum(a * np.sin(2 * math.pi * x / T + q) for a, T, q in zip(amp, per, ph))  # noqa: E731
        out = out + wavy_db * w(f) / np.abs(w(fg)).max()
    return out


def emitter_directivity_v2(cos_theta: np.ndarray, f: np.ndarray, em: dict = EMITTER_V2) -> np.ndarray:
    """Amplitude directivity of the v2 emitter: power = (1 - b) exp(-th^2 / (2 s(f)^2)) + b ((1 + cos th) / 2)^2,
    s(f) = sigma50 (50 kHz / f), b = 10^(floor_db / 10). cos_theta [...], f [F] -> [..., F]."""
    th = np.arccos(np.clip(cos_theta, -1, 1))[..., None]
    s = np.radians(em["sigma50_deg"]) * 50e3 / np.maximum(f, 1.0)
    b = 10 ** (em["floor_db"] / 10)
    pw = (1 - b) * np.exp(-0.5 * (th / s) ** 2) + b * ((1 + np.cos(th)) / 2) ** 2
    return np.sqrt(pw) * (cos_theta[..., None] > -1)


def emitter_axis(em: dict = EMITTER_V2) -> np.ndarray:
    az, el = np.radians(em["axis_azel_deg"])
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def poisson_disc_ellipse(n: int = N_MICS, axes=APERTURE, seed: int = 0, tries: int = 4000) -> np.ndarray:
    """n points (y, z) in the ellipse with semi-axes axes/2, by dart throwing with a shrinking minimum
    distance: the largest r_min for which n points fit (best of `tries` darts per point)."""
    rng = np.random.default_rng(seed)
    ay, az = axes[0] / 2, axes[1] / 2
    r_min = 0.95 * math.sqrt(math.pi * ay * az / n)                        # start: hexagonal-packing scale
    while True:
        pts = []
        for _ in range(tries * n):
            p = rng.uniform(-1, 1, 2)
            if p[0] ** 2 + p[1] ** 2 > 1:
                continue
            q = np.array([p[0] * ay, p[1] * az])
            if all(np.hypot(*(q - o)) >= r_min for o in pts):
                pts.append(q)
                if len(pts) == n:
                    return np.asarray(pts)
        r_min *= 0.97


def make_array(seed: int = 0, err_seed: int = 1, pos_err: float = 0.15e-3, gain_err_db: float = 0.8,
               phase_err_deg: float = 5.0) -> dict:
    """Nominal + true array. Positions [M, 3] in the sensor frame (x = 0 plane)."""
    yz = poisson_disc_ellipse(seed=seed)
    nominal = np.c_[np.zeros(len(yz)), yz]
    rng = np.random.default_rng(err_seed)
    true = nominal + np.c_[np.zeros(len(yz)), pos_err * rng.standard_normal((len(yz), 2))]
    gain = 10 ** (gain_err_db * rng.standard_normal(len(yz)) / 20)
    phase = np.radians(phase_err_deg) * rng.standard_normal(len(yz))
    dmin = min(np.hypot(*(yz[i] - yz[j])) for i in range(len(yz)) for j in range(i))
    return {"nominal": nominal.tolist(), "true": true.tolist(), "gain": gain.tolist(), "phase": phase.tolist(),
            "emitter": list(EMITTER_POS), "emitter_a": EMITTER_A, "aperture": list(APERTURE), "seed": seed,
            "err_seed": err_seed, "min_dist_m": dmin}


def load_array() -> dict:
    """The fixed array of this project (data/array.json; created on first use)."""
    if not ARRAY_FILE.exists():
        ARRAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARRAY_FILE.write_text(json.dumps(make_array(), indent=1))
    a = json.loads(ARRAY_FILE.read_text())
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in a.items()}
