# Vendored into RTISSLAM-3D from RTISSLAM core (sondec), commit 09706bf, BSD-3-Clause.
"""eRTIS image formation (the fixed "observation" pipeline): matched filter, delay-and-sum
beamforming for D directions of interest, envelope detection, range cells.

    x_m(t)  --rfft-->  X_m(f)
    matched filter:    X_m(f) conj(CALL(f)) / sum|CALL|^2           (band F_LO - 2 kHz .. F_HI + 2 kHz)
    DAS (far field, NOMINAL array):  Y_d(f) = (1/M) sum_m X_m(f) exp(-i 2 pi f p_m . u_d / c)
                                     (microphone m is delayed by p_m . u_d / c: an exact fractional
                                      delay, i.e. time-domain DAS done in the frequency domain)
    envelope:          |analytic y_d(t)|^2, computed at complex baseband (fs_bb = L df)
    energyscape:       mean envelope power per range cell, r = c t / 2:  P [R, D]

Mirrors the published eRTIS chain (PDM demodulation -> matched filter -> DAS -> envelope); PDM is not
simulated. The beamformer uses the NOMINAL microphone positions; the true array differs (sensor.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .sensor import C_SOUND, F_HI, F_LO, FS, chirp, load_array
from .sim.echoes import NSIG

NFFT = 18432            # >= NSIG + call length: no circular wrap of the matched filter
L_BB = 8192             # baseband ifft length -> fs_bb = L_BB FS / NFFT = 200 kHz (1.7 mm in range)
BAND = (F_LO - 2e3, F_HI + 2e3)


@dataclass
class RangeGrid:
    r_min: float = 0.2
    dr: float = 0.02
    n_r: int = 240

    def r(self, device="cuda") -> torch.Tensor:
        return self.r_min + (torch.arange(self.n_r, device=device) + 0.5) * self.dr


class DAS:
    def __init__(self, dirs: torch.Tensor, grid: RangeGrid = RangeGrid(), device="cuda", positions=None):
        """dirs [D, 3] directions of interest; positions [M, 3] (default: the NOMINAL array)."""
        self.dev, self.grid = device, grid
        arr = load_array()
        p = np.asarray(arr["nominal"] if positions is None else positions)
        self.M, self.D = len(p), dirs.shape[0]
        f = np.fft.rfftfreq(NFFT, 1 / FS)
        self.k0, self.k1 = int(np.ceil(BAND[0] / f[1])), int(np.floor(BAND[1] / f[1])) + 1
        fb = f[self.k0:self.k1]
        self.fb = torch.as_tensor(fb, dtype=torch.float32, device=device)
        C = np.fft.rfft(chirp(), NFFT)[self.k0:self.k1]
        self.MF = torch.as_tensor(np.conj(C) / np.sum(np.abs(C) ** 2) * NFFT, dtype=torch.complex64, device=device)
        pu = torch.as_tensor(p, dtype=torch.float32, device=device) @ dirs.T.float()     # [M, D] p_m . u_d
        ph = -2 * math.pi * self.fb[:, None, None] * pu[None] / C_SOUND                  # [K, M, D]
        self.W = torch.polar(torch.full_like(ph, 1.0 / self.M), ph)                       # [K, M, D]
        assert self.k1 - self.k0 <= L_BB
        self.fs_bb = L_BB * FS / NFFT
        # range cell edges in baseband samples (two-way time from the emitter/array)
        edges = grid.r_min + grid.dr * np.arange(grid.n_r + 1)
        self.edges = torch.as_tensor(np.round(edges * 2 / C_SOUND * self.fs_bb).astype(np.int64), device=device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        """x [B, M, NSIG] microphone signals -> P [B, R, D] energyscape (linear envelope power)."""
        out = []
        for i0 in range(0, x.shape[0], chunk):
            X = torch.fft.rfft(x[i0:i0 + chunk], NFFT)[..., self.k0:self.k1] * self.MF    # [b, M, K]
            Y = torch.bmm(X.permute(2, 0, 1), self.W)                                     # [K, b, D]
            Z = torch.zeros(X.shape[0], self.D, L_BB, dtype=torch.complex64, device=self.dev)
            Z[..., :Y.shape[0]] = Y.permute(1, 2, 0)
            env = torch.fft.ifft(Z, dim=-1) * 2                                           # analytic, baseband
            p = env.real ** 2 + env.imag ** 2                                             # [b, D, L]
            del Z, env
            cs = torch.nn.functional.pad(torch.cumsum(p.double(), -1), (1, 0))
            a, b = self.edges[:-1], self.edges[1:]
            out.append(((cs[..., b] - cs[..., a]) / (b - a).double()).float().transpose(1, 2))
        return torch.cat(out)
