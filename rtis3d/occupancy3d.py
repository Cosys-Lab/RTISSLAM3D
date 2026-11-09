"""3D log-odds occupancy grid from eRTIS detection sets, on the GPU:

  occupied evidence   every detection adds l_occ * weight, spread as its beam footprint: samples of a Gaussian that is
                      thin along the ray (sigma_r) and wide across it (r * sigma_az, r * sigma_el), so that views from
                      many positions concentrate on the real surfaces
  free space          along every direction, the voxels from r_min up to the first echo minus `gap` get -l_free
                      ("empty up to the first echo"); directions without any echo are carved only up to r_empty
                      (a specular surface seen at grazing incidence gives no echo, so no echo does not mean empty)
  result              L (log-odds, clipped), with voxel centres; occupied = L >= occ_thresh

    G = Occupancy3D(bounds_min, bounds_max, res=0.1)
    G.add(T_world_sensor, pts_sensor, w, U)        # one capture: detections (sensor frame) and the direction set U
    L = G.L                                        # [nx, ny, nz]
"""
from __future__ import annotations

import math

import numpy as np
import torch


class Occupancy3D:
    def __init__(self, lo, hi, res=0.1, l_occ=0.6, l_free=0.12, l_min=-2.0, l_max=5.0, sig_r=0.03, sig_az_deg=3.0,
                 sig_el_deg=5.0, samples=24, r_min=0.5, r_empty=1.0, r_free_max=6.0, gap=0.15, step=None, device="cuda"):
        self.lo, self.res = np.asarray(lo, float), res
        self.n = np.ceil((np.asarray(hi, float) - self.lo) / res).astype(int)
        self.L = torch.zeros(tuple(self.n), dtype=torch.float32, device=device)
        self.dev = device
        self.l_occ, self.l_free, self.l_min, self.l_max = l_occ, l_free, l_min, l_max
        self.sig = (sig_r, math.radians(sig_az_deg), math.radians(sig_el_deg))
        self.K, self.r_min, self.r_empty, self.r_free_max, self.gap = samples, r_min, r_empty, r_free_max, gap
        self.step = step or res * 0.8
        g = torch.Generator(device="cpu").manual_seed(0)
        self.eps = torch.randn(samples, 3, generator=g).to(device)              # fixed footprint samples

    def _index(self, P):
        ijk = torch.floor((P - torch.as_tensor(self.lo, dtype=P.dtype, device=P.device)) / self.res).long()
        n = torch.as_tensor(self.n, device=P.device)
        ok = ((ijk >= 0) & (ijk < n)).all(1)
        ijk = ijk[ok]
        return (ijk[:, 0] * self.n[1] + ijk[:, 1]) * self.n[2] + ijk[:, 2], ok

    def add(self, T, pts, w, U, dir_index=None):
        dev = self.dev
        R = torch.as_tensor(T[:3, :3], dtype=torch.float32, device=dev)
        t = torch.as_tensor(T[:3, 3], dtype=torch.float32, device=dev)
        Lf = self.L.view(-1)
        # ---- free space: per direction, up to the first echo - gap (or r_empty without echo)
        Ut = torch.as_tensor(U, dtype=torch.float32, device=dev)
        first = torch.full((len(U),), float("inf"), device=dev)
        if len(pts):
            p = torch.as_tensor(pts, dtype=torch.float32, device=dev)
            r = torch.linalg.norm(p, dim=1)
            di = torch.as_tensor(dir_index, device=dev) if dir_index is not None else torch.argmax((p / r[:, None]) @ Ut.T, dim=1)
            first = first.scatter_reduce(0, di, r, reduce="amin")
        reach = torch.where(torch.isfinite(first), torch.clamp(first - self.gap, max=self.r_free_max),
                            torch.full_like(first, self.r_empty))
        s = torch.arange(self.r_min, self.r_free_max, self.step, device=dev)
        m = s[None, :] < reach[:, None]                                             # [D, S]
        dd, ss = torch.nonzero(m, as_tuple=True)
        P = (Ut[dd] * s[ss, None]) @ R.T + t
        idx, _ = self._index(P)
        if len(idx):
            idx = torch.unique(idx)                                                 # one vote per voxel per capture
            Lf.index_add_(0, idx, torch.full((len(idx),), -self.l_free, device=dev))
        # ---- occupied: beam footprints of the detections
        if len(pts):
            u = p / r[:, None]
            e_az = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(r)], 1)
            e_az = e_az / (torch.linalg.norm(e_az, dim=1, keepdim=True) + 1e-9)
            e_el = torch.cross(u, e_az, dim=1)
            sr, sa, se = self.sig
            off = (self.eps[None, :, 0:1] * sr * u[:, None] + self.eps[None, :, 1:2] * (r[:, None, None] * sa) * e_az[:, None]
                   + self.eps[None, :, 2:3] * (r[:, None, None] * se) * e_el[:, None])          # [n, K, 3]
            Q = (p[:, None] + off).reshape(-1, 3) @ R.T + t
            ww = torch.as_tensor(w, dtype=torch.float32, device=dev)
            ww = (0.4 + 0.6 * torch.clamp(torch.log10(ww / ww.max() + 1e-12) / 2.5 + 1, 0, 1))  # 25 dB -> 0.4..1
            vals = (self.l_occ / self.K * ww)[:, None].expand(-1, self.K).reshape(-1)
            idx, ok = self._index(Q)
            Lf.index_add_(0, idx, vals[ok])
        self.L.clamp_(self.l_min, self.l_max)

    def centres(self, mask):
        ijk = torch.nonzero(mask).float()
        return (ijk.cpu().numpy() + 0.5) * self.res + self.lo
