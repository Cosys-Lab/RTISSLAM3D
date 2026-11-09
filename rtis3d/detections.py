"""Per-capture 3D detection sets in the sensor frame (no poses needed), the input of the 3D acoustic odometry, the
3D place recognition and the 3D map.

    det = Detector(U, ranges, sigma_deg=12, gain=5, peak_db=25)
    pts, w, info = det(volume_db)      # pts [n, 3] sensor frame (x fwd, y left, z up), w [n] CFAR excess (linear)

Steps: CFAR on the sphere (heat kernel, rtis3d.laplacian), cells within peak_db of the capture's strongest output,
r in [r_min, r_max]; then a ground plane fitted to the near-field ground echoes (robust least squares, plane
n . x = d with n near "up"), and the detections more than `below` m under it dropped: the ground-multipath mirrors.
The plane also gives the sensor's height above the ground and its tilt relative to the local ground.
"""
from __future__ import annotations

import numpy as np

from .laplacian import HeatKernel, sphere_cfar


class Detector:
    def __init__(self, U, ranges, sigma_deg=12.0, gain=5.0, peak_db=25.0, r_min=0.5, r_max=8.0, el_scale=1.7,
                 below=0.15, h_prior=0.32, device="cuda"):
        import torch
        self.U, self.r = np.asarray(U, np.float32), np.asarray(ranges, np.float32)
        self.H = HeatKernel(self.U, sigma_deg=sigma_deg, el_scale=el_scale)
        self.gain, self.peak, self.below, self.h_prior = gain, 10 ** (-peak_db / 10), below, h_prior
        self.keep_r = (self.r >= r_min) & (self.r <= r_max)
        self.pos = torch.as_tensor(self.r[:, None, None] * self.U[None], device=device)          # [R, D, 3]
        self.dev = device

    def ground_plane(self, P, w):
        """Robust plane through the low near-field detections (sensor frame). -> (n [3], d) with n . x = d, or None."""
        r = np.linalg.norm(P, axis=1)
        m = (P[:, 2] < -0.1) & (r < 3.0) & (P[:, 2] > -self.h_prior - 0.6)
        if m.sum() < 15:
            return None
        X, ww = P[m], np.sqrt(w[m])
        n, d = np.array([0, 0, 1.0]), -self.h_prior
        for _ in range(6):                                                       # iteratively reweighted (Huber-like)
            res = X @ n - d
            k = 0.08
            wr = ww * np.where(np.abs(res) < k, 1.0, k / np.abs(res))
            c = (wr[:, None] * X).sum(0) / wr.sum()
            C = (wr[:, None] * (X - c)).T @ (X - c)
            evals, evecs = np.linalg.eigh(C)
            n = evecs[:, 0] * np.sign(evecs[2, 0])
            d = float(n @ c)
        if n[2] < np.cos(np.radians(30)):                                        # not a ground-like plane
            return None
        return n, d

    def __call__(self, vol_db):
        import torch
        P = torch.as_tensor(10 ** (np.asarray(vol_db, np.float32) / 10), device=self.dev)
        Q = sphere_cfar(P, self.H, gain=self.gain)
        Q[~torch.as_tensor(self.keep_r, device=self.dev)] = 0
        thr = Q.max() * self.peak
        ri, di = torch.nonzero(Q > thr, as_tuple=True)
        pts = self.pos[ri, di].cpu().numpy()
        w = Q[ri, di].cpu().numpy()
        info = {"n_raw": int(len(pts)), "plane": None, "height": None, "n_mirror": 0}
        pl = self.ground_plane(pts, w)
        if pl is not None:
            n, d = pl
            below = pts @ n - d < -self.below
            info.update(plane=(n.tolist(), d), height=float(-d), n_mirror=int(below.sum()))
            pts, w = pts[~below], w[~below]
        return pts.astype(np.float32), w.astype(np.float32), info
