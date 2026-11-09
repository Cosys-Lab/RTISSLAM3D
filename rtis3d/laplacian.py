"""Graph Laplacian on the sampled directions, heat-kernel smoothing and a CFAR detector on the sphere.

The heat kernel exp(-t L) of the Laplace-Beltrami operator is the natural "local mean" on the sphere (its spherical-
harmonics form multiplies degree l by exp(-t l(l+1))). On the sampled hemisphere it is replaced by the random-walk graph
Laplacian of a k-nearest-neighbour graph, L = I - D^-1 W, whose heat kernel is row-stochastic (a weighted local mean that
preserves constants) and needs no special treatment at the hemisphere's edge. The edge weights use a metric matched to
the beam: elevation distances divided by `el_scale` (the beam is ~1.7x wider in elevation than in azimuth), so the
smoothing follows the point-spread function instead of being isotropic.

    H = HeatKernel(U, sigma_deg=10, el_scale=1.7)       # dense [D, D], precomputed once (eigendecomposition)
    B = H.smooth(P)                                     # P [R, D] linear power -> local mean over directions per range
    det = sphere_cfar(P, H, gain=2.0, range_sigma=3.0)  # P+ = max(P - gain * background, 0)

sigma_deg -> infinity: the background becomes the mean over all directions per range, i.e. the range-wise CFAR (RW-CFAR)
of RTISSLAM; small sigma_deg: a local cell-averaging CFAR on the sphere.
"""
from __future__ import annotations

import math

import numpy as np

from .sphere import az_el


def knn_weights(U: np.ndarray, k: int = 10, el_scale: float = 1.0):
    """Symmetric kNN weight matrix [D, D] (dense) with Gaussian weights of the beam-matched angular distance."""
    az, el = az_el(U)
    D = len(U)
    daz = (az[None] - az[:, None] + np.pi) % (2 * np.pi) - np.pi
    d_az = daz * np.cos(0.5 * (el[None] + el[:, None]))
    d_el = (el[None] - el[:, None]) / el_scale
    d2 = d_az ** 2 + d_el ** 2
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    h2 = np.median(d2[np.arange(D)[:, None], nn])          # bandwidth: median kNN distance
    W = np.zeros((D, D))
    rows = np.repeat(np.arange(D), k)
    W[rows, nn.ravel()] = np.exp(-d2[rows, nn.ravel()] / h2)
    return np.maximum(W, W.T)


class HeatKernel:
    def __init__(self, U: np.ndarray, sigma_deg: float = 10.0, el_scale: float = 1.0, k: int = 10, t: float | None = None):
        self.U, self.el_scale = np.asarray(U, float), el_scale
        W = knn_weights(self.U, k, el_scale)
        d = W.sum(1)
        S = W / np.sqrt(d[:, None] * d[None])                # D^-1/2 W D^-1/2
        lam, V = np.linalg.eigh(np.eye(len(d)) - S)          # symmetric normalized Laplacian
        self._lam, self._V, self._d = np.clip(lam, 0, None), V, d
        self.t = t if t is not None else self.t_for_sigma(sigma_deg)
        self.K = self.kernel(self.t)
        self.sigma_deg = self.spread_deg(self.K)

    def kernel(self, t: float) -> np.ndarray:
        """Random-walk heat kernel exp(-t (I - D^-1 W)) = D^-1/2 V exp(-t lam) V^T D^1/2 (row-stochastic)."""
        V, s = self._V, np.sqrt(self._d)
        return (V * np.exp(-t * self._lam)) @ V.T * (s[None] / s[:, None])

    def row(self, t: float, i: int) -> np.ndarray:
        """One row of the kernel (O(D^2) instead of O(D^3))."""
        V, s = self._V, np.sqrt(self._d)
        return (V[i] * np.exp(-t * self._lam)) @ V.T * (s / s[i])

    def spread_deg(self, K, i: int | None = None) -> float:
        """RMS angular radius (deg) of a kernel row (K: the full kernel, or one row), default the broadside direction."""
        if i is None:
            i = int(np.argmax(self.U[:, 0]))
        w = np.clip(K[i] if np.ndim(K) == 2 else K, 0, None)
        ang = np.arccos(np.clip(self.U @ self.U[i], -1, 1))
        return float(np.degrees(np.sqrt(np.sum(w * ang ** 2) / w.sum())))

    def t_for_sigma(self, sigma_deg: float) -> float:
        lo, hi = 1e-3, 1e5
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            i0 = int(np.argmax(self.U[:, 0]))
            if self.spread_deg(self.row(mid, i0), i0) < sigma_deg:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def smooth(self, P):
        """[..., D] -> [..., D] local weighted mean over directions (numpy or torch)."""
        try:
            import torch
            if torch.is_tensor(P):
                K = getattr(self, "_Kt", None)
                if K is None or K.device != P.device:
                    self._Kt = K = torch.as_tensor(self.K.T, dtype=P.dtype, device=P.device)
                return P @ K
        except ImportError:
            pass
        return P @ self.K.T


def sphere_cfar(P, H: HeatKernel, gain: float = 2.0, range_sigma: float = 3.0):
    """CFAR on the sphere: background = heat-kernel local mean over directions per range, then Gaussian-smoothed along
    range (sigma in range cells, as RTISSLAM's smoothed RW-CFAR); P+ = max(P - gain * background, 0). P [R, D] linear
    (numpy or torch on the GPU)."""
    import torch
    t = torch.as_tensor(P, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu") if not torch.is_tensor(P) else P
    B = H.smooth(t)
    if range_sigma > 0:
        n = int(math.ceil(3 * range_sigma))
        g = torch.exp(-0.5 * (torch.arange(-n, n + 1, device=t.device, dtype=t.dtype) / range_sigma) ** 2)
        g = (g / g.sum()).view(1, 1, -1)
        B = torch.nn.functional.conv1d(torch.nn.functional.pad(B.T.unsqueeze(1), (n, n), mode="replicate"), g).squeeze(1).T
    out = torch.clamp(t - gain * B, min=0)
    return out if torch.is_tensor(P) else out.cpu().numpy()
