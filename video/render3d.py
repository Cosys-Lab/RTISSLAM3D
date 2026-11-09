"""Small perspective renderer for the 3D videos: point splats (far to near, depth fog) and anti-aliased 3D lines
(clipped at the near plane), camera given as eye [3] and Rcw [3, 3] (rows: forward, left, up in the world).

    V = View(W, H, fov_deg)
    img = V.points(P, C, size_m, eye, Rcw)                     # new image
    V.lines(img, A, B, colour, px, eye, Rcw)                    # segments A[i] -> B[i] drawn on top
    V.triads(img, T [n, 3, 4], length_m, px, eye, Rcw)          # x red, y green, z blue axes of 6-DoF poses
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def look_at(eye, target, up=(0, 0, 1.0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    l = np.cross(up, f); l /= np.linalg.norm(l)
    return np.stack([f, l, np.cross(f, l)])


class View:
    def __init__(self, W, H, fov_deg=60.0, near=0.3, bg=255):
        self.W, self.H, self.near, self.bg = W, H, near, bg
        self.f = W / 2 / math.tan(math.radians(fov_deg / 2))

    def _proj(self, q):
        return self.W / 2 - self.f * q[:, 1] / q[:, 0], self.H / 2 - self.f * q[:, 2] / q[:, 0]

    def points(self, P, C, size, eye, Rcw, rmax=4, fog=None, scale=None, img=None):
        img = np.full((self.H, self.W, 3), self.bg, np.uint8) if img is None else img
        if len(P) == 0:
            return img
        q = (P - eye) @ Rcw.T
        m = q[:, 0] > self.near
        q, C = q[m], C[m].astype(np.float32)
        sc = scale[m] if scale is not None else 1.0
        u, v = self._proj(q)
        u, v = np.round(u).astype(int), np.round(v).astype(int)
        r = np.clip(np.round(size * sc * self.f / q[:, 0]).astype(int), 0, rmax)
        ok = (u > -rmax) & (u < self.W + rmax) & (v > -rmax) & (v < self.H + rmax)
        u, v, r, C, dep = u[ok], v[ok], r[ok], C[ok], q[ok, 0]
        if fog:
            a = np.exp(-dep / fog)[:, None]
            C = a * C + (1 - a) * self.bg
        C = C.astype(np.uint8)
        o = np.argsort(-dep)
        u, v, r, C = u[o], v[o], r[o], C[o]
        for du in range(-rmax, rmax + 1):
            for dv in range(-rmax, rmax + 1):
                s = r >= max(abs(du), abs(dv))
                uu, vv = u[s] + du, v[s] + dv
                k = (uu >= 0) & (uu < self.W) & (vv >= 0) & (vv < self.H)
                img[vv[k], uu[k]] = C[s][k]
        return img

    def lines(self, img, A, B, colour, px, eye, Rcw):
        if len(A) == 0:
            return img
        a, b = (np.asarray(A) - eye) @ Rcw.T, (np.asarray(B) - eye) @ Rcw.T
        ia, ib = a[:, 0] < self.near, b[:, 0] < self.near
        keep = ~(ia & ib)
        a, b, ia, ib = a[keep], b[keep], ia[keep], ib[keep]
        t = np.where(ia | ib, (self.near - a[:, 0]) / (b[:, 0] - a[:, 0] + 1e-12), 0)[:, None]
        cut = a + t * (b - a)
        a = np.where(ia[:, None], cut, a); b = np.where(ib[:, None], cut, b)
        ua, va = self._proj(a); ub, vb = self._proj(b)
        col = tuple(int(c) for c in colour)
        sh = 4
        for x0, y0, x1, y1 in zip(ua, va, ub, vb):
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1e5:
                continue
            cv2.line(img, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), col, px, cv2.LINE_AA, sh)
        return img

    def polyline(self, img, P, colour, px, eye, Rcw):
        P = np.asarray(P)
        P = P[np.isfinite(P).all(1)]
        return self.lines(img, P[:-1], P[1:], colour, px, eye, Rcw) if len(P) > 1 else img

    def triads(self, img, T, length, px, eye, Rcw):
        T = np.asarray(T)
        if len(T) == 0:
            return img
        o = T[:, :3, 3]
        for ax, col in ((0, (220, 30, 30)), (1, (30, 170, 30)), (2, (30, 60, 230))):
            self.lines(img, o, o + length * T[:, :3, ax], col, px, eye, Rcw)
        return img
