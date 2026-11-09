# Vendored into RTISSLAM-3D from RTISSLAM core (sondec), commit 09706bf, BSD-3-Clause.
"""Coherent echo synthesis for the 32-microphone eRTIS array (GPU), deliberately richer than the
deconvolver's forward model.

Per pulse and microphone m, in the frequency domain:

    X_m(f) = CALL(f) a_m e^{i phi_m} sum_n sqrt(sigma_n) g_obj(f) D_em(u_e,n, f) 10^(-alpha(f)(r1+r2)/20) / (r1 r2)
                                      x exp(-i 2 pi f (r1 + r2) / c)

r1 = emitter -> point, r2 = point -> microphone m (exact spherical wavefronts: near field included),
u_e,n = direction of the point seen from the emitter (piston directivity D_em), a_m e^{i phi_m} the TRUE
microphone gain and phase error, the TRUE (perturbed) microphone positions. sigma_n as in worlds.py
(area (d cos^2 + s cos^2p), incidence from the sensor). Points hidden behind occluders are dropped.
Then x_m(t) = irfft(X_m) at FS = 450 kHz, plus white sensor noise.

The world, occlusion and backscatter code is the Recognizer's (bat3d/sim/echoes.py), unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..sensor import (C_SOUND, FS, absorption_db_per_m, chirp, emitter_axis, emitter_directivity_v2,
                      emitter_response_db, load_array, piston_directivity)

MAX_R = 6.0           # m; reflectors beyond are not synthesised
NSIG = 16384          # samples (36.4 ms, ~6.2 m of range): the eRTIS recording window
BLOCK = 2048          # samples of delay per synthesis block
NB = 4096             # synthesis grid per block (block + 2.5 ms call + tails)
F_SYN = (17e3, 83e3)  # Hz, synthesised band (the call has no energy outside)


def _seg_seg_dist2(A, d1, C, d2):
    """Squared distance between segments A + s d1 and C + t d2 (s, t in [0, 1]); broadcast [..., 3]."""
    r = A - C
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    b = (d1 * d2).sum(-1); c = (d1 * r).sum(-1)
    den = (a * e - b * b).clamp_min(1e-12)
    s = ((b * f - c * e) / den).clamp(0, 1)
    t = (b * s + f) / e.clamp_min(1e-12)
    t_c = t.clamp(0, 1)
    s = torch.where(t != t_c, ((b * t_c - c) / a.clamp_min(1e-12)).clamp(0, 1), s)
    p = A + s[..., None] * d1 - C - t_c[..., None] * d2
    return (p * p).sum(-1)


class ArraySynth:
    def __init__(self, world: dict, device="cuda", noise: float = 1e-4, min_range: float = 0.1,
                 exclude: tuple = (), spread_exp: float = 4.0, emitter: str = "v1"):
        """emitter: "v1" (3 mm piston on x, flat transmit response) | "v2" (realistic: 4.5 mm piston, axis
        misaligned, band-pass transmit response with ripple; sensor.EMITTER_V2)."""
        self.dev, self.noise = device, noise
        self.min_range, self.exclude, self.spread_exp = float(min_range), tuple(exclude), float(spread_exp)
        arr = load_array()
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
        self.mic = T(arr["true"])                                            # [M, 3] TRUE positions
        self.mic_g = T(arr["gain"] * np.exp(1j * arr["phase"]), torch.complex64)
        self.emitter = T(arr["emitter"])
        self.emitter_a = float(arr["emitter_a"])
        self.em_axis = T(np.array([1.0, 0.0, 0.0]))
        self.em_kind = emitter
        if emitter in ("v2", "v3"):
            self.em_axis = T(emitter_axis())
        self.M = len(arr["true"])
        f = np.fft.rfftfreq(NB, 1 / FS)
        self.kA, self.kB = int(np.ceil(F_SYN[0] / f[1])), int(np.floor(F_SYN[1] / f[1])) + 1
        fb = f[self.kA:self.kB]
        self.fb_np = fb
        self.fb = T(fb)
        self.kint = torch.arange(self.kA, self.kB, device=device)
        self.kflt = self.kint.float()
        self.alpha = T(absorption_db_per_m(fb))
        self.call = chirp()
        tx = (10 ** (emitter_response_db(fb, wavy_db=10.0 if emitter == "v3" else 0.0) / 20) if emitter in ("v2", "v3")
              else np.ones_like(fb))
        self.CALL = T(np.fft.rfft(self.call, NB)[self.kA:self.kB] * tx, torch.complex64)   # acoustic output
        # emitter directivity on a cos-theta table (1001 x nb), linear interpolation
        self.ct = np.linspace(0, 1, 1001)
        self.Dem = T(emitter_directivity_v2(self.ct, fb) if emitter in ("v2", "v3") else piston_directivity(self.ct, fb, self.emitter_a))
        self.set_world(world)

    def set_world(self, w: dict):
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)  # noqa: E731
        self.pos, self.nrm, self.area, self.mat = T(w["pos"]), T(w["normal"]), T(w["area"]), T(w["mat"])
        self.two = T(w["two"], torch.bool)
        self.obj = T(w["obj"], torch.long)
        self.iso = self.nrm.abs().sum(-1) == 0
        size, tilt = T(w["obj_size"]), T(w["obj_tilt"])
        ka = 2 * math.pi * self.fb[None] / C_SOUND * size[:, None]
        ray = torch.where(torch.isfinite(ka), ka ** 2 / (1 + ka ** 2), torch.ones_like(ka))
        self.gobj = ray * torch.pow(10.0, tilt[:, None] * (self.fb[None] - 60e3) / 60e3 / 20)   # [n_obj, nb]
        self.boxes, self.caps = T(w["boxes"]), T(w["capsules"])
        names = np.asarray(w.get("obj_name", np.array([], dtype=str)))
        drop = np.isin(names, list(self.exclude)) if len(names) else np.zeros(0, bool)
        self.excl = (T(drop, torch.bool)[self.obj] if drop.any()
                     else torch.zeros(len(self.pos), dtype=torch.bool, device=self.dev))

    def visible(self, p: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """p [N, 3] points, o [3] sensor -> bool [N]: the segment o -> p (shortened by 1 cm) hits no occluder."""
        N = p.shape[0]
        vis = torch.ones(N, dtype=torch.bool, device=self.dev)
        d = p - o
        L = d.norm(dim=-1, keepdim=True)
        dd = d * (1 - 0.01 / L.clamp_min(0.02))
        for i0 in range(0, N, 16384):
            sl = slice(i0, i0 + 16384)
            seg = dd[sl]
            if len(self.boxes):
                b = self.boxes
                c, h, yaw = b[:, :3], b[:, 3:6], b[:, 6]
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                o_rel = o[None] - c
                ol = torch.stack([cy * o_rel[:, 0] + sy * o_rel[:, 1], -sy * o_rel[:, 0] + cy * o_rel[:, 1], o_rel[:, 2]], -1)
                sx, sy_, sz = seg[:, 0:1], seg[:, 1:2], seg[:, 2:3]
                dl = torch.stack([cy[None] * sx + sy[None] * sy_, -sy[None] * sx + cy[None] * sy_, sz.expand(-1, len(b))], -1)
                inv = 1.0 / torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
                t1 = (-h[None] - ol[None]) * inv; t2 = (h[None] - ol[None]) * inv
                tmin = torch.minimum(t1, t2).amax(-1).clamp_min(0.0)
                tmax = torch.maximum(t1, t2).amin(-1).clamp_max(1.0)
                vis[sl] &= ~(tmin < tmax).any(-1)
            if len(self.caps):
                cp = self.caps
                A = o[None, None].expand(seg.shape[0], len(cp), 3)
                d2 = _seg_seg_dist2(A, seg[:, None], cp[None, :, :3], (cp[:, 3:6] - cp[:, :3])[None])
                vis[sl] &= ~(d2 < (cp[:, 6] ** 2)[None]).any(-1)
        return vis

    def _emitter_gain(self, cos_e: torch.Tensor) -> torch.Tensor:
        x = cos_e.clamp(0, 1) * (len(self.ct) - 1)
        i0 = x.floor().long().clamp(0, len(self.ct) - 2)
        w = (x - i0)[:, None]
        return self.Dem[i0] * (1 - w) + self.Dem[i0 + 1] * w

    @torch.no_grad()
    def pulse(self, R: torch.Tensor, t: torch.Tensor, top_k: int = 512, points: dict | None = None):
        """Sensor pose (R_WS [3, 3], t_WS [3]) -> (x [M, NSIG] received signals without noise,
        contributions dict for evaluation: the top_k echoes by range-compensated power, with range r and
        head-frame direction u from the array centre).
        points: optional {"p": [N, 3] sensor-frame positions, "amp": [N]} instead of the world (tests)."""
        if points is None:
            v = self.pos - t
            r0 = v.norm(dim=-1)
            u = (v @ R) / r0[:, None].clamp_min(1e-6)                        # sensor frame
            keep = (r0 > self.min_range) & (r0 < MAX_R) & (u[:, 0] > 0) & ~self.excl
            idx = torch.nonzero(keep)[:, 0]
            idx = idx[self.visible(self.pos[idx], t)]
            ps = (self.pos[idx] - t) @ R                                     # [n, 3] sensor frame
            v, r0 = v[idx], r0[idx]
            cosi = -(self.nrm[idx] * v).sum(-1) / r0
            cosi = torch.where(self.two[idx], cosi.abs(), cosi.clamp_min(0))
            m = self.mat[idx]
            sig = torch.where(self.iso[idx], m[:, 0], m[:, 0] * cosi ** 2 + m[:, 1] * cosi.clamp_min(1e-6) ** (2 * m[:, 2]))
            amp0 = torch.sqrt(self.area[idx] * sig)
            gob = self.gobj[self.obj[idx]]
        else:
            ps = torch.as_tensor(points["p"], dtype=torch.float32, device=self.dev)
            amp0 = torch.as_tensor(points["amp"], dtype=torch.float32, device=self.dev)
            idx = torch.arange(len(ps), device=self.dev)
            gob = torch.ones(len(ps), len(self.fb), device=self.dev)
        r0 = ps.norm(dim=-1)
        u = ps / r0[:, None]
        ve = ps - self.emitter
        r1 = ve.norm(dim=-1)
        Dem = self._emitter_gain((ve @ self.em_axis) / r1)                   # [n, nb] (angle from the emitter axis)
        base = amp0[:, None] * gob * Dem                                     # [n, nb]
        nblk = NSIG // BLOCK + 1
        acc = torch.zeros(nblk * self.M, len(self.fb), dtype=torch.complex64, device=self.dev)
        mid = torch.arange(self.M, device=self.dev)
        powr = torch.zeros(len(ps), device=self.dev)
        CH = max(1, 12_000_000 // (self.M * len(self.fb)))
        for i0 in range(0, len(ps), CH):
            sl = slice(i0, i0 + CH)
            r2 = (ps[sl, None] - self.mic[None]).norm(dim=-1)                # [c, M]
            rt = r1[sl, None] + r2
            A = (base[sl, None] * torch.pow(10.0, -self.alpha[None, None] * rt[..., None] / 20)
                 / ((r1[sl, None] * r2) ** (self.spread_exp / 4))[..., None])   # [c, M, nb]
            powr[sl] = (A[:, 0] ** 2).mean(-1)
            tau = rt.double() / C_SOUND * FS
            blk = (tau / BLOCK).floor().long().clamp_max(nblk - 1)
            rel = tau - blk.double() * BLOCK
            m_int = rel.floor().long(); frac = (rel - rel.floor()).float()
            ph = (torch.remainder(m_int[..., None] * self.kint, NB).float() + frac[..., None] * self.kflt) * (2 * math.pi / NB)
            Z = torch.polar(A, -ph)                                          # [c, M, nb]
            acc.index_add_(0, (blk * self.M + mid).flatten(), Z.flatten(0, 1))
        acc = acc.view(nblk, self.M, -1) * (self.CALL * self.mic_g[:, None])
        Xb = torch.zeros(nblk, self.M, NB // 2 + 1, dtype=torch.complex64, device=self.dev)
        Xb[..., self.kA:self.kB] = acc
        xb = torch.fft.irfft(Xb, NB)                                         # [nblk, M, NB]
        x = torch.zeros(self.M, (nblk + 1) * BLOCK + NB, dtype=torch.float32, device=self.dev)
        for b in range(nblk):
            x[:, b * BLOCK: b * BLOCK + NB] += xb[b]
        x = x[:, :NSIG]
        k = min(top_k, len(ps))
        top = torch.topk(powr * r0 ** self.spread_exp, k).indices if k else torch.zeros(0, dtype=torch.long, device=self.dev)
        contrib = {"idx": idx[top].int(), "r": r0[top], "u": u[top], "pow": powr[top]}
        return x, contrib

    @torch.no_grad()
    def add_noise(self, x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        return x + self.noise * torch.randn(x.shape, device=x.device, generator=gen)
