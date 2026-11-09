"""3D place recognition for the eRTIS: keyframe submaps, a height-banded polar descriptor with yaw by circular shift,
and verification by 4-DoF submap-to-submap registration (roll / pitch are known from the IMU, so every submap is
expressed level, in a heading-only frame at the keyframe).

    kf = Keyframe.build(points_world, weights, T_kf)       # submap (world points of the last few metres) -> level frame
    s, shift = descriptor_match(kf_a.desc, kf_b.desc)        # similarity in [0, 1], yaw of b relative to a (sectors)
    T, info = register4(kf_q.pts, kf_q.w, kf_a.pts, yaw0)    # q -> a, 4-DoF (x, y, z, yaw); info: inliers, rmse, H

Descriptor: rings from r_min to r_max (dr), sectors of 360/n_sec deg, by default one height band from below the ground to
2.6 m (optionally four bands: ground, low, mid, high; they retrieved worse in the forest); cell value = sum of sqrt(weight) of the submap points in the cell, compressed (^gamma) and normalised
per band; similarity = weighted mean over the bands of the circular cross-correlation, best shift taken jointly.
Ring key (sector sums) for a rotation-invariant prefilter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

BANDS = ((-0.35, 0.25), (0.25, 0.75), (0.75, 1.5), (1.5, 2.6))   # height above the base frame (base on the ground)
BAND_W = (0.5, 1.0, 1.0, 1.0)


def rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def level_frame(T):
    """Heading-only frame at T (4x4, world <- base): same position, yaw of T's x axis, no roll / pitch."""
    yaw = math.atan2(T[1, 0], T[0, 0])
    L = np.eye(4); L[:3, :3] = rotz(yaw); L[:3, 3] = T[:3, 3]
    return L


@dataclass
class DescParams:
    r_min: float = 1.0
    r_max: float = 8.0
    dr: float = 0.5
    n_sec: int = 60
    gamma: float = 0.5
    bands: tuple = ((-0.35, 2.6),)  # one band: the polar descriptor of the leveled 3D submap; BANDS: four height bands
                                    # (forest seq1: recall@1 0.59 with one band, 0.47 with four; same SLAM result)


def descriptor(P, w, prm: DescParams = DescParams()):
    """P [n, 3] in the level keyframe frame, w [n] -> D [bands, rings, sectors] (each band unit norm, or zero)."""
    nb, nr = len(prm.bands), int(round((prm.r_max - prm.r_min) / prm.dr))
    D = np.zeros((nb, nr, prm.n_sec))
    r = np.hypot(P[:, 0], P[:, 1])
    ri = np.floor((r - prm.r_min) / prm.dr).astype(int)
    si = np.floor((np.arctan2(P[:, 1], P[:, 0]) + math.pi) / (2 * math.pi) * prm.n_sec).astype(int) % prm.n_sec
    for b, (lo, hi) in enumerate(prm.bands):
        m = (ri >= 0) & (ri < nr) & (P[:, 2] >= lo) & (P[:, 2] < hi)
        np.add.at(D[b], (ri[m], si[m]), np.sqrt(w[m]))
    D = D ** prm.gamma
    n = np.linalg.norm(D.reshape(nb, -1), axis=1)
    D[n > 0] /= n[n > 0, None, None]
    return D


def ring_key(D):
    K = D.sum(2)
    n = np.linalg.norm(K)
    return (K / n).ravel() if n > 0 else K.ravel()


def descriptor_match(Da, Db):
    """-> (similarity, shift): Db rolled by `shift` sectors best matches Da; yaw of b relative to a = shift * 2 pi / n_sec
    (a point at sector s in b's frame is at sector s + shift in a's frame)."""
    Fa, Fb = np.fft.rfft(Da, axis=2), np.fft.rfft(Db, axis=2)
    cc = np.fft.irfft(Fa * np.conj(Fb), n=Da.shape[2], axis=2).sum(1)          # [bands, shifts]
    bw = np.asarray(BAND_W) if len(Da) == len(BAND_W) else np.ones(len(Da))
    w = bw[:, None] * ((np.abs(Da).sum((1, 2)) > 0) & (np.abs(Db).sum((1, 2)) > 0))[:, None]
    if w.sum() == 0:
        return 0.0, 0
    s = (w * cc).sum(0) / w.sum()
    k = int(np.argmax(s))
    return float(s[k]), k


@dataclass
class Keyframe:
    idx: int                     # node index in the pose graph
    cap: int                     # capture index
    pts: np.ndarray              # submap in the level keyframe frame
    w: np.ndarray
    desc: np.ndarray
    key: np.ndarray

    @staticmethod
    def build(idx, cap, P_world, w, T_kf, prm: DescParams = DescParams(), r_max=None):
        L = level_frame(T_kf)
        P = (P_world - L[:3, 3]) @ L[:3, :3]
        rm = r_max or prm.r_max
        m = np.hypot(P[:, 0], P[:, 1]) < rm
        P, w = P[m], w[m]
        D = descriptor(P, w, prm)
        return Keyframe(idx, cap, P.astype(np.float32), w.astype(np.float32), D, ring_key(D))


def voxel_down(P, w, res=0.1):
    """Weighted voxel means (res m) -> (P, w summed)."""
    if len(P) == 0:
        return P, w
    k = np.floor(P / res).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    ws = np.bincount(inv, w)
    Q = np.stack([np.bincount(inv, w * P[:, i]) for i in range(3)], 1) / ws[:, None]
    return Q, ws


def register4(Pq, wq, Pa, yaw0=0.0, t0=(0.0, 0.0, 0.0), dists=(1.5, 1.0, 0.6, 0.4, 0.3), iters=6, k=5, s0=0.08,
              cauchy=2.0, ground_w=0.3):
    """4-DoF registration of the query submap Pq (level query frame) onto the anchor submap Pa (level anchor frame):
    T (anchor <- query) with yaw about z and translation; point to local-distribution (k-NN mean and covariance of the
    anchor points), Cauchy weights, coarse-to-fine association distance. Ground points (z < 0.25) are down-weighted:
    they fix z but hardly x, y. -> (T 4x4, info: inliers (fraction of query points within 0.3 m), rmse, H [4x4])."""
    tree = cKDTree(Pa)
    x = np.array([*t0, yaw0], float)
    wq = np.sqrt(wq / (wq.max() + 1e-30)) * np.where(Pq[:, 2] < 0.25, ground_w, 1.0)
    H = np.eye(4) * 1e-6
    for md in dists:
        for _ in range(iters):
            R = rotz(x[3]); q = Pq @ R.T + x[:3]
            d, idx = tree.query(q, k=k, distance_upper_bound=md)
            ok = np.isfinite(d).sum(1) >= 3
            if ok.sum() < 20:
                return None, {"inliers": 0.0, "rmse": np.inf, "H": H, "n": 0}
            ii = np.where(ok)[0]
            dd, nb = d[ii], idx[ii]
            fin = np.isfinite(dd)                                              # [m, k]
            M = Pa[np.where(fin, nb, 0)]                                       # [m, k, 3]
            cnt = fin.sum(1)[:, None]
            mu = (M * fin[..., None]).sum(1) / cnt
            Dv = (M - mu[:, None]) * fin[..., None]
            C = np.einsum("mki,mkj->mij", Dv, Dv) / np.maximum(cnt - 1, 1)[:, :, None] + s0 ** 2 * np.eye(3)
            Ci = np.linalg.inv(C)
            e = q[ii] - mu
            rw = wq[ii] / (1 + np.einsum("mi,mij,mj->m", e, Ci, e) / cauchy ** 2)
            rp = Pq[ii] @ R.T
            J = np.zeros((len(ii), 3, 4)); J[:, :, :3] = np.eye(3); J[:, 0, 3], J[:, 1, 3] = -rp[:, 1], rp[:, 0]
            CJ = Ci @ J                                                        # [m, 3, 4]
            H = np.einsum("m,mia,mib->ab", rw, J, CJ) + np.eye(4) * 1e-6
            g = -np.einsum("m,mia,mi->a", rw, CJ, e)
            dx = np.linalg.solve(H, g)
            x += dx
            if np.linalg.norm(dx[:3]) < 1e-3 and abs(dx[3]) < 1e-4:
                break
    R = rotz(x[3]); q = Pq @ R.T + x[:3]
    d, _ = tree.query(q, k=1, distance_upper_bound=0.3)
    inl = np.isfinite(d)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = x[:3]
    return T, {"inliers": float(inl.mean()), "rmse": float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf,
               "H": H, "n": int(len(Pq)),
               "inliers_above": float(inl[Pq[:, 2] >= 0.25].mean()) if (Pq[:, 2] >= 0.25).any() else 0.0}


def build_keyframes(P, caps, pts, w, st, T_sb, kf_m=1.0, submap_m=4.0, prm: DescParams = DescParams()):
    """Keyframes every kf_m of travel along the odometry poses P [N, 4, 4] (base, one per capture caps[N]); each
    submap = the detections (pts / w / start: rtis3d/detections.py, sensor frame) of the past submap_m of travel,
    0.1 m voxel means, in the keyframe's level frame. -> (indices into P [K], arc length [N], [Keyframe])."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2, 3], axis=0), axis=1))]
    sel, last = [], -1e9
    for j in range(len(caps)):
        if s[j] - last >= kf_m and s[j] >= submap_m:
            sel.append(j); last = s[j]
    KF = []
    for n, j in enumerate(sel):
        js = np.where((s > s[j] - submap_m) & (s <= s[j]) & (np.arange(len(s)) <= j))[0]
        Q, W = [], []
        for i in js:
            k = caps[i]
            p = pts[st[k]:st[k + 1]] @ T_sb[:3, :3].T + T_sb[:3, 3]
            Q.append(p @ P[i, :3, :3].T + P[i, :3, 3]); W.append(w[st[k]:st[k + 1]])
        Q, W = voxel_down(np.vstack(Q), np.concatenate(W), 0.1)
        KF.append(Keyframe.build(n, int(caps[j]), Q, W, P[j], prm))
    return np.asarray(sel), s, KF
