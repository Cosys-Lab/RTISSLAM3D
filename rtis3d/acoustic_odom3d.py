"""3D acoustic odometry: each capture's detection set (base_link frame) registered against a local submap of the
previous captures, 6-DoF, with the wheel + IMU motion as prior.

    ao = AcousticOdometry3D()
    T_rel, cov, info = ao.step(pts_base, w, T_prior_rel, Sigma_prior)   # relative motion since the previous step

Model: current pose T = T_last @ T_pred @ Exp(xi), xi = (rho, phi) in the body frame. For each current point p (body),
q = T p is matched to the mean of its k nearest submap points within `max_dist`; residual e = q - mu with covariance
C = R B(p) R^T + N + s0^2 I, B: the beam covariance of the detection (range sigma_r along the ray, r sigma_az and
r sigma_el across it), N: the spread of the matched submap points. Gauss-Newton with re-association, Cauchy weights on
the Mahalanobis distance, and the prior xi ~ N(0, Sigma_prior) (wheel distance and IMU rotation). The covariance of the
result is the inverse Hessian; info reports the matches and how much the sonar moved the estimate away from the prior.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_se3(xi):
    rho, phi = xi[:3], xi[3:]
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-9:
        R, V = np.eye(3) + K, np.eye(3) + 0.5 * K
    else:
        a, b = np.sin(th) / th, (1 - np.cos(th)) / th ** 2
        R = np.eye(3) + a * K + b * K @ K
        V = np.eye(3) + b * K + (1 - a) / th ** 2 * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, V @ rho
    return T


def beam_cov(p_sensor, sig_r=0.03, sig_az=np.radians(3.0), sig_el=np.radians(5.0)):
    """[n, 3, 3] covariance of detections at p (sensor frame: x fwd, y left, z up)."""
    r = np.linalg.norm(p_sensor, axis=1)
    u = p_sensor / r[:, None]
    e_az = np.stack([-u[:, 1], u[:, 0], np.zeros(len(u))], 1)
    e_az /= np.linalg.norm(e_az, axis=1, keepdims=True) + 1e-12
    e_el = np.cross(u, e_az)
    return (sig_r ** 2 * np.einsum("ni,nj->nij", u, u) + ((r * sig_az) ** 2)[:, None, None] * np.einsum("ni,nj->nij", e_az, e_az)
            + ((r * sig_el) ** 2)[:, None, None] * np.einsum("ni,nj->nij", e_el, e_el))


class AcousticOdometry3D:
    def __init__(self, submap_frames: int = 20, k: int = 6, max_dist: float = 0.6, s0: float = 0.05, iters: int = 8,
                 cauchy: float = 2.5, min_matches: int = 25, T_sb=None, beam=(0.03, 3.0, 5.0)):
        self.n, self.k, self.max_dist, self.s0, self.iters, self.c = submap_frames, k, max_dist, s0, iters, cauchy
        self.min_matches = min_matches
        self.T_sb = np.eye(4) if T_sb is None else np.asarray(T_sb)    # sensor -> base (mount)
        self.beam = (beam[0], np.radians(beam[1]), np.radians(beam[2]))
        self.frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (pose in odom, points base, beam cov base)
        self.pose = np.eye(4)

    def _submap(self):
        if not self.frames:
            return None, None
        Q, C = [], []
        for T, p, B in self.frames:
            Q.append(p @ T[:3, :3].T + T[:3, 3])
            C.append(np.einsum("ij,njk,lk->nil", T[:3, :3], B, T[:3, :3]))
        return np.vstack(Q), np.concatenate(C)

    def step(self, pts_sensor, w, T_prior_rel, Sigma_prior):
        R_sb, t_sb = self.T_sb[:3, :3], self.T_sb[:3, 3]
        p = pts_sensor @ R_sb.T + t_sb                                    # body (base_link) frame
        B = np.einsum("ij,njk,lk->nil", R_sb, beam_cov(pts_sensor, *self.beam), R_sb)
        T_pred = self.pose @ T_prior_rel
        M, CM = self._submap()
        info = {"matches": 0, "used_sonar": False}
        xi = np.zeros(6)
        Pinv = np.linalg.inv(Sigma_prior)
        H = Pinv.copy()
        if M is not None and len(M) > 50 and len(p) > 10:
            tree = cKDTree(M)
            ww = np.sqrt(w / (w.max() + 1e-30))
            for it in range(self.iters):
                T = T_pred @ exp_se3(xi)
                R, t = T[:3, :3], T[:3, 3]
                q = p @ R.T + t
                d, idx = tree.query(q, k=self.k, distance_upper_bound=self.max_dist)
                ok = np.isfinite(d).sum(1) >= 2
                if ok.sum() < self.min_matches:
                    break
                H = Pinv.copy()
                g = -Pinv @ xi
                for i in np.where(ok)[0]:
                    nb = idx[i][np.isfinite(d[i])]
                    mu = M[nb].mean(0)
                    N = np.cov(M[nb].T) if len(nb) > 2 else np.zeros((3, 3))
                    Ci = R @ B[i] @ R.T + N + self.s0 ** 2 * np.eye(3)
                    Ci_inv = np.linalg.inv(Ci)
                    e = q[i] - mu
                    m2 = float(e @ Ci_inv @ e)
                    rw = ww[i] / (1 + m2 / self.c ** 2)                    # Cauchy
                    J = np.hstack([R, -R @ skew(p[i])])
                    H += rw * J.T @ Ci_inv @ J
                    g -= rw * J.T @ Ci_inv @ e
                dxi = np.linalg.solve(H, g)
                xi = xi + dxi
                info["matches"] = int(ok.sum())
                if np.linalg.norm(dxi) < 1e-5:
                    break
            info["used_sonar"] = info["matches"] >= self.min_matches
            if not info["used_sonar"]:
                xi = np.zeros(6); H = Pinv.copy()
        T_new = T_pred @ exp_se3(xi)
        T_rel = np.linalg.inv(self.pose) @ T_new
        cov = np.linalg.inv(H)
        info["xi"] = xi.tolist()
        self.pose = T_new
        self.frames.append((T_new, p, B))
        if len(self.frames) > self.n:
            self.frames.pop(0)
        return T_rel, cov, info
