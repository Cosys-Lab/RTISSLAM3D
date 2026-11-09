"""Full 3D sonar SLAM (6-DoF poses, incremental): keyframes every `kf_m` of travel, GTSAM Pose3 graph with iSAM2.

Factors per keyframe i:
  odometry   between keyframes i-1 and i: the 3D acoustic odometry (rtis3d/acoustic_odom3d.py: sonar translation,
             gyro rotation), sigmas growing with the distance travelled
  attitude   the IMU's up direction in the body frame (Pose3AttitudeFactor): absolute roll / pitch
  loops      3D place recognition (rtis3d/place3d.py): candidates gated by the current estimate (horizontal distance
             < gate_m, at least sep_m of travel back), ranked by the descriptor, verified by 4-DoF submap registration
             (inliers above the ground >= min_inliers, rmse <= max_rmse); a level-frame relative pose (x, y, z, yaw)
             turned into a Pose3 between factor with the IMU roll / pitch of both ends; Cauchy kernel

    S = Slam3D()
    S.add(kf, T_odom_rel, T_odom_abs)       # Keyframe (place3d), relative odometry since the last keyframe, odom pose
    P = S.poses()                           # [N, 4, 4] base poses (gravity frame)
"""
from __future__ import annotations

import math

import gtsam
import numpy as np

from .place3d import descriptor_match, level_frame, register4, rotz

X = gtsam.symbol_shorthand.X


def _p3(T):
    return gtsam.Pose3(np.asarray(T, float))


class Slam3D:
    def __init__(self, sig_trans=(0.02, 0.05), sig_trans_z=None, sig_rot_deg=(0.15, 0.25), sig_att_deg=1.5, gate_m=5.0, sep_m=20.0,
                 top=5, min_inliers=0.85, max_rmse=0.12, sig_loop=(0.10, 0.05, 1.0), sig_loop_rp_deg=3.0,
                 cauchy=1.0, n_sec=60, loop_every=1):
        self.sig_tz = sig_trans_z or sig_trans                    # body vertical (sonar ground + IMU): may be tighter
        self.sig_trans, self.sig_rot = sig_trans, (math.radians(sig_rot_deg[0]), math.radians(sig_rot_deg[1]))
        self.sig_att = math.radians(sig_att_deg)
        self.gate, self.sep, self.top = gate_m, sep_m, top
        self.min_inl, self.max_rmse = min_inliers, max_rmse
        self.sig_loop, self.sig_loop_rp = sig_loop, math.radians(sig_loop_rp_deg)
        self.cauchy, self.n_sec, self.loop_every = cauchy, n_sec, loop_every
        p = gtsam.ISAM2Params()
        p.setRelinearizeThreshold(0.01)
        p.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(p)
        self.kfs, self.s, self.est = [], [], []
        self.links, self.tried = [], 0
        self.hist = []                       # the estimate after every keyframe ([i + 1, 3, 4], float32): for the videos

    def _rp(self, T):
        """Roll / pitch part of a pose's rotation (yaw stripped)."""
        R = T[:3, :3]
        return rotz(-math.atan2(R[1, 0], R[0, 0])) @ R

    def add(self, kf, T_rel, T_odo, ds):
        """kf: place3d.Keyframe; T_rel: odometry since the previous keyframe; T_odo: odometry pose (its roll / pitch are
        the IMU's); ds: distance travelled since the previous keyframe."""
        i = len(self.kfs)
        g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
        if i == 0:
            v.insert(X(0), _p3(T_odo))
            g.add(gtsam.PriorFactorPose3(X(0), _p3(T_odo), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))))
            self.s.append(0.0)
        else:
            prev = self.est[-1]
            v.insert(X(i), _p3(prev @ T_rel))
            st, sr = self.sig_trans[0] + self.sig_trans[1] * ds, self.sig_rot[0] + self.sig_rot[1] * ds
            sz = self.sig_tz[0] + self.sig_tz[1] * ds
            g.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), _p3(T_rel), gtsam.noiseModel.Diagonal.Sigmas(np.r_[[sr] * 3, st, st, sz])))
            self.s.append(self.s[-1] + ds)
        up = T_odo[:3, :3].T @ np.array([0, 0, 1.0])
        # Pose3AttitudeFactor(key, nZ = up in the world, model, bRef = measured up in the body): error of R bRef vs nZ
        g.add(gtsam.Pose3AttitudeFactor(X(i), gtsam.Unit3(np.array([0, 0, 1.0])), gtsam.noiseModel.Isotropic.Sigma(2, self.sig_att), gtsam.Unit3(up)))
        self.kfs.append((kf, self._rp(T_odo)))
        self.isam.update(g, v)
        self._refresh()
        if i % self.loop_every == 0:
            self._loops(i)
        self.hist.append(np.asarray(self.est, np.float32)[:, :3, :])

    def _refresh(self):
        r = self.isam.calculateEstimate()
        self.est = [r.atPose3(X(j)).matrix() for j in range(len(self.kfs))]

    def _loops(self, q):
        s = np.asarray(self.s)
        P = np.array([T[:3, 3] for T in self.est])
        cand = np.where((s < s[q] - self.sep) & (np.linalg.norm(P[:, :2] - P[q, :2], axis=1) < self.gate))[0]
        if len(cand) == 0:
            return
        kq, rpq = self.kfs[q]
        sc = np.array([descriptor_match(self.kfs[c][0].desc, kq.desc) for c in cand])
        best = None
        for o in np.argsort(-sc[:, 0])[:self.top]:
            c = cand[o]
            ka, rpa = self.kfs[c]
            # two initial guesses: the descriptor yaw, and the current estimate of the relative level pose
            Le = np.linalg.inv(level_frame(self.est[c])) @ level_frame(self.est[q])
            inits = [(sc[o, 1] * 2 * math.pi / self.n_sec, (0.0, 0.0, 0.0)), (math.atan2(Le[1, 0], Le[0, 0]), tuple(Le[:3, 3]))]
            for yaw0, t0 in inits:
                self.tried += 1
                T, info = register4(kq.pts, kq.w, ka.pts, yaw0, t0)
                if T is None or info["inliers_above"] < self.min_inl or info["rmse"] > self.max_rmse:
                    continue
                if best is None or info["inliers_above"] > best[2]["inliers_above"]:
                    best = (c, T, info, rpa)
        if best is None:
            return
        c, T, info, rpa = best
        # level relative pose -> base relative pose: B_a^-1 B_q = Ra^-1 (La^-1 Lq) Rq (R: roll / pitch parts)
        Ra = np.eye(4); Ra[:3, :3] = rpa
        Rq = np.eye(4); Rq[:3, :3] = rpq
        rel = np.linalg.inv(Ra) @ T @ Rq
        sx, sz, syaw = self.sig_loop
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(self.cauchy),
                                               gtsam.noiseModel.Diagonal.Sigmas(np.r_[self.sig_loop_rp, self.sig_loop_rp, math.radians(syaw), sx, sx, sz]))
        g = gtsam.NonlinearFactorGraph()
        g.add(gtsam.BetweenFactorPose3(X(int(c)), X(int(q)), _p3(rel), noise))
        self.isam.update(g, gtsam.Values())
        self.isam.update()
        self._refresh()
        self.links.append((int(q), int(c), info["inliers_above"], info["rmse"], *T[:3, 3], math.atan2(T[1, 0], T[0, 0])))

    def poses(self):
        return np.asarray(self.est)
