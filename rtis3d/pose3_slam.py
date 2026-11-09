"""3D pose graph (GTSAM Pose3) for a wheeled robot with an IMU and the eRTIS: the 2D RTISSLAM result lifted to 6-DoF.

Nodes: the RTISSLAM pose-graph nodes (drive pulses). Factors:
  odometry   between consecutive nodes: rotation from the bias-corrected 3-axis gyro (attitude filter), translation
             (ds, 0, 0) in the body frame from the wheel distance ds (a wheeled robot moves along its x axis): height
             follows from pitch x distance
  attitude   the attitude filter's up direction in the body frame (Pose3AttitudeFactor), absolute roll / pitch
  links      RTISSLAM's loop-closure links (query, anchor): planar relative pose from the 2D RTISSLAM solution, the same
             height at the same place (sigma_z), roll / pitch left free (the attitude factors hold them)
  prior      the first node
Everything here is sonar + proprioceptive (wheel, IMU); the lidar is used for scoring only (scripts/run_pose3.py).
"""
from __future__ import annotations

import math

import gtsam
import numpy as np

X = gtsam.symbol_shorthand.X


def pose3(R, t):
    return gtsam.Pose3(gtsam.Rot3(np.asarray(R, float)), gtsam.Point3(*np.asarray(t, float)))


def rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def build_and_solve(R_att, ds, P2d, links, sig_rot_step=(0.004, 0.004, 0.004), sig_trans=(0.03, 0.02, 0.02),
                    sig_trans_frac=0.03, sig_att_deg=1.5, sig_link_xy=0.15, sig_link_yaw_deg=2.0, sig_link_z=0.10,
                    sig_link_rp_deg=30.0, use_links=True, use_attitude=True, init=None):
    """R_att [N, 3, 3] body -> gravity-aligned frame (filter; its yaw is the gyro yaw), ds [N] signed wheel distance from
    node i-1 to i, P2d [N, 3] RTISSLAM final (x, y, yaw) (initial values and link measurements), links [K, 2].
    -> (poses [N, 4, 4], graph info)."""
    N = len(ds)
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    # initial values: x, y, yaw from RTISSLAM 2D; roll / pitch from the filter; z integrated from pitch x distance
    rp_only = np.array([rotz(-math.atan2(R[1, 0], R[0, 0])) @ R for R in R_att])            # strip the filter's yaw
    z = np.cumsum(np.r_[0, [ds[i] * -rp_only[i][2, 0] for i in range(1, N)]])
    for i in range(N):
        if init is not None:
            v.insert(X(i), gtsam.Pose3(init[i]))
        else:
            v.insert(X(i), pose3(rotz(P2d[i, 2]) @ rp_only[i], (P2d[i, 0], P2d[i, 1], z[i])))
    g.add(gtsam.PriorFactorPose3(X(0), v.atPose3(X(0)), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))))
    for i in range(1, N):
        dR = R_att[i - 1].T @ R_att[i]                                                         # gyro rotation increment
        d = abs(ds[i])
        st = np.r_[sig_rot_step, [sig_trans[0] + sig_trans_frac * d, sig_trans[1] + sig_trans_frac * d, sig_trans[2] + sig_trans_frac * d]]
        g.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), pose3(dR, (ds[i], 0, 0)), gtsam.noiseModel.Diagonal.Sigmas(st)))
    if use_attitude:
        na = gtsam.noiseModel.Isotropic.Sigma(2, math.radians(sig_att_deg))
        for i in range(N):
            up_body = R_att[i].T @ np.array([0, 0, 1.0])
            # nZ = up in the world, bRef = measured up in the body (the arguments were swapped before 2026-10-14 evening)
            g.add(gtsam.Pose3AttitudeFactor(X(i), gtsam.Unit3(np.array([0, 0, 1.0])), na, gtsam.Unit3(up_body)))
    n_l = 0
    if use_links and len(links):
        nl = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Diagonal.Sigmas(np.r_[
            [math.radians(sig_link_rp_deg)] * 2, math.radians(sig_link_yaw_deg), sig_link_xy, sig_link_xy, sig_link_z]))
        for q, a in links:
            # planar relative pose (anchor -> query) from the 2D solution, expressed level (roll = pitch = 0)
            dx, dy = P2d[q, 0] - P2d[a, 0], P2d[q, 1] - P2d[a, 1]
            ca, sa = math.cos(P2d[a, 2]), math.sin(P2d[a, 2])
            rel = gtsam.Pose3(gtsam.Rot3.Yaw(P2d[q, 2] - P2d[a, 2]), gtsam.Point3(ca * dx + sa * dy, -sa * dx + ca * dy, 0.0))
            g.add(gtsam.BetweenFactorPose3(X(int(a)), X(int(q)), rel, nl))
            n_l += 1
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    res = gtsam.LevenbergMarquardtOptimizer(g, v, params).optimize()
    P = np.array([res.atPose3(X(i)).matrix() for i in range(N)])
    return P, {"nodes": N, "links": n_l, "error_initial": g.error(v), "error_final": g.error(res)}


def solve_height(ds, pitch, links, sig_step=(0.003, 0.03), sig_link=0.05):
    """Heights z [N] by sparse least squares: steps z_i - z_(i-1) = ds_i sin(pitch_i) (sigma 3 mm + 3 % of ds),
    links z_q = z_a (same place, same ground; sigma_link), z_0 = 0. Loops make the heights consistent at revisits;
    a planar tilt of the whole map is not observable from them (z = e x has the same height at the same place)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    N = len(ds)
    rows, cols, vals, b, w = [], [], [], [], []
    r = 0
    for i in range(1, N):
        rows += [r, r]; cols += [i, i - 1]; vals += [1.0, -1.0]; b.append(ds[i] * math.sin(pitch[i]))
        w.append(1.0 / (sig_step[0] + sig_step[1] * abs(ds[i]))); r += 1
    for q, a in links:
        rows += [r, r]; cols += [int(q), int(a)]; vals += [1.0, -1.0]; b.append(0.0); w.append(1.0 / sig_link); r += 1
    rows.append(r); cols.append(0); vals.append(1.0); b.append(0.0); w.append(1e3); r += 1
    W = np.asarray(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, N)).multiply(W[:, None]).tocsr()
    return spl.lsqr(A, np.asarray(b) * W, atol=1e-10, btol=1e-10, iter_lim=50000)[0]


def decoupled(R_att, ds, P2d, links, **kw):
    """6-DoF poses [N, 4, 4]: x, y, yaw from the 2D SLAM, roll / pitch from the attitude filter, z from solve_height."""
    N = len(ds)
    rp_only = np.array([rotz(-math.atan2(R[1, 0], R[0, 0])) @ R for R in R_att])
    pitch = -np.arcsin(np.clip(rp_only[:, 2, 0], -1, 1))
    z = solve_height(ds, pitch, links, **kw)
    P = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        P[i, :3, :3] = rotz(P2d[i, 2]) @ rp_only[i]
        P[i, :3, 3] = (P2d[i, 0], P2d[i, 1], z[i])
    return P
