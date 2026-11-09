"""Unit tests of the RTISSLAM 3D building blocks that are easy to get silently wrong (synthetic data, no recordings).

    python -m pytest tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.metrics import ate, fit4, rotz  # noqa: E402
from rtis3d.place3d import DescParams, descriptor, descriptor_match, register4  # noqa: E402


def forest(seed=0, n_trunks=25, r=7.5):
    """Synthetic submap: trunks (vertical lines of points) and some ground, in a level frame."""
    rng = np.random.default_rng(seed)
    P = []
    for _ in range(n_trunks):
        c = rng.uniform(-r, r, 2)
        z = rng.uniform(0.3, 2.2, 30)
        P.append(np.c_[np.full(30, c[0]) + rng.normal(0, 0.05, 30), np.full(30, c[1]) + rng.normal(0, 0.05, 30), z])
    P.append(np.c_[rng.uniform(-r, r, (300, 2)), rng.normal(0, 0.03, 300)])
    P = np.vstack(P)
    return P, np.ones(len(P))


def test_descriptor_recovers_heading():
    P, w = forest()
    yaw = math.radians(42.0)
    Q = P @ rotz(-yaw).T                                   # the same place seen with a heading 42 deg further
    prm = DescParams()
    s, shift = descriptor_match(descriptor(P, w, prm), descriptor(Q, w, prm))
    est = (shift * 360 / prm.n_sec + 180) % 360 - 180
    assert s > 0.9
    assert abs(est - 42.0) <= 6.0 + 1e-9                   # one sector


def test_register4_recovers_pose():
    P, w = forest(1)
    T = np.eye(4); T[:3, :3] = rotz(math.radians(8)); T[:3, 3] = (0.6, -0.4, 0.05)
    Q = (P - T[:3, 3]) @ T[:3, :3]                         # query points, so that T maps the query onto P
    Te, info = register4(Q, w, P, yaw0=0.0)
    assert info["inliers_above"] > 0.9
    assert np.linalg.norm(Te[:2, 3] - T[:2, 3]) < 0.05
    assert abs(math.atan2(Te[1, 0], Te[0, 0]) - math.radians(8)) < math.radians(0.5)


def test_ate_is_invariant_to_yaw_and_translation():
    rng = np.random.default_rng(2)
    L = np.tile(np.eye(4), (50, 1, 1)); L[:, :3, 3] = np.cumsum(rng.normal(0, 1, (50, 3)), 0)
    F = np.eye(4); F[:3, :3] = rotz(1.0); F[:3, 3] = (3, -2, 1)
    r, e = ate(np.einsum("ij,njk->nik", F, L), L)
    assert r["ate_xy_m"] < 1e-9 and r["z_rmse_m"] < 1e-9
    R, t = fit4(L[:, :3, 3], L[:, :3, 3])
    assert np.allclose(R, np.eye(3)) and np.allclose(t, 0)


def test_attitude_factor_convention():
    """gtsam.Pose3AttitudeFactor(key, nZ = up in the world, model, bRef = measured up in the body) is zero at the truth
    (the arguments were once swapped, which tilted the map by 5 deg)."""
    import gtsam
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [10, -4, 90], degrees=True).as_matrix()
    up_body = R.T @ np.array([0, 0, 1.0])
    v = gtsam.Values(); v.insert(0, gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(0, 0, 0)))
    n = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    right = gtsam.Pose3AttitudeFactor(0, gtsam.Unit3(np.array([0, 0, 1.0])), n, gtsam.Unit3(up_body))
    wrong = gtsam.Pose3AttitudeFactor(0, gtsam.Unit3(up_body), n, gtsam.Unit3(np.array([0, 0, 1.0])))
    assert right.error(v) < 1e-12
    assert wrong.error(v) > 1e-3


def test_gravity_align_levels_a_tilted_reference():
    from rtis3d.attitude import gravity_align

    class A:                                              # minimal attitude stub: robot standing still, level
        t = np.linspace(0, 10, 200)

        def gravity_meas(self, motion_correction=True):
            return np.tile([0, 0, 9.81], (len(self.t), 1))
    tilt = Rotation_from_rotvec([0.02, -0.015, 0.0])
    R_ref = np.tile(tilt, (len(A.t), 1, 1))               # a reference whose world is tilted by ~1.4 deg
    Rg = gravity_align(A(), A.t, R_ref)
    assert np.allclose(Rg @ tilt @ np.array([0, 0, 1.0]), [0, 0, 1.0], atol=1e-9)


def Rotation_from_rotvec(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()
