"""Attitude (roll, pitch) of a wheeled robot from its IMU (and wheel speed), no lidar.

    est = Attitude(t_imu, gyro, acc, t_wheel, v_wheel, still_s=8.0)
    R = est.complementary(tau=10.0, motion_correction=True)   # [N, 3, 3] body -> gravity-aligned frame (yaw from gyro)
    roll, pitch = est.roll_pitch(R)

Gyro: 3-axis, bias = mean over the first still_s seconds (robot standing still); if the wheels move in that time, the
mean over all wheel standstills of at least 3 s. Gravity: the accelerometer measures
specific force f = a - g; while driving, a = dv/dt along x + v x omega (centripetal), so with the wheel speed v (body x)
the motion part is removed: f_corr = f - (dv/dt, v * omega_z, -v * omega_y). Complementary filter on SO(3): integrate the
bias-corrected gyro, and pull the estimated gravity direction towards the measured one with time constant tau.
"""
from __future__ import annotations

import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def roll_pitch_from_gravity(f):
    """f: specific force in body (at rest +g up: (0, 0, +9.81)) -> roll, pitch (ZYX)."""
    return np.arctan2(f[..., 1], f[..., 2]), np.arctan2(-f[..., 0], np.hypot(f[..., 1], f[..., 2]))


def rot_rp(roll, pitch):
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0, cr, -sr], [-sp, cp * sr, cp * cr]])


class Attitude:
    def __init__(self, t_imu, gyro, acc, t_wheel=None, v_wheel=None, still_s: float = 8.0):
        self.t = np.asarray(t_imu, float)
        start = still = self.t < self.t[0] + still_s
        self.bias_from = "start"
        if t_wheel is not None and np.abs(np.interp(self.t[still], t_wheel, v_wheel)).max() > 0.01:
            # the recording starts while driving (Lokken tunnel): bias from all standstills of >= 3 s the wheels report
            stops = self.standstills(t_wheel, v_wheel)
            if stops.any():
                still, self.bias_from = stops, "standstills"
        self.bias = np.asarray(gyro, float)[still].mean(0)
        self.w = np.asarray(gyro, float) - self.bias
        self.f = np.asarray(acc, float)
        self.g_scale = 9.80665 / np.linalg.norm(self.f[still].mean(0))      # accelerometer scale from the standstill
        self.f = self.f * self.g_scale
        if t_wheel is not None:
            v = np.interp(self.t, t_wheel, v_wheel)
            k = max(1, int(round(0.2 / np.median(np.diff(self.t)))))
            v = np.convolve(v, np.ones(k) / k, "same")
            self.v, self.dv = v, np.gradient(v, self.t)
        else:
            self.v = self.dv = np.zeros(len(self.t))
        self.r0, self.p0 = roll_pitch_from_gravity(self.f[start].mean(0))   # initial attitude: always the start

    def standstills(self, t_wheel, v_wheel, v_max=0.005, min_s=3.0, margin_s=0.5):
        """IMU samples in wheel standstills (|v| < v_max) of at least min_s, margin_s cut at both ends."""
        moving = np.abs(np.interp(self.t, t_wheel, np.abs(v_wheel))) >= v_max
        out = np.zeros(len(self.t), bool)
        d = np.diff(np.r_[1, moving.astype(int), 1])
        for a, b in zip(np.where(d == -1)[0], np.where(d == 1)[0]):
            m = (self.t >= self.t[a] + margin_s) & (self.t <= self.t[b - 1] - margin_s)
            if self.t[b - 1] - self.t[a] >= min_s:
                out |= m
        return out

    def gravity_meas(self, motion_correction: bool = True, avg_s: float = 0.0):
        f = self.f.copy()
        if motion_correction:
            f[:, 0] -= self.dv
            f[:, 1] -= self.v * self.w[:, 2]
            f[:, 2] += self.v * self.w[:, 1]
        if avg_s > 0:
            k = max(1, int(round(avg_s / np.median(np.diff(self.t)))))
            f = np.stack([np.convolve(f[:, i], np.ones(k) / k, "same") for i in range(3)], 1)
        return f

    def complementary(self, tau: float | None = 10.0, motion_correction: bool = True):
        """-> R [N, 3, 3] body -> gravity frame; tau None: gyro only."""
        f = self.gravity_meas(motion_correction)
        R = rot_rp(self.r0, self.p0)
        out = np.empty((len(self.t), 3, 3))
        out[0] = R
        dt = np.diff(self.t, prepend=self.t[0])
        for i in range(1, len(self.t)):
            R = R @ expm_so3(self.w[i] * dt[i])
            if tau is not None:
                g_b = R.T @ np.array([0, 0, 1.0])                 # estimated up in body
                m = f[i] / (np.linalg.norm(f[i]) + 1e-9)          # measured up in body
                err = np.cross(g_b, m)                            # rotation taking estimate to measurement (body)
                R = R @ expm_so3(-err * dt[i] / tau)
            out[i] = R
            if i % 2000 == 0:                                     # re-orthonormalize
                u, _, vt = np.linalg.svd(R); R = u @ vt
        return out

    @staticmethod
    def roll_pitch(R):
        return np.arctan2(R[:, 2, 1], R[:, 2, 2]), -np.arcsin(np.clip(R[:, 2, 0], -1, 1))


def gravity_align(A: Attitude, t_ref, R_ref, t_min: float | None = None):
    """Rotation Rg [3, 3] that makes a reference trajectory's world frame gravity-aligned: R_ref [N, 3, 3] body -> world
    at times t_ref (e.g. a lidar SLAM, which knows no gravity); the motion-corrected accelerometer f (A.gravity_meas),
    rotated into that world and averaged over the whole drive (the driving accelerations average out), is the true up;
    Rg maps it onto (0, 0, 1). Apply as Rg @ R_ref and Rg @ p_ref."""
    f = A.gravity_meas(True)
    ok = (A.t > t_ref[0]) & (A.t < t_ref[-1])
    if t_min is not None:
        ok &= A.t > t_min
    i = np.clip(np.searchsorted(t_ref, A.t[ok]), 0, len(t_ref) - 1)
    u = np.einsum("nij,nj->ni", R_ref[i], f[ok] / np.linalg.norm(f[ok], axis=1, keepdims=True)).mean(0)
    u /= np.linalg.norm(u)
    z = np.array([0, 0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(u @ z)
    if s < 1e-12:
        return np.eye(3)
    return expm_so3(v / s * np.arctan2(s, c))
