"""Directions on the frontal hemisphere and operators on them.

    U = frontal_hemisphere(spacing_deg=3.0)        # [D, 3] unit vectors, x forward, y left, z up (sensor frame)
    az, el = az_el(U)

Uniform sampling: a Fibonacci (golden-angle) lattice on the sphere, kept where x > 0. Every direction covers the same
solid angle (to O(1/D)), so averages over directions are averages over solid angle. The eRTIS array is planar (in the
y-z plane), so it cannot tell front from back: the frontal hemisphere is the whole information.
"""
from __future__ import annotations

import math

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = math.pi * (3 - math.sqrt(5)) * i
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def frontal_hemisphere(spacing_deg: float = 3.0, min_x: float = 0.0) -> np.ndarray:
    """~2 pi / spacing^2 directions with x > min_x (forward hemisphere)."""
    n = int(round(4 * math.pi / math.radians(spacing_deg) ** 2))
    U = fibonacci_sphere(n)
    return U[U[:, 0] > min_x]


def az_el(U: np.ndarray):
    """-> azimuth (rad, + left), elevation (rad, + up)."""
    return np.arctan2(U[:, 1], U[:, 0]), np.arcsin(np.clip(U[:, 2], -1, 1))


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1))
