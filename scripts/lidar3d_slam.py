"""6-DoF lidar SLAM of a CHIRP recording with KISS-SLAM (KISS-ICP odometry + map-closure loop closing + pose-graph
optimization; PRBonn), as the 3D reference for the sonar work.

    python scripts/lidar3d_slam.py <dataset.h5> <out.npz> [--every 1] [--split 25] [--max-scans N]
(Needs kiss-icp, kiss-slam and h5py, best in a separate Python environment: see the README.)

Scans: lidar_00 points [x, y, z] in the lidar frame (no per-point times in CHIRP, so no de-skewing: at 0.4 m/s and
10 Hz the skew is <= 4 cm, plus up to ~1.5 deg while turning), 1..60 m. Local maps every --split m of travel (default
25: the forest drive has tight loops); closures between local maps; final fine-grained optimization.
Output: t [N] (scan times, robot clock), T [N, 4, 4] lidar poses in the map frame (first scan = identity),
T_odom [N, 4, 4] KISS-ICP odometry only (before loop closing), closures [K, 2] (local-map ids), lidar_to_base [4, 4],
map_points [M, 3] (0.2 m voxel map of the optimized cloud, every 5th scan), config (JSON).
"""
from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--split", type=float, default=25.0)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=60.0)
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--lidar", default="lidar_00", help="lidar sensor group in the CHIRP file")
    ap.add_argument("--overlap", type=float, default=None, help="loop closer overlap threshold (KISS-SLAM default 0.4)")
    ap.add_argument("--inliers", type=int, default=None, help="loop detector inlier threshold (default 5)")
    ap.add_argument("--hamming", type=int, default=None, help="loop detector Hamming distance threshold (default 50)")
    a = ap.parse_args()
    from kiss_slam.config.config import KissSLAMConfig
    from kiss_slam.slam import KissSLAM
    cfg = KissSLAMConfig()
    cfg.odometry.preprocessing.deskew = False
    cfg.odometry.preprocessing.min_range = a.min_range
    cfg.odometry.preprocessing.max_range = a.max_range
    cfg.odometry.mapping.voxel_size = a.max_range / 100.0   # KISS-ICP's default rule, explicit once the range changes
    cfg.local_mapper.splitting_distance = a.split
    if a.overlap is not None:
        cfg.loop_closer.overlap_threshold = a.overlap
    if a.inliers is not None:
        cfg.loop_closer.detector.inliers_threshold = a.inliers
    if a.hamming is not None:
        cfg.loop_closer.detector.hamming_distance_threshold = a.hamming
    slam = KissSLAM(cfg)
    f = h5py.File(a.h5, "r")
    li = f[f"sensors/{a.lidar}"]
    P, C, t_all = li["data/points"], li["data/point_counts"], li["data/timestamps"][:]
    T_lo = li["calibration/to_origin"][:]
    od = next(g for k, g in f["sensors"].items() if k.startswith("odometry"))
    T_bo = od["calibration/to_origin"][:]
    lidar_to_base = np.linalg.inv(T_bo) @ T_lo
    idx = np.arange(0, len(t_all), a.every)
    if a.max_scans:
        idx = idx[:a.max_scans]
    T_odom = []
    t0 = time.time()
    for j, k in enumerate(idx):
        pts = np.asarray(P[k][:int(C[k]), :3], np.float64)
        pts = pts[np.isfinite(pts).all(1)]
        slam.process_scan(pts, np.empty(0))
        T_odom.append(np.copy(slam.odometry.last_pose))      # within the current local map
        if (j + 1) % 1000 == 0:
            print(f"  {j + 1}/{len(idx)} scans, {len(slam.get_closures())} closures, {time.time() - t0:.0f} s", flush=True)
    slam.generate_new_node()
    poses, _ = slam.fine_grained_optimization()
    T = np.asarray(poses)
    # odometry-only trajectory: chain the local trajectories with their un-optimized relative motions
    T_chain, base = [], np.eye(4)
    for node in slam.local_map_graph.local_maps():
        for rel in (node.local_trajectory if not T_chain else node.local_trajectory[1:]):
            T_chain.append(base @ rel)
        base = base @ node.local_trajectory[-1]
    # global voxel map from the optimized poses (every 5th scan)
    from kiss_icp.voxelization import voxel_down_sample  # noqa: E402
    cloud = []
    for j in range(0, len(idx), 5):
        k = idx[j]
        pts = np.asarray(P[k][:int(C[k]), :3], np.float64)
        pts = pts[np.isfinite(pts).all(1)]
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > a.min_range) & (r < 30)]
        cloud.append(voxel_down_sample(pts @ T[j][:3, :3].T + T[j][:3, 3], 0.1))
    M = voxel_down_sample(np.vstack(cloud), 0.2)
    np.savez_compressed(a.out, t=t_all[idx], T=T, T_odom=np.asarray(T_chain[:len(T)]),
                        closures=np.asarray(slam.get_closures(), int).reshape(-1, 2), lidar_to_base=lidar_to_base,
                        map_points=M.astype(np.float32),
                        config=json.dumps({"split_m": a.split, "every": a.every, "min_range": a.min_range,
                                           "max_range": a.max_range, "deskew": False,
                                           "loop_closer": cfg.loop_closer.model_dump()}))
    print(f"{a.out}: {len(T)} poses, {len(slam.get_closures())} closures, map {len(M)} points, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
