# RTISSLAM-3D

Full 6-DoF SLAM with an in-air 3D imaging sonar.

RTISSLAM-3D estimates the 3D trajectory of a wheeled robot and builds a 3D map of its surroundings with an
[eRTIS](https://www.uantwerpen.be/en/research-groups/cosys-lab/) in-air 3D imaging sonar, an IMU and wheel encoders.
A CFAR detector on the sphere reduces every sonar image to a sparse set of 3D detections. These feed a 3D acoustic
odometry that models the shape of the sonar beam, place recognition with polar submap descriptors, and the
verification of loop closures by 3D registration, all combined in a 6-DoF pose graph with attitude factors.

On a 545 m drive through a forest, with a 3D lidar SLAM as reference, the loop closures reduced the horizontal error of
the acoustic odometry from 1.74 m to 0.28 m, with a height error of 0.12 m, and 94.6 % of the 3D sonar map lies within
0.3 m of the lidar map. Supplementary video: https://youtu.be/GHdZd9Brqog

## What is in this repository

| Path | Content |
|---|---|
| `rtis3d/` | the library: `sphere` (direction sets), `laplacian` (heat-kernel CFAR on the sphere), `detections`, `attitude`, `acoustic_odom3d`, `odometry3d`, `place3d`, `slam3d`, `occupancy3d`, `poses`, `metrics`, `dataset` |
| `scripts/pipeline3d.py` | the whole pipeline for one recording, in one command |
| `scripts/` | the stages (`real_volumes`, `extract_detections`, `run_ao3d`, `run_slam3d`, `build_occ3d`, `lidar3d_slam`), evaluations (`eval_*`, `ablations3d`), the report (`make_report3d`), the paper figures (`paper_figures`) and earlier studies (`sim_psf3d`, `overlay3d`, `grid3d_eval`, `run_pose3`) |
| `video/` | videos of the SLAM being built, of the 3D map and of a drive through the sonar and lidar maps |
| `rtisreal/` | the real-sensor front end (imaging of CHIRP recordings with RTIS CUDA, preprocessing) |
| `sondec/` | sensor model and delay-and-sum helpers from the 2D RTISSLAM code |
| `docs/` | `pipeline3d.md` (processing a new recording, step by step), `rw_cfar_spec.md` (the CFAR detector) |
| `tests/` | unit tests of the core |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two packages from the Cosys-Lab are needed for real recordings and are not on PyPI: `chirppy` (the reader for CHIRP
HDF5 recordings) and `rtiscuda` (GPU beamforming of RTIS measurements). Contact the Cosys-Lab for access. The tests and
most of the library run without them.

The optional 3D lidar reference uses KISS-ICP / KISS-SLAM, best installed in a separate environment
(`requirements-lidar.txt`), which the pipeline finds through the environment variable `RTIS3D_LIDAR_PYTHON`.

## Usage

```
python scripts/pipeline3d.py <data>/<site>/raw/<file>.h5 --name <name> --recording "<Site>, Sequence <k>"
```

The input is a CHIRP recording with an eRTIS, an IMU and wheel odometry (optionally a 3D lidar and a camera). Every
stage writes its output under `<data>/<site>/threed/<name>/` and is skipped when that output exists. See
`docs/pipeline3d.md` for the stages, outputs, run times and what to check.

```
python -m pytest tests -q
```

## Citation

If you use this code, please cite:

J. Steckel, V. Katter and W. Jansen, "RTISSLAM-3D: Full 6-DoF SLAM With an In-Air 3D Imaging Sonar," submitted to the
IEEE, 2026 (preprint on arXiv).

## License

BSD 3-Clause, see `LICENSE`. The modules in `rtisreal/` and `sondec/` come from the lab's RTISSLAM projects, under the
same license (provenance in each file).

Cosys-Lab, Faculty of Applied Engineering, University of Antwerp, and Flanders Make. Contact: jan.steckel@uantwerpen.be
