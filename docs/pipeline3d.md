# Processing a new recording with RTISSLAM-3D

From a CHIRP recording (eRTIS + IMU + wheel odometry; optionally a 3D lidar and a GoPro) to the 6-DoF trajectory, the
3D map, the scores, three videos and a PDF report, in one command. The forest recording (`forest_seq1`, site
`norway`) is the worked example; its report is `report/forest_seq1/main.pdf`.

## 0. Put the data in place (Linux GPU machine)

```
<data>/<site>/raw/<file>.h5                 # CHIRP file (read-only: chmod a-w)
<data>/<site>/raw/<file>_video_00.mp4       # GoPro sibling, if any
```

## 1. Run the pipeline

```
cd RTISSLAM3D
python scripts/pipeline3d.py <data>/<site>/raw/<file>.h5 \
    --name <name> --recording "<Site>, Sequence <k>"
```

About 35 minutes for a 23-minute drive (RTX PRO 5000): imaging 8 min, lidar SLAM ~8 min (if a lidar is present and not
done yet), odometry 4 min, SLAM 4 min, videos ~10 min. Every stage writes its output and is skipped on a re-run, so the
command can be repeated after an interruption; `--force slam,map,videos,report` re-runs stages, `--stages` runs a
subset. Outputs:

| what | where |
|---|---|
| everything per recording | `<data>/<site>/threed/<name>/` (`pipeline.json`: sensors, stage times; `logs/`) |
| trajectory (one pose per capture, keyframes, loop closures) | `threed/<name>/slam3d.npz`, scores in `slam3d.json` |
| 3D map (0.1 m log-odds occupancy, voxel centres + evidence) | `threed/<name>/occ3d.npz`, scores in `occ3d.json` |
| videos, 1080p H.264 | `results/<name>/` |
| report | `report/<name>/main.pdf` (+ `numbers.json` with every number) |

Optional afterwards (needs the lidar): `scripts/ablations3d.py --work .../threed/<name>` (14 SLAM variants in parallel,
~8 min) and `--force report` to put the ablation table in the report. A hand-written `report/<name>/notes.tex` (findings
of that recording) is included in the report if present and never overwritten.

## 2. What to check

* **Sensors** (printed first, also in `pipeline.json`): sonar = the file's `reference_sensor`, IMU = the `imu_*` with the
  most samples, wheels = the `odometry_*` whose description says wheel/encoder, lidar / video = first `lidar_*` /
  `video_*`. Override with `RTIS3D_SONAR`, `RTIS3D_IMU`, `RTIS3D_WHEEL`, `RTIS3D_LIDAR`.
* **Clock offset** (stage `clock`): the eRTIS clock is matched to the robot clock by its yaw rate. The coarse guess is
  the difference of the sonar and wheel start times, searched +-30 s. Accept if the two halves agree within ~0.1 s
  (printed; plot in `cache/<name>_clock_sync.png`). Forest: 409980.680 s, halves 35 ms apart. If it fails, pass
  `--offset <s>` by hand.
* **Standstill at the start**: the gyro bias is the mean of the first 8 s, and the attitude filter starts 9 s in. A
  recording that starts while driving gets a biased gyro.
* **Detections** (report section 3): ~140 per capture after ~18 % mirrors removed, ground plane in > 90 % of the
  captures, sensor height constant.
* **SLAM**: number of loops; with a lidar, the link errors (median should be < 0.1 m, none > 0.5 m) and the trajectory
  figure. If no loop closes: the odometry may drift more than the 5 m gate between revisits (`run_slam3d.py --gate 8`).

## 3. Parameters

Tuned on forest seq1 and **frozen**: for an honest number on a new recording, do not re-tune them on it.

| stage | parameter | value |
|---|---|---|
| imaging | directions, beamformer | Fibonacci hemisphere 3 deg (2293), DMAS5 + CF, 2 cm cells to 9.7 m |
| detections | CFAR on the sphere | heat kernel 12 deg, elevation scale 1.7, gain 5, 25 dB below the peak, 0.5-8 m |
| detections | ground mirrors | plane from near-field low echoes, drop > 0.15 m below it |
| odometry | priors per step | rotation 0.05 deg (gyro), along x 0.01 m + 5 % of ds, lateral 0.05 m, vertical 0.005 m |
| odometry | submap, beam | 20 captures; sigma range 3 cm, azimuth 3 deg, elevation 5 deg |
| SLAM | keyframes, submaps | every 1 m; past 4 m of detections, 0.1 m voxel means, leveled |
| SLAM | descriptor | rings 1-8 m by 0.5 m, 60 sectors, one height band, gamma 0.5 |
| SLAM | loop candidates | within 5 m of the estimate, >= 20 m of travel back, top 5 by descriptor |
| SLAM | acceptance | >= 85 % of above-ground points within 0.3 m, RMS <= 0.12 m |
| SLAM | factors | odometry 2 cm + 5 %/m, 0.15 deg + 0.25 deg/m; attitude 1.5 deg; loops 0.10 / 0.05 m, 1 deg yaw, Cauchy |
| map | occupancy | 0.1 m, l_occ 4 (beam footprint), l_free 0.12, occupied at log-odds >= 2 |

Sensitivity on forest seq1 (ablations): inlier threshold 0.75-0.95 and gate 3-12 m all give 0.27-0.29 m horizontal ATE.

## 4. Without a lidar

Everything runs (`--no-lidar` forces it); there are no scores, the SLAM and the map stay in their own (gravity-aligned)
frame, and only the SLAM build video is made. The lidar is never used by the SLAM itself, only for scoring.

## 5. Pieces, for working on one stage

```
scripts/run_ao3d.py        h5 dets --offset s --out prefix [--kiss]          3D acoustic odometry
scripts/run_slam3d.py      h5 dets ao.npz --offset s [--kiss] --out prefix    SLAM (+ ablation switches)
scripts/build_occ3d.py     dets h5 --slam slam3d.npz [--kiss] --out prefix    3D map
scripts/eval_place3d.py    h5 dets ao.npz --kiss k --offset s                 place recognition without gate
scripts/eval_ao3d.py       h5 dets kiss --offset s                            odometry against the lidar
video/make_slam3d_video.py slam3d.npz h5 dets --out mp4 [--title]             SLAM build video
scripts/make_report3d.py   --name n --work dir                                report
scripts/paper_figures.py   --work dir                                         paper figures and tables
```
