"""Engineering report (PDF) of one RTISSLAM-3D pipeline run, from its outputs: numbers, figures and text.

    python scripts/make_report3d.py --name <name> --work <data>/<site>/threed/<name>
        [--recording "Forest, sequence 1"] [--no-pdf]
Writes report/<name>/: numbers.json (every number in the text), figures/*.pdf|png, main.tex and main.pdf (latexmk).
A hand-written report/<name>/notes.tex (findings, history, caveats of this recording) is included if it exists and is
never overwritten. With a lidar reference the report scores everything against it; without, it reports the SLAM only.
Ablations (scripts/ablations3d.py output in <work>/ablation/summary.json) are included when present.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rtis3d  # noqa: E402,F401
from rtis3d.metrics import ate, fit4, rpe  # noqa: E402

MAG = (177 / 255, 0, 145 / 255)


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def tex_escape(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace(">=", r"$\geq$")
            .replace("<=", r"$\leq$"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "pdf.fonttype": 42})
    work = Path(a.work)
    out = ROOT / "report" / a.name
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    st = json.loads((work / "pipeline.json").read_text())
    site = Path(st["site"])
    label = a.recording or a.name.replace("_", " ")
    h5 = st["h5"]
    num = {"name": a.name, "recording": label, "h5": h5, "sensors": st["sensors"], "stage_seconds": st.get("seconds", {})}
    inv = json.loads((site / "inventory" / f"{a.name}.json").read_text())
    sens = {s["name"]: s for s in inv["sensors"]}
    so = sens[st["sensors"]["sonar"]]
    num["captures"] = so["n"]; num["rate_hz"] = so["rate_hz_median"]; num["duration_s"] = so["duration_s"]
    num["max_range_m"] = so.get("max_range_m")
    clock = site / "cache" / f"{a.name}_clock_sync.json"
    if clock.exists():
        c = json.loads(clock.read_text())
        num["clock"] = {k: c[k] for k in ("offset_s", "corr", "offset_first_half", "offset_second_half")}
        if (site / "cache" / f"{a.name}_clock_sync.png").exists():
            shutil.copy(site / "cache" / f"{a.name}_clock_sync.png", figs / "clock_sync.png")
    offset = num["clock"]["offset_s"] if "clock" in num else None
    # ---- detections
    D = np.load(work / "detections.npz")
    npc = np.diff(D["start"])
    num["detections"] = {"per_capture_median": float(np.median(npc)), "plane_found": float(np.mean(np.isfinite(D["height"]))),
                         "sensor_height_m": float(np.nanmedian(D["height"])),
                         "mirrors_removed": float(D["n_mirror"].sum() / max(1, D["n_raw"].sum()))}
    # ---- SLAM + odometry
    Z = np.load(work / "slam3d.npz")
    R = json.loads((work / "slam3d.json").read_text())
    sel, K, P = Z["kf_index"], Z["kf_poses"], Z["odom"]
    O = np.load(work / "ao3d.npz")
    has_ref = "lidar" in Z.files
    num["slam"] = {k: R[k] for k in ("keyframes", "links", "registrations", "seconds")}
    s_kf = np.r_[0, np.cumsum(np.linalg.norm(np.diff(K[:, :2, 3], axis=0), axis=1))]
    num["distance_m"] = float(s_kf[-1])
    links = Z["links"]
    if has_ref:
        L = Z["lidar"]
        G = L[sel]
        num["slam3d"] = R["slam3d"]; num["odometry_kf"] = R["odometry"]; num["link_err"] = R.get("link_err", {})
        Pp = O["P_prior"]
        num["odometry"] = {"sonar": {f"rpe_{s:g}m": rpe(P, L, s) for s in (5.0, 20.0)},
                           "wheel_gyro": {f"rpe_{s:g}m": rpe(Pp, L, s) for s in (5.0, 20.0)}}
        num["odometry"]["sonar"]["dead_reckoning"] = ate(P, L)[0]
        num["odometry"]["wheel_gyro"]["dead_reckoning"] = ate(Pp, L)[0]
    occ_j = json.loads((work / "occ3d.json").read_text())
    if "thr_2" in occ_j:
        num["map"] = occ_j["thr_2"]
    abl = work / "ablation" / "summary.json"
    if abl.exists():
        num["ablation"] = json.loads(abl.read_text())
    (out / "numbers.json").write_text(json.dumps(num, indent=1, default=float))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    if has_ref:
        Rk, tk = fit4(K[:, :3, 3], G[:, :3, 3]); Q = K[:, :3, 3] @ Rk.T + tk
        Ro, to = fit4(P[sel][:, :3, 3], G[:, :3, 3]); Qo = P[sel][:, :3, 3] @ Ro.T + to
    else:
        Q, Qo = K[:, :3, 3], P[sel][:, :3, 3]
    ax[0].plot(Qo[:, 0], Qo[:, 1], c="tab:orange", lw=0.8, label="3D Acoustic Odometry")
    if has_ref:
        ax[0].plot(G[:, 0, 3], G[:, 1, 3], c="k", lw=0.8, label="3D Lidar SLAM (Reference)")
    for q, c_ in links[:, :2].astype(int) if len(links) else []:
        ax[0].plot(Q[[q, c_], 0], Q[[q, c_], 1], c=MAG, lw=0.5, alpha=0.6)
    ax[0].plot(Q[:, 0], Q[:, 1], c="tab:blue", lw=1.0, label="RTISSLAM-3D")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Y (m)"); ax[0].legend(fontsize=7, loc="best")
    ax[0].set_title(f"Trajectory, {len(links)} Loop Closures (Magenta)")
    if has_ref:
        ax[1].plot(s_kf, G[:, 2, 3], c="k", lw=0.8, label="Lidar")
    ax[1].plot(s_kf, Qo[:, 2], c="tab:orange", lw=0.8, label="Odometry")
    ax[1].plot(s_kf, Q[:, 2], c="tab:blue", lw=0.9, label="RTISSLAM-3D")
    ax[1].set_xlabel("Distance Driven (m)"); ax[1].set_ylabel("Height (m)"); ax[1].set_title("Height"); ax[1].legend(fontsize=7)
    if has_ref:
        e_s = np.linalg.norm(Q[:, :2] - G[:, :2, 3], axis=1); e_o = np.linalg.norm(Qo[:, :2] - G[:, :2, 3], axis=1)
        ax[2].plot(s_kf, e_o, c="tab:orange", lw=0.8, label="Odometry, Horizontal")
        ax[2].plot(s_kf, e_s, c="tab:blue", lw=0.9, label="RTISSLAM-3D, Horizontal")
        ax[2].plot(s_kf, np.abs(Q[:, 2] - G[:, 2, 3]), c="tab:blue", lw=0.6, ls="--", label="RTISSLAM-3D, Height")
        ax[2].set_xlabel("Distance Driven (m)"); ax[2].set_ylabel("Error (m)"); ax[2].set_title("Error vs Lidar (4-DoF Fit)")
        ax[2].legend(fontsize=7)
    else:
        ax[2].axis("off")
    fig.tight_layout(); fig.savefig(figs / "trajectory.pdf"); plt.close(fig)
    if has_ref and len(links):
        le = Z["link_err"]
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].hist(le[:, 0] * 100, bins=40, color=MAG); ax[0].set_xlabel("Horizontal Error (cm)"); ax[0].set_ylabel("Loop Closures")
        ax[0].set_title("Loop Closure Error vs Lidar")
        ax[1].hist(le[:, 2], bins=40, color="tab:blue"); ax[1].set_xlabel("Yaw Error (deg)"); ax[1].set_title("Loop Closure Yaw Error")
        fig.tight_layout(); fig.savefig(figs / "links.pdf"); plt.close(fig)
    occ = np.load(work / "occ3d.npz")
    fig, ax = plt.subplots(1, 2 if "lidar" in occ.files else 1, figsize=(12, 5) if "lidar" in occ.files else (6, 5), squeeze=False)
    Sx = occ["sonar_occ"]
    from scipy.ndimage import minimum_filter
    def hag(Pts, ref):                                    # height above the lowest point of the 1 m cell (and neighbours)
        lo = ref[:, :2].min(0) - 2
        ij = np.floor((ref[:, :2] - lo)).astype(int); g = np.full(ij.max(0) + 3, np.inf)
        np.minimum.at(g, (ij[:, 0], ij[:, 1]), ref[:, 2]); g = minimum_filter(g, 3)
        k = np.clip(np.floor(Pts[:, :2] - lo).astype(int), 0, np.array(g.shape) - 1)
        h = Pts[:, 2] - g[k[:, 0], k[:, 1]]
        return np.where(np.isfinite(h), h, 0)
    ref = occ["lidar"] if "lidar" in occ.files else Sx
    o = np.argsort(Sx[:, 2])
    sc = ax[0, 0].scatter(Sx[o, 0], Sx[o, 1], c=np.clip(hag(Sx[o], ref), 0, 2.5), s=0.3, cmap="viridis", rasterized=True)
    ax[0, 0].plot(occ["sensor_xyz"][:, 0], occ["sensor_xyz"][:, 1], c=MAG, lw=0.4)
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("Sonar 3D Occupancy (0.1 m), Top View"); ax[0, 0].set_xlabel("X (m)"); ax[0, 0].set_ylabel("Y (m)")
    if "lidar" in occ.files:
        Lm = occ["lidar"]; hl = hag(Lm, Lm); m = hl < 3.0
        o2 = np.argsort(Lm[m, 2])
        ax[0, 1].scatter(Lm[m][o2, 0], Lm[m][o2, 1], c=np.clip(hl[m][o2], 0, 2.5), s=0.3, cmap="viridis", rasterized=True)
        ax[0, 1].set_aspect("equal"); ax[0, 1].set_title("3D Lidar Map up to 3 m (Reference)"); ax[0, 1].set_xlabel("X (m)")
        ax[0, 1].set_xlim(ax[0, 0].get_xlim()); ax[0, 1].set_ylim(ax[0, 0].get_ylim())
    fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.7, label="Height Above Ground (m)")
    fig.savefig(figs / "map.pdf", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(npc, bins=60, color="tab:blue"); ax[0].set_xlabel("Detections per Capture"); ax[0].set_title("3D Detections")
    ax[1].plot((D["t"] - D["t"][0]), D["height"], ".", ms=1, c="tab:blue"); ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Height (m)"); ax[1].set_title("Sensor Height Above the Fitted Ground Plane")
    fig.tight_layout(); fig.savefig(figs / "detections.pdf", dpi=200); plt.close(fig)
    frame = work / "video" / "slam3d_build.mp4"
    has_frame = False
    if frame.exists():
        import imageio_ffmpeg
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-sseof", "-8", "-i", str(frame), "-frames:v", "1",
                            str(figs / "video_frame.png")])
        has_frame = r.returncode == 0

    # ---- LaTeX
    T = []
    w = T.append
    w(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,xcolor,hyperref,enumitem,array}
\graphicspath{{figures/}}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue!50!black,citecolor=black}
\setlist{nosep,leftmargin=*}
\newcommand{\code}[1]{\texttt{\detokenize{#1}}}
""")
    w(r"\title{RTISSLAM-3D: Full 6-DoF Sonar SLAM\\[0.3em]\large " + tex_escape(label)
      + f" (eRTIS, {num['distance_m']:.0f}\\,m, {num['duration_s'] / 60:.0f}\\,min)" + "}\n")
    w(r"\author{Claude (Linux GPU session, VM20) for Jan Steckel}" + "\n" + r"\date{\today}" + "\n\\begin{document}\n\\maketitle\n")
    w(r"\section*{Summary}" + "\n")
    w(r"\noindent\textbf{System.} RTISSLAM-3D estimates the 6-DoF trajectory of a wheeled robot and a 3D occupancy map from "
      r"an eRTIS imaging sonar, an IMU and the wheel odometry; nothing else. 3D acoustic odometry registers every capture's "
      r"3D detection set to a local submap (sonar translation, gyro rotation); every 1\,m a keyframe enters a GTSAM Pose3 "
      r"graph (iSAM2) with IMU attitude factors; 3D place recognition (polar descriptors of leveled 3D submaps, candidates near "
      r"the current estimate, verified by 4-DoF submap registration) closes loops; the map is a 3D log-odds occupancy grid on "
      r"the final poses." + "\n\n")
    if has_ref:
        s, od = num["slam3d"], num["odometry_kf"]
        w(r"\noindent\textbf{Result} (against a 3D lidar SLAM, gravity-aligned; 4-DoF fit since gravity is known):" + "\n")
        w(r"\begin{center}\begin{tabular}{lrrrrr}\toprule" + "\n")
        w(r" & horiz.\ ATE & max & height RMSE & roll / pitch & tilt (6-DoF fit) \\ \midrule" + "\n")
        w(f"3D acoustic odometry (no loops) & {f(od['ate_xy_m'])}\\,m & {f(od['xy_max_m'])}\\,m & {f(od['z_rmse_m'])}\\,m & "
          f"{f(od['roll_rmse_deg'], 1)} / {f(od['pitch_rmse_deg'], 1)}$^\\circ$ & {f(od['fit6_tilt_deg'], 1)}$^\\circ$ \\\\\n")
        w(f"\\textbf{{RTISSLAM-3D}} ({R['links']} loops) & \\textbf{{{f(s['ate_xy_m'])}\\,m}} & {f(s['xy_max_m'])}\\,m & "
          f"\\textbf{{{f(s['z_rmse_m'])}\\,m}} & {f(s['roll_rmse_deg'], 1)} / {f(s['pitch_rmse_deg'], 1)}$^\\circ$ & {f(s['fit6_tilt_deg'], 1)}$^\\circ$ \\\\\n")
        w(r"\bottomrule\end{tabular}\end{center}" + "\n")
        le = num.get("link_err", {})
        if le:
            w(f"Loop closures: median error {100 * le['xy_median_m']:.1f}\\,cm horizontally, {100 * le['z_median_m']:.1f}\\,cm in height, "
              f"{le['yaw_median_deg']:.2f}$^\\circ$ in yaw; {le['over_0.5m']} of {R['links']} off by more than 0.5\\,m. ")
        if "map" in num:
            m = num["map"]
            w(f"3D map: precision {f(m['precision'])} against the lidar map (0.3\\,m tolerance), "
              f"{f(m['low_0.25_2m']['precision'])} for trunks and vegetation (0.25--2\\,m above the ground).\n\n")
    else:
        w(f"\\noindent\\textbf{{Result.}} {R['keyframes']} keyframes, {R['links']} loop closures. No reference in this recording, "
          "so no scores.\n\n")
    w(r"\noindent\textbf{Parameters} are the ones tuned on the forest recording (sequence 1) and frozen; on any other recording "
      r"these numbers are an unbiased test.""\n\n")

    w(r"\section{The Recording}" + "\n")
    w(f"CHIRP file \\code{{{Path(h5).name}}}: sonar \\code{{{st['sensors']['sonar']}}} ({num['captures']} captures at "
      f"{num['rate_hz']:.2f}\\,Hz, {f(num['max_range_m'], 1)}\\,m records), IMU \\code{{{st['sensors']['imu']}}}, wheel odometry "
      f"\\code{{{st['sensors']['wheel']}}}, lidar \\code{{{st['sensors']['lidar']}}}, video \\code{{{st['sensors']['video']}}}. "
      f"Driven: {num['distance_m']:.0f}\\,m in {num['duration_s'] / 60:.1f}\\,min.\n")
    if "clock" in num:
        c = num["clock"]
        w(f"\n\\noindent\\textbf{{Clock.}} The eRTIS clock offset is {c['offset_s']:.3f}\\,s (sonar yaw rate against odometry yaw "
          f"rate, correlation {c['corr']:.2f}; the two halves of the drive give {c['offset_first_half']:.3f} and "
          f"{c['offset_second_half']:.3f}\\,s).\n")
    w(r"\section{Pipeline}" + "\nOne command (\\code{scripts/pipeline3d.py}); stage times on the Linux GPU machine (RTX PRO 5000):\n")
    w(r"\begin{center}\begin{tabular}{lr}\toprule stage & time \\ \midrule" + "\n")
    for k, v in num["stage_seconds"].items():
        w(f"{k} & {v:.0f}\\,s \\\\\n")
    w(r"\bottomrule\end{tabular}\end{center}" + "\n")
    w(r"\section{3D Imaging and Detections}" + "\n")
    d = num["detections"]
    w(f"Every capture is imaged once in 3D (DMAS5 with coherence factor over 2293 directions of the frontal hemisphere, 2\\,cm "
      f"range cells), CFAR on the sphere (heat kernel), cells within 25\\,dB of the capture's strongest. A ground plane is fitted "
      f"to the near-field ground echoes and detections more than 0.15\\,m below it (ground multipath mirrors) are removed. "
      f"Median {d['per_capture_median']:.0f} detections per capture; ground plane found in {100 * d['plane_found']:.0f}\\,\\% of the "
      f"captures, sensor height {d['sensor_height_m']:.3f}\\,m; {100 * d['mirrors_removed']:.0f}\\,\\% of the raw detections removed "
      f"as mirrors (figure~\\ref{{fig:det}}).\n")
    w(r"\begin{figure}[h]\centering\includegraphics[width=0.8\linewidth]{detections.pdf}\caption{Detections Per Capture and the "
      r"Sensor Height From the Ground Plane.}\label{fig:det}\end{figure}" + "\n")
    if has_ref:
        o = num["odometry"]
        w(r"\section{3D Acoustic Odometry}" + "\n")
        w("Per capture, the detection set is registered to a submap of the previous 20 captures (point-to-distribution with the "
          "beam-shaped covariance of each detection, Cauchy weights), with the wheel distance and the gyro rotation as prior. The "
          "sonar gives the horizontal translation, the gyro the rotation (when the sonar also estimates the rotation, it drifts: "
          "forest 3.1$^\\circ$ to 17$^\\circ$ per 20\\,m), and the vertical motion follows the wheels and the pitch (a tight 5\\,mm "
          "per step prior: the robot moves along its body $x$ axis).\n")
        w(r"\begin{center}\begin{tabular}{lrrrr}\toprule & RPE 5\,m & RPE 20\,m & height RPE 20\,m & dead reckoning (horiz.) \\ \midrule" + "\n")
        for k, lab in (("wheel_gyro", "wheel + gyro"), ("sonar", "\\textbf{3D acoustic odometry}")):
            w(f"{lab} & {f(o[k]['rpe_5m']['trans_rmse_m'])}\\,m & {f(o[k]['rpe_20m']['trans_rmse_m'])}\\,m & "
              f"{f(o[k]['rpe_20m']['z_rmse_m'])}\\,m & {f(o[k]['dead_reckoning']['ate_xy_m'])}\\,m \\\\\n")
        w(r"\bottomrule\end{tabular}\end{center}" + "\n")
    w(r"\section{SLAM}" + "\n")
    w(f"{R['keyframes']} keyframes (every 1\\,m), {R['registrations']} candidate registrations, {R['links']} accepted loop closures, "
      f"{R['seconds']:.0f}\\,s for the whole drive (single CPU thread). Figure~\\ref{{fig:traj}}.\n")
    cap = (r"RTISSLAM-3D Against the 3D Lidar Reference. Left: trajectories (4-DoF fit) with the loop closures; middle: height; "
           r"right: error along the drive." if has_ref else
           r"RTISSLAM-3D and the 3D Acoustic Odometry (No Reference). Left: trajectories with the loop closures; right: height.")
    w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{trajectory.pdf}\caption{" + cap + r"}\label{fig:traj}\end{figure}" + "\n")
    if (figs / "links.pdf").exists():
        w(r"\begin{figure}[h]\centering\includegraphics[width=0.75\linewidth]{links.pdf}\caption{Loop Closures Checked Against the "
          r"Lidar: Relative Pose Error of Every Accepted Link.}\label{fig:links}\end{figure}" + "\n")
    w(r"\section{3D Map}" + "\n")
    w("Log-odds occupancy at 0.1\\,m on the SLAM poses: every detection adds evidence spread as its beam footprint (thin along "
      "the ray, wide across it), free space is carved up to the first echo of each direction; occupied at log-odds $\\ge$\\,2.\n")
    if "map" in num:
        m = num["map"]
        w(r"\begin{center}\begin{tabular}{lrrr}\toprule band & voxels & precision & recall (observed) \\ \midrule" + "\n")
        for b, lab in (("ground", "ground ($<$\\,0.25\\,m)"), ("low_0.25_2m", "trunks, vegetation (0.25--2\\,m)"), ("high_2m+", "canopy ($>$\\,2\\,m)")):
            w(f"{lab} & {m[b]['voxels']} & {f(m[b]['precision'])} & {f(m[b]['recall'])} \\\\\n")
        w(f"all & {m['occupied_voxels']} & {f(m['precision'])} & {f(m['recall_observed'])} \\\\\n")
        w(r"\bottomrule\end{tabular}\end{center}" + "\n")
        w("Recall is low by nature: the sonar sees the near side of a trunk, a lidar point cloud sees everything within its "
          "range; the canopy is outside the sonar's view.\n")
    cap = (r"The Sonar 3D Map Next to the Lidar Map, Top View, Coloured by Height Above the Ground." if "lidar" in occ.files else
           r"The Sonar 3D Map, Top View, Coloured by Height Above the Ground.")
    w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{map.pdf}\caption{" + cap + r"}\label{fig:map}\end{figure}" + "\n")
    if "ablation" in num:
        w(r"\section{Ablations and Threshold Sensitivity}" + "\n")
        w(r"\begin{center}\begin{tabular}{lrrrr}\toprule variant & loops & horiz.\ ATE & height RMSE & links $>$\,0.5\,m \\ \midrule" + "\n")
        for k, v in num["ablation"].items():
            w(f"{tex_escape(v.get('label', k))} & {v['links']} & {f(v['ate_xy_m'])}\\,m & {f(v['z_rmse_m'])}\\,m & {v.get('over_0.5m', '--')} \\\\\n")
        w(r"\bottomrule\end{tabular}\end{center}" + "\n")
    w(r"\section{Videos}" + "\n")
    w(f"In \\code{{results/{a.name}/}} (1080p): the SLAM being built step by step, and with a lidar and a GoPro also a "
      f"drive-through of the sonar and lidar maps and a rotating view of the 3D map.\n")
    if has_frame:
        w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{video_frame.png}\caption{Last Frame of the SLAM Video.}\end{figure}" + "\n")
    if (out / "notes.tex").exists():
        w(r"\input{notes}" + "\n")
    w(r"\section{Reproduce}" + "\n" + r"\begin{verbatim}" + "\n" + f"python scripts/pipeline3d.py \\\n    {h5} \\\n"
      f"    --name {a.name}\n" + r"\end{verbatim}" + "\n")
    w(r"\end{document}" + "\n")
    (out / "main.tex").write_text("".join(T))
    if not a.no_pdf:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"], cwd=out, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(out / "main.pdf")


if __name__ == "__main__":
    main()
