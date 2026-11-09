"""3D place recognition (rtis3d/place3d.py) against the 3D lidar: keyframe submaps built from the 3D acoustic
odometry (the past `--submap` m of detections, no lidar), every query compared with all keyframes at least `--sep` m
of travel earlier (no position gate: the descriptor alone), top candidates verified by 4-DoF registration.

    python scripts/eval_place3d.py <dataset.h5> <detections.npz> <ao3d npz> [--kf 1.0] [--submap 4]
        [--out prefix]
Truth: the lidar poses at the same captures (in the ao3d npz). A candidate is a true place if the two keyframes are
within --true-m (horizontal) of each other. Writes <out>.json (recall@1/@5, top-1 PR, verification statistics),
<out>.npz (per query: candidates, scores, registration results) and <out>.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtis3d  # noqa: E402,F401
from rtis3d.place3d import BANDS, DescParams, build_keyframes, descriptor_match, level_frame, register4  # noqa: E402
from rtis3d.poses import _mount  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("dets")
    ap.add_argument("ao")
    ap.add_argument("--kf", type=float, default=1.0)
    ap.add_argument("--submap", type=float, default=4.0)
    ap.add_argument("--sep", type=float, default=20.0)
    ap.add_argument("--true-m", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--n-sec", type=int, default=60)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--height-bands", action="store_true", help="descriptor with four height bands")
    ap.add_argument("--kiss", default=None, help="lidar SLAM npz for the truth (else the ao npz must hold `lidar`)")
    ap.add_argument("--offset", type=float, default=None, help="eRTIS clock offset (with --kiss)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = np.load(a.dets)
    pts, w, st = D["pts"], D["w"], D["start"]
    Z = np.load(a.ao)
    P, caps = Z["P_ao"], Z["caps"]
    if a.kiss:
        from rtis3d.odometry3d import load_attitude
        from rtis3d.poses import lidar_reference
        t = Z["t"] if "t" in Z.files else D["t"][caps] + a.offset
        Lg = lidar_reference(a.kiss, t, load_attitude(a.h5)[0])
    else:
        Lg = Z["lidar"]
    T_sb, _ = _mount(a.h5)
    prm = DescParams(n_sec=a.n_sec, gamma=a.gamma, **({"bands": BANDS} if a.height_bands else {}))
    t0 = time.time()
    sel, s, KF = build_keyframes(P, caps, pts, w, st, T_sb, a.kf, a.submap, prm)
    print(f"{len(KF)} keyframes, {np.mean([len(k.pts) for k in KF]):.0f} submap points, {time.time() - t0:.0f} s", flush=True)
    G = Lg[sel]                                                    # truth (lidar, base, gravity frame) at the keyframes
    sk = s[sel]
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Lg[:, :2, 3], axis=0), axis=1))][sel]
    rows = []
    n_pos = 0
    hit1 = hit5 = 0
    for q in range(len(KF)):
        cand = np.where(sl < sl[q] - a.sep)[0]
        if len(cand) == 0:
            continue
        dtrue = np.linalg.norm(G[cand, :2, 3] - G[q, :2, 3], axis=1)
        has_pos = bool((dtrue < a.true_m).any())
        sc = np.array([descriptor_match(KF[c].desc, KF[q].desc) for c in cand])
        order = np.argsort(-sc[:, 0])
        n_pos += has_pos
        if has_pos:
            hit1 += dtrue[order[0]] < a.true_m
            hit5 += (dtrue[order[:5]] < a.true_m).any()
        for rank, o in enumerate(order[:a.top]):
            c = cand[o]
            yaw0 = sc[o, 1] * 2 * math.pi / a.n_sec
            T, info = register4(KF[q].pts, KF[q].w, KF[c].pts, yaw0)
            # true anchor <- query in the level frames
            Lt = np.linalg.inv(level_frame(G[c])) @ level_frame(G[q])
            if T is not None:
                E = np.linalg.inv(Lt) @ T
                et, ey = float(np.linalg.norm(E[:2, 3])), float(abs(math.degrees(math.atan2(E[1, 0], E[0, 0]))))
                ez = float(abs(E[2, 3]))
            else:
                et = ey = ez = np.inf
            rows.append((q, c, rank, sc[o, 0], dtrue[o], info["inliers"], info.get("inliers_above", 0.0), info["rmse"], et, ez, ey,
                         float(np.linalg.eigvalsh(info["H"][:3, :3] if T is not None else np.eye(3))[0]), has_pos))
        if q % 50 == 0:
            print(f"  {q}/{len(KF)}, queries with a true place {n_pos}, recall@1 {hit1 / max(1, n_pos):.2f}, "
                  f"@5 {hit5 / max(1, n_pos):.2f}, {time.time() - t0:.0f} s", flush=True)
    R = np.array(rows, dtype=float)
    cols = ["q", "c", "rank", "score", "dtrue", "inliers", "inliers_above", "rmse", "err_xy", "err_z", "err_yaw", "h_min", "has_pos"]
    res = {"keyframes": len(KF), "queries_with_true_place": int(n_pos), "recall_at_1": hit1 / max(1, n_pos),
           "recall_at_5": hit5 / max(1, n_pos), "kf_m": a.kf, "submap_m": a.submap, "sep_m": a.sep, "true_m": a.true_m}
    good = (R[:, 8] < 0.5) & (R[:, 10] < 5)                       # registration within 0.5 m / 5 deg of the truth
    res["verify_candidates"] = int(len(R)); res["verify_correct"] = int(good.sum())
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        acc = R[:, 6] >= thr
        res[f"accept_inliers_above_{thr:g}"] = {"accepted": int(acc.sum()), "correct": int((acc & good).sum()),
                                               "precision": float((acc & good).sum() / max(1, acc.sum()))}
    print(json.dumps(res, indent=1))
    out = Path(a.out) if a.out else Path(a.ao).with_name("place3d")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), rows=R, cols=np.array(cols), sel=sel, caps=caps[sel], s=sk)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    top = R[:, 2] == 0
    for m, c, lab in ((R[:, 4] < a.true_m, "tab:green", "true place"), (R[:, 4] >= a.true_m, "tab:red", "other place")):
        ax[0].hist(R[top & m, 3], bins=40, alpha=0.6, color=c, label=lab)
        ax[1].hist(R[m, 6], bins=40, alpha=0.6, color=c, label=lab)
    ax[0].set_title("descriptor score, top-1 candidate"); ax[0].legend()
    ax[1].set_title(f"registration inliers (above ground), top-{a.top}"); ax[1].legend()
    ax[2].scatter(R[:, 6], np.clip(R[:, 8], 0, 5), s=4, c=np.where(good, "tab:green", "tab:red"))
    ax[2].set_xlabel("inliers above ground"); ax[2].set_ylabel("registration error xy (m), clipped at 5")
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=110)


if __name__ == "__main__":
    main()
