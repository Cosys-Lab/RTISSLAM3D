"""The supplementary video of the RTISSLAM-3D paper: title, method (the overview figure of the compiled paper), the raw
sonar data, the SLAM being built, the drive through the sonar and lidar maps, the final 3D map, the
results, and the end slide.

    python video/make_supplementary_video.py --parts <dir> --out <mp4>

<dir> holds the rendered segments (same 1920 x 1080 layout):
    sonar.mp4   video/make_sonar_sequence_video.py <work> --k0 3300 --n 200
    build.mp4   video/make_slam3d_video.py
    drive.mp4   video/make_drive_through_video.py
    map3d.mp4   video/make_3d_map_video.py
With COSYS_VIDEO_ANON=1 (double-anonymous review) all parts and slides are rendered without logos and without the end
slide. The slides are rendered here; the segments' own title, intro and end slides are cut, and all parts are joined at
15 fps with ffmpeg.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosys_video as cv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FPS = 15
TITLE = "RTISSLAM-3D: Full 6-DoF SLAM\nWith an In-Air 3D Imaging Sonar"


def raster(pdf, page=1, dpi=250, clip_pt=None):
    """PDF page (or a clip [x0, y0, x1, y1] in points) -> RGB array, white margins trimmed."""
    import imageio.v2 as iio
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cmd = ["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile"]
        if clip_pt is not None:
            s = dpi / 72
            x0, y0, x1, y1 = clip_pt
            cmd += ["-x", str(int(x0 * s)), "-y", str(int(y0 * s)), "-W", str(int((x1 - x0) * s)), "-H", str(int((y1 - y0) * s))]
        subprocess.run(cmd + [str(pdf), f"{d}/p"], check=True)
        img = iio.imread(f"{d}/p.png")[..., :3]
    nz = np.where((img < 245).any(2))
    return img[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]


def overview_figure():
    """Fig. 1 of the compiled paper (TikZ): the region above its caption on its page."""
    pdf = PAPER / "main.pdf"
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    for p in range(1, n + 1):
        txt = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-bbox", str(pdf), "-"], capture_output=True, text=True).stdout
        words = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', txt)
        for i, w in enumerate(words[:-1]):
            if w[4] == "FIGURE" and words[i + 1][4] == "1.":
                y_cap = float(w[1])
                return raster(pdf, p, 300, (30, 52, 546, y_cap - 4))
    raise RuntimeError("Fig. 1 not found in paper/main.pdf")


def slide_figure(head, fig_img, bullets, wrap=100, fig_h=0.50):
    """White slide: serif heading, a figure, bullets below it, logos."""
    import matplotlib.pyplot as plt
    import textwrap
    fig = cv.new_fig()
    fig.text(0.5, 0.92, head, ha="center", va="center", fontsize=38)
    ax = fig.add_axes([0.05, 0.85 - fig_h, 0.90, fig_h]); ax.imshow(fig_img, interpolation="lanczos"); ax.axis("off")
    y = 0.85 - fig_h - 0.07
    for b in bullets:
        for j, w in enumerate(textwrap.wrap(b, wrap)):
            fig.text(0.5, y, w, fontsize=24, ha="center")
            y -= 0.04
        y -= 0.012
    cv.logos(fig)
    img = cv.grab(fig); plt.close(fig); return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = Path(a.parts)
    sl = parts / "slides"; sl.mkdir(exist_ok=True)

    def still(name, img, sec, fade_in=True, fade_out=True):
        wr = cv.writer(sl / f"{name}.mp4", fps=FPS)
        cv.hold(wr, img, sec, fps=FPS, fade_in=fade_in, fade_out=fade_out); wr.close()
        return sl / f"{name}.mp4"

    s_title = still("title", cv.slide_title(TITLE), 4, fade_in=False)
    s_method = still("method", slide_figure("RTISSLAM-3D", overview_figure(), [], fig_h=0.62), 7)
    s_res = still("results", slide_figure(
        "Results Against the 3D Lidar Reference", raster(PAPER / "figures" / "trajectory.pdf", dpi=250),
        ["Horizontal error 0.28 m, height error 0.12 m over a 545 m drive."], fig_h=0.60), 7)
    s_end = still("end", cv.slide_end(), 3, fade_out=False)

    seq = [s_title, s_method, parts / "sonar.mp4", parts / "build.mp4", parts / "drive.mp4", parts / "map3d.mp4",
           s_res] + ([] if cv.ANON else [s_end])
    # the segment videos carry their own title (and intro) and end slides: one title and one end slide in total
    trim = {parts / "build.mp4": (11.0, 3.0), parts / "drive.mp4": (3.0, 3.0), parts / "map3d.mp4": (3.0, 3.0)}
    for p in seq:
        if not p.exists():
            raise FileNotFoundError(p)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for p in seq:
        cmd += ["-i", str(p)]
    def duration(p):
        err = subprocess.run([ff, "-i", str(p)], capture_output=True, text=True).stderr
        h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
        return 3600 * int(h) + 60 * int(m) + float(sec)
    flt = ""
    for i, p in enumerate(seq):
        cut = ""
        if p in trim:
            t0, t1 = trim[p]
            cut = f"trim=start={t0}:end={duration(p) - t1:.3f},setpts=PTS-STARTPTS,"
        flt += f"[{i}:v]{cut}scale=1920:1080,fps={FPS},format=yuv420p,setsar=1[v{i}];"
    flt += "".join(f"[v{i}]" for i in range(len(seq))) + f"concat=n={len(seq)}:v=1:a=0[out]"
    cmd += ["-filter_complex", flt, "-map", "[out]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-movflags", "+faststart", str(a.out)]
    subprocess.run(cmd, check=True)
    print(a.out)


if __name__ == "__main__":
    main()
