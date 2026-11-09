"""Building blocks for Cosys-Lab research videos (see ../SKILL.md).

Copy this file into the project (e.g. scripts/cosys_video.py) together with ../assets, then write a short
project script that calls these functions. Running this file directly renders a 20 s demo with dummy data:

    python cosys_video.py --out demo.mp4

Requires numpy, matplotlib, imageio, imageio-ffmpeg (and PyMuPDF only for crop_pdf_figure).
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"           # copied from 2026 - Skill Drafts/video-steckel-style (7c102cc)
W, H, DPI = 1920, 1080, 120                      # 1080p; DPI only sets how font sizes map to pixels
FPS = 10                                          # data rate of the sensor; slides are held at the same rate
MAGENTA = (177 / 255, 0, 145 / 255)               # UAntwerp magenta, sampled from the lab's DMAS video
import os as _os
ANON = _os.environ.get("COSYS_VIDEO_ANON", "") not in ("", "0")   # double-anonymous review: no logos, no lab end slide
def _sans():
    """Calibri if installed, else Lato (the base of Calibri's metric clone Carlito), else DejaVu Sans."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Calibri", "Carlito", "Lato") if n in names), "DejaVu Sans")


SANS = {"fontname": _sans()}                     # title slide (closest available font to the UAntwerp house font)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


# ------------------------------------------------------------------------------------------------ basics
def new_fig(bg="white"):
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=bg)


def grab(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def writer(path, fps=FPS):
    """H.264, 1920x1080 (divisible by 8, so no rescaling), yuv420p by default"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8)


def fade(img, a):
    """blend towards white (a = 1: image, a = 0: white)"""
    return img if a >= 1 else (255 - a * (255 - img.astype(np.float32))).astype(np.uint8)


def hold(wr, img, sec, fps=FPS, fade_sec=0.6, fade_in=True, fade_out=True):
    """write a still for `sec` seconds with a fade from / to white"""
    n = int(round(sec * fps)); nf = max(1, int(round(fade_sec * fps)))
    for i in range(n):
        a = min(1.0, (i + 1) / nf if fade_in else 1.0, (n - i) / nf if fade_out else 1.0)
        wr.append_data(fade(img, a))


def logo(name):
    return imageio.imread(ASSETS / name)


def put(fig, img, x, y, h):
    """place an image with its lower-left corner at (x, y) and height h (figure fractions); returns width"""
    w = h * img.shape[1] / img.shape[0] * H / W
    ax = fig.add_axes([x, y, w, h]); ax.imshow(img, interpolation="lanczos"); ax.axis("off")
    return w


def logos(fig, white=False):
    """UAntwerp / Cosys-Lab bottom left, Flanders Make bottom right (on every white frame); none when ANON"""
    if ANON:
        return
    sfx = "_white" if white else ""
    put(fig, logo(f"uantwerp_cosys_logo{sfx}.png"), 0.02, 0.025, 0.075)
    fm = logo(f"flanders_make_logo{sfx}.png")
    put(fig, fm, 0.98 - 0.11 * fm.shape[1] / fm.shape[0] * H / W, 0.005, 0.11)


# ------------------------------------------------------------------------------------------------ slides
def slide_title(title: str) -> np.ndarray:
    """magenta background, white bold sans title, white logos at the bottom"""
    fig = new_fig(MAGENTA)
    txt = fig.text(0.5, 0.56, title, ha="center", va="center", fontsize=46, color="white", fontweight="bold",
                   linespacing=1.35, **SANS)
    fig.canvas.draw()                                   # shrink until the widest line fits in 90 % of the width
    while txt.get_window_extent().width > 0.9 * W and txt.get_fontsize() > 20:
        txt.set_fontsize(txt.get_fontsize() - 1)
        fig.canvas.draw()
    if not ANON:
        put(fig, logo("uantwerp_cosys_logo_white.png"), 0.195, 0.12, 0.085)
        fm = logo("flanders_make_logo_white.png")
        put(fig, fm, 0.80 - 0.12 * fm.shape[1] / fm.shape[0] * H / W, 0.105, 0.12)
    img = grab(fig); plt.close(fig); return img


def slide_text(head: str, bullets: list[str], figure: np.ndarray | None = None, wrap=84) -> np.ndarray:
    """white slide, serif heading, optional figure (e.g. the overview figure of the paper), wrapped bullets"""
    fig = new_fig()
    fig.text(0.5, 0.9, head, ha="center", va="center", fontsize=40)
    y = 0.76
    if figure is not None:
        ax = fig.add_axes([0.06, 0.47, 0.88, 0.36]); ax.imshow(figure, interpolation="lanczos"); ax.axis("off")
        y = 0.42
    for b in bullets:
        for j, w in enumerate(textwrap.wrap(b, wrap)):
            fig.text(0.1, y, ("•  " if j == 0 else "    ") + w, fontsize=23)
            y -= 0.052
        y -= 0.015
    logos(fig)
    img = grab(fig); plt.close(fig); return img


def slide_end() -> np.ndarray:
    """the colour UAntwerp / Cosys-Lab logo, centred on white (an empty white frame when ANON)"""
    fig = new_fig()
    if ANON:
        img = grab(fig); plt.close(fig); return img
    lg = logo("uantwerp_cosys_logo_onwhite.png")
    put(fig, lg, 0.5 - 0.5 * 0.2 * lg.shape[1] / lg.shape[0] * H / W, 0.4, 0.2)
    img = grab(fig); plt.close(fig); return img


def crop_pdf_figure(pdf, page_text: str, caption_start: str, keys: tuple[str, ...], dpi=330) -> np.ndarray:
    """raster crop of a (vector) figure from the compiled paper: the text blocks containing `keys` on the page
    that contains `page_text`, above the caption that starts with `caption_start` (e.g. "Fig. 2.")"""
    import fitz
    page = next(p for p in fitz.open(pdf) if page_text in p.get_text())
    blocks = page.get_text("blocks")
    cap = next(b for b in blocks if b[4].startswith(caption_start))
    blk = [b for b in blocks if any(k in b[4] for k in keys) and b[3] < cap[1]]
    r = fitz.Rect(min(b[0] for b in blk) - 8, min(b[1] for b in blk) - 8, max(b[2] for b in blk) + 8,
                  max(b[3] for b in blk) + 6)
    pix = page.get_pixmap(dpi=dpi, clip=r)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


# ------------------------------------------------------------------------------------------------ data frames
class DataLayout:
    """The main frame: two-line serif title on top, status column (bold label: value) on the left, a context
    view below it (camera image or top view of the scene), a grid of titled panels with one dB colorbar per
    row on the right, logos in the bottom corners, a grey note line at the bottom.

        lay = DataLayout(title, ["Room", "Pulse"], panels=["DAS", ..., "Proposed"], ncol=4, highlight="Proposed")
        lay.ctx      -> axes for the context view (draw once, update artists per frame)
        lay.axes[k]  -> axes of panel k (create the image artist once, then set_data per frame)
        lay.status("Pulse", "7 of 150"); frame = lay.frame()
    """

    def __init__(self, title, status_labels, panels, ncol=4, highlight=None, cbar_label="Normalized acoustic energy (dB)",
                 vmin=-25, vmax=0, cmap="inferno", note=""):
        self.fig = fig = new_fig()
        fig.text(0.5, 0.925, title, ha="center", va="center", fontsize=30, linespacing=1.2)
        self._st = {}
        for i, lab in enumerate(status_labels):
            fig.text(0.02, 0.78 - 0.042 * i, f"{lab}:", fontsize=19, fontweight="bold")
            self._st[lab] = fig.text(0.15, 0.78 - 0.042 * i, "", fontsize=19)
        self.ctx = fig.add_axes([0.035, 0.19, 0.175, 0.37])
        nrow = int(np.ceil(len(panels) / ncol))
        x0, x1 = 0.235, 0.925
        pw, top, bot = (x1 - x0) / ncol, 0.83, 0.13
        ph = (top - bot) / nrow
        self.axes = []
        for k, lab in enumerate(panels):
            r, c = divmod(k, ncol)
            ax = fig.add_axes([x0 + c * pw + 0.003, top - (r + 1) * ph + 0.02, pw - 0.006, ph - 0.05])
            hl = lab == highlight
            ax.set_title(lab, fontsize=19, fontweight="bold", color=MAGENTA if hl else "k", pad=4)
            self.axes.append(ax)
        self.cmap = plt.get_cmap(cmap).copy(); self.cmap.set_bad("white")
        for r in range(nrow):
            cax = fig.add_axes([0.935, top - (r + 1) * ph + 0.055, 0.009, ph - 0.11])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=self.cmap), cax=cax)
            cb.ax.tick_params(labelsize=11); cb.set_label(cbar_label, fontsize=11)
        if note:
            fig.text(0.5, 0.075, note, ha="center", fontsize=13, color="0.4")
        logos(fig)

    def status(self, label, value):
        self._st[label].set_text(value)

    def frame(self):
        return grab(self.fig)

    def close(self):
        plt.close(self.fig)


# ------------------------------------------------------------------------------------------------ demo
def demo(out):
    rng = np.random.default_rng(0)
    wr = writer(out)
    hold(wr, slide_title("Title of the Paper on Two Lines\nWith the Second Line Here"), 4, fade_in=False)
    hold(wr, slide_text("What is shown", ["First bullet: the sensor and the scene.",
                                          "Second bullet: the compared methods, and which one is the proposed one.",
                                          "Third bullet: what the markers mean (reference only)."]), 5)
    panels = ["DAS", "DMAS", "DMAS + CF", "DMAS3 + CF", "Baseline A", "Baseline B", "Oracle", "Proposed"]
    lay = DataLayout("Title of the Paper on Two Lines\nWith the Second Line Here", ["Sequence", "Frame"], panels,
                     highlight="Proposed", note="circles: ground truth (reference only)")
    lay.ctx.plot(np.cumsum(rng.normal(size=60)), np.cumsum(rng.normal(size=60)), c="tab:blue")
    lay.ctx.set_title("scene (top view)", fontsize=17, fontweight="bold")
    ims = [ax.imshow(np.zeros((90, 180)), cmap=lay.cmap, vmin=-25, vmax=0, origin="lower") for ax in lay.axes]
    for ax in lay.axes:
        ax.set_xticks([]); ax.set_yticks([])
    n = 60
    for i in range(n):
        lay.status("Sequence", "1 of 1"); lay.status("Frame", f"{i + 1} of {n}")
        for k, im in enumerate(ims):
            im.set_data(np.clip(10 * np.log10(rng.random((90, 180)) ** (8 + 4 * k) + 1e-6), -25, 0))
        nf = 6; a = min(1.0, (i + 1) / nf, (n - i) / nf)
        wr.append_data(fade(lay.frame(), a))
    lay.close()
    hold(wr, slide_end(), 3, fade_out=False)
    wr.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="demo.mp4")
    demo(ap.parse_args().out)
