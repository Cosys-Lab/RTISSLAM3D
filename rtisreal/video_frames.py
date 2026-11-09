# Vendored into RTISSLAM-3D from RTISSLAM Real (rtisreal), commit 5c0dc2c.
"""Frames of a CHIRP video sensor (e.g. the GoPro, `video_00`: an MP4 next to the .h5) by robot time, decoded
sequentially: seek once to the start, then read forward. For renders whose frame times only increase (one reader per
render chunk), this is far faster than random access into a long HEVC file.

    rd = VideoFrames(ds.get_video(0), t_first, width=960)
    img = rd.at(t)          # RGB uint8 [h, w, 3] of the frame closest to robot time t (t non-decreasing)
    rd.close()

Timing: frame k is at video.effective_timestamps[k] (robot clock, converter's sync: GoPro forest, +-0.11 s).
Before the first / after the last frame: the first / last frame.
"""
from __future__ import annotations

import numpy as np


class VideoFrames:
    def __init__(self, video, t_first: float, width: int = 960):
        import imageio_ffmpeg
        self.ts = np.asarray(video.effective_timestamps, float)
        self.path = video.video_path
        w, h = video.resolution
        self.size = (width, int(round(h * width / w / 2)) * 2)
        self.k0 = int(np.clip(np.searchsorted(self.ts, t_first) - 1, 0, len(self.ts) - 1))
        pts = self.ts[self.k0] - self.ts[0]
        self.gen = imageio_ffmpeg.read_frames(self.path, input_params=["-ss", f"{pts:.4f}"],
                                              output_params=["-vf", f"scale={self.size[0]}:{self.size[1]}"],
                                              pix_fmt="rgb24")
        meta = next(self.gen)                                   # first item: metadata
        self.k = self.k0 - 1                                    # index of the frame last read
        self.frame = None
        self.done = False

    def _read(self):
        try:
            b = next(self.gen)
        except StopIteration:
            self.done = True
            return False
        self.frame = np.frombuffer(b, np.uint8).reshape(self.size[1], self.size[0], 3)
        self.k += 1
        return True

    def at(self, t: float) -> np.ndarray:
        """Frame closest to robot time t (t must not decrease between calls)."""
        target = int(np.clip(np.argmin(np.abs(self.ts - t)), 0, len(self.ts) - 1))
        while (self.frame is None or self.k < target) and not self.done:
            self._read()
        return self.frame

    def close(self):
        try:
            self.gen.close()
        except Exception:
            pass
