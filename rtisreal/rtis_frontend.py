# Vendored into RTISSLAM-3D from RTISSLAM Real (rtisreal), commit 5c0dc2c.
"""Real-sensor imaging front end: eRTIS measurements -> 2D energyscapes on the RTISSLAM grid, with RTIS CUDA.

RTIS CUDA (Cosys-Lab, rtis-software/rtiscuda, >= 2.6.2) is the lab's eRTIS processing chain, the one rtisdev runs on
the sensor's host: PDM demodulation (512-tap FIR, decimation 10: 4.5 MHz -> 450 kHz), matched filter (FFT, circular,
with a reference spectrum the caller supplies), D(M)AS(-CF) on INTEGER sample delays, then an envelope. This module
uses RTIS CUDA up to and including the beamformer (|DMAS_n x CF| per direction at 450 kHz) and forms the energyscape
itself exactly as sondec does (|.| -> brick-wall low-pass at LP_HZ -> mean power per 2 cm range cell -> dB), so the
energyscapes are the same quantity as in simulation and every tuned RTISSLAM parameter keeps its meaning.

The matched filter reference is sondec's (band 18..82 kHz of the call, normalized to unit peak at zero lag), so with
the same call RTIS CUDA and sondec differ only in the delays (integer vs exact fractional) and in the PDM front end.
Demodulated signals are zero-padded by one call length (+ BF_DROP) before RTIS CUDA: its matched filter is a circular
FFT correlation over the recording length, and without padding the direct path (emitter -> microphones, by far the
strongest arrival) wraps into the last call length of the record (real eRTIS, 10 m records: a bright band at 9.3..9.8 m).
The beamformer's output sample i is the array centre's time i + shift_d (RTIS CUDA references the earliest
microphone), compensated per direction when the range cells are formed.

    imager = RtisCudaImager(RtisSetup.from_layout("eRTIS_v3D1", chirp=rtisdev_chirp(...)), ranges, azimuths)
    es_db = imager(raw_uint32)              # one measurement, raw PDM words (as stored by rtisdev)
    es_db = imager.from_signals(x)          # or microphone signals after PDM demodulation [M, N] (the simulator's)

    with CHIRPDataset(path) as ds:          # a CHIRP dataset (Cosys-Lab wjansen/chirp, chirppy): signals are stored
        ac, tx = ds.get_acoustic_sensor(1), ds.get_acoustic_transmitter(1)     # already PDM-demodulated (450 kHz)
        imager = RtisCudaImager(RtisSetup.from_chirp(ac, tx))
        es_db = imager.from_signals(ac[i]["signals"].T)

Geometry: sensor frame x forward, y left, z up (sondec, rtissim, CHIRP); azimuth positive to the left. CHIRP stores
the receiver positions in that frame ([x, y, z], calibration/receiver_relative_translations), so from_chirp needs no
convention. The eRTIS layout files (rtisreal/ertis_layouts, converted from rtisdev) give the microphones in the array
plane as (u, v); here u -> y, v -> z, which is what rtisdev's own steering implies (and chirppy's example confirms:
CHIRP [x, y, z] -> rtisdev [y, z, x]); if ever mirrored, use RtisSetup.from_layout(..., flip_u=True).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sondec.das import RangeGrid
from sondec.sensor import C_SOUND

LAYOUT_DIR = Path(__file__).resolve().parent / "ertis_layouts"
FS_PDM = 4.5e6
PDM_DECIM = 10
PDM_FC = 100e3          # rtisdev: 100 kHz PDM low-pass at 4.5 MHz
BF_DROP = 512           # rtisdev: samples dropped at the end of the beamformer output
LP_HZ = 30e3            # envelope low-pass, as sondec/dmas.py (simulation); rtisdev uses 1 kHz. Real eRTIS with robot
                        # interference: a lower cut-off averages the interference along range (lp_hz argument)
BAND = (18e3, 82e3)     # matched filter band, as sondec/das.py (call band 20..80 kHz +- 2 kHz)
METHODS = {"das": (1, False), "dascf": (1, True), "dmas2": (2, False), "dmas2cf": (2, True),
           "dmas3": (3, False), "dmas3cf": (3, True), "dmas4": (4, False), "dmas4cf": (4, True),
           "dmas5": (5, False), "dmas5cf": (5, True)}


def rtisdev_chirp(f_max=80e3, f_min=20e3, fs=450e3, duration_ms=2.5, win_pct=10.0) -> np.ndarray:
    """The eRTIS call as rtisdev generates it (RTISGenerateSettings.fm_sweep): hyperbolic FM sweep f_max -> f_min,
    Hann-tapered win_pct % at both ends. rtisdev's premade recordings: 20..80 or 25..50 kHz, 2.5 ms (and 'long' /
    'short' variants); the recording's own settings decide."""
    T = duration_ms * 1e-3
    t = np.arange(0, T + 1 / fs, 1 / fs)
    sig = np.sin(2 * np.pi * (T / (1 / f_min - 1 / f_max)) * (np.log(1 / f_max + t * (1 / f_min - 1 / f_max) / T) - np.log(1 / f_max)))
    n_win = int(np.round(sig.size * 2 * win_pct / 100))
    w = np.hanning(n_win + 2)[1:-1]
    on, off = w[:int(np.ceil(n_win / 2))], w[int(np.ceil(n_win / 2)):]
    win = np.ones(sig.size)
    win[:on.size], win[win.size - off.size:] = on, off
    return sig * win


@dataclass
class RtisSetup:
    """What one recording says about the sensor: microphone positions in the sensor frame [M, 3] (m), the call (at
    fs_adc), the PDM samples per measurement and the disabled microphones."""
    positions: np.ndarray
    chirp: np.ndarray
    n_pdm: int = 163840                 # PDM samples per microphone per measurement (rtisdev "microphoneSamples")
    fs_pdm: float = FS_PDM
    disabled: list = field(default_factory=list)
    name: str = ""
    t0: float = 0.0                     # s, time of signal sample 0 after the emission (CHIRP converter select_range:
                                        # min_range * 2 / 343; not stored in the file, so it must be given)

    @property
    def fs_adc(self) -> float:
        return self.fs_pdm / PDM_DECIM

    @property
    def n_adc(self) -> int:
        return self.n_pdm // PDM_DECIM

    @classmethod
    def from_layout(cls, layout: str = "eRTIS_v3D1", chirp: np.ndarray | None = None, flip_u: bool = False, **kw):
        d = json.loads((LAYOUT_DIR / f"{layout}.json").read_text())
        uv = np.array([[np.nan if c is None else c for c in row] for row in d["uv"]], float)
        p = np.zeros((len(uv), 3))
        p[:, 1] = -uv[:, 0] if flip_u else uv[:, 0]
        p[:, 2] = uv[:, 1]
        p[np.isnan(p)] = 0.0
        return cls(positions=p, chirp=rtisdev_chirp() if chirp is None else np.asarray(chirp, float),
                   disabled=list(d["disabled"]), name=layout, **kw)

    @classmethod
    def from_chirp(cls, acoustic, transmitter, n_samples: int | None = None, t0: float = 0.0):
        """From chirppy's AcousticSensor / AcousticTransmitter (one eRTIS of a CHIRP dataset). Receiver positions are
        centred over the enabled microphones in the array plane (y, z), as chirppy's rtis example and rtisdev do."""
        p = np.array(acoustic.get_receiver_positions(), float)
        dis = np.where(np.isnan(p).any(1))[0].tolist()
        p[:, 1:] -= np.nanmean(p[:, 1:], 0)
        p[np.isnan(p)] = 0.0
        fs = float(acoustic.sample_rates)
        if float(transmitter.attrs.get("sample_rate_hz", fs)) != fs:
            raise ValueError("transmitter and receiver sample rates differ: resample the emitted signal first")
        n = int(n_samples if n_samples is not None else acoustic.signals.shape[1])
        return cls(positions=p, chirp=np.asarray(transmitter.get_emitted_signal(), float), n_pdm=n * PDM_DECIM,
                   fs_pdm=fs * PDM_DECIM, disabled=dis, name=str(acoustic.attrs.get("device_model", "chirp")), t0=t0)


class RtisCudaImager:
    """One RTIS CUDA pipeline per input kind (raw PDM words / demodulated signals), preloaded on first use."""
    _count = 0

    def __init__(self, setup: RtisSetup, ranges=None, azimuths=None, method: str = "dmas3cf", directions=None,
                 grid: RangeGrid = RangeGrid(), device: str = "cuda", lp_hz: float = LP_HZ,
                 lp_shape: str = "brick"):
        import rtiscuda
        self.rc, self.setup, self.method, self.dev = rtiscuda, setup, method, device
        self.order, self.cf = METHODS[method]
        # directions: horizontal fan of azimuths (default, 181 over -90..90 deg), or any set of unit vectors [D, 3]
        # in the sensor frame (x forward, y left, z up), e.g. a uniform frontal hemisphere for 3D imaging
        if directions is not None:
            u3 = np.asarray(directions, float)
            u3 = u3 / np.linalg.norm(u3, axis=1, keepdims=True)
            self.az, self.el = np.arctan2(u3[:, 1], u3[:, 0]), np.arcsin(np.clip(u3[:, 2], -1, 1))
        else:
            self.az = np.radians(np.linspace(-90, 90, 181)) if azimuths is None else np.asarray(azimuths, float)
            self.el = np.zeros_like(self.az)
            u3 = np.c_[np.cos(self.az), np.sin(self.az), np.zeros(len(self.az))]
        self.directions = u3
        self.grid = grid
        if ranges is not None:
            assert np.allclose(ranges, grid.r("cpu").numpy(), atol=1e-4), "ranges must be the RangeGrid's cell centres"
        self.M, self.D, self.n_rec = len(setup.positions), len(self.az), setup.n_adc
        # signals path: zero padding against the circular matched filter wrap (raw PDM words cannot be padded:
        # there the range grid must stay below n_rec - call length)
        self.pad = int(np.ceil((len(setup.chirp) + BF_DROP) / 64.0)) * 64
        self.N = self.n_rec + self.pad
        fs = setup.fs_adc
        # integer delays: mic m advanced by d_m = a_m - min a, a_m = -p_m . u / c * fs (its arrival time re. the centre)
        u = self.directions
        a = -(setup.positions @ u.T) / C_SOUND * fs                                       # [M, D]
        ai = np.round(a)
        self.delays = (ai - ai.min(0)).astype(np.int32)                                    # [M, D] (rtiscuda layout)
        self.shift = -ai.min(0)                                                            # [D] samples: centre time - i
        # matched filter reference (sondec's): conj(C) on the band, unit zero-lag peak; rtiscuda divides by N after irfft
        C = np.fft.rfft(setup.chirp, self.N)
        f = np.fft.rfftfreq(self.N, 1 / fs)
        band = (f >= BAND[0]) & (f <= BAND[1])
        H = np.where(band, np.conj(C), 0) / np.sum(np.abs(C[band]) ** 2) * self.N
        self.mf_re = np.ascontiguousarray(np.tile(H.real, (self.M, 1)), np.float32)
        self.mf_im = np.ascontiguousarray(np.tile(H.imag, (self.M, 1)), np.float32)
        self.pdm_filter = __import__("scipy.signal", fromlist=["firwin"]).firwin(512, PDM_FC / (setup.fs_pdm / 2)).astype(np.float32)
        self.n_bf = self.N - BF_DROP
        # envelope low-pass and range cells, per direction shifted to the array centre's time
        # envelope low-pass, zero phase (acausal: no range offset): "brick" (sondec, simulation), "gauss" (-3 dB at
        # lp_hz) or "filtfilt" (|H|^2 of a 2nd-order Butterworth at lp_hz = scipy filtfilt(butter(2)), as CLAIT).
        # A brick wall at a few kHz rings the direct path through the whole record.
        self.lp_hz, self.lp_shape = lp_hz, lp_shape
        fr = np.fft.rfftfreq(self.n_bf, 1 / fs)
        if lp_shape == "brick":
            H = (fr <= lp_hz).astype(float)
        elif lp_shape == "gauss":
            H = np.exp(-np.log(2) * (fr / lp_hz) ** 2)
        elif lp_shape == "filtfilt":
            from scipy.signal import butter, freqz
            b, a_ = butter(2, lp_hz / (fs / 2))
            H = np.abs(freqz(b, a_, worN=fr, fs=fs)[1]) ** 2
        else:
            raise ValueError(lp_shape)
        self.lp = torch.as_tensor(H, dtype=torch.float32, device=device)
        edges = grid.r_min + grid.dr * np.arange(grid.n_r + 1)
        e = np.round((edges[None] * 2 / C_SOUND - setup.t0) * fs - self.shift[:, None]).astype(np.int64)  # [D, R+1]
        assert e.min() >= 0 and e.max() <= self.n_rec - self.shift.max(), \
            f"range grid {edges[0]:.2f}..{edges[-1]:.2f} m outside the record (t0 {setup.t0 * 1e3:.2f} ms, {self.n_rec} samples)"
        self.e0 = torch.as_tensor(e[:, :-1], device=device)
        self.e1 = torch.as_tensor(e[:, 1:], device=device)
        RtisCudaImager._count += 1
        self.ids = {k: f"rtisslam{RtisCudaImager._count}_{k}" for k in ("raw", "sig")}
        self.loaded: set = set()

    def _preload(self, raw: bool) -> str:
        pid = self.ids["raw" if raw else "sig"]
        if pid not in self.loaded:
            z = np.zeros((1, 1), np.float32)
            dis = np.full(64, 64, np.uint32)
            dis[:len(self.setup.disabled)] = self.setup.disabled
            if raw:
                raise NotImplementedError("raw PDM input: the matched filter would need the unpadded length")
            self.rc.preload(self.pdm_filter, z, z, z, self.mf_re, self.mf_im, self.delays, self.M, self.D,
                            self.N * PDM_DECIM, PDM_DECIM, BF_DROP, 10, 0, 1, np.float32(2), np.float32(0.5), pid,
                            int(raw), 0, 1, 1, 0, 0, 0, dis, self.order, int(self.cf))
            self.loaded.add(pid)
        return pid

    def close(self):
        for pid in self.loaded:
            self.rc.unloadone(pid)
        self.loaded.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def beamform(self, data: np.ndarray) -> np.ndarray:
        """Raw PDM words uint32 [n_pdm] or signals float [M, N] at fs_adc -> |DMAS_n (x CF)| [D, N - BF_DROP]."""
        raw = data.dtype == np.uint32
        if raw:
            assert data.size == self.setup.n_pdm, (data.size, self.setup.n_pdm)
        else:
            assert data.shape == (self.M, self.n_rec), (data.shape, (self.M, self.n_rec))
            data = np.ascontiguousarray(np.pad(data.astype(np.float32), ((0, 0), (0, self.pad))))
        return self.rc.processing(data, int(raw), self.D, self.n_bf, self._preload(raw))

    @torch.no_grad()
    def energyscape(self, bf: np.ndarray) -> np.ndarray:
        """|beamformer| [D, T] -> [R, D] dB: brick-wall low-pass at lp_hz, mean power per range cell (sondec)."""
        y = torch.as_tensor(bf, dtype=torch.float32, device=self.dev)
        env = torch.fft.irfft(torch.fft.rfft(y, dim=-1) * self.lp, self.n_bf, dim=-1).clamp_min(0)
        cs = torch.nn.functional.pad(torch.cumsum(env.double() ** 2, -1), (1, 0))
        P = (cs.gather(1, self.e1) - cs.gather(1, self.e0)) / (self.e1 - self.e0).double()
        return (10 * torch.log10(P.float() + 1e-30)).T.contiguous().cpu().numpy()

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.energyscape(self.beamform(np.asarray(raw, np.uint32)))

    def from_signals(self, x) -> np.ndarray:
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        return self.energyscape(self.beamform(x.astype(np.float32)))


class InterferenceNuller:
    """Removes stationary directional interference (the robot's own drive electronics) from the microphone signals,
    before any beamformer: per frequency, the cross-spectral matrix of the echo-poor part of many captures (beyond
    ~1.5 m; echoes are sparse there, the interference is always on) is eigen-decomposed, and its dominant eigenvectors
    (eigenvalue > tau x the median, at most kmax) are projected out, P(f) = I - U U^H. The interference sits at fixed
    positions relative to the array, so one projector serves the whole recording. Echoes from those same directions
    are attenuated too (forest recording: a few directions per frequency band).

        nul = InterferenceNuller.fit(captures)          # iterable of [N, M] signals (CHIRP layout)
        x_clean = nul(x)                                # [M, N]
    """

    def __init__(self, P: np.ndarray, L: int, fs: float, info: dict | None = None):
        self.P, self.L, self.fs, self.info = np.asarray(P), int(L), float(fs), info or {}

    @classmethod
    def fit(cls, captures, fs: float = 450e3, start: int = 4000, L: int = 256, tau: float = 4.0, kmax: int = 4):
        win = np.hanning(L)
        R, cnt = None, 0
        for x in captures:
            x = np.asarray(x, float)[start:]
            fr = np.lib.stride_tricks.sliding_window_view(x, L, axis=0)[::L // 2]          # [F, M, L]
            S = np.fft.rfft(fr * win, axis=-1)
            r = np.einsum("fmk,fnk->kmn", S, S.conj())
            R = r if R is None else R + r
            cnt += S.shape[0]
        R /= cnt
        w, V = np.linalg.eigh(R)
        P, ks = [], []
        for i in range(len(w)):
            e, U = w[i][::-1], V[i][:, ::-1]
            k = int(min(kmax, (e > tau * np.median(e)).sum()))
            P.append(np.eye(R.shape[1]) - U[:, :k] @ U[:, :k].conj().T)
            ks.append(k)
        return cls(np.array(P), L, fs, {"tau": tau, "kmax": kmax, "start": start, "captures": cnt, "k": ks})

    def save(self, path):
        np.savez(path, P=self.P, L=self.L, fs=self.fs, k=np.asarray(self.info.get("k", [])))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["P"], int(z["L"]), float(z["fs"]), {"k": z["k"].tolist()})

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        M, N = x.shape
        n = N + 4 * self.L                                       # padding: no circular wrap of the projection
        X = np.fft.rfft(x, n, axis=1)
        kb = np.clip(np.round(np.fft.rfftfreq(n, 1 / self.fs) / self.fs * self.L).astype(int), 0, len(self.P) - 1)
        Pt = torch.as_tensor(self.P[kb], dtype=torch.complex64, device="cuda" if torch.cuda.is_available() else "cpu")
        Y = torch.einsum("kmn,nk->mk", Pt, torch.as_tensor(X, dtype=torch.complex64, device=Pt.device))
        return np.fft.irfft(Y.cpu().numpy(), n, axis=1)[:, :N].astype(np.float32)


def direct_path_sample(x: np.ndarray, chirp: np.ndarray, search: int = 3000) -> int:
    """Sample of the direct path (emitter -> microphones) in a capture [M, N]: peak of the matched filter of the
    microphone mean over the first `search` samples."""
    m = np.asarray(x, float)[:, :search].mean(0)
    return int(np.argmax(np.abs(np.correlate(m, np.asarray(chirp, float), "valid"))))


def align_direct_path(x: np.ndarray, chirp: np.ndarray, ref: int, max_shift: int = 200) -> tuple[np.ndarray, int]:
    """Shift a capture [M, N] so that its direct path lands on sample `ref` (the recording's usual value). The direct
    path has a fixed geometry, so a different arrival is a timing jitter between the emission and the start of the
    recording (forest recording: ~2 % of the captures 33 samples late). Zero fill; shifts beyond max_shift are not
    trusted and left alone. Returns (x, shift)."""
    d = ref - direct_path_sample(x, chirp)
    if d == 0 or abs(d) > max_shift:
        return x, 0
    y = np.zeros_like(x)
    if d > 0:
        y[:, d:] = x[:, :-d]
    else:
        y[:, :d] = x[:, -d:]
    return y, d


class RealPreprocessor:
    """The preprocessing of real eRTIS captures before beamforming, fitted once per recording:
        1. align the capture to its direct path (timing jitter between emission and recording start),
        2. subtract the static background: the mean of many aligned captures, i.e. everything fixed relative to the
           sensor (direct path emitter -> microphones and its ringing, echoes of the robot itself; forest recording:
           93 % of the power of the first 0.75 m, < 2 % beyond). Echoes of the moving scene, ground included, average
           out and stay,
        3. null the stationary directional interference (InterferenceNuller).

        pre = RealPreprocessor.fit(ac, tx)          # chirppy acoustic sensor / transmitter; ~40 s
        pre.save(path); pre = RealPreprocessor.load(path)
        x_clean = pre(np.asarray(ac.signals[k]).T)  # [M, N]
    """

    def __init__(self, chirp, dp_ref: int, background: np.ndarray, nuller: InterferenceNuller):
        self.chirp, self.dp_ref, self.B, self.nuller = np.asarray(chirp, float), int(dp_ref), np.asarray(background, np.float32), nuller

    @classmethod
    def fit(cls, acoustic, transmitter, n_captures: int = 300, **nuller_kw):
        chirp = np.asarray(transmitter.get_emitted_signal(), float)
        caps = np.linspace(0, len(acoustic) - 1, n_captures).astype(int)
        X = [np.asarray(acoustic.signals[i]).T for i in caps]
        dp_ref = int(np.median([direct_path_sample(x, chirp) for x in X[::5]]))
        X = [align_direct_path(x, chirp, dp_ref)[0] for x in X]
        B = np.mean(X, axis=0).astype(np.float32)
        nul = InterferenceNuller.fit(((x - B).T for x in X), fs=float(acoustic.sample_rates), **nuller_kw)
        return cls(chirp, dp_ref, B, nul)

    def save(self, path):
        P = self.nuller
        np.savez(path, chirp=self.chirp, dp_ref=self.dp_ref, B=self.B, P=P.P, L=P.L, fs=P.fs, k=np.asarray(P.info.get("k", [])))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["chirp"], int(z["dp_ref"]), z["B"], InterferenceNuller(z["P"], int(z["L"]), float(z["fs"]), {"k": z["k"].tolist()}))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x, _ = align_direct_path(np.asarray(x, np.float32), self.chirp, self.dp_ref)
        return self.nuller(x - self.B)
