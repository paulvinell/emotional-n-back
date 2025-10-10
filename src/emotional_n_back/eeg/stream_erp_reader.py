# erp_stream.py
# Single-channel OSC → online ERP extraction (P1/N1/N200/P300/LPP)
# Robust to missing fs, chunk overlap/gaps, and unaligned event timing.
# MIT License.

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

# ----------------------- Configuration -----------------------

# Default component windows (seconds) and polarity
COMPONENT_SPECS: List[Tuple[str, Tuple[float, float], str]] = [
    ("P1", (0.080, 0.130), "pos"),
    ("N1", (0.120, 0.180), "neg"),
    ("N200", (0.180, 0.300), "neg"),
    ("P300", (0.300, 0.600), "pos"),
    ("LPP", (0.400, 0.800), "pos_mean"),  # mean amplitude over the window
]

# ----------------------- Utilities ---------------------------


@dataclass
class RingBuffer:
    """Fixed-size ring buffer for single-channel float64 samples."""

    capacity: int
    buf: NDArray[np.float64]
    start_idx: int = 0  # global sample index corresponding to buf[0]
    write_pos: int = 0
    n_written: int = 0

    @classmethod
    def with_capacity(cls, capacity: int) -> "RingBuffer":
        return cls(capacity=capacity, buf=np.zeros(capacity, dtype=np.float64))

    def append(self, x: NDArray[np.float64]) -> None:
        """Append 1-D array of samples."""
        n = int(x.shape[0])
        for i in range(n):
            self.buf[self.write_pos] = float(x[i])
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.n_written += 1
            if self.n_written > self.capacity:
                self.start_idx += 1

    def has_range(self, start_idx: int, end_idx: int) -> bool:
        """Return True if [start_idx, end_idx) is fully available."""
        earliest = self.start_idx
        latest = self.start_idx + min(self.n_written, self.capacity)
        return start_idx >= earliest and end_idx <= latest

    def get_range(self, start_idx: int, end_idx: int) -> NDArray[np.float64]:
        """Materialize [start_idx, end_idx) into a 1-D array."""
        L = end_idx - start_idx
        out = np.empty(L, dtype=np.float64)
        for i in range(L):
            idx = (start_idx - self.start_idx + i) % self.capacity
            out[i] = self.buf[idx]
        return out


@dataclass
class RunningAverage:
    """Incremental mean for ERP updates (1-D)."""

    n: int
    avg: NDArray[np.float64]

    @classmethod
    def init_like(cls, length: int) -> "RunningAverage":
        return cls(n=0, avg=np.zeros(length, dtype=np.float64))

    def update(self, x: NDArray[np.float64]) -> None:
        self.n += 1
        self.avg += (x - self.avg) / self.n


# ----------------------- Stream Epocher ----------------------


class StreamEpocher:
    """
    Online epocher for single-channel EEG.

    - Maintains a ring buffer of recent samples (a bit larger than tmin..tmax).
    - Accepts chunks of raw samples; applies zero-phase band-pass per chunk.
    - Accepts events; as soon as post-stim samples exist, it extracts an epoch,
      baseline-corrects, and updates a running ERP per condition.
    - Extracts P1 / N1 / N200 / P300 peaks, and LPP mean on each update.
    """

    def __init__(
        self,
        fs: float,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
        hp: float = 0.1,
        lp: float = 30.0,
        extra_seconds: float = 2.0,
        component_specs: Optional[List[Tuple[str, Tuple[float, float], str]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        assert fs > 0, "Sampling rate fs must be positive"
        self.fs = float(fs)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.baseline = baseline
        self.logger = logger or logging.getLogger(__name__)
        self.component_specs = component_specs or COMPONENT_SPECS

        # Buffer sized for epoch window + a safety margin
        cap = int((tmax - tmin + extra_seconds) * fs)
        cap = max(cap, 1)
        self.rb = RingBuffer.with_capacity(capacity=cap)

        # Global sample index = number of samples ingested so far
        self.global_idx = 0

        # Pending events: list[(global_sample_idx, code)]
        self.events: List[Tuple[int, str]] = []

        # Incremental ERP per code
        self.running: Dict[str, RunningAverage] = {}

        # Pre-compute filters
        nyq = max(fs / 2.0, 1.0)
        low = max(hp, 0.01) / nyq
        high = min(lp, nyq - 1e-6) / nyq
        if not (0 < low < high < 1):
            raise ValueError(f"Invalid band [{hp}, {lp}] for fs={fs}")
        self.b, self.a = butter(4, [low, high], btype="band")

        # Epoch time base (constant length)
        self.n_pre = int(round(-tmin * fs))
        self.n_post = int(round(tmax * fs))
        self.epoch_len = self.n_pre + self.n_post
        self.t = np.arange(self.epoch_len) / fs + tmin

    # ------------------ ingesting data & events ------------------

    def ingest_chunk(
        self,
        samples: NDArray[np.float64],
        offset_samples: Optional[int] = None,
    ) -> None:
        """
        Ingest a raw chunk of samples. Optionally specify offset_samples to
        correct for gaps/overlaps vs. the previous chunk.
        """
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Samples must be a 1-D array")

        # Realign global index if sender provides a relative offset
        if offset_samples is not None:
            # positive offset = gap (advance); negative = overlap (rewind)
            self.global_idx = max(0, self.global_idx + int(offset_samples))
            # Note: ring buffer start index advances automatically as we append

        # Zero-phase band-pass, robust pad length
        padlen = max(3 * max(len(self.a), len(self.b)), min(31, x.size - 1))
        if padlen > 0 and x.size > padlen:
            x = filtfilt(self.b, self.a, x, axis=0, padlen=padlen)

        self.rb.append(x)
        self.global_idx += x.size

    def ingest_event(
        self,
        code: str,
        at_offset: Optional[int] = None,
        at_global: Optional[int] = None,
    ) -> None:
        """
        Register an event. Choose *one* timing mode:
        - at_global: absolute sample index (preferred)
        - at_offset: sample offset inside the NEXT arriving chunk
        - neither: event will be timestamped at the current end (fallback)
        """
        if at_global is not None:
            idx = max(0, int(at_global))
        elif at_offset is not None:
            # Place at current end + provided offset (can be negative/positive)
            idx = max(0, self.global_idx + int(at_offset))
        else:
            idx = self.global_idx  # best-effort fallback

        self.events.append((idx, str(code)))

    # ------------------ epoching & ERP updates ------------------

    def _baseline_correct(self, epoch: NDArray[np.float64]) -> NDArray[np.float64]:
        b0, b1 = self.baseline
        if b0 is None:
            b0 = self.tmin
        if b1 is None:
            b1 = 0.0
        ib0 = int(round((b0 - self.tmin) * self.fs))
        ib1 = int(round((b1 - self.tmin) * self.fs))
        if ib1 <= ib0:
            return epoch
        base = float(epoch[ib0:ib1].mean())
        return epoch - base

    def _score_components(
        self, erp: NDArray[np.float64]
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, (w0, w1), pol in self.component_specs:
            m = (self.t >= w0) & (self.t <= w1)
            if not m.any():
                out[name] = {"amp": float("nan"), "lat": float("nan")}
                continue
            seg = erp[m]
            tt = self.t[m]
            if pol == "pos":
                i = int(np.argmax(seg))
                amp = float(seg[i])
                lat = float(tt[i])
            elif pol == "neg":
                i = int(np.argmin(seg))
                amp = float(seg[i])
                lat = float(tt[i])
            else:  # pos_mean == LPP style
                amp = float(seg.mean())
                lat = float((tt[0] + tt[-1]) / 2.0)
            out[name] = {"amp": amp, "lat": lat}
        return out

    def materialize_ready_epochs(
        self,
    ) -> List[Dict[str, Union[str, int, Dict[str, Dict[str, float]]]]]:
        """
        Try to realize any epochs whose full tmin..tmax window is now available.
        For each realized epoch, update the running ERP and return a summary dict.
        """
        updates: List[Dict[str, Union[str, int, Dict[str, Dict[str, float]]]]] = []
        need_pre = self.n_pre
        need_post = self.n_post

        # iterate over a copy; remove as we go
        for ev_idx, code in list(self.events):
            start = ev_idx - need_pre
            end = ev_idx + need_post
            if start < 0:
                continue
            if self.rb.has_range(start, end):
                epoch = self.rb.get_range(start, end)
                epoch = self._baseline_correct(epoch)

                ra = self.running.get(code)
                if ra is None:
                    ra = RunningAverage.init_like(self.epoch_len)
                ra.update(epoch)
                self.running[code] = ra

                comps = self._score_components(ra.avg)
                updates.append({"code": code, "n": ra.n, "components": comps})
                self.events.remove((ev_idx, code))

        return updates


# ----------------------- OSC Server Wrapper -------------------


class OscErpServer:
    """
    Minimal OSC server that accepts:
      /eeg   [samples] | [fs, samples] | [fs, samples, offset_samples]
      /event [code] | [code, at_offset] | [code, "global:", at_global]
    Emits JSON summaries to stdout as soon as running ERPs update.
    """

    def __init__(
        self,
        host: str,
        port: int,
        fs_fallback: float,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        self.logger = logger or logging.getLogger("osc_erp")
        self._fs_hint = float(fs_fallback)
        self.epocher: Optional[StreamEpocher] = None

        self._q = queue.Queue()  # queue of callables to serialize ingestion

        # Build OSC dispatcher
        disp = Dispatcher()
        disp.map("/eeg", self._handle_eeg)
        disp.map("/event", self._handle_event)

        self._server = ThreadingOSCUDPServer((host, port), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

        # Periodic worker thread to process queue & produce updates
        self._worker = threading.Thread(target=self._work_loop, daemon=True)

        self._tmin, self._tmax, self._baseline = tmin, tmax, baseline

    # ------------------ OSC handlers ------------------

    def _handle_eeg(self, addr: str, *args):
        """Accept /eeg messages in flexible forms."""
        try:
            if len(args) == 1:
                samples = np.asarray(args[0], dtype=np.float64)
                fs = self._fs_hint
                offset = None
            elif len(args) == 2:
                # could be (fs, samples) OR (samples, offset)
                if isinstance(args[0], (int, float)):
                    fs = float(args[0])
                    samples = np.asarray(args[1], dtype=np.float64)
                    offset = None
                else:
                    samples = np.asarray(args[0], dtype=np.float64)
                    fs = self._fs_hint
                    offset = int(args[1])
            else:
                fs = float(args[0])
                samples = np.asarray(args[1], dtype=np.float64)
                offset = int(args[2])

            if self.epocher is None:
                self.epocher = StreamEpocher(
                    fs=fs, tmin=self._tmin, tmax=self._tmax, baseline=self._baseline
                )
            self._fs_hint = fs
            self._q.put(lambda: self.epocher.ingest_chunk(samples, offset))
        except Exception as e:
            self.logger.exception("Failed to handle /eeg: %s", e)

    def _handle_event(self, addr: str, *args):
        """Accept /event messages in flexible forms."""
        try:
            code = str(args[0]) if args else "event"
            at_offset: Optional[int] = None
            at_global: Optional[int] = None
            if len(args) >= 2:
                if isinstance(args[1], str) and args[1].lower().startswith("global"):
                    at_global = int(args[2]) if len(args) >= 3 else None
                else:
                    at_offset = int(args[1])

            if self.epocher is None:
                # No samples yet; stage event at current end (0)
                self.epocher = StreamEpocher(
                    fs=self._fs_hint,
                    tmin=self._tmin,
                    tmax=self._tmax,
                    baseline=self._baseline,
                )
            self._q.put(
                lambda: self.epocher.ingest_event(
                    code, at_offset=at_offset, at_global=at_global
                )
            )
        except Exception as e:
            self.logger.exception("Failed to handle /event: %s", e)

    # ------------------ worker & lifecycle ------------------

    def _work_loop(self):
        while True:
            try:
                fn = self._q.get(timeout=0.05)
                fn()
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.exception("Worker error: %s", e)

            # Try to materialize epochs and print updates
            if self.epocher is not None:
                updates = self.epocher.materialize_ready_epochs()
                for u in updates:
                    print(json.dumps({"type": "erp_update", **u}), flush=True)

            time.sleep(0.005)

    def start(self):
        self._thread.start()
        self._worker.start()

    def serve_forever(self):
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self._server.shutdown()
            self._server.server_close()
