# erp_stream.py
# Single-channel OSC → online ERP extraction (P1/N1/N200/P300/LPP)
# Robust to missing fs, chunk overlap/gaps, and unaligned event timing.

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

# ----------------------- Configuration -----------------------

# Default component windows (seconds) and polarity
COMPONENT_SPECS: List[Tuple[str, Tuple[float, float], str]] = [
    ("P1", (0.080, 0.130), "pos"),
    ("N1", (0.120, 0.180), "neg"),
    ("N200", (0.180, 0.300), "neg"),
    ("P300", (0.300, 0.600), "pos"),
    ("LPP", (0.400, 0.800), "pos_mean"),  # mean amplitude over the window
]

# Component scoring filter bands (Hz)
COMPONENT_SCORING_SPECS = {
    "P1": (1.0, 20.0),
    "N1": (1.0, 20.0),
    "N200": (1.0, 12.0),
    "P300": (0.1, 12.0),
    "LPP": (0.1, 8.0),
}

# ----------------------- Utilities ---------------------------


@dataclass
class PendingEvent:
    ev_idx: int
    code: str
    pending_components: List[str]
    base_metrics: Optional[Tuple[float, float]] = None


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
        if end_idx <= start_idx:
            return False
        earliest = self.start_idx
        latest = self.start_idx + min(self.n_written, self.capacity)
        return start_idx >= earliest and end_idx <= latest

    def get_range(self, start_idx: int, end_idx: int) -> NDArray[np.float64]:
        """Materialize [start_idx, end_idx) into a 1-D array."""
        if end_idx <= start_idx:
            raise ValueError("end_idx must be greater than start_idx")
        if not self.has_range(start_idx, end_idx):
            raise ValueError("Requested range not fully available in buffer")
        L = end_idx - start_idx
        out = np.empty(L, dtype=np.float64)
        for i in range(L):
            idx = (start_idx - self.start_idx + i) % self.capacity
            out[i] = self.buf[idx]
        return out


# ----------------------- Stream Epocher ----------------------


class StreamEpocher:
    """
    Online epocher for single-channel EEG.

    - Maintains a ring buffer of recent samples.
    - Accepts chunks of raw samples; applies causal band-pass filtering.
    - Accepts events and publishes ERP components (P1, N1, etc.) as soon as their
      respective time windows are available.
    - Optionally maintains a running average of the full ERP waveform.
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
        components_to_calculate: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        on_publish: Optional[Callable[[dict], None]] = None,
        use_artifact_detection: bool = True,
        blink_threshold_mult: Tuple[float, float] = (10.0, 7.0),
        emg_threshold_mult: Tuple[float, float] = (8.0, 5.0),
    ) -> None:
        assert fs > 0, "Sampling rate fs must be positive"
        self.fs = float(fs)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.baseline = baseline
        self.logger = logger or logging.getLogger(__name__)
        self.on_publish = on_publish
        self.use_artifact_detection = use_artifact_detection
        self.blink_threshold_mult_high, self.blink_threshold_mult_low = (
            blink_threshold_mult
        )
        self.emg_threshold_mult_high, self.emg_threshold_mult_low = emg_threshold_mult

        # Store component specs in a dict for easy lookup
        specs_to_use = component_specs or COMPONENT_SPECS
        if components_to_calculate:
            specs_to_use = [s for s in specs_to_use if s[0] in components_to_calculate]
        self.component_specs = {s[0]: s for s in specs_to_use}

        # Pre-compute scoring filters
        self.scoring_filters = {}
        for comp_name, (low, high) in COMPONENT_SCORING_SPECS.items():
            if comp_name in self.component_specs:
                nyq = max(self.fs / 2.0, 1.0)
                l, h = max(low, 0.01) / nyq, min(high, nyq - 1e-6) / nyq
                if 0 < l < h < 1:
                    self.scoring_filters[comp_name] = butter(
                        2, [l, h], btype="band", output="sos"
                    )

        # Buffer sized for epoch window + a safety margin
        cap = int((tmax - tmin + extra_seconds) * fs)
        cap = max(cap, 1)
        self.rb = RingBuffer.with_capacity(capacity=cap)

        # Global sample index = number of samples ingested so far
        self.global_idx = 0

        # Pending events to be processed
        self.events: List[PendingEvent] = []

        # Pre-compute filters
        nyq = max(fs / 2.0, 1.0)
        low = max(hp, 0.01) / nyq
        high = min(lp, nyq - 1e-6) / nyq
        if not (0 < low < high < 1):
            raise ValueError(f"Invalid band [{hp}, {lp}] for fs={fs}")
        self.sos = butter(4, [low, high], btype="band", output="sos")
        self.filt_zi = np.zeros((self.sos.shape[0], 2))

        if self.use_artifact_detection:
            self.sos_blink = butter(
                2, [max(0.1, 0.5) / nyq, 6.0 / nyq], btype="band", output="sos"
            )
            self.zi_blink = np.zeros((self.sos_blink.shape[0], 2))
            lo_emg = 20.0 / nyq
            hi_emg = min(70.0, nyq - 1e-6) / nyq
            self.sos_emg = butter(2, [lo_emg, hi_emg], btype="band", output="sos")
            self.zi_emg = np.zeros((self.sos_emg.shape[0], 2))

        # Epoch time base (constant length)
        self.n_pre = int(round(-tmin * fs))
        self.n_post = int(round(tmax * fs))
        self.epoch_len = self.n_pre + self.n_post

    # ------------------ new artifact detection helpers ------------------

    def _baseline_indices(self, ev_idx):
        b0 = self.baseline[0] if self.baseline[0] is not None else self.tmin
        b1 = self.baseline[1] if self.baseline[1] is not None else 0.0
        i0 = ev_idx + int(round(b0 * self.fs))
        i1 = ev_idx + int(round(b1 * self.fs))
        return i0, i1

    def _baseline_metrics(self, i0, i1):
        if not self.rb.has_range(i0, i1) or (i1 - i0) < int(0.12 * self.fs):
            return None  # not enough baseline to judge
        base_raw = self.rb.get_range(i0, i1)
        bb = sosfilt(self.sos_blink, base_raw.copy())
        ee = sosfilt(self.sos_emg, base_raw.copy())
        mad_low = 1.4826 * np.median(np.abs(bb - np.median(bb))) + 1e-6
        w = max(1, int(0.05 * self.fs))
        if len(ee) < w:
            return None
        rms = np.sqrt(np.convolve(ee**2, np.ones(w) / w, mode="valid"))
        med_rms = np.median(rms) + 1e-6
        return mad_low, med_rms

    def _blink_exceeds(self, y_blink, thr):
        w = max(1, int(0.03 * self.fs))  # ~30 ms
        if len(y_blink) < w:
            return False
        env = np.convolve(np.abs(y_blink), np.ones(w) / w, mode="same")
        return bool(np.any(env > thr))

    def _emg_exceeds(self, y_emg, base_med_rms, mult):
        w = max(1, int(0.08 * self.fs))  # ~80 ms
        if len(y_emg) < w:
            return False
        rms = np.sqrt(np.convolve(y_emg**2, np.ones(w) / w, mode="same"))
        thr = mult * base_med_rms
        above = rms > thr
        # ≥40 ms continuous above-threshold
        k = max(1, int(0.04 * self.fs))
        if len(above) < k:
            return False
        return bool(
            np.any(np.convolve(above.astype(int), np.ones(k), mode="same") >= k)
        )

    def _artifact_ratios(self, y_blink, y_emg, mad_low, med_rms):
        w_b = max(1, int(0.03 * self.fs))
        env_b = np.convolve(np.abs(y_blink), np.ones(w_b) / w_b, mode="same")
        w_e = max(1, int(0.08 * self.fs))
        rms_e = np.sqrt(np.convolve(y_emg**2, np.ones(w_e) / w_e, mode="same"))
        return float(env_b.max() / (mad_low + 1e-6)), float(
            rms_e.max() / (med_rms + 1e-6)
        )

    # ------------------ ingesting data & events ------------------

    def ingest_chunk(
        self,
        samples: NDArray[np.float64],
    ) -> None:
        """
        Ingest a raw chunk of samples.
        """
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Samples must be a 1-D array")

        # Main ERP-band filtering
        y, self.filt_zi = sosfilt(self.sos, x, zi=self.filt_zi)
        self.rb.append(y)
        self.global_idx += y.size

        # After each chunk, check if any components are ready
        self._check_and_publish_components()

    def ingest_event(self, code: str) -> None:
        """
        Register an event. The event is timestamped at the current end of the stream.
        """
        self.ingest_event_at(self.global_idx, code)

    def ingest_event_at(self, global_sample_idx: int, code: str) -> None:
        """
        Register an event with a precise sample index.
        """
        pending_components = list(self.component_specs.keys())
        event = PendingEvent(
            ev_idx=int(global_sample_idx),
            code=str(code),
            pending_components=pending_components,
        )
        self.events.append(event)

    # ------------------ epoching & ERP updates ------------------

    def _baseline_correct(self, epoch: NDArray[np.float64]) -> NDArray[np.float64]:
        b0, b1 = self.baseline
        if b0 is None:
            b0 = self.tmin
        if b1 is None:
            b1 = 0.0
        ib0 = int(round((b0 - self.tmin) * self.fs))
        ib1 = int(round((b1 - self.tmin) * self.fs))
        if ib1 <= ib0 or ib1 > len(epoch):
            return epoch  # or raise/log
        base = float(epoch[ib0:ib1].mean())
        return epoch - base

    def _score_component(
        self, erp: NDArray[np.float64], name: str, t: NDArray[np.float64]
    ) -> Dict[str, float]:
        """Scores a single component from a (potentially partial) epoch."""
        _, (w0, w1), pol = self.component_specs[name]
        m = (t >= w0) & (t <= w1)
        if not m.any():
            return {"amp": float("nan"), "lat": float("nan")}

        seg = erp[m]
        tt = t[m]
        if pol == "pos":
            i = int(np.argmax(seg))
            amp, lat = float(seg[i]), float(tt[i])
        elif pol == "neg":
            i = int(np.argmin(seg))
            amp, lat = float(seg[i]), float(tt[i])
        else:  # pos_mean
            amp = float(seg.mean())
            lat = float((tt[0] + tt[-1]) / 2.0)
        return {"amp": amp, "lat": lat}

    def _check_and_publish_components(self) -> None:
        """
        Check all pending events and publish any components that have become ready.
        """
        self.events.sort(key=lambda e: e.ev_idx)

        for event in self.events:
            if event.base_metrics is None and self.use_artifact_detection:
                i0, i1 = self._baseline_indices(event.ev_idx)
                event.base_metrics = self._baseline_metrics(i0, i1)

            # --- 1. Event-level artifact processing (if enabled) ---
            event_clean_flag: Union[str, bool] = True
            if self.use_artifact_detection:
                base_metrics = event.base_metrics
                if base_metrics is None:
                    event_clean_flag = "unknown"
                else:
                    mad_low, med_rms = base_metrics
                    i0, i1 = self._baseline_indices(event.ev_idx)

                    # Final baseline check for event-level flag
                    # A) Baseline-only detection, no pad/back-contamination
                    base_raw = self.rb.get_range(i0, i1)
                    bb = sosfilt(self.sos_blink, base_raw.copy())
                    ee = sosfilt(self.sos_emg, base_raw.copy())

                    trim = int(0.05 * self.fs)  # ~50 ms
                    if len(bb) > 2 * trim:
                        bb = bb[trim:-trim]
                    if len(ee) > 2 * trim:
                        ee = ee[trim:-trim]

                    # Blink baseline check with persistence (no pads applied beyond the baseline slice itself)
                    w_b = max(1, int(0.03 * self.fs))
                    env_b = np.convolve(np.abs(bb), np.ones(w_b) / w_b, mode="same")
                    blink_in_baseline = bool(
                        np.any(env_b > self.blink_threshold_mult_high * mad_low)
                    )

                    # EMG baseline check with persistence
                    w_e = max(1, int(0.08 * self.fs))
                    rms_e = np.sqrt(np.convolve(ee**2, np.ones(w_e) / w_e, mode="same"))
                    thr_e = self.emg_threshold_mult_high * med_rms
                    above_e = rms_e > thr_e
                    k = max(1, int(0.04 * self.fs))  # ≥40 ms continuous
                    emg_in_baseline = bool(
                        len(above_e) >= k
                        and np.any(
                            np.convolve(above_e.astype(int), np.ones(k), "same") >= k
                        )
                    )

                    # Decide baseline clean WITHOUT using artifact_events pads
                    event_clean_flag = not (blink_in_baseline or emg_in_baseline)

            # --- 2. Per-component processing ---
            remaining_components = []
            for comp_name in event.pending_components:
                spec = self.component_specs[comp_name]
                _, (w0, w1), _ = spec
                comp_end_idx = event.ev_idx + int(round(w1 * self.fs))

                if not self.rb.has_range(event.ev_idx - self.n_pre, comp_end_idx):
                    remaining_components.append(comp_name)
                    continue

                # Determine cleanliness for this component
                is_clean = True
                reason = None
                final_clean_flag = event_clean_flag

                if self.use_artifact_detection:
                    if event_clean_flag == "unknown":
                        is_clean = False
                        reason = "baseline_unavailable"
                        final_clean_flag = False
                    elif event_clean_flag is False:
                        is_clean = False
                        reason = "baseline_contaminated"
                    elif event_clean_flag is True:
                        # Direct component window artifact detection
                        comp_start_idx = event.ev_idx + int(round(w0 * self.fs))
                        comp_end_idx = event.ev_idx + int(round(w1 * self.fs))

                        min_len = max(int(0.12 * self.fs), 1)  # ~120 ms
                        if (comp_end_idx - comp_start_idx) < min_len:
                            is_clean = False
                            reason = "component_window_too_short"
                            final_clean_flag = "unknown"

                        elif self.rb.has_range(comp_start_idx, comp_end_idx):
                            comp_raw = self.rb.get_range(comp_start_idx, comp_end_idx)
                            base_metrics = event.base_metrics
                            if base_metrics is not None:
                                mad_low, med_rms = base_metrics
                                y_blink = sosfilt(self.sos_blink, comp_raw)
                                y_emg = sosfilt(self.sos_emg, comp_raw)
                                if self._blink_exceeds(
                                    y_blink, self.blink_threshold_mult_low * mad_low
                                ) or self._emg_exceeds(
                                    y_emg, med_rms, self.emg_threshold_mult_low
                                ):
                                    is_clean = False
                                    reason = "component_window_contaminated"
                                    final_clean_flag = False

                # Always score the component
                epoch_all = self.rb.get_range(event.ev_idx - self.n_pre, comp_end_idx)
                t_epoch = np.arange(len(epoch_all)) / self.fs + self.tmin
                epoch = self._baseline_correct(epoch_all)

                # Apply component-specific scoring filter if available
                if comp_name in self.scoring_filters:
                    scoring_epoch = sosfilt(self.scoring_filters[comp_name], epoch)
                else:
                    scoring_epoch = epoch

                comp_data = self._score_component(scoring_epoch, comp_name, t_epoch)

                if not is_clean:
                    comp_data["error"] = {
                        "status": "artifact_detected",
                        "reason": reason,
                    }
                    if reason == "baseline_contaminated" and event.base_metrics:
                        mad_low, med_rms = event.base_metrics
                        base_raw = self.rb.get_range(
                            *self._baseline_indices(event.ev_idx)
                        )
                        bb = sosfilt(self.sos_blink, base_raw.copy())
                        ee = sosfilt(self.sos_emg, base_raw.copy())
                        b_ratio, e_ratio = self._artifact_ratios(
                            bb, ee, mad_low, med_rms
                        )
                        comp_data["error"]["blink_ratio"] = b_ratio
                        comp_data["error"]["emg_ratio"] = e_ratio

                    if reason == "component_window_contaminated" and event.base_metrics:
                        mad_low, med_rms = event.base_metrics
                        b_ratio, e_ratio = self._artifact_ratios(
                            y_blink, y_emg, mad_low, med_rms
                        )
                        comp_data["error"]["blink_ratio"] = b_ratio
                        comp_data["error"]["emg_ratio"] = e_ratio

                update = {
                    "code": event.code,
                    "event_idx": event.ev_idx,
                    "clean": final_clean_flag,
                    "component": {comp_name: comp_data},
                }
                if self.on_publish:
                    self.on_publish(update)

            event.pending_components = remaining_components

        self.events = [e for e in self.events if e.pending_components]


# ----------------------- OSC Server Wrapper -------------------


class OscErpServer:
    """
    Minimal OSC server that accepts:
      /eeg   [samples]
    Events are ingested via the `ingest_event` method.
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
        components_to_calculate: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        on_update: Optional[Callable[[dict], None]] = None,
        eeg_started: Optional[threading.Event] = None,
    ) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        self.logger = logger or logging.getLogger("osc_erp")
        self._fs_hint = float(fs_fallback)
        self.epocher: Optional[StreamEpocher] = None
        self.on_update = on_update
        self.eeg_started = eeg_started

        self._q = queue.Queue()  # queue of callables to serialize ingestion

        # Build OSC dispatcher
        disp = Dispatcher()
        disp.map("/eeg", self._handle_eeg)

        self._server = ThreadingOSCUDPServer((host, port), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

        # Periodic worker thread to process queue & produce updates
        self._worker = threading.Thread(target=self._work_loop, daemon=True)

        self._tmin, self._tmax, self._baseline = tmin, tmax, baseline
        self._components_to_calculate = components_to_calculate

    def _publish_update(self, update: dict):
        print(json.dumps({"type": "erp_update", **update}), flush=True)
        if self.on_update:
            self.on_update(update)

    def ingest_event(self, code: str):
        """Ingest an event from within the same process."""
        if self.epocher is None:
            self.epocher = StreamEpocher(
                fs=self._fs_hint,
                tmin=self._tmin,
                tmax=self._tmax,
                baseline=self._baseline,
                on_publish=self._publish_update,
                components_to_calculate=self._components_to_calculate,
            )
        self._q.put(lambda: self.epocher.ingest_event(code))

    # ------------------ OSC handlers ------------------

    def _handle_eeg(self, addr: str, *args):
        """Accepts /eeg [samples] messages."""
        if self.eeg_started and not self.eeg_started.is_set():
            self.eeg_started.set()
        try:
            samples = np.asarray(args, dtype=np.float64)

            if self.epocher is None:
                self.epocher = StreamEpocher(
                    fs=self._fs_hint,
                    tmin=self._tmin,
                    tmax=self._tmax,
                    baseline=self._baseline,
                    on_publish=self._publish_update,
                    components_to_calculate=self._components_to_calculate,
                )
            self._q.put(lambda: self.epocher.ingest_chunk(samples))
        except Exception as e:
            self.logger.exception("Failed to handle /eeg: %s", e)

    # ------------------ worker & lifecycle ------------------

    def _work_loop(self):
        """
        Data-driven worker loop.
        - Blocks waiting for data/events from the queue.
        - Processes all available items.
        - Component publication is triggered by the StreamEpocher itself.
        """
        while True:
            try:
                # Block until the first item is available
                fn = self._q.get()
                fn()

                # Process all other currently available items
                while not self._q.empty():
                    try:
                        fn = self._q.get_nowait()
                        fn()
                    except queue.Empty:
                        break  # Should not happen with this logic, but for safety

            except Exception as e:
                self.logger.exception("Worker error processing queue: %s", e)

    def start(self):
        self._thread.start()
        self._worker.start()

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()

    def serve_forever(self):
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
