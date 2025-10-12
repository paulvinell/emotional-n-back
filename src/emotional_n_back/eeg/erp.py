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
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

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

# ----------------------- Utilities ---------------------------


@dataclass
class PendingEvent:
    ev_idx: int
    code: str
    pending_components: List[str]


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
    ) -> None:
        assert fs > 0, "Sampling rate fs must be positive"
        self.fs = float(fs)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.baseline = baseline
        self.logger = logger or logging.getLogger(__name__)
        self.on_publish = on_publish

        # Store component specs in a dict for easy lookup
        specs_to_use = component_specs or COMPONENT_SPECS
        if components_to_calculate:
            specs_to_use = [s for s in specs_to_use if s[0] in components_to_calculate]
        self.component_specs = {s[0]: s for s in specs_to_use}

        # Buffer sized for epoch window + a safety margin
        cap = int((tmax - tmin + extra_seconds) * fs)
        cap = max(cap, 1)
        self.rb = RingBuffer.with_capacity(capacity=cap)

        # Global sample index = number of samples ingested so far
        self.global_idx = 0

        # Pending events to be processed. These are typically markers for stimuli
        # presented in the experiment (e.g., image onset, sound onset). Each event
        # is tracked with its sample index, a string code, and the status of its
        # ERP components.
        self.events: List[PendingEvent] = []

        # Pre-compute filters
        nyq = max(fs / 2.0, 1.0)
        low = max(hp, 0.01) / nyq
        high = min(lp, nyq - 1e-6) / nyq
        if not (0 < low < high < 1):
            raise ValueError(f"Invalid band [{hp}, {lp}] for fs={fs}")
        self.sos = butter(4, [low, high], btype="band", output="sos")
        self.filt_zi = np.zeros((self.sos.shape[0], 2))

        # Epoch time base (constant length)
        self.n_pre = int(round(-tmin * fs))
        self.n_post = int(round(tmax * fs))
        self.epoch_len = self.n_pre + self.n_post

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

    def _is_clean(self, epoch, p2p_thresh=150.0, slope_thresh=75.0):
        if np.ptp(epoch) > p2p_thresh:  # large blink/motion
            return False
        if np.max(np.abs(np.diff(epoch))) > slope_thresh:  # EMG burst
            return False
        return True

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
        # Sort events by timestamp to ensure we process the earliest one first
        self.events.sort(key=lambda e: e.ev_idx)

        for event in self.events:
            # Baseline window must be available
            b0, b1 = self.baseline
            baseline_start_t = b0 if b0 is not None else self.tmin
            baseline_end_t = b1 if b1 is not None else 0.0
            baseline_start_idx = event.ev_idx + int(round(baseline_start_t * self.fs))
            baseline_end_idx = event.ev_idx + int(round(baseline_end_t * self.fs))

            if baseline_end_idx <= baseline_start_idx or not self.rb.has_range(
                baseline_start_idx, baseline_end_idx
            ):
                continue  # Wait for more data for baseline

            # --- Per-component processing ---
            remaining_components = []
            for comp_name in event.pending_components:
                spec = self.component_specs[comp_name]
                _, (w0, w1), _ = spec

                # Determine required window for this component
                epoch_start = event.ev_idx - self.n_pre
                comp_end_idx = event.ev_idx + int(round(w1 * self.fs))

                if not self.rb.has_range(epoch_start, comp_end_idx):
                    remaining_components.append(comp_name)
                    continue  # Not ready yet

                # Extract partial epoch
                partial_epoch = self.rb.get_range(epoch_start, comp_end_idx)
                t_partial = np.arange(len(partial_epoch)) / self.fs + self.tmin

                if not self._is_clean(partial_epoch):
                    # Publish artifact skip message
                    update = {
                        "code": event.code,
                        "event_idx": event.ev_idx,
                        "component": {comp_name: {"status": "skipped_artifact"}},
                    }
                    if self.on_publish:
                        self.on_publish(update)
                    continue  # Skip this component for this event

                # Baseline correct and score
                partial_epoch = self._baseline_correct(partial_epoch)
                comp_data = self._score_component(partial_epoch, comp_name, t_partial)

                # Publish
                update = {
                    "code": event.code,
                    "event_idx": event.ev_idx,
                    "component": {comp_name: comp_data},
                }
                if self.on_publish:
                    self.on_publish(update)

            event.pending_components = remaining_components

        # Clean up events with no pending components
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
