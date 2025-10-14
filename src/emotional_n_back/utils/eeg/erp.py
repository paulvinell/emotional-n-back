# erp_stream.py
# Single-channel OSC → online ERP extraction (P1/N1/N200/P300/LPP)
# Robust to missing fs, chunk overlap/gaps, and unaligned event timing.

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

from emotional_n_back.utils.streaming.buffer import RingBuffer
from emotional_n_back.utils.streaming.resampler import Resampler

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

# Numerical stability constant
EPS = 1e-6

# ----------------------- Utilities ---------------------------


@dataclass
class PendingEvent:
    ev_idx: int
    code: str
    pending_components: List[str]
    base_metrics: Optional[Tuple[float, float]] = None
    base_artifacts: Optional[Tuple[NDArray, NDArray]] = None


# ----------------------- Stream Epocher ----------------------


class StreamEpocher:
    """
    Online epocher for single-channel EEG, operating on a uniform timeline.

    - Maintains ring buffers of uniformly sampled data.
    - Accepts chunks of uniformly sampled data; applies causal filtering.
    - Accepts events and publishes ERP components (P1, N1, etc.) as soon as their
      respective time windows are available.
    - Rejects epochs that contain gaps (NaNs) or are too close to them.
    """

    def __init__(
        self,
        fs_target: float,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
        component_specs: Optional[List[Tuple[str, Tuple[float, float], str]]] = None,
        components_to_calculate: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        on_publish: Optional[Callable[[dict], None]] = None,
        use_artifact_detection: bool = True,
        blink_threshold_mult: Tuple[float, float] = (10.0, 7.0),
        emg_threshold_mult: Tuple[float, float] = (8.0, 5.0),
        notch_hz: float = 50.0,
        notch_q: float = 30.0,
        trim_s: float = 0.05,
        blink_env_s: float = 0.03,
        emg_rms_s: float = 0.08,
        warmup_s: float = 0.1,
    ) -> None:
        assert fs_target > 0, "Target sampling rate must be positive"
        if notch_hz not in (50.0, 60.0):
            raise ValueError("notch_hz must be 50.0 or 60.0")
        if notch_hz >= fs_target * 0.45:
            raise ValueError(
                f"notch_hz ({notch_hz}) must be less than 0.45 * fs_target ({0.45 * fs_target})"
            )

        self.fs = float(fs_target)  # Use fs_target internally
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
        self.trim_s = trim_s
        self.blink_env_s = blink_env_s
        self.emg_rms_s = emg_rms_s
        self.warmup_s = warmup_s

        # Store component specs in a dict for easy lookup
        specs_to_use = component_specs or COMPONENT_SPECS
        if components_to_calculate:
            specs_to_use = [s for s in specs_to_use if s[0] in components_to_calculate]
        self.component_specs = {s[0]: s for s in specs_to_use}

        # Pre-compute scoring filters at the target sampling rate
        self.scoring_filters = {}
        for comp_name, (low, high) in COMPONENT_SCORING_SPECS.items():
            if comp_name in self.component_specs:
                nyq = max(self.fs / 2.0, 1.0)
                l, h = max(low, 0.01) / nyq, min(high, nyq - 1e-6) / nyq
                if 0 < l < h < 1:
                    self.scoring_filters[comp_name] = butter(
                        2, [l, h], btype="band", output="sos"
                    )
                else:
                    self.logger.warning(
                        "Cannot build scoring filter for %s with band [%.2f, %.2f] Hz at fs=%.2f",
                        comp_name,
                        low,
                        high,
                        self.fs,
                    )

        # Buffer sized for epoch window + a safety margin
        buffer_duration_s = tmax - tmin + 2.0  # 2s safety margin
        cap = int(buffer_duration_s * self.fs)
        self.rb_raw = RingBuffer.with_capacity(capacity=cap)
        self.rb_erp = RingBuffer.with_capacity(capacity=cap)

        # Per-component forward-filtered buffers and filter states
        self.rb_comp = {}
        self.zi_comp = {}
        for comp_name, sos in self.scoring_filters.items():
            self.rb_comp[comp_name] = RingBuffer.with_capacity(capacity=cap)
            self.zi_comp[comp_name] = np.zeros((sos.shape[0], 2))

        # Global sample index on the uniform timeline
        self.global_idx = 0

        # Pending events to be processed
        self.events: List[PendingEvent] = []

        # Deque to store recent gap end times
        self.recent_gaps: deque = deque(maxlen=100)

        # --- Filters for the uniform timeline ---
        hp, lp = 0.1, 30.0  # Fixed ERP band for now
        nyq = max(self.fs / 2.0, 1.0)

        # Notch filter
        b_notch, a_notch = iirnotch(notch_hz, notch_q, fs=self.fs)
        self.sos_notch = tf2sos(b_notch, a_notch)
        self.filt_zi_notch = np.zeros((self.sos_notch.shape[0], 2))

        # ERP band-pass filter
        low = max(hp, 0.01) / nyq
        high = min(lp, nyq - 1e-6) / nyq
        self.sos_erp = butter(4, [low, high], btype="band", output="sos")
        self.filt_zi_erp = np.zeros((self.sos_erp.shape[0], 2))

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
        self.n_pre = int(round(-tmin * self.fs))
        self.n_post = int(round(tmax * self.fs))
        self.epoch_len = self.n_pre + self.n_post

        # Logging throttle
        self._last_log_time = 0
        self._log_interval_s = 5.0  # Log max once every 5s

    def has_nan_or_gap(self, start_idx: int, end_idx: int) -> bool:
        """Check if a window in the raw uniform buffer contains NaNs."""
        if not self.rb_raw.has_range(start_idx, end_idx):
            return True  # Data not even available
        epoch_raw = self.rb_raw.get_range(start_idx, end_idx)
        return bool(np.isnan(epoch_raw).any())

    def reset_filters(self):
        """Reset all causal filter states after a gap."""
        self.filt_zi_notch.fill(0)
        self.filt_zi_erp.fill(0)
        if self.use_artifact_detection:
            self.zi_blink.fill(0)
            self.zi_emg.fill(0)
        for zi in self.zi_comp.values():
            zi.fill(0)

    def set_recent_gaps(self, recent_gaps: deque):
        """Set the deque of recent gap end times."""
        self.recent_gaps = recent_gaps

    def is_in_warmup(self, start_idx: int, end_idx: int) -> bool:
        """Check if a window overlaps with a post-gap warmup period."""
        start_time = start_idx / self.fs
        end_time = end_idx / self.fs
        for gap_end_t in self.recent_gaps:
            warmup_end_t = gap_end_t + self.warmup_s
            if start_time < warmup_end_t and end_time > gap_end_t:
                return True
        return False

    # ------------------ new artifact detection helpers ------------------

    def _baseline_indices(self, ev_idx):
        b0 = self.baseline[0] if self.baseline[0] is not None else self.tmin
        b1 = self.baseline[1] if self.baseline[1] is not None else 0.0
        i0 = ev_idx + int(round(b0 * self.fs))
        i1 = ev_idx + int(round(b1 * self.fs))
        return i0, i1

    def _baseline_metrics(self, i0, i1):
        if not self.rb_raw.has_range(i0, i1) or (i1 - i0) < int(0.12 * self.fs):
            return None  # not enough baseline to judge
        base_raw = self.rb_raw.get_range(i0, i1)
        # **Mandatory**: Notch filter before artifact band extraction
        base_notched = sosfilt(self.sos_notch, base_raw.copy())

        bb = sosfilt(self.sos_blink, base_notched.copy())
        ee = sosfilt(self.sos_emg, base_notched.copy())
        mad_low = 1.4826 * np.median(np.abs(bb - np.median(bb))) + EPS
        w = max(1, int(self.emg_rms_s * self.fs))
        if len(ee) < w:
            return None
        rms = np.sqrt(np.convolve(ee**2, np.ones(w) / w, mode="valid"))
        med_rms = np.median(rms) + EPS
        return mad_low, med_rms

    def _blink_exceeds(self, y_blink, thr):
        w = max(1, int(self.blink_env_s * self.fs))
        if len(y_blink) < w:
            return False
        env = np.convolve(np.abs(y_blink), np.ones(w) / w, mode="same")
        return bool(np.any(env > thr))

    def _emg_exceeds(self, y_emg, base_med_rms, mult):
        w = max(1, int(self.emg_rms_s * self.fs))
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
        w_b = max(1, int(self.blink_env_s * self.fs))
        env_b = np.convolve(np.abs(y_blink), np.ones(w_b) / w_b, mode="same")
        w_e = max(1, int(self.emg_rms_s * self.fs))
        rms_e = np.sqrt(np.convolve(y_emg**2, np.ones(w_e) / w_e, mode="same"))
        return float(env_b.max() / (mad_low + EPS)), float(
            rms_e.max() / (med_rms + EPS)
        )

    def ingest_chunk(self, samples: NDArray[np.float64]) -> None:
        """Ingest a chunk of uniformly sampled data."""
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("Samples must be a 1-D array")

        # Store raw uniform data
        self.rb_raw.append(x)

        # Filter and store ERP-band data
        y, self.filt_zi_notch = sosfilt(self.sos_notch, x, zi=self.filt_zi_notch)
        y, self.filt_zi_erp = sosfilt(self.sos_erp, y, zi=self.filt_zi_erp)
        self.rb_erp.append(y)

        # Streamed forward-pass for each component's scoring filter
        for comp_name, sos in self.scoring_filters.items():
            yc, self.zi_comp[comp_name] = sosfilt(sos, y, zi=self.zi_comp[comp_name])
            self.rb_comp[comp_name].append(yc)

        self.global_idx += x.size
        self._check_and_publish_components()

    def ingest_event_at(self, ev_idx: int, code: str) -> None:
        """Register an event with a precise uniform sample index."""
        event = PendingEvent(
            ev_idx=int(ev_idx),
            code=str(code),
            pending_components=list(self.component_specs.keys()),
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
        # Drop events whose latest needed sample is already out of buffer
        earliest = self.rb_erp.start_idx
        self.events = [e for e in self.events if (e.ev_idx + self.n_post) > earliest]

        # Optional: hard cap to prevent runaway growth
        MAX_EVENTS = 2048
        if len(self.events) > MAX_EVENTS:
            overflow = len(self.events) - MAX_EVENTS
            self.logger.warning(
                "Pruning %d excess events (cap=%d)", overflow, MAX_EVENTS
            )
            self.events = self.events[-MAX_EVENTS:]

        self.events.sort(key=lambda e: e.ev_idx)

        for event in self.events:
            if event.base_metrics is None and self.use_artifact_detection:
                i0, i1 = self._baseline_indices(event.ev_idx)
                event.base_metrics = self._baseline_metrics(i0, i1)

            # --- 1. Event-level artifact processing (if enabled) ---
            event_clean_flag: Union[str, bool] = True
            if self.use_artifact_detection:
                i0, i1 = self._baseline_indices(event.ev_idx)
                if self.has_nan_or_gap(i0, i1):
                    event_clean_flag = False
                    reason = "gap_in_baseline"
                elif self.is_in_warmup(i0, i1):
                    event_clean_flag = False
                    reason = "post_gap_warmup"
                else:
                    event.base_metrics = self._baseline_metrics(i0, i1)
                    base_metrics = event.base_metrics
                    if base_metrics is None:
                        event_clean_flag = "unknown"
                    else:
                        mad_low, med_rms = base_metrics

                        # Final baseline check for event-level flag
                        # A) Baseline-only detection, no pad/back-contamination
                        base_raw = self.rb_raw.get_range(i0, i1)
                        base_notched = sosfilt(self.sos_notch, base_raw.copy())
                        bb = sosfilt(self.sos_blink, base_notched.copy())
                        ee = sosfilt(self.sos_emg, base_notched.copy())
                        event.base_artifacts = (bb, ee)  # Cache for ratio calculation

                        trim = int(self.trim_s * self.fs)
                        if len(bb) > 2 * trim:
                            bb = bb[trim:-trim]
                        if len(ee) > 2 * trim:
                            ee = ee[trim:-trim]

                        # Blink baseline check with persistence (no pads applied beyond the baseline slice itself)
                        w_b = max(1, int(self.blink_env_s * self.fs))
                        env_b = np.convolve(np.abs(bb), np.ones(w_b) / w_b, mode="same")
                        blink_in_baseline = bool(
                            np.any(env_b > self.blink_threshold_mult_high * mad_low)
                        )

                        # EMG baseline check with persistence
                        w_e = max(1, int(self.emg_rms_s * self.fs))
                        rms_e = np.sqrt(
                            np.convolve(ee**2, np.ones(w_e) / w_e, mode="same")
                        )
                        thr_e = self.emg_threshold_mult_high * med_rms
                        above_e = rms_e > thr_e
                        k = max(1, int(0.04 * self.fs))  # ≥40 ms continuous
                        emg_in_baseline = bool(
                            len(above_e) >= k
                            and np.any(
                                np.convolve(above_e.astype(int), np.ones(k), "same")
                                >= k
                            )
                        )

                        # Decide baseline clean WITHOUT using artifact_events pads
                        event_clean_flag = not (blink_in_baseline or emg_in_baseline)

            # --- 2. Per-component processing ---
            remaining_components = []
            for comp_name in event.pending_components:
                spec = self.component_specs[comp_name]
                _, (w0, w1), _ = spec
                comp_start_idx = event.ev_idx + int(round(w0 * self.fs))
                comp_end_idx = event.ev_idx + int(round(w1 * self.fs))

                if not self.rb_erp.has_range(event.ev_idx - self.n_pre, comp_end_idx):
                    remaining_components.append(comp_name)
                    continue

                # Determine cleanliness for this component
                is_clean = True
                reason = None
                final_clean_flag = event_clean_flag

                # Check for gaps first, as they are a type of artifact
                if self.has_nan_or_gap(comp_start_idx, comp_end_idx):
                    self.logger.warning(
                        "Processing component %s for event %s with gap in window.",
                        comp_name,
                        event.code,
                    )
                    is_clean = False
                    reason = "gap_in_component"
                    final_clean_flag = False

                if self.use_artifact_detection and is_clean:  # Only check others if no gap
                    if event_clean_flag == "unknown":
                        now = time.time()
                        if now - self._last_log_time > self._log_interval_s:
                            self.logger.warning(
                                "Event %s at %d: baseline metrics unavailable, marking as unclean.",
                                event.code,
                                event.ev_idx,
                            )
                            self._last_log_time = now
                        is_clean = False
                        reason = "baseline_unavailable"
                        final_clean_flag = "unknown"
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

                        elif self.rb_raw.has_range(comp_start_idx, comp_end_idx):
                            comp_raw = self.rb_raw.get_range(
                                comp_start_idx, comp_end_idx
                            )
                            base_metrics = event.base_metrics
                            if base_metrics is not None:
                                mad_low, med_rms = base_metrics
                                # Notch first, then check for artifacts
                                comp_notched = sosfilt(self.sos_notch, comp_raw)
                                y_blink = sosfilt(self.sos_blink, comp_notched)
                                y_emg = sosfilt(self.sos_emg, comp_notched)

                                # Trim edges to avoid filter ringing artifacts
                                trim = int(self.trim_s * self.fs)
                                if len(y_blink) > 2 * trim:
                                    y_blink = y_blink[trim:-trim]
                                if len(y_emg) > 2 * trim:
                                    y_emg = y_emg[trim:-trim]

                                if self._blink_exceeds(
                                    y_blink, self.blink_threshold_mult_low * mad_low
                                ) or self._emg_exceeds(
                                    y_emg, med_rms, self.emg_threshold_mult_low
                                ):
                                    is_clean = False
                                    reason = "component_window_contaminated"
                                    final_clean_flag = False

                # Always score the component from the pre-filtered ERP buffer
                epoch_all = self.rb_erp.get_range(
                    event.ev_idx - self.n_pre, comp_end_idx
                )
                t_epoch = np.arange(len(epoch_all)) / self.fs + self.tmin

                # Get forward-filtered data and apply a reverse pass to approximate zero-phase
                if comp_name in self.rb_comp:
                    fwd_slice = self.rb_comp[comp_name].get_range(
                        event.ev_idx - self.n_pre, comp_end_idx
                    )

                    # --- Baseline correct the component-filtered slice ---
                    b0, b1 = self.baseline
                    if b0 is None:
                        b0 = self.tmin
                    if b1 is None:
                        b1 = 0.0
                    ib0 = int(round((b0 - self.tmin) * self.fs))
                    ib1 = int(round((b1 - self.tmin) * self.fs))
                    if ib1 > ib0 and ib1 <= len(fwd_slice):
                        baseline_val = float(fwd_slice[ib0:ib1].mean())
                        fwd_slice -= baseline_val

                    if len(fwd_slice) >= 3:
                        sos = self.scoring_filters[comp_name]
                        zi_reverse = np.zeros((sos.shape[0], 2))
                        rev = fwd_slice[::-1].copy()
                        rev_filt, _ = sosfilt(sos, rev, zi=zi_reverse)
                        scoring_epoch = rev_filt[::-1]
                    else:
                        scoring_epoch = fwd_slice  # Too short, use forward-pass only
                else:
                    scoring_epoch = self._baseline_correct(epoch_all)  # Fallback

                comp_data = self._score_component(scoring_epoch, comp_name, t_epoch)

                if not is_clean:
                    comp_data["error"] = {
                        "status": "artifact_detected",
                        "reason": reason,
                    }
                    if (
                        reason == "baseline_contaminated"
                        and event.base_metrics
                        and event.base_artifacts
                    ):
                        mad_low, med_rms = event.base_metrics
                        bb, ee = event.base_artifacts
                        b_ratio, e_ratio = self._artifact_ratios(
                            bb, ee, mad_low, med_rms
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
    Minimal OSC server that accepts /eeg messages, resamples the data to a
    uniform timeline, and forwards it to the StreamEpocher.
    """

    def __init__(
        self,
        host: str,
        port: int,
        fs_target: float = 256.0,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
        components_to_calculate: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        on_update: Optional[Callable[[dict], None]] = None,
        eeg_started: Optional[threading.Event] = None,
        use_artifact_detection: bool = True,
        notch_hz: float = 50.0,
        notch_q: float = 30.0,
        gap_threshold_s: float = 0.02,
        warmup_s: float = 0.1,
    ) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        self.logger = logger or logging.getLogger("osc_erp")
        self.epocher: Optional[StreamEpocher] = None
        self.on_update = on_update
        self.eeg_started = eeg_started

        self._q = queue.Queue()  # For serializing calls to the epocher

        # Instantiate the resampler
        self.resampler = Resampler(
            fs_target=fs_target,
            gap_threshold_s=gap_threshold_s,
            warmup_s=warmup_s,
            on_resampled_chunk=self._handle_resampled_chunk,
            on_gap=lambda: self._q.put(self.epocher.reset_filters),
        )

        # Instantiate the epocher
        self.epocher = StreamEpocher(
            fs_target=fs_target,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            on_publish=self._publish_update,
            components_to_calculate=components_to_calculate,
            use_artifact_detection=use_artifact_detection,
            notch_hz=notch_hz,
            notch_q=notch_q,
            warmup_s=warmup_s,
        )

        # Build OSC dispatcher
        disp = Dispatcher()
        disp.map("/eeg", self._handle_eeg)

        self._server = ThreadingOSCUDPServer((host, port), disp)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._worker = threading.Thread(target=self._work_loop, daemon=True)

    def _handle_resampled_chunk(self, chunk: NDArray[np.float64]):
        """Callback for when the resampler has a new chunk of uniform data."""
        if self.epocher:
            self._q.put(lambda: self.epocher.ingest_chunk(chunk))

    def _publish_update(self, update: dict):
        """Add stream health info and publish the update."""
        update["stream_health"] = {
            "fs_target": self.resampler.fs_target,
            "fs_obs": self.resampler.fs_obs,
            "drift_ratio": self.resampler.drift_ratio,
            # Add gap info here later
        }
        print(json.dumps({"type": "erp_update", **update}), flush=True)
        if self.on_update:
            self.on_update(update)

    def ingest_event(self, code: str):
        t_event = time.time()
        idx = self.resampler.transport_to_uniform_idx(t_event)

        if idx is None:
            # Resampler not anchored yet; retry once it produces output
            self._q.put(lambda: self._ingest_event_when_ready(t_event, code))
        else:
            self._q.put(lambda: self.epocher.ingest_event_at(idx, code))

    def _ingest_event_when_ready(self, t_event: float, code: str):
        idx = self.resampler.transport_to_uniform_idx(t_event)
        if idx is None:
            # still not ready; requeue
            self._q.put(lambda: self._ingest_event_when_ready(t_event, code))
            return
        self.epocher.ingest_event_at(idx, code)

    def _handle_eeg(self, addr: str, *args):
        """Accepts /eeg [samples] messages."""
        if self.eeg_started and not self.eeg_started.is_set():
            self.eeg_started.set()
        try:
            samples = np.asarray(args, dtype=np.float64)
            t0 = time.time()
            dt = 1.0 / self.resampler.fs_target
            t_chunk = t0 + np.arange(len(samples), dtype=np.float64) * dt

            self.resampler.ingest_chunk(t_chunk, samples)
            if self.epocher:
                self.epocher.set_recent_gaps(self.resampler.recent_gaps)

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
