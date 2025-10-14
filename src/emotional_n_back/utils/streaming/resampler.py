from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from emotional_n_back.utils.streaming.buffer import RingBuffer


@dataclass
class Resampler:
    fs_target: float = 256.0
    gap_threshold_s: float = 0.02
    warmup_s: float = 0.1
    ema_alpha: float = 0.01

    # internal state (all times are RELATIVE to t0_transport once set)
    t0_transport: Optional[float] = None
    t0_uniform: Optional[float] = None  # keep for event mapping
    t_prev: Optional[float] = None  # relative time of last raw sample
    x_prev: Optional[float] = None
    t_uniform_next: Optional[float] = None  # next uniform time (relative)
    dt_hat: Optional[float] = None
    fs_obs: float = 0.0
    drift_ratio: float = 1.0
    recent_gaps: deque = field(default_factory=lambda: deque(maxlen=100))

    rb_uniform_raw: RingBuffer = field(init=False)

    on_resampled_chunk: Optional[Callable[[NDArray[np.float64]], None]] = None
    on_gap: Optional[Callable[[], None]] = None

    def __post_init__(self):
        self.rb_uniform_raw = RingBuffer.with_capacity(int(self.fs_target * 10))

    # ---- helpers ----

    def _emit_uniform_segment(
        self, t0: float, x0: float, t1: float, x1: float
    ) -> NDArray[np.float64]:
        """Vectorized emitter for (t0,x0)->(t1,x1) on the RELATIVE axis."""
        if self.t_uniform_next is None:
            return np.empty(0, dtype=np.float64)
        dt = t1 - t0
        if dt <= 0:
            return np.empty(0, dtype=np.float64)

        # how many uniform points fit in (t0, t1] from the current cursor
        n = int(np.floor((t1 - self.t_uniform_next) * self.fs_target + 1e-12))
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        # uniform times (strictly increasing)
        t_u = (
            self.t_uniform_next
            + (np.arange(n, dtype=np.float64) + 1.0) / self.fs_target
        )
        alpha = (t_u - t0) / dt
        x_u = x0 + alpha * (x1 - x0)

        # advance the cursor
        self.t_uniform_next += n / self.fs_target
        return x_u

    def _emit_batched(self, arr: NDArray[np.float64], max_emit: int = 4096):
        """Batch output to avoid blocking."""
        i = 0
        while i < len(arr):
            j = min(i + max_emit, len(arr))
            sub = arr[i:j]
            self.rb_uniform_raw.append(sub)
            if self.on_resampled_chunk:
                self.on_resampled_chunk(sub)
            i = j

    # ---- main API ----

    def ingest_chunk(self, t_chunk: NDArray[np.float64], x_chunk: NDArray[np.float64]):
        """Ingest raw samples with ABSOLUTE transport timestamps; internally convert to RELATIVE."""
        if len(t_chunk) == 0:
            return

        # initialize bases on first call
        if self.t0_transport is None:
            self.t0_transport = float(t_chunk[0])
            self.t0_uniform = 0.0
            self.t_uniform_next = 0.0
            # seed t_prev/x_prev with the first sample and start from the second
            self.t_prev = 0.0
            self.x_prev = float(x_chunk[0])
            t_chunk = t_chunk[1:]
            x_chunk = x_chunk[1:]
            if len(t_chunk) == 0:
                return

        # work in RELATIVE seconds
        t_rel = t_chunk.astype(np.float64) - self.t0_transport

        out = []
        for t, x in zip(t_rel, x_chunk):
            # drop non-monotonic timestamps
            if self.t_prev is not None and t <= self.t_prev:
                continue

            dt = t - (self.t_prev if self.t_prev is not None else t)
            if self.t_prev is None:
                # shouldn't happen after init, but guard anyway
                self.t_prev = t
                self.x_prev = float(x)
                continue

            if dt > self.gap_threshold_s:
                # emit capped NaNs for gap
                num_nans = min(
                    int(np.floor(dt * self.fs_target)), int(2 * self.fs_target)
                )
                if num_nans > 0:
                    out.append(np.full(num_nans, np.nan, dtype=np.float64))
                    if self.t_uniform_next is not None:
                        self.t_uniform_next += num_nans / self.fs_target
                # record gap end (relative time in seconds)
                self.recent_gaps.append(self.t_prev + dt)
                if self.on_gap:
                    self.on_gap()
            else:
                # update drift stats (read-only; do not change fs_target)
                self.dt_hat = (
                    dt
                    if self.dt_hat is None
                    else (self.ema_alpha * dt + (1 - self.ema_alpha) * self.dt_hat)
                )
                if self.dt_hat and self.dt_hat > 0:
                    self.fs_obs = 1.0 / self.dt_hat
                    self.drift_ratio = self.fs_obs / self.fs_target

                # emit uniform segment for (t_prev,x_prev)->(t,x)
                seg = self._emit_uniform_segment(
                    self.t_prev, float(self.x_prev), t, float(x)
                )
                if seg.size:
                    out.append(seg)

            self.t_prev = t
            self.x_prev = float(x)

        if out:
            chunk = np.concatenate(out)
            self._emit_batched(chunk)

    def transport_to_uniform_idx(self, t_event: float) -> Optional[int]:
        """Map ABSOLUTE transport time to UNIFORM sample index."""
        if self.t0_transport is None or self.t0_uniform is None:
            return None
        # relative seconds on uniform axis
        u_time = (t_event - self.t0_transport) + self.t0_uniform
        return int(round(u_time * self.fs_target))
