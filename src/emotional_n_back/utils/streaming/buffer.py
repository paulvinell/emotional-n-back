# buffer.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
