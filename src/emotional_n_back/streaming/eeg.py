import time
import numpy as np
from .base import BaseStreamer
from ..eeg.eeg_generator_morph_bursts import (
    add_random_bursts,
    generate_eeg_variants,
    morph_variants_over_time,
)

class EEGStreamer(BaseStreamer):
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 5005,
        address: str = "/eeg",
        fs: int = 256,
        duration: float = 20.0,
        seed: int = 7,
        rate_per_sec: float = 0.5,
        dur_range: str = "0.2,0.8",
        amp_range: str = "0.15,0.7",
    ):
        super().__init__(ip, port)
        self.address = address
        self.fs = fs
        self.duration = duration
        self.seed = seed
        self.rate_per_sec = rate_per_sec
        self.dur_range = dur_range
        self.amp_range = amp_range

    def stream(self):
        t, X, *_ = generate_eeg_variants(fs=self.fs, duration=self.duration, seed=self.seed)

        waypoints = [
            (0.0, [1.0, 0.0, 0.0]),
            (8.0, [0.2, 0.8, 0.0]),
            (15.0, [0.0, 0.5, 0.5]),
            (self.duration, [0.0, 0.0, 1.0]),
        ]

        x, W = morph_variants_over_time(t, X, waypoints)

        x = add_random_bursts(
            x,
            self.fs,
            seed=11,
            rate_per_sec=self.rate_per_sec,
            dur_range=tuple(map(float, self.dur_range.split(","))),
            amp_range=tuple(map(float, self.amp_range.split(","))),
        )

        print(f"Sending EEG-like signal to {self.address} at {self.client._address}:{self.client._port}")
        try:
            while True:
                for sample in x:
                    self.client.send_message(self.address, float(sample))
                    time.sleep(1 / self.fs)
        except KeyboardInterrupt:
            print("Stopped sending EEG data.")
