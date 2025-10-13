
from .base import BaseStreamer
from .dummy import DummyStreamer
from .eeg_generator import EEGStreamer
from .eeg_file_writer import EEGWriter

def create_streamer(
    mode: str,
    ip: str = "127.0.0.1",
    port: int = 5005,
    address: str = None,
    message: str = "Hello OSC",
    fs: int = 256,
    duration: float = 20.0,
    seed: int = 7,
    rate_per_sec: float = 0.5,
    dur_range: str = "0.2,0.8",
    amp_range: str = "0.15,0.7",
    eeg_path: str = None,
) -> BaseStreamer:
    if eeg_path:
        mode = "eeg_file"

    if mode == "dummy":
        if address is None:
            address = "/some/address"
        return DummyStreamer(ip=ip, port=port, address=address, message=message)
    elif mode == "eeg":
        if address is None:
            address = "/eeg"
        return EEGStreamer(
            ip=ip,
            port=port,
            address=address,
            fs=fs,
            duration=duration,
            seed=seed,
            rate_per_sec=rate_per_sec,
            dur_range=dur_range,
            amp_range=amp_range,
        )
    elif mode == "eeg_file":
        return EEGWriter(eeg_path=eeg_path, ip=ip, port=port)
    else:
        raise ValueError(f"Unknown mode: {mode}")
