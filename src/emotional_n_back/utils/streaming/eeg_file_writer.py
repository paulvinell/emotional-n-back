import time
import numpy as np
from .base import BaseStreamer

class EEGWriter(BaseStreamer):
    def __init__(self, eeg_path: str, ip: str = "127.0.0.1", port: int = 5005):
        super().__init__(ip, port)
        self.eeg_path = eeg_path
        self.sampling_rate, self.data = self._load_data()

    def _load_data(self):
        with open(self.eeg_path, 'r') as f:
            lines = f.readlines()
            sampling_rate = int(lines[0].split(':')[1].strip())
            data = np.loadtxt(lines[1:], usecols=0)
        return sampling_rate, data

    def stream(self):
        chunk_size = 32
        while True:
            for i in range(0, len(self.data), chunk_size):
                chunk = self.data[i:i+chunk_size]
                self.client.send_message("/eeg", chunk.tolist())
                time.sleep(len(chunk) / self.sampling_rate)