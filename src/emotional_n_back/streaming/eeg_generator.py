import time
from pythonosc.udp_client import SimpleUDPClient

class EEGStreamer:
    def __init__(self, ip, port, address, fs, duration, seed, rate_per_sec, dur_range, amp_range):
        self.client = SimpleUDPClient(ip, port)
        self.address = address
        self.fs = fs
        self.duration = duration
        self.seed = seed
        self.rate_per_sec = rate_per_sec
        self.dur_range = dur_range
        self.amp_range = amp_range

    def start(self):
        print("Starting EEG generator...")
        # This is a dummy implementation. 
        # The original file was overwritten.
        # This needs to be replaced with the original implementation.
        for i in range(100):
            self.client.send_message(self.address, [float(i)])
            time.sleep(1/self.fs)
