import threading
import time
from collections import deque
from pythonosc import dispatcher, osc_server

class OSCReader:
    def __init__(self, ip, port, visualize=False, buffer_size=256*5):
        self.ip = ip
        self.port = port
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.server = None

    def _handler(self, address, *args):
        if self.visualize:
            self.data_buffer.append(args[0])
            self.time_buffer.append(time.time())
        else:
            print(f"Received message from {address}: {args}")

    def start(self):
        disp = dispatcher.Dispatcher()
        disp.map("/*", self._handler)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), disp)
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        print(f"Serving on {self.server.server_address}")

        if self.visualize:
            from .visualization import EEGVisualizer
            visualizer = EEGVisualizer(self.data_buffer, self.time_buffer, self.buffer_size)
            visualizer.start()
        else:
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                pass

        self.server.shutdown()
