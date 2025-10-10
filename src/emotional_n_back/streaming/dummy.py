from .base import BaseStreamer

class DummyStreamer(BaseStreamer):
    def __init__(self, ip: str = "127.0.0.1", port: int = 5005, address: str = "/some/address", message: str = "Hello OSC"):
        super().__init__(ip, port)
        self.address = address
        self.message = message

    def stream(self):
        self.client.send_message(self.address, self.message)
        print(f"Sent message '{self.message}' to {self.address} at {self.client._address}:{self.client._port}")
