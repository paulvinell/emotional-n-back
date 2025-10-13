from abc import ABC, abstractmethod
from pythonosc import udp_client

class BaseStreamer(ABC):
    def __init__(self, ip: str = "127.0.0.1", port: int = 5005):
        self.client = udp_client.SimpleUDPClient(ip, port)

    @abstractmethod
    def stream(self):
        pass
