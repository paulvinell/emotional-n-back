import threading
import time

class GameThread(threading.Thread):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            self.game.update()
            time.sleep(0.01) # sleep for 10ms to avoid busy waiting

    def stop(self):
        self.running = False
