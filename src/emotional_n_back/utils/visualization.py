import time
import numpy as np
import matplotlib.pyplot as plt

class EEGVisualizer:
    def __init__(self, data_buffer, time_buffer, buffer_size):
        self.data_buffer = data_buffer
        self.time_buffer = time_buffer
        self.buffer_size = buffer_size

    def start(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(np.zeros(self.buffer_size))
        ax.set_ylim(-3, 3)
        ax.set_xlim(0, 5)
        plt.show()

        while True:
            try:
                if not self.data_buffer:
                    plt.pause(0.01)
                    continue

                # Vertical auto-scaling
                min_val = min(self.data_buffer)
                max_val = max(self.data_buffer)
                margin = (max_val - min_val) * 0.1
                ax.set_ylim(min_val - margin, max_val + margin)

                # Horizontal auto-scaling
                current_time = time.time()
                ax.set_xlim(current_time - 5, current_time)
                
                line.set_data(list(self.time_buffer), list(self.data_buffer))
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            except (KeyboardInterrupt, Exception):
                break
