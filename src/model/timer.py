"""
This module provides a timer utility for measuring execution time and calculating FPS (frames per second).
"""

import time
import torch

def time_synchronized():
    """
    Get synchronized time (suitable for GPU computation).
    Returns:
        float: Current time (seconds).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

class FPSTimer:
    """
    A timer class for measuring code execution time and FPS (frames per second).
    """
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def tic(self):
        """
        Start the timer.
        """
        self.start_time = time_synchronized()

    def toc(self):
        """
        Stop the timer.
        """
        self.end_time = time_synchronized()

    @property
    def time_interval(self):
        """
        Get the time interval between two timings (seconds).
        Returns:
            float: Execution time (seconds).
        """
        return self.end_time - self.start_time

    @property
    def fps(self):
        """
        Calculate frames per second (FPS).
        Returns:
            float: FPS value.
        """
        return round(1.0 / max(self.time_interval, 1e-10), 2)
