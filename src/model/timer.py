import time
import torch


def time_synchronized():
    """
    獲取同步時間（適用於 GPU 運算）。
    Returns:
        float: 當前時間（秒）。
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter_ns() / 1e9


class FPSTimer:
    """
    用於計算程式碼執行時間和 FPS（每秒幀數）的計時器類別。
    """
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def tic(self):
        """
        計時開始。
        """
        self.start_time = time_synchronized()

    def toc(self):
        """
        計時結束。
        """
        self.end_time = time_synchronized()

    @property
    def time_interval(self):
        """
        獲取兩次計時之間的時間間隔（秒）。
        Returns:
            float: 執行時間（秒）。
        """
        return self.end_time - self.start_time

    @property
    def fps(self):
        """
        計算每秒幀數 (FPS)。
        Returns:
            float: FPS 值。
        """
        return round(1.0 / max(self.time_interval, 1e-10), 2)
