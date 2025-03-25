"""
This module contains the TwoCamerasSystemThread class, which handles the operations of the TwoCamerasSystem
in a separate thread using PyQt5's QThread.
"""
import time
import logging

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from src.camera_objects import TwoCamerasSystem

class TwoCamerasSystemThread(QThread):
    """
    A QThread class to handle the TwoCamerasSystem operations in a separate thread.
    """
    rgb_images_signal = pyqtSignal(bool, np.ndarray, np.ndarray)

    def __init__(self, camera_system: TwoCamerasSystem):
        """
        Initializes the TwoCamerasSystemThread.

        Parameters
        ----------
        camera_system : TwoCamerasSystem
            The camera system to be used.
        """
        super().__init__()
        self.camera_system = camera_system
        self.streaming = False
        self._stop_flag = False

    @property
    def width(self) -> int:
        """
        Get the width of the camera system.

        Returns
        -------
        int
            Width of the camera system.
        """
        return self.camera_system.get_width()

    @property
    def height(self) -> int:
        """
        Get the height of the camera system.

        Returns
        -------
        int
            Height of the camera system.
        """
        return self.camera_system.get_height()

    def run(self):
        """
        The main loop of the thread. Continuously captures and emits RGB images if streaming is enabled.
        """
        while not self._stop_flag:
            if self.streaming:
                start_time = time.perf_counter_ns()

                success, left_rgb, right_rgb = self.camera_system.get_rgb_images()
                self.rgb_images_signal.emit(success, left_rgb, right_rgb)

                end_time = time.perf_counter_ns()
                logging.info("Frame acquisition Time: %.2f ms", (end_time - start_time) / 1e6)

    def start_streaming(self):
        """
        Start streaming images from the camera system.
        """
        logging.info("Two camera system thread started.")
        self.streaming = True

    def stop_streaming(self):
        """
        Stop streaming images from the camera system.
        """
        logging.info("Two camera system thread stopped.")
        self.streaming = False

    def stop(self):
        """
        Stop the thread and release the camera system.
        """
        self._stop_flag = True
        self.camera_system.release()
        self.quit()
        self.wait()
