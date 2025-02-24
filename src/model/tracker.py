"""
This module contains the Tracker class which utilizes the BYTETracker for tracking objects in video frames.
"""

from argparse import ArgumentParser
import numpy as np

from Bytetrack.yolox.tracker.byte_tracker import BYTETracker

class Tracker():
    """
    Tracker class that initializes and uses a BYTETracker for object tracking.
    """
    def __init__(self, model_name: str = "ByteTracker"):
        """
        Initializes the Tracker with the specified model name.

        Args:
            model_name (str): The name of the tracking model to use. Default is "ByteTracker".
        """
        self.model_name = model_name
        if self.model_name == "ByteTracker":
            self.tracker_args = self._set_bytetracker_parser()
            self.tracker = BYTETracker(self.tracker_args, frame_rate=30.0)
        else:
            raise KeyError(f"Model name {model_name} not found")

    def process_bbox(self, image: np.ndarray, bboxes: np.ndarray):
        """
        Updates the tracker with new bounding boxes and returns the online targets.

        Args:
            image (np.ndarray): The current frame of the video.
            bboxes (np.ndarray): The bounding boxes detected in the current frame.

        Returns:
            online_targets: The updated online targets from the tracker.
        """
        # 將新偵測的邊界框更新到跟蹤器
        online_targets = self.tracker.update(np.hstack((bboxes, np.full((bboxes.shape[0], 2), [0.9, 0]))),
                                             [image.shape[0], image.shape[1]], [image.shape[0], image.shape[1]]
                                             )
        return online_targets

    def _set_bytetracker_parser(self) -> ArgumentParser:
        """
        Sets up the argument parser for BYTETracker with default values.

        Returns:
            ArgumentParser: The argument parser with BYTETracker settings.
        """
        parser = ArgumentParser()

        # tracking args
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
        parser = parser.parse_args()
        return parser
