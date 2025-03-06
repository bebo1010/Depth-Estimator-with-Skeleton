"""
This module contains the PoseEstimator class which is responsible for detecting and tracking human poses in images.
"""

from argparse import ArgumentParser
import logging
import queue

import numpy as np
import polars as pl

from mmpose.apis import init_model
from mmpose.apis import inference_topdown as vitpose_inference_topdown
from mmpose.structures import merge_data_samples

from .detector import Detector
from .tracker import Tracker
from .point_processing_utils import filter_valid_targets, merge_person_data, smooth_keypoints, update_keypoint_buffer
from .timer import FPSTimer

class PoseEstimator():
    """
    PoseEstimator class for detecting and tracking human poses in images.

    Attributes:
        detector (Detector): The object detection model.
        tracker (Tracker): The tracking model.
        pose2d_estimator: The 2D pose estimation model.
        _person_df (pl.DataFrame): DataFrame to store person data.
        _bbox_buffer (list): Buffer to store bounding boxes.
        _track_id (int): ID of the tracked person.
        _joint_id (int): ID of the joint being tracked.
        _is_detect (bool): Flag to indicate if detection is active.
        fps_timer (FPSTimer): Timer to measure FPS.
        processed_frames (set): Set of processed frame numbers.
        image_buffer (queue.Queue): Buffer to store images.
        kpt_buffer (list): Buffer to store keypoints.
    """

    def __init__(self, detector: Detector, tracker: Tracker, pose_model_name: str = "vit-pose"):
        """
        Initialize the PoseEstimator.

        Args:
            detector (Detector): The object detection model.
            tracker (Tracker): The tracking model.
            pose_model_name (str): The name of the pose model to use.
        """
        logging.info("PoseEstimator initialized with wrapper.")
        self._model_name = pose_model_name

        self.detector = detector
        self.tracker = tracker
        self.pose2d_estimator = self.init_pose2d_estimator(pose_model_name)

        self._person_df = pl.DataFrame(
            {
                "track_id": pl.Series([], dtype=pl.Float64),
                "bbox": pl.Series([], dtype=pl.List(pl.Float64)),
                "area": pl.Series([], dtype=pl.Float64),
                "keypoints": pl.Series([], dtype=pl.List(pl.List(pl.Float64))),
                "frame_number": pl.Series([], dtype=pl.Int64)
            }
        )

        self._bbox_buffer = [None, None]
        self._track_id = None
        self._joint_id = None

        self._is_detect = False

        self.fps_timer = FPSTimer()

        self.processed_frames = set()
        self.image_buffer = queue.Queue(3)
        self.kpt_buffer = []

    def init_pose2d_estimator(self, model_name: str):
        """
        Initialize the 2D pose estimator model.

        Args:
            model_name (str): The name of the pose model to use.

        Returns:
            The initialized pose model.
        """
        if model_name == "vit-pose":
            pose2d_args = self.set_vitpose_parser()
            return init_model(pose2d_args.pose_config, pose2d_args.pose_checkpoint)

        raise KeyError(f"Model name {model_name} not found")

    def process_image(self, image_array: np.ndarray, bbox: np.ndarray) -> list:
        """
        Process an image to detect keypoints.

        Args:
            image_array (np.ndarray): The input image array.
            bbox (np.ndarray): The bounding box coordinates.

        Returns:
            list: The detected keypoints.
        """
        image = image_array[-1]
        pose_results = vitpose_inference_topdown(self.pose2d_estimator, image, bbox)
        data_samples = merge_data_samples(pose_results)
        return data_samples.get('pred_instances', None)

    def set_vitpose_parser(self) -> ArgumentParser:
        """
        Set up the argument parser for the ViTPose model.

        Returns:
            ArgumentParser: The argument parser with the necessary arguments.
        """
        parser = ArgumentParser()
        parser.add_argument('--pose-config',
                            default='./src/model/configs/ViTPose_base_simple_halpe_256x192.py',
                            help='Config file for pose')
        parser.add_argument('--pose-checkpoint',
                            default='./Db/checkpoints/vitpose.pth',
                            help='Checkpoint file for pose')
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args

    def detect_keypoints(self, image: np.ndarray, frame_num: int = None):
        """
        Detect keypoints in the given image.

        Args:
            image (np.ndarray): The input image.
            frame_num (int, optional): The frame number. Defaults to None.

        Returns:
            int: The frames per second (FPS) of the detection process.
        """
        if self.image_buffer.full():
            self.image_buffer.get()
        self.image_buffer.put(image)

        if not self._is_detect:
            return 0
        self.fps_timer.tic()

        if frame_num not in self.processed_frames:
            if frame_num % 5 == 0:
                bboxes = self.detector.process_image(image)
                online_targets = self.tracker.process_bbox(image, bboxes)
                online_bbox, track_ids = filter_valid_targets(online_targets, self._track_id)
                self._bbox_buffer = [online_bbox, track_ids]
            else:
                online_bbox, track_ids = self._bbox_buffer

            pred_instances = self.process_image(np.array(list(self.image_buffer.queue)), online_bbox)
            new_person_df = merge_person_data(pred_instances, track_ids, frame_num)
            new_person_df = smooth_keypoints(self._person_df, new_person_df, track_ids)

            self._person_df = pl.concat([self._person_df, new_person_df])
            self.processed_frames.add(frame_num)

        self.fps_timer.toc()
        elapsed_time_second = self.fps_timer.time_interval
        logging.info("Pose Estimation time for frame %d: %.6f seconds", frame_num, elapsed_time_second)

        if self._joint_id is not None and self._track_id is not None:
            self.kpt_buffer = update_keypoint_buffer(self._person_df, self._track_id, self._joint_id, frame_num)

        return int(self.fps_timer.fps) if int(self.fps_timer.fps) < 1000 else 0

    @property
    def model_name(self):
        """
        Get the name of the pose model.

        Returns:
            str: The name of the pose model.
        """
        return self._model_name

    @property
    def track_id(self):
        """
        Get the ID of the tracked person.

        Returns:
            int: The ID of the tracked person.
        """
        return self._track_id

    @track_id.setter
    def track_id(self, value):
        """
        Set the ID of the tracked person.

        Args:
            value (int): The new track ID.
        """
        if value != self._track_id:
            self._track_id = value
            logging.info("Person ID set to: %d", self._track_id)

    @property
    def joint_id(self):
        """
        Get the ID of the joint being tracked.

        Returns:
            int: The ID of the joint being tracked.
        """
        return self._joint_id

    @joint_id.setter
    def joint_id(self, value):
        """
        Set the ID of the joint being tracked.

        Args:
            value (int): The new joint ID.
        """
        if value != self._joint_id:
            self._joint_id = value
            logging.info("當前關節點: %d", self._joint_id)

    @property
    def is_detect(self):
        """
        Get the detection status.

        Returns:
            bool: The detection status.
        """
        return self._is_detect

    @is_detect.setter
    def is_detect(self, status: bool):
        """
        Set the detection status.

        Args:
            status (bool): The new detection status.
        """
        if status != self._is_detect:
            self._is_detect = status
            logging.info("當前偵測的狀態: %d", self._is_detect)

    @property
    def person_df(self):
        """
        Get the DataFrame containing person data.

        Returns:
            pl.DataFrame: The DataFrame containing person data.
        """
        return self._person_df

    @person_df.setter
    def person_df(self, load_df: pl.DataFrame):
        """
        Set the DataFrame containing person data.

        Args:
            load_df (pl.DataFrame): The new DataFrame containing person data.
        """
        if load_df.is_empty():
            logging.info("讀取資料的狀態: %s", not load_df.is_empty())
            return
        self._person_df = load_df
        self.processed_frames = set(self._person_df['frame_number'])
        logging.info("讀取資料的狀態: %s", not load_df.is_empty())

    def get_person_df(self, frame_num=None, is_select=False, is_kpt=False) -> pl.DataFrame:
        """
        Get the DataFrame containing person data for a specific frame.

        Args:
            frame_num (int, optional): The frame number. Defaults to None.
            is_select (bool, optional): Flag to select specific track ID. Defaults to False.
            is_kpt (bool, optional): Flag to return keypoints only. Defaults to False.

        Returns:
            pl.DataFrame: The DataFrame containing person data.
        """
        if self._person_df.is_empty():
            return pl.DataFrame([])

        condition = pl.Series([True] * len(self._person_df))
        if frame_num is not None:
            condition &= self._person_df["frame_number"] == frame_num

        if is_select and self._track_id is not None:
            condition &= self._person_df["track_id"] == self._track_id

        data = self._person_df.filter(condition)
        if data.is_empty():
            if not is_kpt:
                return pl.DataFrame([])
            if is_kpt:
                return list([])

        if is_kpt:
            data = data["keypoints"].to_list()[0]

        return data

    def update_person_df(self, x: float, y: float, frame_num: int, correct_kpt_idx: int):
        """
        Update the keypoints in the DataFrame.

        Args:
            x (float): The x-coordinate of the keypoint.
            y (float): The y-coordinate of the keypoint.
            frame_num (int): The frame number.
            correct_kpt_idx (int): The index of the keypoint to update.
        """
        if self._person_df is None or self._person_df.is_empty():
            return
        update_keypoint = self._person_df.filter(
            (pl.col("frame_number") == frame_num) &
            (pl.col("track_id") == self._track_id)
        )["keypoints"][0].to_list()
        update_keypoint[correct_kpt_idx] = [x, y] + update_keypoint[correct_kpt_idx][2:]

        self._person_df = self._person_df.with_columns(
            pl.when(
                (pl.col("frame_number") == frame_num) &
                (pl.col("track_id") == self._track_id)
            )
            .then(
              pl.Series("keypoints", [update_keypoint])
            )
            .otherwise(pl.col("keypoints"))
            .alias("keypoints")
        )

    def clear_keypoint_buffer(self):
        """
        Clear the keypoint buffer.
        """
        self.kpt_buffer = []

    def reset(self):
        """
        Reset the PoseEstimator to its initial state.
        """
        self._person_df = pl.DataFrame()
        self._track_id = None
        self._joint_id = None
        self.processed_frames = set()
        self._is_detect = False
        self.kpt_buffer = []
