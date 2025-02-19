import numpy as np
from .detector import Detector
from .tracker import Tracker
from .point_processing_utils import filter_valid_targets, merge_person_data, smooth_keypoints, update_keypoint_buffer
from .timer import FPSTimer
import logging
import queue
import polars as pl
from argparse import ArgumentParser
from mmpose.apis import init_model
from mmpose.apis import inference_topdown as vitpose_inference_topdown
from mmpose.structures import merge_data_samples

class PoseEstimator(object):
    def __init__(self, detector: Detector, tracker: Tracker, pose_model_name: str = "vit-pose"):
        logging.info("PoseEstimator initialized with wrapper.")
        self._model_name = pose_model_name

        self.detector = detector
        self.tracker = tracker
        self.pose2d_estimator = self.init_pose2d_estimator(pose_model_name)

        self._person_df = pl.DataFrame()

        self._bbox_buffer = [None, None]
        self._track_id = None
        self._joint_id = None

        self._is_detect = False

        self.fps_timer = FPSTimer()

        self.processed_frames = set()
        self.image_buffer = queue.Queue(3)
        self.kpt_buffer = []

    def init_pose2d_estimator(self, model_name: str):
        if model_name == "vit-pose":
            pose2d_args = self.set_vitpose_parser()
            return init_model(pose2d_args.pose_config, pose2d_args.pose_checkpoint)
        else:
            raise KeyError(f"Model name {model_name} not found")

    def process_image(self, image_array: np.ndarray, bbox: np.ndarray) -> list:
        image = image_array[-1]
        pose_results = vitpose_inference_topdown(self.pose2d_estimator, image, bbox)
        data_samples = merge_data_samples(pose_results)
        return data_samples.get('pred_instances', None)

    def set_vitpose_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--pose-config', default='./src/model/configs/ViTPose_base_simple_halpe_256x192.py', help='Config file for pose')
        parser.add_argument('--pose-checkpoint', default='./Db/checkpoints/vitpose.pth', help='Checkpoint file for pose')
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args

    def detect_keypoints(self, image: np.ndarray, frame_num: int = None):
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
        logging.info("Pose Estimation time for frame %d: %.2f seconds", frame_num, elapsed_time_second)

        if self._joint_id is not None and self._track_id is not None:
            self.kpt_buffer = update_keypoint_buffer(self._person_df, self._track_id, self._joint_id, frame_num)

        return int(self.fps_timer.fps) if int(self.fps_timer.fps) < 1000 else 0

    @property
    def model_name(self):
        return self._model_name

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, value):
        if value != self._track_id:
            self._track_id = value
            logging.info("Person ID set to: %d", self._track_id)

    @property
    def joint_id(self):
        return self._joint_id

    @joint_id.setter
    def joint_id(self, value):
        if value != self._joint_id:
            self._joint_id = value
            logging.info("當前關節點: %d", self._joint_id)

    @property
    def is_detect(self):
        return self._is_detect

    @is_detect.setter
    def is_detect(self, status: bool):
        if status != self._is_detect:
            self._is_detect = status
            logging.info("當前偵測的狀態: %d", self._is_detect)

    @property
    def person_df(self):
        return self._person_df

    @person_df.setter
    def person_df(self, load_df: pl.DataFrame):
        if load_df.is_empty():
            logging.info("讀取資料的狀態: %s", not load_df.is_empty())
            return
        self._person_df = load_df
        self.processed_frames = {frame_num for frame_num in self._person_df['frame_number']}
        logging.info("讀取資料的狀態: %s", not load_df.is_empty())

    def get_person_df(self, frame_num=None, is_select=False, is_kpt=False) -> pl.DataFrame:
        if self._person_df.is_empty():
            return pl.DataFrame([])

        condition = pl.Series([True] * len(self._person_df))
        if frame_num is not None:
            condition &= self._person_df["frame_number"] == frame_num

        if is_select and self._track_id is not None:
            condition &= self._person_df["track_id"] == self._track_id

        data = self._person_df.filter(condition)
        if data.is_empty():
            return None

        if is_kpt:
            data = data["keypoints"].to_list()[0]

        return data

    def update_person_df(self, x: float, y: float, frame_num: int, correct_kpt_idx: int):
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
        self.kpt_buffer = []

    def reset(self):
        self._person_df = pl.DataFrame()
        self._track_id = None
        self._joint_id = None
        self.processed_frames = set()
        self._is_detect = False
        self.kpt_buffer = []
