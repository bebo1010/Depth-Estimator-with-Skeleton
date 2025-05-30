"""
This module contains the Detector class for object detection using a pre-trained yolo10 model.
"""

from argparse import ArgumentParser

import numpy as np


from mmcv.transforms import Compose
from mmdet.apis import inference_detector, init_detector
from mmpose.evaluation.functional import nms

class Detector():
    """
    A class used to perform object detection using a pre-trained yolo10 model.
    """
    def __init__(self):
        """
        Initializes the Detector with the specified configuration and checkpoint.
        """
        self.detect_args = self._set_detect_parser()
        self.detector = init_detector(self.detect_args.det_config,
                                      self.detect_args.det_checkpoint,
                                      device=self.detect_args.device
                                      )
        self.detector.cfg.test_dataloader.dataset.pipeline[0].type \
                        = 'mmdet.LoadImageFromNDArray'
        self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Processes an image to detect objects and returns bounding boxes.

        Args:
            image (np.ndarray): The input image in numpy array format.

        Returns:
            np.ndarray: The bounding boxes of detected objects.
        """
        # 進行物件偵測
        result = inference_detector(self.detector, image, test_pipeline= self.detector_test_pipeline)
        pred_instances = result.pred_instances
        det_result = pred_instances[pred_instances.scores >self.detect_args.score_thr].cpu().numpy()
        # 篩選指定類別的邊界框
        bboxes = det_result.bboxes[det_result.labels == self.detect_args.det_cat_id]
        scores = det_result.scores[det_result.labels == self.detect_args.det_cat_id]
        bboxes = bboxes[nms(np.hstack((bboxes, scores[:, None])), self.detect_args.nms_thr), :4]
        return bboxes

    def _set_detect_parser(self) -> ArgumentParser:
        """
        Sets up the argument parser for detection configuration.

        Returns:
            ArgumentParser: The argument parser with detection configuration.
        """
        parser = ArgumentParser()
        parser.add_argument('--det-config',
                            default='./src/model/configs/yolox/yolox_tiny_fast_8xb8-300e_coco.py',
                            help='Config file for detection'
                            )
        parser.add_argument('--det-checkpoint',
                            default='./Db/checkpoints/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth',
                            help='Checkpoint file for detection'
                            )
        parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
        parser.add_argument(
            '--score-thr', type=float, default=0.3, help='Bbox score threshold')
        parser.add_argument(
            '--nms-thr',
            type=float,
            default=0.3,
            help='IoU threshold for bounding box NMS')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        args = parser.parse_args()
        return args
