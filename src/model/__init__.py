"""
Initializes the components of the Depth Estimator with Skeleton project.

Imports:
- Detector: Handles object detection.
- Tracker: Manages object tracking.
- PoseEstimator: Estimates the pose of detected objects.
- SkeletonVisualizer: Visualizes the skeleton of a detected object.
- halpe26_keypoint_info: Information about the HALPE26 keypoint.
- draw_points_and_skeleton: Draws points and skeleton connections.

Attributes:
    __all__ (list): List of public objects of this module.
"""

from .detector import Detector
from .tracker import Tracker
from .pose_estimator import PoseEstimator

from .skeleton_visualizer import SkeletonVisualizer
from .keypoint_info import halpe26_keypoint_info
from .skeleton_drawing_utils import draw_points_and_skeleton

__all__ = ["detector", "tracker", "pose_estimator",
           "SkeletonVisualizer",
           "halpe26_keypoint_info", "draw_points_and_skeleton"]
