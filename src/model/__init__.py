"""
This module initializes the components of the Depth Estimator with Skeleton project.
It imports the following classes:
- Detector: Handles object detection.
- Tracker: Manages object tracking.
- PoseEstimator: Estimates the pose of detected objects.
The `__all__` list defines the public interface of the module, specifying the components
that can be imported when using `from module import *`.
Attributes:
    __all__ (list): List of public objects of this module.
"""

from .detector import Detector
from .tracker import Tracker
from .pose_estimator import PoseEstimator

__all__ = ["detector", "tracker", "pose_estimator"]
