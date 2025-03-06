"""
This module provides utility functions for drawing lines on images and applying colormaps to depth images.

Functions:
    draw_lines(image: np.ndarray, step: int, orientation: str) -> None:
        Draw lines on the image.

    apply_colormap(depth_image: Optional[np.ndarray], reference_image: np.ndarray) -> np.ndarray:
        Apply colormap to the depth image.

    draw_aruco_rectangle(image: np.ndarray, corners: np.ndarray, marker_id: int) -> None:
        Draw a rectangle from the 4 corner points with red color and display the marker ID.

    update_aruco_info(marker_id: int, estimated_3d_coords: list, realsense_3d_coords: list,
                      mean_depth_estimated: float, mean_depth_realsense: float) -> str:
        Update ArUco marker information.
"""

from typing import Optional

import cv2
import numpy as np

def draw_lines(image: np.ndarray, step: int, orientation: str) -> None:
    """
    Draw lines on the image.

    Args:
        image (np.ndarray): The image on which to draw lines.
        step (int): The step size between lines.
        orientation (str): The orientation of the lines ('horizontal' or 'vertical').

    Returns:
        None
    """
    for i in range(0, image.shape[0 if orientation == 'horizontal' else 1], step):
        if orientation == 'horizontal':
            cv2.line(image, (0, i), (image.shape[1], i), (0, 0, 255), 1)
        else:
            cv2.line(image, (i, 0), (i, image.shape[0]), (0, 0, 255), 1)

def apply_colormap(depth_image: Optional[np.ndarray], reference_image: np.ndarray) -> np.ndarray:
    """
    Apply colormap to the depth image.

    Args:
        depth_image (Optional[np.ndarray]): The depth image to which the colormap is applied.
        reference_image (np.ndarray): The reference image for size comparison.

    Returns:
        np.ndarray: The color-mapped depth image.
    """
    return np.zeros_like(reference_image) if depth_image is None else \
        cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
