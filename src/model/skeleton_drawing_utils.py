"""
This module provides utility functions for drawing keypoints and skeletons on images.
It includes functions to draw individual points, skeletons, and both combined.

The code probably comes from
https://github.com/stefanopini/simple-HRNet/blob/6bfcdaf5bcb006b945af9735883b198a54f62d4c/misc/visualization.py
"""

import numpy as np
import cv2

import matplotlib.pyplot as plt
from .keypoint_info import halpe26_keypoint_info

def draw_points(image, points, track_idx, color_palette='gist_rainbow', palette_samples=10):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'gist_rainbow'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 10

    Returns:
        A new image with overlaid points
    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)
    for pt in points:
        image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[track_idx % len(colors)]), -1)
    return image

def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0

    Returns:
        A new image with overlaid joints
    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    right_skeleton = halpe26_keypoint_info['right_points_indices']
    left_skeleton = halpe26_keypoint_info['left_points_indices']

    for joint in skeleton:
        pt1, pt2 = points[joint]
        skeleton_color = tuple(colors[person_index % len(colors)])
        skeleton_color = (0, 165, 255)
        if joint in right_skeleton:
            skeleton_color = (240, 176, 0)
        elif joint in left_skeleton:
            skeleton_color = (0, 0, 255)
        image = cv2.line(image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), skeleton_color, 6)
    return image

def draw_points_and_skeleton(image, person_df, skeleton):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn.
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`

    Returns:
        A new image with overlaid joints
    """
    if person_df is None:
        return image
    if person_df.is_empty():
        return image
    person_data = df_to_points(person_df)
    for track_id, points in person_data.items():
        image = draw_skeleton(image, points, skeleton, person_index=track_id)
        image = draw_points(image, points, track_idx=track_id)
    return image

def df_to_points(person_df):
    """
    Converts a DataFrame of person keypoints to a dictionary of points.

    Args:
        person_df: DataFrame containing person keypoints
            Columns: 'track_id', 'keypoints'
            'keypoints' should be a list of (y, x, confidence) tuples

    Returns:
        A dictionary where keys are track IDs and values are numpy arrays of points
    """
    def swap_values(kpts):
        return [[item[1], item[0], item[2]] for item in kpts]

    person_data = {}
    track_ids = person_df['track_id']
    person_kpts = person_df['keypoints']
    for track_id, kpts in zip(track_ids, person_kpts):
        person_data[track_id] = np.array(swap_values(kpts))

    return person_data
