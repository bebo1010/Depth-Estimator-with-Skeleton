"""
Module: epipolar_line_detector
Description: This module contains the EpipolarLineDetector class for detecting epipolar lines using OpenCV.
"""
from typing import Tuple
import logging
import os
import json

import cv2
import numpy as np

class EpipolarLineDetector:
    """
    A class to detect epipolar lines using OpenCV.
    """

    def __init__(self):
        """
        Initializes the EpipolarLineDetector class.
        """
        self.detector = None
        self.detector_index = 0
        self.detectors = [
            ("ORB", cv2.ORB_create()),
            ("SIFT", cv2.SIFT_create()),
            # ("SURF", cv2.xfeatures2d.SURF_create()),
            # ("FAST", cv2.FastFeatureDetector_create()),
            # ("BRIEF", cv2.xfeatures2d.BriefDescriptorExtractor_create()),
            # ("KAZE", cv2.KAZE_create())
        ]
        self.set_feature_detector(self.detectors[self.detector_index][1])

        self._fundamental_matrix = None
        self._homography_left = None
        self._homography_right = None
        self.fundamental_matrix_file = None
        self._homography_ready = False

        logging.info("EpipolarLineDetector initialized with detector: %s", self.detectors[self.detector_index][0])

    @property
    def homography_left(self):
        """
        Getter for the left homography matrix.

        Returns
        -------
        numpy.ndarray
            The left homography matrix.
        """
        return self._homography_left

    @property
    def homography_right(self):
        """
        Getter for the right homography matrix.

        Returns
        -------
        numpy.ndarray
            The right homography matrix.
        """
        return self._homography_right

    @property
    def homography_ready(self):
        """
        Getter for the homography_ready flag.

        Returns
        -------
        bool
            True if the homography matrices are ready, False otherwise.
        """
        return self._homography_ready

    def set_feature_detector(self, detector: cv2.Feature2D) -> None:
        """
        Sets the feature detector to be used.

        For example, you can pass in an instance of cv2.SIFT() or cv2.ORB().

        Parameters
        ----------
        detector : cv2.Feature2D
            The feature detector to be used.

        Example
        -------
        >>> detector = cv2.SIFT_create()
        >>> obj.set_feature_detector(detector)
        """
        self.detector = detector
        logging.info("Feature detector set to: %s", type(detector).__name__)

    def switch_detector(self, direction: str) -> None:
        """
        Switches the feature detector cyclically.

        Parameters
        ----------
        direction : str
            Direction to switch the detector ('n' for next, 'p' for previous).
        """
        if direction == 'n':
            self.detector_index = (self.detector_index + 1) % len(self.detectors)
        elif direction == 'p':
            self.detector_index = (self.detector_index - 1) % len(self.detectors)
        self.set_feature_detector(self.detectors[self.detector_index][1])
        logging.info("Switched feature detector to: %s", self.detectors[self.detector_index][0])

    def set_save_directory(self, directory: str) -> None:
        """
        Sets the directory for saving the fundamental matrix and updates the file path.
        Attempts to load the fundamental matrix from the specified directory.

        Parameters
        ----------
        directory : str
            The directory where the fundamental matrix will be saved.
        """
        self.fundamental_matrix_file = os.path.join(directory, "fundamental_homography.json")
        logging.info("Save directory set to: %s", directory)

        # Attempt to load the fundamental matrix from the JSON file
        if os.path.exists(self.fundamental_matrix_file):
            with open(self.fundamental_matrix_file, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                self._fundamental_matrix = np.array(data["fundamental_matrix"])
                self._homography_left = np.array(data["homography_left"])
                self._homography_right = np.array(data["homography_right"])
                self._homography_ready = True
                logging.info("Loaded fundamental matrix and homography matrices from file: %s",
                             self.fundamental_matrix_file)
        else:
            self._fundamental_matrix = None
            self._homography_left = None
            self._homography_right = None
            self._homography_ready = False
            logging.info("No existing fundamental matrix file found.")

    def draw_epilines_from_scene(self,
                         left_image: np.ndarray,
                         right_image: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects features and computes the epipolar lines for the detected points.

        Parameters
        ----------
        left_image : np.ndarray
            The left input image in which to detect features.
        right_image : np.ndarray
            The right input image in which to detect features.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The left image with epipolar lines drawn and the right image with epipolar lines drawn.
        """
        if self.detector is None:
            raise ValueError("Feature detector is not set. Use set_feature_detector method to set it.")

        logging.info("Detecting features in the left and right images.")
        keypoints_left, descriptors_left = self.detector.detectAndCompute(left_image, None)
        keypoints_right, descriptors_right = self.detector.detectAndCompute(right_image, None)

        points_left = cv2.KeyPoint.convert(keypoints_left)
        points_right = cv2.KeyPoint.convert(keypoints_right)

        logging.info("Matching features between the left and right images.")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract the matched points
        points_left = np.array([points_left[m.queryIdx] for m in good_matches], dtype=np.float32)
        points_right = np.array([points_right[m.trainIdx] for m in good_matches], dtype=np.float32)

        if not self.homography_ready:
            logging.info("Computing fundamental matrix.")
            self._fundamental_matrix, mask = cv2.findFundamentalMat(points_left, points_right,
                                   method=cv2.FM_RANSAC,
                                   ransacReprojThreshold=3,
                                   confidence=0.99,
                                   maxIters=100)

            points_left = points_left[mask.ravel() == 1]
            points_right = points_right[mask.ravel() == 1]

            logging.info("Computing epilines.")
            epilines_left = cv2.computeCorrespondEpilines(points_right, 2, self._fundamental_matrix).reshape(-1, 3)
            epilines_right = cv2.computeCorrespondEpilines(points_left, 1, self._fundamental_matrix).reshape(-1, 3)

            if self._is_good_fundamental_matrix(epilines_left):
                self._compute_homographies(points_left, points_right, left_image.shape)
                self._save_fundamental_matrix()
                self._homography_ready = True
        else:
            logging.info("Using existing fundamental matrix.")
            epilines_left = cv2.computeCorrespondEpilines(points_right, 2, self._fundamental_matrix).reshape(-1, 3)
            epilines_right = cv2.computeCorrespondEpilines(points_left, 1, self._fundamental_matrix).reshape(-1, 3)

        # Limit the number of epipolar lines to 10
        num_lines = min(100, len(epilines_left), len(epilines_right))

        left_image_with_lines = self._draw_epilines(left_image,
                                                    epilines_left[:num_lines], points_left[:num_lines])
        right_image_with_lines = self._draw_epilines(right_image,
                                                     epilines_right[:num_lines], points_right[:num_lines])

        logging.info("Epipolar lines computed and drawn on images.")
        return left_image_with_lines, right_image_with_lines

    def draw_epilines_from_corners(self,
                                      left_image: np.ndarray,
                                      right_image: np.ndarray,
                                      corners_left: np.ndarray,
                                      corners_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute epipolar lines from the corner points of detected ArUco markers.

        Parameters
        ----------
        left_image : np.ndarray
            The left input image.
        right_image : np.ndarray
            The right input image.
        corners_left : np.ndarray
            Corner points from the left image.
        corners_right : np.ndarray
            Corner points from the right image.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The left image with epipolar lines drawn and the right image with epipolar lines drawn.
        """
        logging.info("Computing epilines from corners.")
        # Ensure points are in the correct shape and type
        points_left = np.asarray(corners_left, dtype=np.float32).reshape(-1, 2)
        points_right = np.asarray(corners_right, dtype=np.float32).reshape(-1, 2)

        if not self.homography_ready:
            return left_image, right_image

        epilines_left = cv2.computeCorrespondEpilines(points_right, 2, self._fundamental_matrix).reshape(-1, 3)
        epilines_right = cv2.computeCorrespondEpilines(points_left, 1, self._fundamental_matrix).reshape(-1, 3)

        left_image_with_lines = self._draw_epilines(left_image, epilines_left, points_left)
        right_image_with_lines = self._draw_epilines(right_image, epilines_right, points_right)

        logging.info("Epipolar lines computed and drawn on images from corners.")
        return left_image_with_lines, right_image_with_lines

    def _is_good_fundamental_matrix(self, epilines: np.ndarray) -> bool:
        """
        Checks if more than 50% of the epipolar lines are horizontal.

        Parameters
        ----------
        epilines : np.ndarray
            The epipolar lines to be checked.

        Returns
        -------
        bool
            True if more than 50% of the epipolar lines are horizontal, False otherwise.
        """
        # Count the number of horizontal lines by checking if the slope -a/b is near 0
        horizontal_lines = sum(abs(-line[0] / line[1]) < 0.005 for line in epilines)

        # Check if more than 50% of the lines are horizontal
        return horizontal_lines > 0.5 * len(epilines)

    def _save_fundamental_matrix(self) -> None:
        """
        Saves the fundamental matrix and homography matrices to a JSON file with formatted text.
        """
        fundamental_matrix_list = self._fundamental_matrix.tolist()
        homography_left_list = self.homography_left.tolist()
        homography_right_list = self.homography_right.tolist()
        data = {
            "fundamental_matrix": fundamental_matrix_list,
            "homography_left": homography_left_list,
            "homography_right": homography_right_list
        }

        with open(self.fundamental_matrix_file, 'w', encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

        logging.info("Fundamental matrix and homography matrices saved to JSON file: %s", self.fundamental_matrix_file)

    def _compute_homographies(self,
                                points_left: np.ndarray,
                                points_right: np.ndarray,
                                image_shape: Tuple[int, int]
                                ) -> None:
        """
        Computes the homography matrices for both views using the fundamental matrix.

        Parameters
        ----------
        points_left : np.ndarray
            Points from the left image.
        points_right : np.ndarray
            Points from the right image.
        image_shape : Tuple[int, int]
            Shape of the image.
        """
        logging.info("Computing homography matrices.")
        _, self._homography_left, self._homography_right = cv2.stereoRectifyUncalibrated(
            points_left, points_right, self._fundamental_matrix,
            imgSize=(image_shape[1], image_shape[0])
        )
        logging.info("Homography matrices computed.")

    def _draw_epilines(self, image: np.ndarray, epilines: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Draws the epipolar lines on the image.

        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the epipolar lines.
        epilines : np.ndarray
            The epipolar lines to be drawn.
        points : np.ndarray
            Points corresponding to the epipolar lines.

        Returns
        -------
        np.ndarray
            The image with epipolar lines drawn.
        """
        logging.info("Drawing epipolar lines on the image.")
        image_with_lines = image.copy()
        bright_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)
        ]
        num_colors = len(bright_colors)
        for i, (r, pt) in enumerate(zip(epilines, points)):
            color = bright_colors[i % num_colors]
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [image.shape[1], -(r[2] + r[0] * image.shape[1]) / r[1]])
            image_with_lines = cv2.line(image_with_lines, (x0, y0), (x1, y1), color, 1)
            image_with_lines = cv2.circle(image_with_lines, tuple(map(int, pt)), 5, color, -1)
        logging.info("Epipolar lines drawn.")
        return image_with_lines
