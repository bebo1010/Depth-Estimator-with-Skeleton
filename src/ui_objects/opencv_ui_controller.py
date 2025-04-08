"""
Module for main UI controller.
"""
import os
import time
from datetime import datetime
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.opencv_objects import EpipolarLineDetector, ChessboardCalibrator
from src.camera_objects import TwoCamerasSystem
from src.utils import get_starting_index, setup_directories, setup_logging, save_images, draw_lines, \
    apply_colormap, load_images_from_directory, save_setup_info, load_setup_info, save_skeleton_info_to_csv, \
    load_camera_parameters

# from src.model import Detector, Tracker, PoseEstimator, \
#     SkeletonVisualizer, halpe26_keypoint_info, draw_points_and_skeleton

from src.model import Detector, Tracker, PoseEstimator, \
    halpe26_keypoint_info, draw_points_and_skeleton

from src.model.skeleton_visualizer_matplotlib import SkeletonVisualizer

class OpencvUIController():
    """
    UI controller for skeleton detection application.
    """
    def __init__(self) -> None:
        """
        Initialize UI controller without parameters.
        """
        self.stop_flag = False

        self.base_dir = None
        self.image_index = 0
        self.chessboard_image_index = 0
        self.mouse_coords = {'x': 0, 'y': 0}
        self.window_size = (2000, 960)
        self.matrix_view_size = (1280, 960)
        self._setup_window()

        self.camera_system = None

        self.camera_params = {
            'system_prefix': None,
            'focal_length': None,
            'baseline': None,
            'principal_point': None,
            'width': None,
            'height': None
        }

        self.display_option = {
            'image_mode': False,
            'horizontal_lines': False,
            'vertical_lines': False,
            'epipolar_lines': False,
            'calibration_mode': False,
            'freeze_mode': False,
            'estimation_mode': False,
        }

        self.epipolar_detector = EpipolarLineDetector()

        self.chessboard_calibrator = ChessboardCalibrator()
        # small chessboard pattern size
        self.chessboard_calibrator.pattern_size = (10, 7)
        self.chessboard_calibrator.square_size_mm = 10

        self.image_points = {'left': [], 'right': []}

        self.loaded_images = []
        self.loaded_image_index = 0

        self.frame_number = 0
        detector_model = Detector()
        tracker_model = Tracker()
        self.left_pose_model = PoseEstimator(detector_model, tracker_model, pose_model_name="vit-pose")
        self.right_pose_model = PoseEstimator(detector_model, tracker_model, pose_model_name="vit-pose")

        self.open3d_visualizer = SkeletonVisualizer()

    def set_parameters(self,
                       system_prefix: str,
                       focal_length: float,
                       baseline: float,
                       principal_point: Tuple[int, int]) -> None:
        """
        Set the parameters for the UI controller.

        Parameters
        ----------
        system_prefix : str
            Prefix for the camera system.
        focal_length : float
            Focal length of the camera.
        baseline : float
            Baseline distance between the cameras.
        principal_point : Tuple[int, int]
            Principal point of the camera.

        Returns
        -------
        None
        """
        self.base_dir = os.path.join("Db", f"{system_prefix}_{datetime.now().strftime('%Y%m%d')}")
        left_ir_dir = os.path.join(self.base_dir, "left_skeleton_images")
        left_chessboard_dir = os.path.join(self.base_dir, "left_chessboard_images")

        setup_directories(self.base_dir)
        self.image_index = get_starting_index(left_ir_dir)
        self.chessboard_image_index = get_starting_index(left_chessboard_dir) - 1

        setup_logging(self.base_dir)

        self.camera_params['system_prefix'] = system_prefix
        self.camera_params['focal_length'] = focal_length
        self.camera_params['baseline'] = baseline
        self.camera_params['principal_point'] = principal_point

        # Set the save directory for the fundamental matrix
        self.epipolar_detector.set_save_directory(self.base_dir)

    def set_camera_system(self, camera_system: TwoCamerasSystem) -> None:
        """
        Set the camera system for the application.

        Parameters
        ----------
        camera_system : TwoCamerasSystem
            The camera system to be used.

        Returns
        -------
        None
        """
        self.camera_system = camera_system
        self.camera_params['width'] = self.camera_system.get_width()
        self.camera_params['height'] = self.camera_system.get_height()
        save_setup_info(self.base_dir, self.camera_params)

        parameter_dir = os.path.join("Db", f"{self.camera_params['system_prefix']}_calibration_parameter")
        success, stereo_params = \
            load_camera_parameters(parameter_dir)
        if success:
            self.chessboard_calibrator.stereo_camera_parameters = stereo_params
            self.chessboard_calibrator.initialize_rectification_maps((self.camera_params['width'],
                                                                      self.camera_params['height']))

        width = self.camera_params['width']
        height = self.camera_params['height']
        focal_length = self.camera_params['focal_length']
        cx = self.camera_params['principal_point'][0]
        cy = self.camera_params['principal_point'][1]

        self.open3d_visualizer.set_camera_intrinsics(width, height, focal_length, focal_length, cx, cy)
        self.open3d_visualizer.open_window()

    def start(self) -> None:
        """
        Initialize the OpenCV window and enter a loop to continuously capture and process images.

        Returns
        -------
        None
        """
        cv2.namedWindow("Combined View (2x2)")

        left_color_image, right_color_image = None, None
        first_depth_image, second_depth_image = None, None

        while not self.stop_flag:
            self._update_window_title()

            if self.camera_system and not self.display_option['image_mode']:
                if not self.display_option['freeze_mode']:
                    start_time = time.perf_counter_ns()

                    success, left_color_image, right_color_image = self.camera_system.get_rgb_images()
                    _, first_depth_image, second_depth_image = self.camera_system.get_depth_images()
                    if not success:
                        continue

                    if self.chessboard_calibrator.rectification_ready:
                        left_color_image, right_color_image = \
                            self.chessboard_calibrator.rectify_images(left_color_image, right_color_image)

                    elif self.epipolar_detector.homography_ready:
                        left_color_image = cv2.warpPerspective(left_color_image,
                                                               self.epipolar_detector.homography_left,
                                                            (left_color_image.shape[1], left_color_image.shape[0]))
                        right_color_image = cv2.warpPerspective(right_color_image,
                                                                self.epipolar_detector.homography_right,
                                                            (right_color_image.shape[1], right_color_image.shape[0]))

                    if self.display_option['estimation_mode']:
                        left_detect_fps = self.left_pose_model.detect_keypoints(left_color_image, self.frame_number)
                        right_detect_fps = self.right_pose_model.detect_keypoints(right_color_image, self.frame_number)
                        logging.info("Left Detect FPS: %.2f, Right Detect FPS: %.2f", left_detect_fps, right_detect_fps)

                        self.frame_number += 1

                    end_time = time.perf_counter_ns()
                    logging.info("Full Update Time: %.2f ms", (end_time - start_time) / 1e6)

                if self.display_option['calibration_mode']:
                    self._process_and_draw_chessboard(left_color_image, right_color_image)

                self._process_and_draw_images(left_color_image, right_color_image,
                                              first_depth_image, second_depth_image, self.frame_number-1)

            else:
                self._display_loaded_images()
            # Check for key presses
            key = cv2.pollKey() & 0xFF
            if self._handle_key_presses(key,
                                        left_color_image, right_color_image,
                                        first_depth_image, second_depth_image):
                break

    def _calibrate_cameras(self) -> None:
        """
        Calibrate the cameras using the saved chessboard images.

        Returns
        -------
        None
        """
        if self.image_points['left'] and self.image_points['right']:
            image_size = (self.camera_params['width'], self.camera_params['height'])
            success = self.chessboard_calibrator.calibrate_stereo_camera(self.image_points['left'],
                                                                         self.image_points['right'],
                                                                         image_size)
            if success:
                logging.info("Stereo camera calibration successful.")
                self.chessboard_calibrator.save_parameters("./Db/", self.camera_params['system_prefix'])

        else:
            logging.warning("No chessboard images saved for calibration.")

    def _display_image(self,
                       left_colored: np.ndarray,
                       right_colored: np.ndarray,
                       first_depth_colormap: np.ndarray,
                       second_depth_colormap: np.ndarray,
                       aruco_info: str,
                       mouse_info: str) -> None:
        """
        Display the processed images on the window.

        Parameters
        ----------
        left_colored : np.ndarray
            Colored image of the left camera.
        right_colored : np.ndarray
            Colored image of the right camera.
        first_depth_colormap : np.ndarray
            Color-mapped first depth image.
        second_depth_colormap : np.ndarray
            Color-mapped second depth image.
        aruco_info : str
            Information about detected ArUco markers.
        mouse_info : str
            Information about mouse hover.

        Returns
        -------
        None
        """
        image_width, image_height = self.matrix_view_size

        top_row = np.hstack((left_colored, right_colored))
        bottom_row = np.hstack((first_depth_colormap, second_depth_colormap))
        combined_view = np.vstack((cv2.resize(top_row, (image_width, image_height // 2)),
                                   cv2.resize(bottom_row, (image_width, image_height // 2))))

        # Create a blank image with the desired window size
        window_image = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        window_image.fill(255)  # White background
        window_image[:image_height, :image_width] = combined_view

        # Add ArUco and mouse information to the right side of the window
        x0 = 20
        y0, dy = 30, 30
        cv2.putText(window_image, "Units: mm", (image_width + x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text
        for i, line in enumerate(aruco_info.split('\n')):
            y = y0 + (i + 1) * dy
            cv2.putText(window_image, line, (image_width + x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text
        for i, line in enumerate(mouse_info.split('\n')):
            y = y0 + (i + len(aruco_info.split('\n')) + 1) * dy
            cv2.putText(window_image, line, (image_width + x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  # Black text

        cv2.imshow("Combined View (2x2)", window_image)

    def _handle_key_presses(self, key: int, left_color_image: np.ndarray, right_color_image: np.ndarray,
                            first_depth_image: Optional[np.ndarray], second_depth_image: Optional[np.ndarray]) -> bool:
        """
        Handle key presses for various actions.

        Parameters
        ----------
        key : int
            Key code of the pressed key.
        left_gray_image : np.ndarray
            Grayscale image of the left camera.
        right_gray_image : np.ndarray
            Grayscale image of the right camera.
        first_depth_image : Optional[np.ndarray]
            First depth image.
        second_depth_image : Optional[np.ndarray]
            Second depth image.

        Returns
        -------
        bool
            True if the application should exit, False otherwise.
        """
        # Define actions for each key
        actions = {
            27: self._exit_or_switch_mode,  # ESC key
            ord('s'): lambda: self._save_images(left_color_image, right_color_image,
                                                first_depth_image, second_depth_image),
            ord('h'): lambda: self._toggle_option('horizontal_lines'),
            ord('v'): lambda: self._toggle_option('vertical_lines'),
            ord('e'): lambda: self._toggle_option('epipolar_lines'),
            ord('n'): lambda: self._navigate_images('next'),
            ord('p'): lambda: self._navigate_images('previous'),
            ord('c'): self._toggle_calibration_mode,
            ord('f'): self._toggle_freeze_mode,
            ord('l'): self._load_images,
            ord('m'): self._toggle_model,
        }

        # Execute the corresponding action if the key is in the dictionary
        if key in actions: # lower case keys
            return actions[key]()
        if key + 32 in actions: # upper case keys
            return actions[key + 32]()

        return False

    def _exit_or_switch_mode(self) -> bool:
        """
        Exit the program or switch from image mode to normal stream mode.

        Returns
        -------
        bool
            True if the application should exit, False otherwise.
        """
        if self.display_option['image_mode']:
            self.display_option['image_mode'] = False
            self._update_window_title()
            return False

        self._exit_program()
        return True

    def _navigate_images(self, direction: str) -> bool:
        """
        Navigate through the loaded images or switch detectors based on the current mode.

        Parameters
        ----------
        direction : str
            Direction to navigate ('next' or 'previous').

        Returns
        -------
        bool
            Always returns False.
        """
        if self.display_option['image_mode']:
            if direction == 'next':
                self.loaded_image_index = (self.loaded_image_index + 1) % len(self.loaded_images)
            elif direction == 'previous':
                self.loaded_image_index = (self.loaded_image_index - 1) % len(self.loaded_images)

            self._display_loaded_images()

        elif self.display_option['epipolar_lines']:
            if direction == 'next':
                self.epipolar_detector.switch_detector('n')
            elif direction == 'previous':
                self.epipolar_detector.switch_detector('p')

            self._update_window_title()

        return False

    def _exit_program(self) -> None:
        """
        Terminate the program by releasing the camera system and closing all OpenCV windows.

        Returns
        -------
        None
        """
        logging.info("Program terminated by user.")
        if self.camera_system:
            logging.info("Releasing camera system.")
            self.camera_system.release()
        cv2.destroyAllWindows()
        self.open3d_visualizer.close_window()
        self.stop_flag = True

    def _save_images(self, left_color_image: np.ndarray, right_color_image: np.ndarray,
                     first_depth_image: Optional[np.ndarray], second_depth_image: Optional[np.ndarray]) -> bool:
        """
        Save the provided images based on the current display option.

        Parameters
        ----------
        left_color_image : np.ndarray
            RGB image of the left camera.
        right_color_image : np.ndarray
            RGB image of the right camera.
        first_depth_image : Optional[np.ndarray]
            First depth image.
        second_depth_image : Optional[np.ndarray]
            Second depth image.

        Returns
        -------
        bool
            Always returns False.
        """
        if self.display_option['image_mode']:
            logging.warning("Cannot save images while in image mode.")
            return False

        if self.display_option['calibration_mode']:
            self._save_chessboard_images(left_color_image, right_color_image)
        else:
            save_images(self.base_dir, left_color_image, right_color_image,
                        self.image_index, first_depth_image, second_depth_image,
                        prefix="skeleton")

            # Save 2D and 3D points
            left_skeleton_points = \
                self.left_pose_model.get_person_df(self.frame_number - 1, is_select=True, is_kpt=True)
            right_skeleton_points = \
                self.right_pose_model.get_person_df(self.frame_number - 1, is_select=True, is_kpt=True)
            if len(left_skeleton_points) > 0 and len(right_skeleton_points) > 0:
                _, _, _, _, _, estimated_3d_coords, realsense_3d_coords = self._process_disparity_and_depth(
                    np.array(left_skeleton_points), np.array(right_skeleton_points), first_depth_image)

                skeleton_data = []

                for joint_name, left_point, right_point, estimated_3d_point, realsense_3d_point \
                        in zip(halpe26_keypoint_info["keypoints"].values(),
                                left_skeleton_points, right_skeleton_points,
                                estimated_3d_coords, realsense_3d_coords):
                    skeleton_data.append([joint_name,
                                          left_point[0], left_point[1],
                                          right_point[0], right_point[1],
                                          estimated_3d_point[0], estimated_3d_point[1], estimated_3d_point[2],
                                          realsense_3d_point[0], realsense_3d_point[1], realsense_3d_point[2]])

                save_skeleton_info_to_csv(self.base_dir, self.image_index, skeleton_data)

            self.image_index += 1

        return False

    def _toggle_option(self, option: str) -> bool:
        """
        Toggle the state of a given display option and update the window title.

        Parameters
        ----------
        option : str
            The display option to toggle.

        Returns
        -------
        bool
            Always returns False.
        """

        self.display_option[option] = not self.display_option[option]
        self._update_window_title()

        return False

    def _toggle_calibration_mode(self) -> bool:
        """
        Toggle the calibration mode for the UI.

        Returns
        -------
        bool
            Always returns False.
        """
        if not self.camera_system:
            logging.warning("Calibration mode cannot be activated without a camera system.")
            return False

        self.display_option['calibration_mode'] = not self.display_option['calibration_mode']
        if self.display_option['calibration_mode']:
            self.display_option['freeze_mode'] = False
        if not self.display_option['calibration_mode']:
            self._calibrate_cameras()

        return False

    def _toggle_freeze_mode(self) -> bool:
        """
        Toggle the freeze mode in the display options.

        Returns
        -------
        bool
            Always returns False.
        """

        self.display_option['freeze_mode'] = not self.display_option['freeze_mode']
        if self.display_option['freeze_mode']:
            self.display_option['calibration_mode'] = False

        return False

    def _toggle_model(self) -> bool:
        """
        Toggle the model for the UI.

        Returns
        -------
        bool
            Always returns False.
        """
        if self.display_option['estimation_mode']: # disable model
            self.frame_number = 0

            self.left_pose_model.reset()
            self.right_pose_model.reset()

            self.left_pose_model.is_detect = False
            self.right_pose_model.is_detect = False

            logging.info("Pose estimation models disabled.")

        else: # enable model
            self.left_pose_model.is_detect = True
            self.right_pose_model.is_detect = True

            self.left_pose_model.queued_select = True
            self.right_pose_model.queued_select = True

            logging.info("Pose estimation models enabled.")

        self.display_option['estimation_mode'] = not self.display_option['estimation_mode']

        return False

    def _process_and_draw_chessboard(self, left_color_image: np.ndarray, right_color_image: np.ndarray) -> None:
        """
        Process and draw chessboard corners on the images in calibration mode.

        Parameters
        ----------
        left_color_image : np.ndarray
            RGB image of the left camera.
        right_color_image : np.ndarray
            RGB image of the right camera.

        Returns
        -------
        None
        """
        left_gray_image = cv2.cvtColor(left_color_image, cv2.COLOR_BGR2GRAY)
        right_gray_image = cv2.cvtColor(right_color_image, cv2.COLOR_BGR2GRAY)

        # Define the scale factor
        scale_factor = 2

        # Downsample the images by the scale factor
        left_small = cv2.resize(left_gray_image,
                                (self.camera_params['width'] // scale_factor,
                                 self.camera_params['height'] // scale_factor))

        right_small = cv2.resize(right_gray_image,
                                 (self.camera_params['width'] // scale_factor,
                                  self.camera_params['height'] // scale_factor))

        # Detect chessboard corners on the downsampled images
        ret_left, corners_left_small = self.chessboard_calibrator.detect_chessboard_corners(left_small)
        ret_right, corners_right_small = self.chessboard_calibrator.detect_chessboard_corners(right_small)

        left_colored = cv2.cvtColor(left_gray_image, cv2.COLOR_GRAY2BGR)
        right_colored = cv2.cvtColor(right_gray_image, cv2.COLOR_GRAY2BGR)

        if ret_left and ret_right:
            # Rescale the corners back to the original image size
            corners_left = corners_left_small * scale_factor
            corners_right = corners_right_small * scale_factor

            left_colored = self.chessboard_calibrator.display_chessboard_corners(left_colored, corners_left)
            right_colored = self.chessboard_calibrator.display_chessboard_corners(right_colored, corners_right)

        self._display_image(left_colored, right_colored,
                            np.zeros_like(left_colored), np.zeros_like(right_colored),
                            aruco_info="", mouse_info="")

    def _process_and_draw_images(self,
                                 left_color_image: np.ndarray,
                                 right_color_image: np.ndarray,
                                 first_depth_image: Optional[np.ndarray] = None,
                                 second_depth_image: Optional[np.ndarray] = None,
                                 frame_number: Optional[int] = 0) -> None:
        """
        Process and draw on the images.

        Parameters
        ----------
        left_color_image : np.ndarray
            RGB image of the left camera.
        right_color_image : np.ndarray
            RGB image of the right camera.
        first_depth_image : Optional[np.ndarray]
            First depth image.
        second_depth_image : Optional[np.ndarray]
            Second depth image.
        frame_number : Optional[int]
            Frame number for the images. Default is 0.

        Returns
        -------
        None
        """
        left_display_image = left_color_image.copy()
        right_display_image = right_color_image.copy()

        if self.display_option['horizontal_lines']:
            draw_lines(left_display_image, 20, 'horizontal')
            draw_lines(right_display_image, 20, 'horizontal')

        if self.display_option['vertical_lines']:
            draw_lines(left_display_image, 20, 'vertical')
            draw_lines(right_display_image, 20, 'vertical')

        first_depth_colormap = apply_colormap(first_depth_image, left_display_image)
        second_depth_colormap = apply_colormap(second_depth_image, left_display_image)

        if self.display_option['epipolar_lines']:
            left_display_image, right_display_image = self.epipolar_detector.draw_epilines_from_scene(
                left_display_image, right_display_image)

        left_full_df = self.left_pose_model.get_person_df(frame_number, is_select=True)
        left_display_image = draw_points_and_skeleton(left_display_image, left_full_df)

        right_full_df = self.right_pose_model.get_person_df(frame_number, is_select=True)
        right_display_image = draw_points_and_skeleton(right_display_image, right_full_df)

        left_keypoints = self.left_pose_model.get_person_df(frame_number, is_select=True, is_kpt=True)
        right_keypoints = self.right_pose_model.get_person_df(frame_number, is_select=True, is_kpt=True)

        if all([len(left_keypoints) > 0, len(right_keypoints) > 0]):
            left_keypoints = np.array(left_keypoints)
            right_keypoints = np.array(right_keypoints)

            disparities, mean_disparity, variance_disparity, \
                estimated_depth_mm, realsense_depth_mm, \
                    estimated_3d_coords, _realsense_3d_coords = \
                        self._process_disparity_and_depth(left_keypoints, right_keypoints, first_depth_image)

            logging.info("Frame: %d\n"
                         "Estimated Depth: %s mm\n"
                         "RealSense Depth: %s mm\n"
                        "Disparities: %s\n"
                        "Mean Disparity: %.2f, Variance: %.2f",
                        frame_number, estimated_depth_mm, realsense_depth_mm,
                        disparities.tolist(), mean_disparity, variance_disparity)
            logging.info("============================================================")

            self.open3d_visualizer.update_skeleton_halpe26(estimated_3d_coords)

        # Calculate mouse hover info
        mouse_x, mouse_y = self.mouse_coords['x'], self.mouse_coords['y']
        if first_depth_image is not None:
            scaled_x = int(mouse_x * (self.camera_params['width'] / (self.matrix_view_size[0] // 2)))
            scaled_y = int(mouse_y * (self.camera_params['height'] / (self.matrix_view_size[1] // 2)))

            depth_value = first_depth_image[scaled_y, scaled_x]

            mouse_x_3d = (scaled_x - self.camera_params['principal_point'][0]) \
                            * depth_value / self.camera_params['focal_length']
            mouse_y_3d = (scaled_y - self.camera_params['principal_point'][1]) \
                            * depth_value / self.camera_params['focal_length']
            mouse_info = f"Mouse: ({mouse_x_3d:7.1f}, {mouse_y_3d:7.1f}, {depth_value:7.1f})"
        else:
            mouse_info = "Mouse: (N/A, N/A, N/A)"

        self._display_image(left_display_image, right_display_image,
                            first_depth_colormap, second_depth_colormap,
                            aruco_info="", mouse_info=mouse_info)

    def _process_disparity_and_depth(self,
                                     left_keypoints: np.ndarray,
                                     right_keypoints: np.ndarray,
                                     depth_image: Optional[np.ndarray] = None
                                     ) -> Tuple[np.ndarray, float, float,
                                                np.ndarray, Optional[np.ndarray],
                                                Optional[list], Optional[list]]:
        """
        Calculate disparities, mean, variance, and depth from keypoints.

        Optionally include depth from depth image.

        Parameters
        ----------
        left_keypoints : np.ndarray
            Keypoints of the left image.
        right_keypoints : np.ndarray
            Keypoints of the right image.
        depth_image : Optional[np.ndarray]
            Depth image for calculating depth from image (optional).

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            - Disparities between matching corners.
            - Mean of disparities.
            - Variance of disparities.
            - Calculated depth per corner in mm.
            - Depths at the 4 corner points from depth image (if provided).
            - 3D coordinates from estimated depth.
            - 3D coordinates from depth image (if provided).
        """
        left_xs = left_keypoints[:, 0]
        left_ys = left_keypoints[:, 1]
        right_xs = right_keypoints[:, 0]
        disparities = np.abs(left_xs - right_xs)
        mean_disparity = np.mean(disparities)
        variance_disparity = np.var(disparities)

        estimated_depth_mm = (self.camera_params['focal_length'] * self.camera_params['baseline']) / disparities

        def calculate_3d_coords(xs, ys, depths):
            return [((x - self.camera_params['principal_point'][0]) * depth / self.camera_params['focal_length'],
                    (y - self.camera_params['principal_point'][1]) * depth / self.camera_params['focal_length'], depth)
                    for x, y, depth in zip(xs, ys, depths)]

        estimated_3d_coords = calculate_3d_coords(left_xs, left_ys, estimated_depth_mm)

        realsense_depth_mm = None
        if depth_image is not None:
            realsense_depth_mm = np.zeros_like(estimated_depth_mm)
            for j, (cx, cy) in enumerate(left_keypoints[:, :2]):
                depth_value = depth_image[min(max(int(cy), 0), self.camera_params['height'] - 1),
                                          min(max(int(cx), 0), self.camera_params['width'] - 1)]
                realsense_depth_mm[j] = depth_value
            realsense_3d_coords = calculate_3d_coords(left_xs, left_ys, realsense_depth_mm)
        else:
            # FIX: need to rethink about this solution
            realsense_3d_coords = [(0, 0, 0)] * len(estimated_3d_coords)

        return disparities, mean_disparity, variance_disparity, \
            estimated_depth_mm, realsense_depth_mm, \
            estimated_3d_coords, realsense_3d_coords

    def _save_chessboard_images(self, left_color_image: np.ndarray, right_color_image: np.ndarray) -> None:
        """
        Save chessboard images to disk and store image points for calibration.

        Parameters
        ----------
        left_color_image : np.ndarray
            RGB image of the left camera.
        right_color_image : np.ndarray
            RGB image of the right camera.

        Returns
        -------
        None
        """
        left_gray_image = cv2.cvtColor(left_color_image, cv2.COLOR_BGR2GRAY)
        right_gray_image = cv2.cvtColor(right_color_image, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = self.chessboard_calibrator.detect_chessboard_corners(left_gray_image)
        ret_right, corners_right = self.chessboard_calibrator.detect_chessboard_corners(right_gray_image)

        if ret_left and ret_right:
            self.image_points['left'].append(corners_left)
            self.image_points['right'].append(corners_right)

            self.chessboard_image_index += 1
            save_images(self.base_dir, left_gray_image, right_gray_image,
                        self.chessboard_image_index, prefix="chessboard")

    def _setup_window(self) -> None:
        """
        Setup OpenCV window and set the mouse callback.

        Returns
        -------
        None
        """
        def _mouse_callback(event, x, y, _flags, _param):
            """Update the mouse position."""
            if event == cv2.EVENT_MOUSEMOVE:
                if 0 <= y < self.matrix_view_size[1] // 2:
                    if 0 <= x < self.matrix_view_size[0] // 2:
                        self.mouse_coords['x'], self.mouse_coords['y'] = x, y
                    elif self.matrix_view_size[0] // 2 <= x < self.matrix_view_size[0]:
                        self.mouse_coords['x'], self.mouse_coords['y'] = x - self.matrix_view_size[0] // 2, y
                else:
                    self.mouse_coords['x'], self.mouse_coords['y'] = 0, 0
            elif event == cv2.EVENT_LBUTTONDOWN:
                if not self.display_option['estimation_mode']:
                    return

                click_coord = (x, y)
                if 0 <= y < self.matrix_view_size[1] // 2:
                     # Select the person in left view
                    if 0 <= x < self.matrix_view_size[0] // 2:
                        click_coord = (int(x * (self.camera_params['width'] / (self.matrix_view_size[0] // 2))),
                                       int(y * (self.camera_params['height'] / (self.matrix_view_size[1] // 2))))
                        self.left_pose_model.select_person(click_coord[0], click_coord[1])
                        # Temp solution to sync the track ID, they may not be the same
                        self.right_pose_model.track_id = self.left_pose_model.track_id

                    # Select the person in right view
                    elif self.matrix_view_size[0] // 2 <= x < self.matrix_view_size[0]:
                        click_coord = (int((x - self.matrix_view_size[0] // 2) * \
                                           (self.camera_params['width'] / (self.matrix_view_size[0] // 2))),
                                       int(y * (self.camera_params['height'] / (self.matrix_view_size[1] // 2))))

                        self.right_pose_model.select_person(click_coord[0], click_coord[1])
                        # Temp solution to sync the track ID, they may not be the same
                        self.left_pose_model.track_id = self.right_pose_model.track_id

        # Create a window and set the mouse callback
        cv2.namedWindow("Combined View (2x2)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Combined View (2x2)", *self.window_size)
        cv2.setMouseCallback("Combined View (2x2)", _mouse_callback)

    def _update_window_title(self, setup_name: str = "") -> None:
        """
        Update the window title with the current detector name if epipolar lines are shown.

        Parameters
        ----------
        setup_name : str, optional
            Name of the setup (default is "").

        Returns
        -------
        None
        """
        title = "Combined View (2x2)"
        if setup_name:
            title += f" - {setup_name}"

        if self.display_option['calibration_mode']:
            title += f" - Calibration Mode - Images Saved: {self.chessboard_image_index}"

        elif self.display_option['epipolar_lines']:
            detector_name = self.epipolar_detector.detectors[self.epipolar_detector.detector_index][0]
            title += f" - Detector: {detector_name}"

        elif self.display_option['freeze_mode']:
            title += " - Freeze Mode"

        elif self.display_option['image_mode']:
            total_images = len(self.loaded_images)
            current_index = self.loaded_image_index + 1
            title += f" - Image {current_index}/{total_images}"

        cv2.setWindowTitle("Combined View (2x2)", title)

    def _load_images(self) -> bool:
        """
        Load images from a selected directory.

        Returns
        -------
        bool
            Always returns False.
        """
        _ = QApplication([])
        selected_dir = QFileDialog.getExistingDirectory(None, "Select Directory", "Db/")

        if not selected_dir:
            QMessageBox.critical(None, "Error", "No directory selected.")
            return False

        setup_info = load_setup_info(selected_dir)
        if not setup_info:
            QMessageBox.critical(None, "Error", "Failed to load setup information.")
            return False

        self.camera_params = {
            'system_prefix': setup_info['system_prefix'],
            'focal_length': setup_info['focal_length'],
            'baseline': setup_info['baseline'],
            'principal_point': tuple(setup_info['principal_point']),
            'width': setup_info['width'],
            'height': setup_info['height']
        }
        self.base_dir = selected_dir
        self._update_window_title(self.camera_params['system_prefix'])

        self.open3d_visualizer.set_camera_intrinsics(setup_info['width'], setup_info['height'],
                                                      setup_info['focal_length'], setup_info['focal_length'],
                                                        setup_info['principal_point'][0],
                                                        setup_info['principal_point'][1])
        self.open3d_visualizer.open_window()

        loaded_images, error = load_images_from_directory(selected_dir)
        if error:
            QMessageBox.critical(None, "Error", error)
            return False

        self.loaded_images = loaded_images
        self.loaded_image_index = 0
        self.display_option['image_mode'] = True
        self._display_loaded_images()

        return False

    def _display_loaded_images(self):
        """
        Display the loaded images.

        Returns
        -------
        None
        """
        if not hasattr(self, 'loaded_images') or not self.loaded_images:
            return

        left_image_path, right_image_path, \
        left_depth_image_path, right_depth_image_path = self.loaded_images[self.loaded_image_index]

        left_color_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
        right_color_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
        left_depth_image = np.load(left_depth_image_path) \
            if left_depth_image_path else None
        right_depth_image = np.load(right_depth_image_path) \
            if right_depth_image_path else None

        if left_color_image is None or right_color_image is None:
            QMessageBox.critical(None, "Error", "Failed to load images.")
            return

        left_detect_fps = self.left_pose_model.detect_keypoints(left_color_image, self.loaded_image_index)
        right_detect_fps = self.right_pose_model.detect_keypoints(right_color_image, self.loaded_image_index)
        logging.info("Left Detect FPS: %.2f, Right Detect FPS: %.2f", left_detect_fps, right_detect_fps)

        self._process_and_draw_images(left_color_image, right_color_image,
                                      left_depth_image, right_depth_image,
                                      self.loaded_image_index)
        self._update_window_title(self.camera_params['system_prefix'])
