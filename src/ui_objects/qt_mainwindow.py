"""
This module creates a PyQt5 application with an embedded Open3D view and image display functionality.
"""
from typing import Dict
import logging

import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import pyqtSlot
import win32gui

from src.utils import draw_lines
from src.model import Detector, Tracker, PoseEstimator
from src.model import SkeletonVisualizer, draw_points_and_skeleton

from .two_cameras_system_thread import TwoCamerasSystemThread
from .abstract_tab import AbstractTabWidget

class MainWindow(QtWidgets.QMainWindow):
    """
    Main window class for the application.
    """
    def __init__(self, base_dir: str):
        """
        Initializes the main window, sets up the layout, and starts the Open3D visualizer.
        """
        super().__init__()
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QGridLayout(widget)
        self.setCentralWidget(widget)

        self._initialize_labels(main_layout)
        self._initialize_3d_view(main_layout, widget)
        self._initialize_tabs(main_layout)

        # Set column stretch to achieve 7:3 ratio
        main_layout.setColumnStretch(0, 7)
        main_layout.setColumnStretch(1, 3)

        self.camera_thread: TwoCamerasSystemThread = None

        self.base_dir: str = base_dir

        self.togglable_states = {
            'Stream': False,
            'Model': False,
            'Horizontal': False,
            'Vertical': False,
            'Epipolar Line': False,
            'Freeze Frame': False,
        }

        self.model_variables = {
            'Frame Number': 0,
            'Skip Frame': 10,
        }

        # Connect window close event
        self.closeEvent = self._on_close # pylint: disable=invalid-name

        detector_model = Detector()
        tracker_model = Tracker()
        self.pose_models = {
            "Left": PoseEstimator(detector_model, tracker_model, pose_model_name="vit-pose"),
            "Right": PoseEstimator(detector_model, tracker_model, pose_model_name="vit-pose")
        }
        for pose_model in self.pose_models.values():
            pose_model.skip_frame = self.model_variables['Skip Frame']

    def _initialize_labels(self, layout: QtWidgets.QGridLayout):
        """
        Initializes the image labels and their layout.
        """
        # Create labels for images
        self.left_image_label = QtWidgets.QLabel("Image 1")
        self.right_image_label = QtWidgets.QLabel("Image 2")

        # Make labels scale with the window
        self.left_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.right_image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Create a layout for the labels
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addWidget(self.left_image_label)
        label_layout.addWidget(self.right_image_label)

        # Add the label layout to the main layout
        layout.addLayout(label_layout, 0, 0, 1, 1)

    def _initialize_3d_view(self, layout: QtWidgets.QGridLayout, widget: QtWidgets.QWidget):
        """
        Initializes the 3D view using SkeletonVisualizer.
        """
        # Create a layout for the 3D view
        view_layout = QtWidgets.QHBoxLayout()

        # Create the 3D view using SkeletonVisualizer
        self.skeleton_visualizer = SkeletonVisualizer()
        self.skeleton_visualizer.set_camera_intrinsics(1280, 720, 908.36, 908.36, 614.695, 354.577)
        self.skeleton_visualizer.open_window()

        hwnd = win32gui.FindWindowEx(0, 0, None, "Skeleton Visualizer")
        window = QtGui.QWindow.fromWinId(hwnd)
        self.windowcontainer = self.createWindowContainer(window, widget)

        # Make the 3D view scale with the window
        self.windowcontainer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Add the 3D view to the layout
        view_layout.addWidget(self.windowcontainer)

        # Add the view layout to the main layout
        layout.addLayout(view_layout, 1, 0, 1, 1)

    def _initialize_tabs(self, main_layout: QtWidgets.QLayout):
        """
        Initializes the tab control.
        """
        # Add the tab control
        self.tab_control = QtWidgets.QTabWidget()
        self.tab_control.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.tab_control, 0, 1, 2, 1)

        self.tab_objects: Dict[str, AbstractTabWidget] = {}

    @pyqtSlot(str, bool)
    def handle_toggle_signal(self, name: str, state: bool):
        """
        Slot to handle toggle signals.

        Parameters
        ----------
        name : str
            The name of the toggle.
        state : bool
            The state of the toggle.
        """
        logging.info("Toggle %s changed to %s", name, 'ON' if state else 'OFF')
        # Handle the toggle state change here
        self.togglable_states[name] = state
        if name == "Stream":
            if state:
                self.camera_thread.start_streaming()
            else:
                self.camera_thread.stop_streaming()
        if name == "Model":
            if not state:
                for pose_model in self.pose_models.values():
                    pose_model.disable_detection()
                self.model_variables['Frame Number'] = 0
            else:
                for pose_model in self.pose_models.values():
                    pose_model.enable_detection()

    def _on_close(self, event: QCloseEvent):
        """
        Handles the window close event.

        Args:
            event (QCloseEvent): The close event.
        """
        logging.info("Program terminated by user.")
        if self.camera_thread:
            logging.info("Releasing camera system.")
            self.camera_thread.stop()
        logging.info("Closing 3D skeleton visualizer.")
        self.skeleton_visualizer.close_window()
        event.accept()

    @pyqtSlot(bool, np.ndarray, np.ndarray)
    def display_images(self, success: bool, left_image: np.ndarray, right_image: np.ndarray):
        """
        Displays the given images in the left and right labels.

        Args:
            success (bool): Whether the images were successfully captured.
            left_image (numpy.ndarray): The left image in OpenCV format (RGB).
            right_image (numpy.ndarray): The right image in OpenCV format (RGB).
        """
        if success:
            left_qimage = self._convert_image(left_image)
            right_qimage = self._convert_image(right_image)
            self.left_image_label.setPixmap(QtGui.QPixmap.fromImage(left_qimage))
            self.right_image_label.setPixmap(QtGui.QPixmap.fromImage(right_qimage))

    @pyqtSlot(bool, np.ndarray, np.ndarray)
    def image_processing(self, success: bool, left_image: np.ndarray, right_image: np.ndarray):
        """
        Processes the given images and displays the results.

        Args:
            success (bool): Whether the images were successfully captured.
            left_image (numpy.ndarray): The left image in OpenCV format (RGB).
            right_image (numpy.ndarray): The right image in OpenCV format (RGB).
        """
        # Process the images here
        if success:
            left_display_image = left_image.copy()
            right_display_image = right_image.copy()

            if self.togglable_states['Horizontal']:
                draw_lines(left_display_image, 20, 'horizontal')
                draw_lines(right_display_image, 20, 'horizontal')

            if self.togglable_states['Vertical']:
                draw_lines(left_display_image, 20, 'vertical')
                draw_lines(right_display_image, 20, 'vertical')

            if self.togglable_states['Model']:
                left_detect_fps = self.pose_models['Left'].detect_keypoints(left_image,
                                                                            self.model_variables['Frame Number'])
                right_detect_fps = self.pose_models['Right'].detect_keypoints(right_image,
                                                                              self.model_variables['Frame Number'])

                logging.info("Left Detect FPS: %.2f, Right Detect FPS: %.2f", left_detect_fps, right_detect_fps)
                logging.info("============================================================")

                left_full_df = self.pose_models['Left'].get_person_df(self.model_variables['Frame Number'],
                                                                      is_select=True)
                left_display_image = draw_points_and_skeleton(left_display_image, left_full_df)

                right_full_df = self.pose_models['Right'].get_person_df(self.model_variables['Frame Number'],
                                                                        is_select=True)
                right_display_image = draw_points_and_skeleton(right_display_image, right_full_df)

                self.model_variables['Frame Number'] += 1

            self.display_images(True, left_display_image, right_display_image)

    def add_tab(self, widget: AbstractTabWidget, title: str):
        """
        Adds a new tab to the tab control.

        Args:
            widget (QtWidgets.QWidget): The widget to add as a tab.
            title (str): The title of the tab.
        """
        self.tab_control.addTab(widget, title)
        self.tab_objects[title] = widget

    def _convert_image(self, cv_img: np.ndarray):
        """
        Converts an OpenCV image to QImage.

        Args:
            cv_img (numpy.ndarray): The image in OpenCV format (RGB).

        Returns:
            QImage: The converted image.
        """
        height, width, _ = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        return q_img

    def set_camera_thread(self,
                          camera_thread: TwoCamerasSystemThread
                          ) -> None:
        """
        Set the camera thread for the application.

        Parameters
        ----------
        camera_thread : TwoCamerasSystemThread
            The camera thread to be used.

        Returns
        -------
        None
        """
        # Initialize and start the camera thread
        self.camera_thread = camera_thread
        for tab_name, tab_object in self.tab_objects.items():
            if tab_name in ["Control Tab", "Calibration Tab"]:
                tab_object.width = camera_thread.width
                tab_object.height = camera_thread.height

        self.camera_thread.rgb_images_signal.connect(self.display_images)
        self.camera_thread.start()
        logging.info("Camera thread set.")
