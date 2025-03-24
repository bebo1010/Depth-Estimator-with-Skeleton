"""
This module creates a PyQt5 application with an embedded Open3D view and image display functionality.
"""
try:
    import torch # pylint: disable=unused-import
except ModuleNotFoundError:
    print("PyTorch not installed. Please install PyTorch to run the application.")
    print("Run",
            "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2",
            "--index-url https://download.pytorch.org/whl/cu118 to install PyTorch.")
import sys

import cv2
import numpy as np

from PyQt5 import QtWidgets, QtGui
import win32gui

from src.model import SkeletonVisualizer
from .basic_setting_tab import BasicSettingTabWidget  # Import the TabWidget

class MainWindow(QtWidgets.QMainWindow):
    """
    Main window class for the application.
    """
    def __init__(self):
        """
        Initializes the main window, sets up the layout, and starts the Open3D visualizer.
        """
        super().__init__()
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QGridLayout(widget)
        self.setCentralWidget(widget)

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

        # Create a layout for the 3D view and tab control
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

        # Add the label layout and view layout to the main layout
        main_layout.addLayout(label_layout, 0, 0, 1, 1)
        main_layout.addLayout(view_layout, 1, 0, 1, 1)

        # Add the tab control
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        main_layout.addWidget(self.tabs, 0, 1, 2, 1)

        # Set column stretch to achieve 7:3 ratio
        main_layout.setColumnStretch(0, 7)
        main_layout.setColumnStretch(1, 3)

    def display_images(self, left_image: np.ndarray, right_image: np.ndarray):
        """
        Displays the given images in the left and right labels.

        Args:
            left_image (numpy.ndarray): The left image in OpenCV format (RGB).
            right_image (numpy.ndarray): The right image in OpenCV format (RGB).
        """
        left_qimage = self._convert_image(left_image)
        right_qimage = self._convert_image(right_image)
        self.left_image_label.setPixmap(QtGui.QPixmap.fromImage(left_qimage))
        self.right_image_label.setPixmap(QtGui.QPixmap.fromImage(right_qimage))

    def add_tab(self, widget: QtWidgets.QWidget, title: str):
        """
        Adds a new tab to the tab control.

        Args:
            widget (QtWidgets.QWidget): The widget to add as a tab.
            title (str): The title of the tab.
        """
        self.tabs.addTab(widget, title)

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.setWindowTitle('o3d Embed')
    form.setFixedSize(1920, 1080)  # Fix the window size to 1920x1080
    form.show()

    test_left_image = cv2.imread('left_image1.png')
    test_right_image = cv2.imread('right_image1.png')

    form.display_images(test_left_image, test_right_image)

    # Replace the example tab with TabWidget
    tab_widget = BasicSettingTabWidget()
    form.add_tab(tab_widget, "Control Tab")

    sys.exit(app.exec_())
