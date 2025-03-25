import os
from datetime import datetime
import sys

import cv2

from PyQt5 import QtWidgets

try:
    import torch # pylint: disable=unused-import
except ModuleNotFoundError:
    print("PyTorch not installed. Please install PyTorch to run the application.")
    print("Run",
            "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2",
            "--index-url https://download.pytorch.org/whl/cu118 to install PyTorch.")

from src.utils import setup_directories, setup_logging

from src.ui_objects.qt_mainwindow import MainWindow
from src.ui_objects.basic_setting_tab import BasicSettingTabWidget
from src.ui_objects.calibration_tab import CalibrationTabWidget
from src.ui_objects.two_cameras_system_thread import TwoCamerasSystemThread

if __name__ == '__main__':
    base_dir = os.path.join("Db", f"{datetime.now().strftime('%Y%m%d')}")
    setup_directories(base_dir)
    setup_logging(base_dir)

    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow(base_dir)
    form.setWindowTitle('o3d Embed')
    form.setFixedSize(1920, 1080)  # Fix the window size to 1920x1080
    form.show()

    test_left_image = cv2.imread('left_image1.png')
    test_right_image = cv2.imread('right_image1.png')

    form.display_images(True, test_left_image, test_right_image)

    # Replace the example tab with TabWidget
    basic_tab_widget = BasicSettingTabWidget(base_dir)
    basic_tab_widget.toggle_signal.connect(form.handle_toggle_signal)
    form.add_tab(basic_tab_widget, "Control Tab")

    calibration_tab_widget = CalibrationTabWidget(base_dir)
    form.add_tab(calibration_tab_widget, "Calibration Tab")

    import pyrealsense2 as rs
    from .camera_objects import RealsenseCameraSystem

    context = rs.context()
    connected_devices = context.query_devices()

    WIDTH = 848
    HEIGHT = 480
    cameras = RealsenseCameraSystem(width=WIDTH, height=HEIGHT)

    # Create and start the camera system thread
    camera_thread = TwoCamerasSystemThread(cameras)
    form.set_camera_thread(camera_thread)
    camera_thread.rgb_images_signal.connect(form.display_images)
    camera_thread.start()

    sys.exit(app.exec_())
