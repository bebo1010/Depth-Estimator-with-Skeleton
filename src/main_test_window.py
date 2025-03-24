
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

from src.ui_objects.qt_mainwindow import MainWindow
from src.ui_objects.basic_setting_tab import BasicSettingTabWidget
from src.ui_objects.calibration_tab import CalibrationTabWidget

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
    basic_tab_widget = BasicSettingTabWidget()
    form.add_tab(basic_tab_widget, "Control Tab")

    calibration_tab_widget = CalibrationTabWidget()
    form.add_tab(calibration_tab_widget, "Calibration Tab")

    sys.exit(app.exec_())
