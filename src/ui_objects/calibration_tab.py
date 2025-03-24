"""
This module contains the CalibrationTabWidget class for the Depth Estimator with Skeleton project.
The CalibrationTabWidget class provides a user interface for controlling the chessboard calibration process.
"""
from typing import Tuple

from PyQt5 import QtWidgets

from src.opencv_objects import ChessboardCalibrator

class CalibrationTabWidget(QtWidgets.QWidget):
    """
    A QWidget class for the Depth Estimator with Skeleton project.
    """

    def __init__(self):
        """
        Initialize the CalibrationTabWidget.
        """
        super().__init__()

        # Create the main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Initialize UI components
        self._init_chessboard_calibration(main_layout)
        self._init_parameter_display(main_layout)  # Add parameter settings initialization

        # Initialize variables
        self._init_variables()

    def _init_chessboard_calibration(self, main_layout: QtWidgets.QLayout):
        """
        Initialize the Chessboard Calibration GroupBox.

        Parameters
        ----------
        main_layout : QtWidgets.QLayout
            The main layout of the main window.

        Returns
        -------
        None
        """
        chessboard_calibration_group = QtWidgets.QGroupBox("Chessboard Calibration")
        chessboard_calibration_layout = QtWidgets.QVBoxLayout(chessboard_calibration_group)

        self.start_calibration_button = QtWidgets.QPushButton("Start Calibration")
        self.start_calibration_button.clicked.connect(self._toggle_calibration)

        self.save_images_button = QtWidgets.QPushButton("Save Images")
        self.save_images_button.setDisabled(True)
        self.save_images_button.clicked.connect(self._save_image)
        self.saved_images_label = QtWidgets.QLabel("Saved Images: 0")

        chessboard_calibration_layout.addWidget(self.start_calibration_button)
        chessboard_calibration_layout.addWidget(self.save_images_button)
        chessboard_calibration_layout.addWidget(self.saved_images_label)
        main_layout.addWidget(chessboard_calibration_group)

    def _init_parameter_display(self, main_layout: QtWidgets.QLayout):
        """
        Initialize the Parameter Display GroupBox.

        Parameters
        ----------
        main_layout : QtWidgets.QLayout
            The main layout of the main window.

        Returns
        -------
        None
        """
        parameter_display_group = QtWidgets.QGroupBox("Parameter Display")
        parameter_display_layout = QtWidgets.QVBoxLayout(parameter_display_group)

        # Text input box for parameters
        def set_system_prefix(text):
            self._camera_params['system_prefix'] = text
            print("Updated system_prefix:", self._camera_params['system_prefix'])

        self.parameter_input = QtWidgets.QLineEdit()
        self.parameter_input.setPlaceholderText("Enter setup prefix name")
        self.parameter_input.textChanged.connect(set_system_prefix)
        parameter_display_layout.addWidget(self.parameter_input)

        # Parameter rows
        self.camera_parameters = {}
        self.parameter_display_dict = {}
        parameter_names = ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y", "Width", "Height"]
        for name in parameter_names:
            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            input_box = QtWidgets.QLineEdit()
            input_box.setDisabled(True)  # Default to disabled
            row_layout.addWidget(label)
            row_layout.addWidget(input_box)
            parameter_display_layout.addLayout(row_layout)
            self.camera_parameters[name] = None
            self.parameter_display_dict[name] = input_box

        # Add Save Parameter Button
        self.save_parameter_button = QtWidgets.QPushButton("Save Parameters")
        parameter_display_layout.addWidget(self.save_parameter_button)

        main_layout.addWidget(parameter_display_group)

    def _init_variables(self):
        """
        Initialize the variables for the CalibrationTabWidget.

        Returns
        -------
        None
        """
        self._camera_params = {
            'system_prefix': None,
            'focal_length': None,
            'baseline': None,
            'principal_point': None,
            'width': None,
            'height': None
        }

        pattern_size = (10, 7)
        square_size_mm = 10

        self.chessboard_calibrator = ChessboardCalibrator()
        self.chessboard_calibrator.pattern_size = pattern_size
        self.chessboard_calibrator.square_size_mm = square_size_mm

        self.image_points = {'left': [], 'right': []}

    def _toggle_calibration(self):
        """
        Toggle the calibration process.

        Returns
        -------
        None
        """
        if self.start_calibration_button.text() == "Start Calibration":
            self.start_calibration_button.setText("Stop Calibration")
            self.save_images_button.setDisabled(False)
        else:
            self.start_calibration_button.setText("Start Calibration")
            self.save_images_button.setDisabled(True)

    def _save_image(self):
        """
        Save the current image and update the saved images count.

        Returns
        -------
        None
        """
        current_count = int(self.saved_images_label.text().split(": ")[1])
        current_count += 1
        self.saved_images_label.setText(f"Saved Images: {current_count}")

    @property
    def camera_params(self) -> dict:
        """
        Get the camera parameters.

        Returns
        -------
        dict
            The camera parameters.
        """
        return self._camera_params

    @property
    def system_prefix(self) -> str:
        """
        Get the system prefix.

        Returns
        -------
        str
            The system prefix.
        """
        return self._camera_params['system_prefix']

    @property
    def focal_length(self) -> float:
        """
        Get the focal length.

        Returns
        -------
        float
            The focal length.
        """
        return self._camera_params['focal_length']

    @property
    def baseline(self) -> int:
        """
        Get the baseline.

        Returns
        -------
        int
            The baseline.
        """
        return self._camera_params['baseline']

    @baseline.setter
    def baseline(self, value: int):
        """
        Set the baseline.

        Parameters
        ----------
        value : int
            The new baseline.

        Returns
        -------
        None
        """
        self._camera_params['baseline'] = value

    @property
    def principal_point(self) -> Tuple[int, int]:
        """
        Get the principal point.

        Returns
        -------
        Tuple[int, int]
            The principal point.
        """
        return self._camera_params['principal_point']

    @property
    def width(self) -> int:
        """
        Get the width.

        Returns
        -------
        int
            The width.
        """
        return self._camera_params['width']

    @width.setter
    def width(self, value: int):
        """
        Set the width.

        Parameters
        ----------
        value : int
            The new width.

        Returns
        -------
        None
        """
        self._camera_params['width'] = value

    @property
    def height(self) -> int:
        """
        Get the height.

        Returns
        -------
        int
            The height.
        """
        return self._camera_params['height']

    @height.setter
    def height(self, value: int):
        """
        Set the height.

        Parameters
        ----------
        value : int
            The new height.

        Returns
        -------
        None
        """
        self._camera_params['height'] = value

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """
        Get the pattern size.

        Returns
        -------
        Tuple[int, int]
            The pattern size.
        """
        return self.chessboard_calibrator.pattern_size

    @pattern_size.setter
    def pattern_size(self, value: Tuple[int, int]):
        """
        Set the pattern size.

        Parameters
        ----------
        value : Tuple[int, int]
            The new pattern size.

        Returns
        -------
        None
        """
        self.chessboard_calibrator.pattern_size = value

    @property
    def square_size_mm(self) -> float:
        """
        Get the square size in millimeters.

        Returns
        -------
        float
            The square size in millimeters.
        """
        return self.chessboard_calibrator.square_size_mm

    @square_size_mm.setter
    def square_size_mm(self, value: float):
        """
        Set the square size in millimeters.

        Parameters
        ----------
        value : float
            The new square size in millimeters.

        Returns
        -------
        None
        """
        self.chessboard_calibrator.square_size_mm = value

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tab = CalibrationTabWidget()
    tab.show()
    sys.exit(app.exec_())
