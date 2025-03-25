"""
This module contains the BasicSettingTabWidget class for the Depth Estimator with Skeleton project.
The BasicSettingTabWidget class provides a user interface for controlling model parameters, loading configurations,
and displaying control options for the depth estimator application.
"""

import os
import json

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal
import yaml

from src.utils import get_starting_index, save_setup_info

from .abstract_tab import AbstractTabWidget, CommonMeta

class BasicSettingTabWidget(QtWidgets.QWidget, AbstractTabWidget, metaclass=CommonMeta):
    """
    A tab widget for controlling model parameters, loading configurations, and displaying control options for the
    depth estimator application.
    """
    toggle_signal = pyqtSignal(str, bool)

    def __init__(self, base_dir: str):
        """
        Initialize the BasicSettingTabWidget.
        """
        super().__init__()

        self.base_dir: str = base_dir

        # Create the main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Initialize UI components
        self._init_model_control(main_layout)
        self._init_parameter_settings(main_layout)
        self._init_display_control(main_layout)

        # Initialize parameters
        self._update_parameters()

        # Initialize variables
        self._init_variables()

    @property
    def width(self) -> int:
        """
        Get the width of the camera image.

        Returns
        -------
        float
            The width of the camera image.
        """
        return self._camera_params["Width"]

    @width.setter
    def width(self, new_width: int):
        """
        Set the width of the camera image.

        Parameters
        ----------
        value : int
            The width of the camera image.
        """
        self._camera_params["Width"] = new_width
        self.parameter_display_dict["Width"].setText(f"{new_width:.2f}")

    @property
    def height(self) -> int:
        """
        Get the height of the camera image.

        Returns
        -------
        float
            The height of the camera image.
        """
        return self._camera_params["Height"]

    @height.setter
    def height(self, new_height: int):
        """
        Set the height of the camera image.

        Parameters
        ----------
        value : int
            The height of the camera image.
        """
        self._camera_params["Height"] = new_height
        self.parameter_display_dict["Height"].setText(f"{new_height:.2f}")

    def _init_model_control(self, main_layout: QtWidgets.QLayout):
        """
        Initialize the Model Control GroupBox.

        Parameters
        ----------
        main_layout : QtWidgets.QLayout
            The main layout of the main window.

        Returns
        -------
        None
        """
        model_control_group = QtWidgets.QGroupBox("Model Control")
        model_control_layout = QtWidgets.QVBoxLayout(model_control_group)
        self.model_toggle = QtWidgets.QCheckBox("Model")
        self.select_person_toggle = QtWidgets.QCheckBox("Select Person")  # Added checkbox
        self.reset_model_button = QtWidgets.QPushButton("Reset Model")
        self.stream_button = QtWidgets.QPushButton("Start Stream")  # Added button

        # Connect toggles to signal
        self.model_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Model", checked))
        self.select_person_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Select Person", checked))
        self.stream_button.clicked.connect(self._toggle_stream)

        model_control_layout.addWidget(self.model_toggle)
        model_control_layout.addWidget(self.select_person_toggle)  # Added checkbox to layout
        model_control_layout.addWidget(self.reset_model_button)
        model_control_layout.addWidget(self.stream_button)  # Added button to layout
        main_layout.addWidget(model_control_group)

    def _toggle_stream(self):
        """
        Toggle the stream button text between 'Start Stream' and 'Stop Stream'.
        """
        if self.stream_button.text() == "Start Stream":
            self.stream_button.setText("Stop Stream")
            self.toggle_signal.emit("Stream", True)
        else:
            self.stream_button.setText("Start Stream")
            self.toggle_signal.emit("Stream", False)

    def _init_parameter_settings(self, main_layout: QtWidgets.QLayout):
        """
        Initialize the Parameter Settings GroupBox.

        Parameters
        ----------
        main_layout : QtWidgets.QLayout
            The main layout of the main window.

        Returns
        -------
        None
        """
        parameter_settings_group = QtWidgets.QGroupBox("Parameter Settings")
        parameter_settings_layout = QtWidgets.QVBoxLayout(parameter_settings_group)
        self.parameter_dropdown = QtWidgets.QComboBox()
        self.parameter_dropdown.addItems(["GH3", "ORYX", "Custom"])
        self.parameter_dropdown.currentIndexChanged.connect(self._update_parameters)
        parameter_settings_layout.addWidget(self.parameter_dropdown)

        # Load Parameter Button
        self.load_parameter_button = QtWidgets.QPushButton("Load Parameters")
        self.load_parameter_button.clicked.connect(self._load_parameter)
        parameter_settings_layout.addWidget(self.load_parameter_button)

        # Parameter rows
        self._camera_params = {"System Prefix": None}
        self.parameter_display_dict = {}
        parameter_names = ["Focal Length", "Baseline", "Width", "Height", "Principal Point X", "Principal Point Y"]
        for name in parameter_names:
            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            input_box = QtWidgets.QLineEdit()
            input_box.setValidator(QtGui.QDoubleValidator(0.00, 9999.99, 2))
            row_layout.addWidget(label)
            row_layout.addWidget(input_box)
            parameter_settings_layout.addLayout(row_layout)
            self._camera_params[name] = None
            self.parameter_display_dict[name] = input_box

        main_layout.addWidget(parameter_settings_group)

    def _init_display_control(self, main_layout: QtWidgets.QLayout):
        """
        Initialize the Display Control GroupBox.

        Parameters
        ----------
        main_layout : QtWidgets.QLayout
            The main layout of the main window.

        Returns
        -------
        None
        """
        display_control_group = QtWidgets.QGroupBox("Display Control")
        display_control_layout = QtWidgets.QVBoxLayout(display_control_group)
        self.horizontal_toggle = QtWidgets.QCheckBox("Horizontal")
        self.vertical_toggle = QtWidgets.QCheckBox("Vertical")
        self.epipolar_line_toggle = QtWidgets.QCheckBox("Epipolar Line")
        self.freeze_frame_toggle = QtWidgets.QCheckBox("Freeze Frame")

        # Connect toggles to signal
        self.horizontal_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Horizontal", checked))
        self.vertical_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Vertical", checked))
        self.epipolar_line_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Epipolar Line", checked))
        self.freeze_frame_toggle.toggled.connect(lambda checked: self.toggle_signal.emit("Freeze Frame", checked))

        display_control_layout.addWidget(self.horizontal_toggle)
        display_control_layout.addWidget(self.vertical_toggle)
        display_control_layout.addWidget(self.epipolar_line_toggle)
        display_control_layout.addWidget(self.freeze_frame_toggle)
        main_layout.addWidget(display_control_group)

    def _init_variables(self):
        left_skeleton_dir = os.path.join(self.base_dir, "left_skeleton_images")
        self.image_index = get_starting_index(left_skeleton_dir)

    def _load_width_height_from_config(self, config: dict = None):
        """
        Load the width and height from the configuration.

        Parameters
        ----------
        config : dict
            The configuration dictionary.

        Returns
        -------
        None
        """
        if config is not None:
            pass
        elif self.parameter_dropdown.currentText() == "GH3":
            config = self._load_config('src/camera_config/GH3_camera_config.yaml')
        elif self.parameter_dropdown.currentText() == "ORYX":
            config = self._load_config('src/camera_config/ORYX_camera_config.yaml')

        if config:
            self._camera_params["Width"] = float(config['camera_settings']['width'])
            self._camera_params["Height"] = float(config['camera_settings']['height'])

            for key in ["Width", "Height"]:
                self.parameter_display_dict[key].setText(f"{self._camera_params[key]:.2f}")
                self.parameter_display_dict[key].setDisabled(True)
        else:
            for key in ["Width", "Height"]:
                self.parameter_display_dict[key].clear()
                self._camera_params[key] = None
                self.parameter_display_dict[key].setDisabled(False)

    def _load_camera_parameters(self, stereo_params: dict = None):
        """
        Load the camera parameters from the stereo parameters.

        Parameters
        ----------
        stereo_params : dict
            The stereo parameters dictionary.

        Returns
        -------
        None
        """
        if stereo_params is not None:
            pass
        elif self.parameter_dropdown.currentText() == "GH3":
            stereo_params = self._load_stereo_parameters(
                './Db/GH3_calibration_parameter/stereo_camera_parameters.json'
                )
        elif self.parameter_dropdown.currentText() == "ORYX":
            stereo_params = self._load_stereo_parameters(
                './Db/ORYX_calibration_parameter/stereo_camera_parameters.json'
                )

        if stereo_params:
            self._camera_params["Focal Length"] = float((stereo_params['camera_matrix_left'][0][0] +
                                                            stereo_params['camera_matrix_left'][1][1]) / 2)
            self._camera_params["Baseline"] = float(abs(stereo_params['translation_vector'][0][0]))
            self._camera_params["Principal Point X"] = float(stereo_params['camera_matrix_left'][0][2])
            self._camera_params["Principal Point Y"] = float(stereo_params['camera_matrix_left'][1][2])

            for key in ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y"]:
                self.parameter_display_dict[key].setText(f"{self._camera_params[key]:.2f}")
                self.parameter_display_dict[key].setDisabled(True)
        else:
            for key in ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y"]:
                self.parameter_display_dict[key].clear()
                self._camera_params[key] = None
                self.parameter_display_dict[key].setDisabled(False)

    def _update_parameters(self, _index: int = 0, config: dict = None, stereo_params: dict = None):
        """
        Update the parameters from the configuration and stereo parameters.

        Parameters
        ----------
        _index : int
            The index of the parameter dropdown.
        config : dict
            The configuration dictionary.
        stereo_params : dict
            The stereo parameters dictionary.

        Returns
        -------
        None
        """
        self._camera_params["System Prefix"] = self.parameter_dropdown.currentText()
        self._load_width_height_from_config(config)
        self._load_camera_parameters(stereo_params)

        # Save current setup info
        save_setup_info(self.base_dir, self._camera_params)

    def _load_parameter(self):
        """
        Load parameters from a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             "Load Parameter File", "",
                                                             "JSON Files (*.json);;All Files (*)",
                                                             options=options)
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as file:
                parameters = json.load(file)
                self._camera_params.update(parameters)
                self._update_parameters(stereo_params=parameters)

    def _load_config(self, filepath: str) -> dict:
        """
        Load the configuration from a YAML file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return None

    def _load_stereo_parameters(self, filepath: str) -> dict:
        """
        Load the stereo parameters from a JSON file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            return None

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tab = BasicSettingTabWidget()
    tab.show()
    sys.exit(app.exec_())
