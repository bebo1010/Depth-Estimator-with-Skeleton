"""
This module contains the BasicSettingTabWidget class for the Depth Estimator with Skeleton project.
The BasicSettingTabWidget class provides a user interface for controlling model parameters, loading configurations,
and displaying control options for the depth estimator application.
"""

import json

from PyQt5 import QtWidgets, QtGui
import yaml

class BasicSettingTabWidget(QtWidgets.QWidget):
    """
    A QWidget class for the Depth Estimator with Skeleton project.
    """

    def __init__(self):
        """
        Initialize the BasicSettingTabWidget.
        """
        super().__init__()

        # Create the main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Initialize UI components
        self._init_model_control(main_layout)
        self._init_parameter_settings(main_layout)
        self._init_display_control(main_layout)
        self._init_chessboard_calibration(main_layout)

        # Initialize parameters
        self._load_parameters()

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
        self.reset_model_button = QtWidgets.QPushButton("Reset Model")
        model_control_layout.addWidget(self.model_toggle)
        model_control_layout.addWidget(self.reset_model_button)
        main_layout.addWidget(model_control_group)

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
        self.load_parameter_button = QtWidgets.QPushButton("Load Parameter")
        self.load_parameter_button.clicked.connect(self._load_parameter)
        parameter_settings_layout.addWidget(self.load_parameter_button)

        # Parameter rows
        self.camera_parameters = {}
        self.parameter_display_dict = {}
        parameter_names = ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y", "Width", "Height"]
        for name in parameter_names:
            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            input_box = QtWidgets.QLineEdit()
            input_box.setValidator(QtGui.QDoubleValidator(0.00, 9999.99, 2))
            row_layout.addWidget(label)
            row_layout.addWidget(input_box)
            parameter_settings_layout.addLayout(row_layout)
            self.camera_parameters[name] = None
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
        display_control_layout.addWidget(self.horizontal_toggle)
        display_control_layout.addWidget(self.vertical_toggle)
        display_control_layout.addWidget(self.epipolar_line_toggle)
        display_control_layout.addWidget(self.freeze_frame_toggle)
        main_layout.addWidget(display_control_group)

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

    def _toggle_calibration(self):
        """
        Toggle the calibration process.
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
        """
        current_count = int(self.saved_images_label.text().split(": ")[1])
        current_count += 1
        self.saved_images_label.setText(f"Saved Images: {current_count}")

    def _load_parameters(self, config: dict = None, stereo_params: dict = None):
        """
        Load the parameters from the configuration and stereo parameters.

        Parameters
        ----------
        config : dict
            The configuration dictionary.
        stereo_params : dict
            The stereo parameters dictionary.

        Returns
        -------
        None
        """
        self._load_width_height_from_config(config)
        self._load_camera_parameters(stereo_params)

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
            self.camera_parameters["Width"] = float(config['camera_settings']['width'])
            self.camera_parameters["Height"] = float(config['camera_settings']['height'])

            for key in ["Width", "Height"]:
                self.parameter_display_dict[key].setText(f"{self.camera_parameters[key]:.2f}")
                self.parameter_display_dict[key].setDisabled(True)
        else:
            for key in ["Width", "Height"]:
                self.parameter_display_dict[key].clear()
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
            self.camera_parameters["Focal Length"] = float((stereo_params['camera_matrix_left'][0][0] +
                                                            stereo_params['camera_matrix_left'][1][1]) / 2)
            self.camera_parameters["Baseline"] = float(abs(stereo_params['translation_vector'][0][0]))
            self.camera_parameters["Principal Point X"] = float(stereo_params['camera_matrix_left'][0][2])
            self.camera_parameters["Principal Point Y"] = float(stereo_params['camera_matrix_left'][1][2])

            for key in ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y"]:
                self.parameter_display_dict[key].setText(f"{self.camera_parameters[key]:.2f}")
                self.parameter_display_dict[key].setDisabled(True)
        else:
            for key in ["Focal Length", "Baseline", "Principal Point X", "Principal Point Y"]:
                self.parameter_display_dict[key].clear()
                self.parameter_display_dict[key].setDisabled(False)

    def _update_parameters(self):
        """
        Update the parameters based on the selected configuration.
        """
        self._load_width_height_from_config()
        self._load_camera_parameters()

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
                self.camera_parameters.update(parameters)
                self._load_parameters(stereo_params=parameters)

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
