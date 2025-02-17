"""
Unit tests for the RealsenseCameraSystem class.
"""
import logging

from unittest.mock import MagicMock, patch
import unittest
import coverage

import numpy as np
from src.camera_objects import RealsenseCameraSystem

class TestRealsenseCameraSystem(unittest.TestCase):
    """
    Test suite for the RealsenseCameraSystem class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

        # Patch rs.pipeline
        patcher = patch('src.camera_objects.two_cameras.realsense_camera_system.rs.pipeline')
        self.addCleanup(patcher.stop)
        self.mock_pipeline = patcher.start().return_value

        self.mock_pipeline.wait_for_frames = MagicMock()
        self.camera_system = RealsenseCameraSystem(640, 480)

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_get_grayscale_images_success(self):
        """
        Test successful retrieval of grayscale images.
        """
        mock_frames = MagicMock()
        mock_ir_frame_left = MagicMock()
        mock_ir_frame_right = MagicMock()
        mock_ir_frame_left.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_ir_frame_right.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_frames.get_infrared_frame.side_effect = [mock_ir_frame_left, mock_ir_frame_right]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertTrue(success)
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(right_image)

    def test_get_grayscale_images_failure_left(self):
        """
        Test failure to retrieve left grayscale image.
        """
        mock_frames = MagicMock()
        mock_ir_frame_right = MagicMock()
        mock_ir_frame_right.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_frames.get_infrared_frame.side_effect = [None, mock_ir_frame_right]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertFalse(success)
        self.assertIsNone(left_image)
        self.assertIsNotNone(right_image)

    def test_get_grayscale_images_failure_right(self):
        """
        Test failure to retrieve right grayscale image.
        """
        mock_frames = MagicMock()
        mock_ir_frame_left = MagicMock()
        mock_ir_frame_left.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_frames.get_infrared_frame.side_effect = [mock_ir_frame_left, None]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertFalse(success)
        self.assertIsNotNone(left_image)
        self.assertIsNone(right_image)

    def test_get_grayscale_images_failure(self):
        """
        Test failure to retrieve grayscale images.
        """
        mock_frames = MagicMock()
        mock_frames.get_infrared_frame.side_effect = [None, None]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertFalse(success)
        self.assertIsNone(left_image)
        self.assertIsNone(right_image)

    def test_get_depth_image_success(self):
        """
        Test successful retrieval of depth image.
        """
        mock_frames = MagicMock()
        mock_depth_frame = MagicMock()
        mock_depth_frame.get_data.return_value = np.zeros((480, 640), dtype=np.uint16)
        mock_frames.get_depth_frame.return_value = mock_depth_frame
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, depth_image, _ = self.camera_system.get_depth_images()
        self.assertTrue(success)
        self.assertIsNotNone(depth_image)

    def test_get_depth_image_failure(self):
        """
        Test failure to retrieve depth image.
        """
        mock_frames = MagicMock()
        mock_frames.get_depth_frame.return_value = None
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, depth_image, _ = self.camera_system.get_depth_images()
        self.assertFalse(success)
        self.assertIsNone(depth_image)

    def test_get_width(self):
        """
        Test retrieval of camera width.
        """
        self.assertEqual(self.camera_system.get_width(), 640)

    def test_get_height(self):
        """
        Test retrieval of camera height.
        """
        self.assertEqual(self.camera_system.get_height(), 480)

    def test_release(self):
        """
        Test releasing the camera system.
        """
        self.assertTrue(self.camera_system.release())
        self.mock_pipeline.stop.assert_called_once()

    @patch('src.camera_objects.two_cameras.realsense_camera_system.rs.config')
    def test_init_with_serial_number(self, mock_config):
        """
        Test initialization with a specific serial number.
        """
        serial_number = "123456789"
        mock_config_instance = mock_config.return_value
        RealsenseCameraSystem(640, 480, serial_number)
        mock_config_instance.enable_device.assert_called_once_with(serial_number)

if __name__ == '__main__':
    cov = coverage.Coverage()
    cov.start()

    unittest.main()

    cov.stop()
    cov.save()

    cov.html_report()
    print("Done.")
