"""
Unit tests for the display_utils module.

This module contains unit tests for the functions defined in the display_utils module.
"""
import logging
import unittest

import numpy as np
import cv2

from src.utils.display_utils import draw_lines, apply_colormap

class TestDisplayUtils(unittest.TestCase):
    """
    Test cases for the display_utils module.
    """

    def setUp(self):
        """
        Set up the test case environment.
        """
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level


    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_draw_lines_horizontal(self):
        """
        Test drawing horizontal lines on an image.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_lines(image, 10, 'horizontal')
        for i in range(0, 100, 10):
            self.assertTrue(np.all(image[i, :, :] == [0, 0, 255]))

    def test_draw_lines_vertical(self):
        """
        Test drawing vertical lines on an image.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_lines(image, 10, 'vertical')
        for i in range(0, 100, 10):
            self.assertTrue(np.all(image[:, i, :] == [0, 0, 255]))

    def test_apply_colormap_with_depth_image(self):
        """
        Test applying a colormap to a depth image.
        """
        depth_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        reference_image = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_image = apply_colormap(depth_image, reference_image)
        self.assertEqual(colored_image.shape, reference_image.shape)
        self.assertFalse(np.array_equal(colored_image, reference_image))

    def test_apply_colormap_without_depth_image(self):
        """
        Test applying a colormap when no depth image is provided.
        """
        reference_image = np.zeros((100, 100, 3), dtype=np.uint8)
        colored_image = apply_colormap(None, reference_image)
        self.assertTrue(np.array_equal(colored_image, reference_image))

if __name__ == '__main__':
    unittest.main()
