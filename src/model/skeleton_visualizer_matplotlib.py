"""
This module provides a visualization tool for displaying a 3D skeleton using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rcParams["figure.raise_window"] = False  # Prevent the GUI window from stealing focus

from .keypoint_info import halpe26_keypoint_info

def halpe26_to_3d_visualizer_format(joints):
    """
    Convert the joint coordinates from halpe26 format to the format used in skeleton_visualizer_matplotlib.py.

    Parameters:
    joints (np.ndarray): An array of joint coordinates in halpe26 format.

    Returns:
    np.ndarray: An array of joint coordinates formatted for skeleton_visualizer_matplotlib.py.
    """
    converted_joints = np.zeros((len(halpe26_keypoint_info["keypoints"]), 3))
    for i, _ in halpe26_keypoint_info["keypoints"].items():
        converted_joints[i] = joints[i]
    return converted_joints

def normalize_joints(joints):
    """
    Normalize the joint coordinates to a range of [0, 1].

    Parameters:
    joints (np.ndarray): An array of joint coordinates.

    Returns:
    np.ndarray: An array of normalized joint coordinates.
    """
    min_vals = np.min(joints, axis=0)
    max_vals = np.max(joints, axis=0)
    normalized_joints = (joints - min_vals) / (max_vals - min_vals)
    return normalized_joints

def switch_axes(joints):
    """
    Switch the axes of the joint coordinates by multiplying y by -1 and z by -1,
    while x remains the same.

    Parameters:
    joints (np.ndarray): An array of joint coordinates.

    Returns:
    np.ndarray: An array of joint coordinates with switched axes.
    """
    switched_joints = joints.copy()
    switched_joints[:, 1] = -joints[:, 1]  # y to -y
    switched_joints[:, 2] = -joints[:, 2]  # z to -z
    return switched_joints

class SkeletonVisualizer:
    """
    A class to visualize a 3D skeleton using Matplotlib.
    """

    def __init__(self):
        """
        Initialize the SkeletonVisualizer with predefined skeleton links and colors.
        """
        # Define skeleton links
        self.skeleton_links = {
            "main_body": [(0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10)],  # Main body
            "left_limbs": [(4, 5), (5, 6), (8, 11), (11, 12), (12, 13)],  # Left limbs
            "right_limbs": [(1, 2), (2, 3), (8, 14), (14, 15), (15, 16)]  # Right limbs
        }

        # Define colors
        self.colors = {
            "main_body": "yellow",
            "left_limbs": "blue",
            "right_limbs": "red"
        }

        # Initialize Matplotlib figure and 3D axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def set_camera_intrinsics(self, width, height, fx, fy, cx, cy):
        """
        Set the camera intrinsics for the visualizer (not applicable in Matplotlib).
        """
        pass  # Matplotlib does not use camera intrinsics

    def open_window(self):
        """
        Open the visualization window and set the background color.
        """
        self.ax.set_facecolor((0.1, 0.2, 0.4))  # Set background color

    def create_skeleton(self, joints):
        """
        Create a skeleton representation for visualization.

        Parameters:
        joints (np.ndarray): An array of joint coordinates.
        """
        for link_type, link_bones in self.skeleton_links.items():
            for bone in link_bones:
                start, end = joints[bone[0]], joints[bone[1]]
                self.ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=self.colors[link_type]
                )

    def update_skeleton(self, joints):
        """
        Update the skeleton visualization with new joint coordinates.

        Parameters:
        joints (np.ndarray): An array of new joint coordinates.
        """
        self.ax.cla()  # Clear the axis
        switched_joints = switch_axes(joints)
        self.create_skeleton(switched_joints)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.001)  # Use non-blocking pause to avoid stealing focus

    def update_skeleton_halpe26(self, joints):
        """
        Update the skeleton visualization with new joint coordinates in halpe26 format.

        Parameters:
        joints (np.ndarray): An array of new joint coordinates.
        """
        converted_joints = halpe26_to_3d_visualizer_format(joints)
        self.update_skeleton(converted_joints)

    def close_window(self):
        """
        Close the visualization window.
        """
        plt.close(self.fig)

    def save_figure(self, filename):
        """
        Save the current figure to a file.

        Parameters:
        filename (str): The path to save the figure.
        """
        self.fig.savefig(filename)

    def run(self):
        """
        Run the visualization window.
        """
        plt.show()

def main():
    """
    Main function to create and run the SkeletonVisualizer with example data.
    """
    # Example human-like skeleton data (list of joints and list of bones connecting the joints)
    joints = np.array([
        [0, 0, 0],    # Pelvis (root)
        [0.1, -0.2, 0], # Right hip
        [0.1, -0.5, 0], # Right knee
        [0.1, -0.8, 0], # Right foot
        [-0.1, -0.2, 0], # Left hip
        [-0.1, -0.5, 0], # Left knee
        [-0.1, -0.8, 0], # Left foot
        [0, 0.2, 0],    # Spine
        [0, 0.5, 0],    # Thorax
        [0, 0.7, 0],    # Neck base
        [0, 0.9, 0],    # Head
        [-0.2, 0.5, 0], # Left shoulder
        [-0.3, 0.3, 0], # Left elbow
        [-0.4, 0.1, 0], # Left wrist
        [0.2, 0.5, 0],  # Right shoulder
        [0.3, 0.3, 0],  # Right elbow
        [0.4, 0.1, 0]   # Right wrist
    ])

    # Create the visualizer and update the skeleton
    visualizer = SkeletonVisualizer()
    visualizer.open_window()
    visualizer.update_skeleton(joints)

    # Add coordinate axes to the visualizer
    visualizer.ax.quiver(0, 0, 0, 1, 0, 0, color='red', label='X-axis')
    visualizer.ax.quiver(0, 0, 0, 0, 1, 0, color='green', label='Y-axis')
    visualizer.ax.quiver(0, 0, 0, 0, 0, 1, color='blue', label='Z-axis')

    visualizer.run()
    visualizer.close_window()

if __name__ == "__main__":
    main()