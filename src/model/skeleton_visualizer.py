"""
This module provides a visualization tool for displaying a 3D skeleton using Open3D.
"""

import open3d as o3d
import numpy as np

from .keypoint_info import halpe26_keypoint_info

def halpe26_to_3d_visualizer_format(joints):
    """
    Convert the joint coordinates from halpe26 format to the format used in skeleton_visualizer.py.

    Parameters:
    joints (np.ndarray): An array of joint coordinates in halpe26 format.

    Returns:
    np.ndarray: An array of joint coordinates formatted for skeleton_visualizer.py.
    """
    converted_joints = np.zeros((len(halpe26_keypoint_info["keypoints"]), 3))
    for i, _ in halpe26_keypoint_info["keypoints"].items():
        converted_joints[i] = joints[i]
    return converted_joints

class SkeletonVisualizer:
    """
    A class to visualize a 3D skeleton using Open3D.
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
            "main_body": [1.0, 1.0, 0.0],  # Yellow
            "left_limbs": [0.0, 0.0, 1.0],  # Blue
            "right_limbs": [1.0, 0.0, 0.0]  # Red
        }

        # Create a visualizer
        self.vis = o3d.visualization.Visualizer()

    def open_window(self):
        """
        Open the visualization window and set the background color.
        """
        self.vis.create_window()
        # Set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.2, 0.4])

    def create_skeleton(self, joints):
        """
        Create a LineSet object representing the skeleton.

        Parameters:
        joints (np.ndarray): An array of joint coordinates.

        Returns:
        o3d.geometry.LineSet: The LineSet object representing the skeleton.
        """
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(joints)

        bones = []
        colors = []

        for link_type, link_bones in self.skeleton_links.items():
            for bone in link_bones:
                bones.append(bone)
                colors.append(self.colors[link_type])

        lines.lines = o3d.utility.Vector2iVector(bones)
        lines.colors = o3d.utility.Vector3dVector(colors)
        return lines

    def update_skeleton(self, joints):
        """
        Update the skeleton visualization with new joint coordinates.

        Parameters:
        joints (np.ndarray): An array of new joint coordinates.
        """
        skeleton = self.create_skeleton(joints)
        self.vis.clear_geometries()
        self.vis.add_geometry(skeleton)
        self.vis.poll_events()
        self.vis.update_renderer()

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
        self.vis.destroy_window()

    def run(self):
        """
        Run the visualization window.
        """
        self.vis.run()

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

    # example for updating skeleton
    # new_joints = joints + np.random.uniform(-0.1, 0.1, joints.shape)
    # visualizer.update_skeleton(new_joints)

    visualizer.run()
    visualizer.close_window()

if __name__ == "__main__":
    main()
