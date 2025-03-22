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

        self.camera_param = None

    def set_camera_intrinsics(self, width, height, fx, fy, cx, cy):
        """
        Set the camera intrinsics for the visualizer.

        Parameters:
        width (int): The width of the camera image.
        height (int): The height of the camera image.
        fx (float): The focal length in the x direction.
        fy (float): The focal length in the y direction.
        cx (float): The principal point in the x direction.
        cy (float): The principal point in the y direction.
        """
        self.vis.create_window(window_name = "Skeleton Visualizer",
                               width = width, height = height)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        camera_param = o3d.camera.PinholeCameraParameters()
        camera_param.intrinsic = intrinsic
        camera_param.extrinsic = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.]
        ])

        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_param, allow_arbitrary=True)

        self.camera_param = ctr.convert_to_pinhole_camera_parameters()

    def _lock_camera(self, vis):
        """
        Lock the camera parameters (intrinsic and extrinsic) to prevent changes during animation.
        """
        if self.camera_param is not None:
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.camera_param)  # Restore locked view
        return False

    def open_window(self):
        """
        Open the visualization window and set the background color.
        """
        # Set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.2, 0.4])

        # Register animation callback
        self.vis.register_animation_callback(self._lock_camera)

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
        switched_joints = switch_axes(joints)
        skeleton = self.create_skeleton(switched_joints)
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

    example_3d_skeleton = np.array([
        (57.03156971835551, -595.0352854452603, 1476.6691980747444),
        (96.65673373615542, -803.5955170084982, 1990.3015408556528),
        (38.69015152481671, -706.2527024565902, 1740.5182446426218),
        (115.62106789014622, -748.5432410364158, 1831.0617456060095),
        (-27.258405742199272, -668.8772286960423, 1632.02345369355),
        (154.58558322411523, -544.5787075635226, 1573.0108485181584),
        (-188.0999417934467, -608.9320031043495, 1698.555135301704),
        (230.80798048747712, -310.3555509516973, 1718.1375851154044),
        (-228.49525882233306, -277.0932643140445, 1513.533025526289),
        (308.4996037782877, -46.79709052076081, 1758.806184840437),
        (-259.97706192684416, -12.4524730928275, 1467.8196371252632),
        (147.02207536150073, -4.51888004327426, 1604.2469988279888),
        (-53.14386238571871, -1.7838259889943207, 1612.133350665107),
        (140.45717704694206, 403.32388419065984, 1643.4593486728509),
        (-36.77614621504896, 375.86946807910886, 1488.5953411229825),
        (110.9557113094114, 625.0018743954829, 1625.699577955158),
        (-42.435465492307884, 576.6306197359146, 1487.0997899816143),
        (37.60720392230901, -662.3350847397389, 1776.5450672709003),
        (16.980110473530345, -605.2935781701651, 1596.5246150785447),
        (50.588905911167345, -25.3865965173077, 1618.076796350072),
        (199.68383166601453, 828.6296920583469, 2079.3844296652323),
        (-0.17395024979276438, 659.430011498075, 1639.6602723481315),
        (214.3549306930692, 767.1546293257898, 1929.6118887317305),
        (-53.02788202915577, 743.2910918307446, 1827.0645527210393),
        (85.17602291121894, 615.8369467544037, 1568.041844008314),
        (-53.64923860832503, 590.4554731140408, 1493.8412742908035)
    ])

    # Create the visualizer and update the skeleton
    visualizer = SkeletonVisualizer()
    visualizer.set_camera_intrinsics(1280, 720, 908.36, 908.36, 614.695, 354.577)
    visualizer.open_window()
    visualizer.update_skeleton(joints)

    visualizer.update_skeleton_halpe26(example_3d_skeleton)

    # Add coordinate axes to the visualizer
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    visualizer.vis.add_geometry(axes)

    visualizer.run()
    visualizer.close_window()

if __name__ == "__main__":
    main()
