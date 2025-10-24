#!/usr/bin/env python3
"""
Utility functions and classes for motion operations.

This module provides functionality for:
- Creating object attachment configurations
"""

from typing import Any, Dict, List, Optional

from geometry_msgs.msg import Pose, Vector3
from visualization_msgs.msg import Marker


def create_object_attachment_config(
    gripper_pose_object: Dict[str, Any],
    shape: str = 'CUBOID',
    scale: Optional[List[float]] = None,
    mesh_file_path: str = '',
    frame_id: str = 'grasp_frame'
) -> Marker:
    """
    Create an object attachment configuration as a visualization marker.

    Args:
    ----
    gripper_pose_object
        The pose of the gripper relative to object
        (dict with 'position' and 'orientation')
    shape
        The shape of the object ('SPHERE', 'CUBOID', or 'CUSTOM_MESH')
    scale
        List containing [x, y, z] scale values
    mesh_file_path
        Path to the mesh file if shape is 'CUSTOM_MESH'
    frame_id
        The frame ID for the marker

    Returns
    -------
    Marker
        A marker object representing the attachment configuration

    """
    # Create a marker for object attachment
    marker = Marker()

    # Create the pose for the marker
    pose = Pose()
    pose.position.x = gripper_pose_object['position'][0]
    pose.position.y = gripper_pose_object['position'][1]
    pose.position.z = gripper_pose_object['position'][2]
    pose.orientation.x = gripper_pose_object['orientation'][0]
    pose.orientation.y = gripper_pose_object['orientation'][1]
    pose.orientation.z = gripper_pose_object['orientation'][2]
    pose.orientation.w = gripper_pose_object['orientation'][3]
    marker.pose = pose

    # Set the scale
    marker_scale = Vector3()
    if scale and len(scale) == 3:
        marker_scale.x = scale[0]
        marker_scale.y = scale[1]
        marker_scale.z = scale[2]
    else:
        marker_scale.x = 0.05
        marker_scale.y = 0.05
        marker_scale.z = 0.1
    marker.scale = marker_scale

    # Set the marker type based on the shape
    if shape == 'SPHERE':
        marker.type = Marker.SPHERE
    elif shape == 'CUBOID':
        marker.type = Marker.CUBE
    elif shape == 'CUSTOM_MESH':
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = mesh_file_path
    else:
        print(f'Unknown object shape: {shape}, defaulting to CUBE')
        marker.type = Marker.CUBE

    # Set other marker properties
    marker.header.frame_id = frame_id
    marker.frame_locked = True
    marker.color.r = 1.0
    marker.color.a = 1.0

    return marker
