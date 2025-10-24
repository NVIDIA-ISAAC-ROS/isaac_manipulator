# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Launch file for object detection testing."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for object detection testing."""
    return LaunchDescription([
        Node(
            package='isaac_manipulator_orchestration',
            executable='object_detection_server',
            name='object_detection_server',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ]
        )
    ])
