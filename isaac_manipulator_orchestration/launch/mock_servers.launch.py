#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch file for the mock perception and motion servers."""
    return LaunchDescription([
        Node(
            package='isaac_manipulator_orchestration',
            executable='robot_tf_broadcaster',
            name='robot_tf_broadcaster',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='pose_est_server',
            name='pose_est_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='object_detection_server',
            name='object_detection_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='object_selector_server',
            name='object_selector_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='add_mesh_server',
            name='add_mesh_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='gripper_command_server',
            name='gripper_command_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='attach_object_server',
            name='attach_object_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='assign_name_server',
            name='assign_name_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='motion_plan_server',
            name='motion_plan_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='execute_trajectory_server',
            name='execute_trajectory_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='controller_manager_server',
            name='controller_manager_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_manipulator_orchestration',
            executable='publish_static_planning_scene_server',
            name='publish_static_planning_scene_server',
            output='screen',
            emulate_tty=True,
        ),
    ])
