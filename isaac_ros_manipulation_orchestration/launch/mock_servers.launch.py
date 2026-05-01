#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch file for the mock perception and motion servers."""
    return LaunchDescription([
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='robot_tf_broadcaster',
            name='robot_tf_broadcaster',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='pose_est_server',
            name='pose_est_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='object_detection_server',
            name='object_detection_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='object_selector_server',
            name='object_selector_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='add_mesh_server',
            name='add_mesh_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='gripper_command_server',
            name='gripper_command_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='attach_object_server',
            name='attach_object_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='assign_name_server',
            name='assign_name_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='motion_plan_server',
            name='motion_plan_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='execute_trajectory_server',
            name='execute_trajectory_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='controller_manager_server',
            name='controller_manager_server',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='isaac_ros_manipulation_orchestration',
            executable='publish_static_planning_scene_server',
            name='publish_static_planning_scene_server',
            output='screen',
            emulate_tty=True,
        ),
    ])
