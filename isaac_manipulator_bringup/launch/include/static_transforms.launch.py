# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    camera_type = str(context.perform_substitution(LaunchConfiguration('camera_type')))

    world_pose_hawk_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '-0.95662777', '--y', '0.62544195', '--z', '0.58333933',
                   '--qx', '0.07346877', '--qy', '0.13587378', '--qz', '-0.56931322', '--qw', '0.80747948',
                   '--frame-id', 'world', '--child-frame-id', 'camera'],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"hawk"'])),
    )

    world_pose_realsense_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '-0.70078', '--y', '0.593563', '--z', '0.994128',
                   '--qx', '0.261831', '--qy', '0.215316', '--qz', '-0.622901', '--qw', '0.705037',
                   '--frame-id', 'world', '--child-frame-id', 'camera_1_link'],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"realsense"'])),
    )

    world_pose_target1_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '-0.7', '--y', '0.3', '--z', '0.4',
                   '--qx', '1.0', '--qy', '0.0', '--qz', '0.0', '--qw', '0.0',
                   '--frame-id', 'world', '--child-frame-id', 'target1_frame'],
    )

    world_pose_target2_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '-0.7', '--y', '-0.3', '--z', '0.4',
                   '--qx', '1.0', '--qy', '0.0', '--qz', '0.0', '--qw', '0.0',
                   '--frame-id', 'world', '--child-frame-id', 'target2_frame'],
    )

    return [
        world_pose_realsense_camera, world_pose_hawk_camera,
        world_pose_target1_frame, world_pose_target2_frame,
    ]


def generate_launch_description():
    return (LaunchDescription([OpaqueFunction(function=launch_setup)]))
