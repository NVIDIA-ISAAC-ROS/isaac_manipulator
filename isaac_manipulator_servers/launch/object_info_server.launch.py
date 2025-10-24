# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import isaac_manipulator_ros_python_utils.constants as constants
from isaac_ros_launch_utils.all_types import (
    GroupAction, LoadComposableNodes
)

import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file to bring up isaac manipulator object info server."""
    launch_args = [
        DeclareLaunchArgument(
            'standalone',
            default_value='false',
            choices=['true', 'false'],
            description='Whether to use this launch file to start the object info server in \
                standalone mode (good for debugging) or load into existing container.',
        ),
        DeclareLaunchArgument(
            'object_detection_backend',
            default_value='RT_DETR',
            description='Object detection backend for the Object Info Server',
        ),
        DeclareLaunchArgument(
            'pose_estimation_backend',
            default_value='FOUNDATION_POSE',
            description='Pose estimation backend for the Object Info Server',
        ),
        DeclareLaunchArgument(
            'segmentation_backend',
            default_value='SEGMENT_ANYTHING',
            description='Segmentation backend for the Object Info Server',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for the container',
        ),
    ]

    object_info_node = ComposableNode(
        name='object_info_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectInfoServer',
        parameters=[{
                'object_detection_backend': LaunchConfiguration('object_detection_backend'),
                'pose_estimation_backend': LaunchConfiguration('pose_estimation_backend'),
                'segmentation_backend': LaunchConfiguration('segmentation_backend'),
            }]
    )

    # When standalone is false, load into existing container
    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[object_info_node],
        condition=UnlessCondition(LaunchConfiguration('standalone'))
    )

    # When standalone is false, load into existing container
    original_launch = GroupAction(
        actions=[load_composable_nodes],
        condition=UnlessCondition(LaunchConfiguration('standalone'))
    )

    # When standalone is true, create standalone container for object info server
    object_info_container = ComposableNodeContainer(
        name='object_info_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[object_info_node],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[],
        output='screen',
        condition=IfCondition(LaunchConfiguration('standalone'))
    )

    return launch.LaunchDescription(launch_args + [original_launch, object_info_container])
