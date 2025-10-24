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

import os

from ament_index_python.packages import get_package_share_directory

import isaac_manipulator_ros_python_utils.constants as constants

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):

    realsense_config = os.path.join(
        get_package_share_directory(
            'isaac_manipulator_bringup'), 'config', 'sensors', 'realsense_calibration.yaml'
    )

    realsense_node = ComposableNode(
        namespace='',
        name='camera_1',
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config]
    )

    calibration_container = ComposableNodeContainer(
        name=constants.MANIPULATOR_CONTAINER_NAME,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', 'error'],
        composable_node_descriptions=[
            realsense_node,
        ],
    )

    return [
        calibration_container,
    ]


def generate_launch_description():
    LaunchConfiguration('camera_type')
    launch_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='REALSENSE',
        choices=['REALSENSE'],
        description='Camera sensor used for calibration'
    )

    return LaunchDescription([launch_arg, OpaqueFunction(function=launch_setup)])
