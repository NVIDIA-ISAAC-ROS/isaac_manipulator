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

import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    default_params_file = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_object_following'),
        'config', 'object_following_params.yaml'
    )

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to the YAML file with object following workflow parameters'),
        DeclareLaunchArgument(
            'target_tool_frame',
            default_value='goal_frame',
            description='The target TF frame that the robot tool should move toward '
                        '(overrides value in params_file)'),
        DeclareLaunchArgument(
            'time_dilation_factor',
            default_value='0.2',
            description='Speed scaling factor for the planned trajectory '
                        '(overrides value in params_file)'),
    ]

    params_file = LaunchConfiguration('params_file')
    target_tool_frame = LaunchConfiguration('target_tool_frame')
    time_dilation_factor = LaunchConfiguration('time_dilation_factor')

    object_following_node = Node(
        package='isaac_ros_manipulation_object_following',
        namespace='',
        executable='object_following_node',
        name='object_following_node',
        parameters=[
            params_file,
            {
                'target_tool_frame': target_tool_frame,
                'time_dilation_factor': time_dilation_factor,
            },
        ],
        output='screen',
        on_exit=Shutdown()
    )

    return (launch.LaunchDescription(launch_args + [object_following_node]))
