# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        get_package_share_directory('isaac_ros_manipulation_pose_to_pose'),
        'config', 'pose_to_pose_params.yaml'
    )

    launch_args = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to the YAML file with pose-to-pose workflow parameters'),
        DeclareLaunchArgument(
            'time_dilation_factor',
            default_value='0.2',
            description='Speed scaling factor for the planned trajectory '
                        '(overrides value in params_file)'),
    ]

    params_file = LaunchConfiguration('params_file')
    time_dilation_factor = LaunchConfiguration('time_dilation_factor')

    pose_to_pose_node = Node(
        package='isaac_ros_manipulation_pose_to_pose',
        namespace='',
        executable='pose_to_pose_node',
        name='pose_to_pose_node',
        parameters=[
            params_file,
            {'time_dilation_factor': time_dilation_factor},
        ],
        output='screen',
        on_exit=Shutdown()
    )

    return launch.LaunchDescription(launch_args + [pose_to_pose_node])
