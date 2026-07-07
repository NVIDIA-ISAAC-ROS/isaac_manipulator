# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Robot driver routing launch file.

This launch file reads the 'robot_launch_file_path' parameter from the workflow configuration
and forwards all parameters to the specified robot-specific driver launch file.

This enables support for multiple robot types by routing to their respective driver launch files
based on configuration, rather than hardcoding a specific robot driver.
"""

from typing import List

from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params
from isaac_ros_manipulation_ros_python_utils.launch_utils import get_str_variable

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context: LaunchContext, *args, **kwargs) -> List:
    """
    Set up the launch by routing to the robot-specific driver launch file.

    Args
    ----
        context (LaunchContext): Launch context containing configuration parameters.

    Returns
    -------
        List: List containing the included robot driver launch description.

    """
    manipulator_workflow_config_path = get_str_variable(context, 'manipulator_workflow_config')
    params = load_yaml_params(manipulator_workflow_config_path)

    robot_launch_file_path = params.get('robot_launch_file_path')

    if robot_launch_file_path is None:
        raise ValueError(
            "Missing 'robot_launch_file_path' in workflow configuration. "
            "Please specify the path to your robot's driver launch file."
        )

    # Include the robot-specific driver launch file and forward all parameters
    driver_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_launch_file_path),
        launch_arguments={key: str(value) for key, value in params.items()}.items()
    )

    return [driver_nodes]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'manipulator_workflow_config',
            description='Path to the yaml file which hosts configuration values for the '
                        'manipulator workflow including robot_launch_file_path.'
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
