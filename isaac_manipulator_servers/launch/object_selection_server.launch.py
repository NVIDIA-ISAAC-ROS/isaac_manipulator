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
import isaac_manipulator_ros_python_utils.constants as constants
from isaac_ros_launch_utils.all_types import (
    GroupAction, LoadComposableNodes
)

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file to bring up isaac manipulator object selection server."""
    launch_args = [
        DeclareLaunchArgument(
            'action_name',
            default_value='get_selected_object',
            description='Action server name',
        ),
        DeclareLaunchArgument(
            'selection_policy',
            default_value='first',
            description='Selection policy for object selection',
        ),
    ]

    object_selection_node = ComposableNode(
        name='object_selection_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectSelectionServer',
        parameters=[{
                'action_name': LaunchConfiguration('action_name'),
                'selection_policy': LaunchConfiguration('selection_policy'),
        }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[object_selection_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription(launch_args + [final_launch])
