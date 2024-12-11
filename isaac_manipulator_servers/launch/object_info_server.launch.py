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
import isaac_manipulator_ros_python_utils.constants as constants
from isaac_ros_launch_utils.all_types import (
    GroupAction, LoadComposableNodes
)

import launch
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file to bring up isaac manipulator pick and place server."""
    object_info_node = ComposableNode(
        name='object_info_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectInfoServer',
        parameters=[{
                'action_name': 'object_info',
            }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[object_info_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription([final_launch])
