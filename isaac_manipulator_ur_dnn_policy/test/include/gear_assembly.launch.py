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

"""This file is used to test the gear assembly inference policy."""

from typing import List

import isaac_manipulator_ros_python_utils as manipulator_utils

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_context import LaunchContext
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    """
    Test gear assembly inference policy.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        List[Node]: List of nodes

    """
    # Set up container for our nodes
    gear_assembly_config = manipulator_utils.GearAssemblyConfig(context=context)
    nodes = manipulator_utils.get_gear_assembly_nodes(gear_assembly_config,
                                                      use_sim_time=False)
    return nodes


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
