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
"""Script to get the robot in the home position in Isaac Sim."""

# ros2 topic pub /stop_joint_commands std_msgs/msg/Bool "data: true"

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation


world = World()

stage = world.stage
robot = Articulation(prim_paths_expr='/World/ur10e_robotiq2f_140_ROS')

potential_ik_solutions = []
ranked_sorted_indexes = {}

index = ranked_sorted_indexes['sorted_indexes'][0]
position = potential_ik_solutions[index]['position'][:6]

# This script should be run in the Isaac Sim Script Editor.
robot.set_joint_position_targets(
    positions=position,
    joint_names=[
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
)
