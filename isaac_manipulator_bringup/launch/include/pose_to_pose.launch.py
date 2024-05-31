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
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    goal_setter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('isaac_ros_moveit_goal_setter'),
                         'launch', 'isaac_ros_goal_setter.launch.py')
        ),
    )

    pose_to_pose_node = Node(
        package='isaac_ros_moveit_goal_setter',
        namespace='',
        executable='pose_to_pose.py',
        name='pose_to_pose_node',
        output='screen',
        parameters=[{
            'target_frames': ['target1_frame', 'target2_frame']
        }]
    )

    return launch.LaunchDescription([goal_setter_launch, pose_to_pose_node])
