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

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file to bring up isaac manipulator pick and place server."""
    dope_node = ComposableNode(
        name='dope_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::DopeServer',
        parameters=[{
                'action_name': 'dope',
                'in_img_topic_name': 'image_rect',
                'out_img_topic_name': 'dope_server/image_rect',
                'in_camera_info_topic_name': 'camera_info_rect',
                'out_camera_info_topic_name': 'dope_server/camera_info',
                'in_pose_estimate_topic_name': 'detections',
                'out_pose_estimate_topic_name': 'dope_server/poses',
            }]
    )

    dope_launch_container = ComposableNodeContainer(
        name='dope_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[dope_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'dope_server:=info']
    )

    return launch.LaunchDescription([dope_launch_container])
