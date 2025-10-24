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
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch file to bring up isaac manipulator pick and place server."""
    launch_args = [
        DeclareLaunchArgument(
            'dope_action_name',
            default_value='dope',
            description='Action name for the DOPE server',
        ),
        DeclareLaunchArgument(
            'dope_in_img_topic_name',
            default_value='image_rect',
            description='Input image topic name',
        ),
        DeclareLaunchArgument(
            'dope_out_img_topic_name',
            default_value='dope_server/image_rect',
            description='Output image topic name',
        ),
        DeclareLaunchArgument(
            'dope_in_camera_info_topic_name',
            default_value='camera_info_rect',
            description='Input camera info topic name',
        ),
        DeclareLaunchArgument(
            'dope_out_camera_info_topic_name',
            default_value='dope_server/camera_info',
            description='Output camera info topic name',
        ),
        DeclareLaunchArgument(
            'dope_in_pose_estimate_topic_name',
            default_value='detections',
            description='Input pose estimate topic name',
        ),
        DeclareLaunchArgument(
            'dope_out_pose_estimate_topic_name',
            default_value='dope_server/poses',
            description='Output pose estimate topic name',
        ),
        DeclareLaunchArgument(
            'dope_sub_qos',
            default_value='SENSOR_DATA',
            description='Subscription QoS profile for the DOPE server',
        ),
        DeclareLaunchArgument(
            'dope_pub_qos',
            default_value='DEFAULT',
            description='Publication QoS profile for the DOPE server',
        ),
    ]

    dope_node = ComposableNode(
        name='dope_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::DopeServer',
        parameters=[{
                'action_name':  LaunchConfiguration('dope_action_name'),
                'in_img_topic_name': LaunchConfiguration('dope_in_img_topic_name'),
                'out_img_topic_name': LaunchConfiguration('dope_out_img_topic_name'),
                'in_camera_info_topic_name': LaunchConfiguration('dope_in_camera_info_topic_name'),
                'out_camera_info_topic_name': LaunchConfiguration(
                    'dope_out_camera_info_topic_name'),
                'in_pose_estimate_topic_name': LaunchConfiguration(
                    'dope_in_pose_estimate_topic_name'),
                'out_pose_estimate_topic_name': LaunchConfiguration(
                    'dope_out_pose_estimate_topic_name'),
                'sub_qos': LaunchConfiguration('dope_sub_qos'),
                'pub_qos': LaunchConfiguration('dope_pub_qos')
            }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[dope_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription(launch_args + [final_launch])
