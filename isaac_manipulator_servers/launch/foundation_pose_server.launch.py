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
            'fp_action_name',
            default_value='estimate_pose_foundation_pose',
            description='Action name for the Foundation Pose Server',
        ),
        DeclareLaunchArgument(
            'fp_in_img_topic_name',
            default_value='/camera_1/color/image_raw',
            description='Input image topic name',
        ),
        DeclareLaunchArgument(
            'fp_out_img_topic_name',
            default_value='foundation_pose_server/camera_1/color/image_raw',
            description='Output image topic name',
        ),
        DeclareLaunchArgument(
            'fp_in_camera_info_topic_name',
            default_value='/resize/camera_info',
            description='Input camera info topic name',
        ),
        DeclareLaunchArgument(
            'fp_out_camera_info_topic_name',
            default_value='foundation_pose_server/resize/camera_info',
            description='Output camera info topic name',
        ),
        DeclareLaunchArgument(
            'fp_in_depth_topic_name',
            default_value='/camera_1/aligned_depth_to_color/image_raw',
            description='Input depth topic name',
        ),
        DeclareLaunchArgument(
            'fp_out_depth_topic_name',
            default_value='foundation_pose_server/camera_1/aligned_depth_to_color/image_raw',
            description='Output depth topic name',
        ),
        DeclareLaunchArgument(
            'fp_out_bbox_topic_name',
            default_value='foundation_pose_server/bbox',
            description='Output bounding box topic name',
        ),
        DeclareLaunchArgument(
            'fp_in_pose_estimate_topic_name',
            default_value='/pose_estimation/output',
            description='Input pose estimate topic name',
        ),
        DeclareLaunchArgument(
            'fp_out_pose_estimate_topic_name',
            default_value='foundation_pose_server/pose_estimation/output',
            description='Output pose estimate topic name',
        ),
    ]

    foundation_pose_server_node = ComposableNode(
        name='foundation_pose_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::FoundationPoseServer',
        parameters=[{
                'action_name': LaunchConfiguration('fp_action_name'),
                'in_img_topic_name': LaunchConfiguration('fp_in_img_topic_name'),
                'out_img_topic_name': LaunchConfiguration('fp_out_img_topic_name'),
                'in_camera_info_topic_name': LaunchConfiguration('fp_in_camera_info_topic_name'),
                'out_camera_info_topic_name': LaunchConfiguration('fp_out_camera_info_topic_name'),
                'in_depth_topic_name': LaunchConfiguration('fp_in_depth_topic_name'),
                'out_depth_topic_name': LaunchConfiguration('fp_out_depth_topic_name'),
                'out_bbox_topic_name': LaunchConfiguration('fp_out_bbox_topic_name'),
                'in_pose_estimate_topic_name':
                    LaunchConfiguration('fp_in_pose_estimate_topic_name'),
                'out_pose_estimate_topic_name':
                    LaunchConfiguration('fp_out_pose_estimate_topic_name')
        }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[foundation_pose_server_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription(launch_args + [final_launch])
