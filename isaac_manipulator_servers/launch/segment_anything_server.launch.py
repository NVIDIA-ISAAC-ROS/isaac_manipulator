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
    """Launch file to bring up isaac manipulator segment anything server."""
    launch_args = [
        DeclareLaunchArgument(
            'is_sam2',
            default_value='False',
            description='True if using SAM2, False if using SAM1',
        ),
        DeclareLaunchArgument(
            'sam_action_name',
            default_value='segment_anything',
            description='Action name for the Segment Anything Server',
        ),
        DeclareLaunchArgument(
            'sam_in_img_topic_name',
            default_value='in_camera',
            description='Input image topic name',
        ),
        DeclareLaunchArgument(
            'sam_out_img_topic_name',
            default_value='segment_anything_server/out_camera',
            description='Output image topic name',
        ),
        DeclareLaunchArgument(
            'sam_in_camera_info_topic_name',
            default_value='in_camera_info',
            description='Input camera info topic name',
        ),
        DeclareLaunchArgument(
            'sam_out_camera_info_topic_name',
            default_value='segment_anything_server/out_camera_info',
            description='Output camera info topic name',
        ),
        DeclareLaunchArgument(
            'sam_in_segmentation_mask_topic_name',
            default_value='in_segmentation_mask',
            description='Input segmentation mask topic name',
        ),
        DeclareLaunchArgument(
            'sam_in_detections_topic_name',
            default_value='in_detections',
            description='Input detections topic name',
        ),
        DeclareLaunchArgument(
            'sam_input_qos',
            default_value='SENSOR_DATA',
            description='Subscription QoS profile for the Segment Anything Server',
        ),
        DeclareLaunchArgument(
            'sam_result_and_output_qos',
            default_value='DEFAULT',
            description='Publication QoS profile for the Segment Anything Server',
        ),
        DeclareLaunchArgument(
            'sam_out_detections_topic_name',
            default_value='segment_anything_server/detections_initial_guess',
            description='Output detections topic name',
        ),
    ]

    segment_anything_server_node = ComposableNode(
        name='segment_anything_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::SegmentAnythingServer',
        parameters=[{
                'is_sam2': LaunchConfiguration('is_sam2'),
                'action_name': LaunchConfiguration('sam_action_name'),
                'in_img_topic_name': LaunchConfiguration('sam_in_img_topic_name'),
                'out_img_topic_name': LaunchConfiguration('sam_out_img_topic_name'),
                'in_camera_info_topic_name': LaunchConfiguration('sam_in_camera_info_topic_name'),
                'out_camera_info_topic_name':
                    LaunchConfiguration('sam_out_camera_info_topic_name'),
                'in_segmentation_mask_topic_name':
                    LaunchConfiguration('sam_in_segmentation_mask_topic_name'),
                'in_detections_topic_name':
                    LaunchConfiguration('sam_in_detections_topic_name'),
                'out_detections_topic_name':
                    LaunchConfiguration('sam_out_detections_topic_name'),
                'input_qos': LaunchConfiguration('sam_input_qos'),
                'result_and_output_qos': LaunchConfiguration('sam_result_and_output_qos')
        }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[segment_anything_server_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription(launch_args + [final_launch])
