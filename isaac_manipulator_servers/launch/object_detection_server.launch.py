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
            'obj_action_name',
            default_value='detect_objects',
            description='Action server name',
        ),
        DeclareLaunchArgument(
            'obj_input_img_topic_name',
            default_value='/camera_1/color/image_raw',
            description='Image topic as input for object detection server',
        ),
        DeclareLaunchArgument(
            'obj_output_img_topic_name',
            default_value='/object_detection_server/image_rect',
            description='Image topic as output for object detection server',
        ),
        DeclareLaunchArgument(
            'obj_input_detections_topic_name',
            default_value='/detections',
            description='The topic name for input detections to parse',
        ),
        DeclareLaunchArgument(
            'obj_output_detections_topic_name',
            default_value='object_detection_server/detections_output',
            description='The topic name for output detections to parse',
        ),
    ]

    object_detection_node = ComposableNode(
        name='object_detection_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectDetectionServer',
        parameters=[{
                'action_name': LaunchConfiguration('obj_action_name'),
                'input_img_topic_name': LaunchConfiguration('obj_input_img_topic_name'),
                'output_img_topic_name': LaunchConfiguration('obj_output_img_topic_name'),
                'input_detections_topic_name': LaunchConfiguration(
                    'obj_input_detections_topic_name'),
                'output_detections_topic_name': LaunchConfiguration(
                    'obj_output_detections_topic_name')
            }]
    )

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[object_detection_node],
    )
    final_launch = GroupAction(
        actions=[
            load_composable_nodes
        ],
    )

    return launch.LaunchDescription(launch_args + [final_launch])
