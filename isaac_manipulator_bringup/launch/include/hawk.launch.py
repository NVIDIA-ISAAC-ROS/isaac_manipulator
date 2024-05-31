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

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    correlated_timestamp_driver_node = ComposableNode(
        package='isaac_ros_correlated_timestamp_driver',
        plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
        name='correlated_timestamp_driver',
        parameters=[{'use_time_since_epoch': False,
                     'nvpps_dev_name': '/dev/nvpps0'}])

    hawk_node = ComposableNode(
        name='hawk_node',
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{'module_id': 0,
                     'input_qos': 'SENSOR_DATA',
                     'output_qos': 'SENSOR_DATA',
                     'enable_statistics': True,
                     'topics_list': ['left/image_raw'],
                     'expected_fps_list': [30.0],
                     'jitter_tolerance_us': 30000}],
        remappings=[
            ('/hawk_front/correlated_timestamp', '/correlated_timestamp')
        ]
    )

    drop_node = ComposableNode(
        name='drop_node',
        package='isaac_ros_nitros_topic_tools',
        plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'SENSOR_DATA',
            'X': 5,
            'Y': 30,
            'mode': 'stereo',
            'sync_queue_size': 100
        }],
        remappings=[
            ('image_1', 'left/image_raw'),
            ('camera_info_1', 'left/camera_info'),
            ('image_1_drop', 'left/image_raw_drop'),
            ('camera_info_1_drop', 'left/camera_info_drop'),
            ('image_2', 'right/image_raw'),
            ('camera_info_2', 'right/camera_info'),
            ('image_2_drop', 'right/image_raw_drop'),
            ('camera_info_2_drop', 'right/camera_info_drop'),
        ]
    )

    load_nodes = LoadComposableNodes(
        target_container='manipulation_container',
        composable_node_descriptions=[
            correlated_timestamp_driver_node,
            hawk_node,
            drop_node,
        ],
    )

    final_launch = GroupAction(
        actions=[
            load_nodes,
        ],
    )

    return LaunchDescription([
        final_launch,
    ])
