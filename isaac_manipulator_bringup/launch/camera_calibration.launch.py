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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

import isaac_manipulator_ros_python_utils.constants as constants


def launch_setup(context, *args, **kwargs):
    camera_type = context.perform_substitution(LaunchConfiguration('camera_type'))

    if camera_type == 'hawk':
        rgb_image_width = 1920
        rgb_image_height = 1200
    else:
        rgb_image_width = 1280
        rgb_image_height = 720
    
    correlated_timestamp_driver_node = ComposableNode(
        package='isaac_ros_correlated_timestamp_driver',
        plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
        name='correlated_timestamp_driver',
        parameters=[{'use_time_since_epoch': False,
                     'nvpps_dev_name': '/dev/nvpps0'}],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', '==', '"hawk"']))
    )

    hawk_node = ComposableNode(
        name='hawk_node',
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{'module_id': 0,
                     'input_qos': 'DEFAULT',
                     'output_qos': 'DEFAULT',
                     'enable_statistics': True,
                     'topics_list': ['left/image_raw'],
                     'expected_fps_list': [30.0],
                     'jitter_tolerance_us': 30000}],
        remappings=[
            ('/hawk_front/correlated_timestamp', '/correlated_timestamp')
        ],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', '==', '"hawk"']))
    )

    hawk_left_rectify_node = ComposableNode(
        name='hawk_left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': rgb_image_width,
            'output_height': rgb_image_height,
            'input_qos': 'DEFAULT',
            'output_qos': 'DEFAULT'
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('image_rect', 'left/rect/image'),
            ('camera_info_rect', 'left/rect/camera_info')
        ],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', '==', '"hawk"']))
    )
    
    realsense_config = os.path.join(
        get_package_share_directory(
            'isaac_manipulator_bringup'), 'config', 'sensors', 'realsense_calibration.yaml'
    )
    
    realsense_node = ComposableNode(
        namespace='camera_1',
        name='camera_1',
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', '==', '"realsense"']))
    )

    calibration_container = ComposableNodeContainer(
        name=constants.MANIPULATOR_CONTAINER_NAME,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', 'error'],
        composable_node_descriptions=[
            correlated_timestamp_driver_node,
            hawk_node,
            hawk_left_rectify_node,
            realsense_node,
        ],
    )

    return [
        calibration_container,
    ]


def generate_launch_description():
    LaunchConfiguration('camera_type')
    launch_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='hawk',
        choices=['hawk', 'realsense'],
        description='Camera sensor used for calibration'
    )

    return LaunchDescription([launch_arg, OpaqueFunction(function=launch_setup)])
