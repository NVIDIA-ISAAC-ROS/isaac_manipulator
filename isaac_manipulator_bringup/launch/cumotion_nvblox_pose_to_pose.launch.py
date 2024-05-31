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
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    camera_type = str(context.perform_substitution(LaunchConfiguration('camera_type')))
    hawk_depth_mode = str(context.perform_substitution(LaunchConfiguration('hawk_depth_mode')))
    rgb_image_width = LaunchConfiguration('rgb_image_width')
    rgb_image_height = LaunchConfiguration('rgb_image_height')
    depth_image_width = LaunchConfiguration('depth_image_width')
    depth_image_height = LaunchConfiguration('depth_image_height')
    cumotion_depth_image_topics = LaunchConfiguration('cumotion_depth_image_topics')
    cumotion_depth_camera_infos = LaunchConfiguration('cumotion_depth_camera_infos')
    nvblox_rgb_image_topic = LaunchConfiguration('nvblox_rgb_image_topic')
    nvblox_rgb_camera_info = LaunchConfiguration('nvblox_rgb_camera_info')
    nvblox_depth_image_topic = LaunchConfiguration('nvblox_depth_image_topic')
    nvblox_depth_camera_info = LaunchConfiguration('nvblox_depth_camera_info')
    rviz_config_file = ''

    if camera_type == 'hawk':
        if hawk_depth_mode == 'ess_light':
            depth_image_width = 480
            depth_image_height = 288
        else:
            depth_image_width = 960
            depth_image_height = 576
        rgb_image_width = 1920
        rgb_image_height = 1200
        cumotion_depth_image_topics = '["/depth_image"]'
        cumotion_depth_camera_infos = '["/rgb/camera_info"]'
        nvblox_rgb_image_topic = '/rgb/image_rect_color'
        nvblox_rgb_camera_info = '/rgb/camera_info'
        nvblox_depth_camera_info = '/rgb/camera_info'
        rviz_config_file = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'rviz', 'hawk.rviz')

    else:
        depth_image_width = 1280
        depth_image_height = 720
        rgb_image_width = 1280
        rgb_image_height = 720
        cumotion_depth_image_topics = '["/camera_1/aligned_depth_to_color/image_raw"]'
        cumotion_depth_camera_infos = '["/camera_1/aligned_depth_to_color/camera_info"]'
        nvblox_rgb_image_topic = '/camera_1/color/image_raw'
        nvblox_rgb_camera_info = '/camera_1/color/camera_info'
        nvblox_depth_camera_info = '/camera_1/aligned_depth_to_color/camera_info'
        rviz_config_file = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'rviz', 'realsense.rviz')

    nvblox_depth_image_topic = '/cumotion/camera_1/world_depth'

    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/realsense.launch.py']
        ),
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"realsense"'])),
    )

    hawk_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/hawk.launch.py']
        ),
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"hawk"'])),
    )

    ess_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/ess.launch.py']
        ),
        launch_arguments={
            'ess_mode': str(hawk_depth_mode),
            'image_width': str(rgb_image_width),
            'image_height': str(rgb_image_height),
            'ess_model_width': str(depth_image_width),
            'ess_model_height': str(depth_image_height),
        }.items(),
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"hawk"'])),
    )

    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/cumotion.launch.py']
        ),
        launch_arguments={
            'depth_image_topics': cumotion_depth_image_topics,
            'depth_camera_infos': cumotion_depth_camera_infos,
        }.items(),
    )

    nvblox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/nvblox.launch.py']
        ),
        launch_arguments={
            'rgb_image_topic': nvblox_rgb_image_topic,
            'rgb_camera_info': nvblox_rgb_camera_info,
            'depth_image_topic': nvblox_depth_image_topic,
            'depth_camera_info': nvblox_depth_camera_info,
        }.items(),
    )

    pose_to_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/pose_to_pose.launch.py']
        ),
    )

    static_transform_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/static_transforms.launch.py']
        ),
        launch_arguments={
            'camera': camera_type,
        }.items(),
    )

    rviz2_node = Node(
        name='rviz2',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(context.launch_configurations['launch_rviz'])
    )

    manipulation_container = Node(
        name='manipulation_container',
        package='rclcpp_components',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', 'error'],
    )

    return [
        manipulation_container,
        hawk_launch,
        ess_launch,
        realsense_launch,
        cumotion_launch,
        nvblox_launch,
        pose_to_pose_launch,
        static_transform_launch,
        rviz2_node,
    ]


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'camera_type',
            default_value='hawk',
            choices=['hawk', 'realsense'],
            description='Camera sensor to use for this example'),
        DeclareLaunchArgument(
            'hawk_depth_mode',
            default_value='ess_light',
            choices=['ess_light', 'ess_full'],
            description='Depth mode for Hawk camera'),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            choices=['true', 'false'],
            description='Launch RViz visualization tool'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
