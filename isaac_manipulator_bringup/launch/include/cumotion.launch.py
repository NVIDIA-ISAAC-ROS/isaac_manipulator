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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'robot_file_name', default_value='kinova_gen3.yml',
            description='The file path that describes robot'),
        DeclareLaunchArgument(
            'time_dilation_factor', default_value='0.25',
            description='Speed scaling factor for the planner'),
        DeclareLaunchArgument(
            'distance_threshold', default_value='0.15',
            description='Maximum distance from a given collision sphere (in meters) at which to mask points in the robot segmenter'),
        DeclareLaunchArgument(
            'time_sync_slop', default_value='0.1',
            description='Maximum allowed delay (in seconds) for which depth image and joint state messages are considered synchronized in the robot segmenter'),
    ]

    robot_file_name = LaunchConfiguration('robot_file_name')
    time_dilation_factor = LaunchConfiguration('time_dilation_factor')
    depth_image_topics = LaunchConfiguration('depth_image_topics')
    depth_camera_infos = LaunchConfiguration('depth_camera_infos')
    distance_threshold = LaunchConfiguration('distance_threshold')
    time_sync_slop = LaunchConfiguration('time_sync_slop')

    cumotion_launch_path = os.path.join(get_package_share_directory('isaac_ros_cumotion'),
                                        'launch', 'isaac_ros_cumotion.launch.py')

    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([cumotion_launch_path]),
        launch_arguments={
                          'cumotion_planner.robot': robot_file_name,
                          'cumotion_planner.voxel_dims': '[2.0, 2.0, 2.0]',
                          'cumotion_planner.grid_position': '[0.0, 0.0, 0.0]',
                          'cumotion_planner.time_dilation_factor': time_dilation_factor,
                          'cumotion_planner.end_effector_link': 'tool_frame',
                          'cumotion_planner.read_esdf_world': 'True',
                          'cumotion_planner.publish_curobo_world_as_voxels': 'True',
                          'cumotion_planner.override_moveit_scaling_factors': 'True',
                          }.items())

    robot_segmenter_launch_path = os.path.join(get_package_share_directory('isaac_ros_cumotion'),
                                               'launch', 'robot_segmentation.launch.py')

    robot_segmenter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_segmenter_launch_path]),
        launch_arguments={
                          'robot_segmenter.robot': robot_file_name,
                          'robot_segmenter.depth_image_topics': depth_image_topics,
                          'robot_segmenter.depth_camera_infos': depth_camera_infos,
                          'robot_segmenter.distance_threshold': distance_threshold,
                          'robot_segmenter.time_sync_slop': time_sync_slop,
                        }.items())

    return (launch.LaunchDescription(launch_args + [cumotion_launch, robot_segmenter_launch]))
