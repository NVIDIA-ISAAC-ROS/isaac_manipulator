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
from launch.actions import GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    nvblox_base_config = os.path.join(
        get_package_share_directory(
            'isaac_manipulator_bringup'), 'config', 'nvblox', 'nvblox_base.yaml'
    )

    rgb_image_topic = LaunchConfiguration('rgb_image_topic')
    rgb_camera_info = LaunchConfiguration('rgb_camera_info')
    depth_image_topic = LaunchConfiguration('depth_image_topic')
    depth_camera_info = LaunchConfiguration('depth_camera_info')

    nvblox_node = ComposableNode(
        name='nvblox_node',
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        remappings=[
            ('/camera_0/color/image', rgb_image_topic),
            ('/camera_0/color/camera_info', rgb_camera_info),
            ('/camera_0/depth/image', depth_image_topic),
            ('/camera_0/depth/camera_info', depth_camera_info),
        ],
        parameters=[
            nvblox_base_config,
            {
                'num_cameras': 1,
                'global_frame': 'base_link',
                'static_mapper.esdf_slice_height': 0.0,
                'static_mapper.esdf_slice_min_height': -0.1,
                'static_mapper.esdf_slice_max_height': 0.3,
                'dynamic_mapper.esdf_slice_height': 0.0,
                'dynamic_mapper.esdf_slice_min_height': -0.1,
                'dynamic_mapper.esdf_slice_max_height': 0.3,
                'esdf_mode': 0,
            },
        ],
    )

    load_nodes = LoadComposableNodes(
        target_container='manipulation_container',
        composable_node_descriptions=[
            nvblox_node,
        ],
    )

    final_launch = GroupAction(
        actions=[
            load_nodes,
        ],
    )

    return (launch.LaunchDescription([final_launch]))
