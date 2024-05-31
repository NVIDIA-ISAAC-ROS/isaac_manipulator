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
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    realsense_launch_path = os.path.join(get_package_share_directory(
        'realsense2_camera'), 'launch', 'rs_launch.py')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([realsense_launch_path]),
        launch_arguments={'camera_name': 'camera_1',
                          'align_depth.enable': 'True',
                          'spatial_filter.enable': 'True',
                          'depth_module.profile': '1280x720x15'
                          }.items(),
    )

    return LaunchDescription([
        realsense_launch,
    ])
