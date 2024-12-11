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

from typing import List
import os
import yaml

from isaac_ros_launch_utils.all_types import Action, ComposableNode, LaunchDescription
import isaac_ros_launch_utils as lu

import isaac_manipulator_ros_python_utils.constants as constants


def load_config_dict(config_path: str):
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)
    return config_dict


def add_realsense_drivers(args: lu.ArgumentContainer) -> List[Action]:
    actions = []
    num_cameras = int(args.num_cameras)
    camera_ids_config_name = str(args.camera_ids_config_name)
    assert num_cameras <= 2, 'Currently we only support up to 2 realsense cameras.'

    # Get the realsense camera config paths
    realsense_config = lu.get_path('isaac_manipulator_bringup', 'config/sensors/realsense.yaml')
    camera_ids_config_path = lu.get_path(
        'isaac_manipulator_bringup',
        f'config/sensors/realsense_camera_ids/{camera_ids_config_name}.yaml')
    unspecified_ids_config_path = lu.get_path(
        'isaac_manipulator_bringup', 'config/sensors/realsense_camera_ids/unspecified.yaml')

    # Get the camera IDs (for repeatable numbering of the cameras)
    if os.path.exists(camera_ids_config_path):
        camera_ids_dict = load_config_dict(camera_ids_config_path)
        actions.append(
            lu.log_info(
                ["Loading the realsense camera ids of the '",
                 camera_ids_config_name, "' config."]))
    else:
        camera_ids_dict = load_config_dict(unspecified_ids_config_path)

    driver_nodes = []
    for i in range(num_cameras):
        camera_name = f'camera_{i+1}'
        driver_nodes.append(
            ComposableNode(
                namespace=camera_name,
                name=camera_name,
                package='realsense2_camera',
                plugin='realsense2_camera::RealSenseNodeFactory',
                parameters=[
                    realsense_config, {
                        'camera_name': camera_name,
                        'serial_no': camera_ids_dict[camera_name]['serial_no'],
                    }
                ]))

    if args.run_standalone:
        actions.append(lu.component_container(args.container_name))
    actions.append(lu.load_composable_nodes(args.container_name, driver_nodes))
    actions.append(lu.log_info(["Started '", str(num_cameras), "' RealSense camera drivers."]))

    # Visualization
    if args.run_standalone:
        actions.append(
            lu.include(
                'isaac_manipulator_bringup',
                'launch/visualization/visualization.launch.py',
                launch_arguments={'camera_type': 'realsense'},
            ))

    return actions


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('num_cameras', 1)
    args.add_arg('camera_ids_config_name', '')
    args.add_arg('container_name', constants.MANIPULATOR_CONTAINER_NAME)
    args.add_arg('run_standalone', 'False')
    args.add_opaque_function(add_realsense_drivers)
    return LaunchDescription(args.get_launch_actions())
