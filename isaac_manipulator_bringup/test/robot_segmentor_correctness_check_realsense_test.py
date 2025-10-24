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
"""Tests the correctness of the robot segmentor for Realsense setup."""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_manipulator_ros_python_utils.test_utils import (
    IsaacManipulatorRobotSegmentorCorrectnessCheckTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = os.path.join(ISAAC_ROS_WS, 'robot_segmentor_correctness_realsense_check')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Robot Segmentor nodes for testing."""
    IsaacManipulatorRobotSegmentorCorrectnessCheckTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch')
    isaac_manipulator_test_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'test', 'include')

    params = get_params_from_config_file_set_in_env(RUN_TEST)
    params['camera_type'] = 'REALSENSE'
    params['num_cameras'] = '1'
    params['workflow_type'] = 'OBJECT_FOLLOWING'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    xrdf_file_path = ''
    xrdf_collision_geometry_name = ''
    if RUN_TEST:
        node_startup_delay = 12.0
        xrdf_file_path = params['cumotion_xrdf_file_path']
        gripper_type = params['gripper_type']
        ur_type = params['ur_type']
        xrdf_collision_geometry_name = f'{ur_type}_{gripper_type}_collision_spheres'
        # This starts object following without object following nodes.
        # It starts the pose to pose so that we get planning calls every few seconds
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_test_include_dir,
                 '/cumotion.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir,
                    '/drivers/ur_robotiq_driver.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir,
                    '/sensors/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return IsaacManipulatorRobotSegmentorCorrectnessCheckTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        use_sim_time=True,
        output_dir=OUTPUT_DIR,
        timeout_seconds_for_service_call=30.0,
        links_to_check=['forearm_link', 'upper_arm_link'],
        xrdf_file_path=xrdf_file_path,
        intrinsics_topic='/camera_1/aligned_depth_to_color/camera_info',
        mask_topic='/cumotion/camera_1/robot_mask',
        joint_states_topic='/joint_states',
        depth_topic='/cumotion/camera_1/world_depth',
        raw_depth_topic='/camera_1/aligned_depth_to_color/image_raw',
        num_samples=1000,
        buffer_distance_for_collision_spheres=0.1,
        depth_image_is_float16=True,
        xrdf_collision_geometry_name=xrdf_collision_geometry_name,
    )
