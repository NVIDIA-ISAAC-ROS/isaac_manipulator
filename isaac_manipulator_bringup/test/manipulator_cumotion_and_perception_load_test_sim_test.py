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
"""
Tests the performance of the Isaac Manipulator system under maximum load.

This test runs the Isaac Manipulator system with the pose to pose node and the
object following nodes running all the time. It then measures the performance of the
system under maximum load by checking the latency of the pose estimation node as a proxy for the
overall system performance.
"""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    load_yaml_params
)
from isaac_manipulator_ros_python_utils.test_utils import IsaacManipulatorLoadEstimationTest
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest
from vision_msgs.msg import Detection3DArray


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'isaac_sim'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = f'/{ISAAC_ROS_WS}/manipulator_load_test_sim'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Cumotion, perception and nvblox nodes for testing."""
    IsaacManipulatorLoadEstimationTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch')
    isaac_manipulator_test_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'test', 'include')
    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'params',
        'sim_launch_params.yaml'
    )
    params = load_yaml_params(test_yaml_config)

    params['workflow_type'] = 'OBJECT_FOLLOWING'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 12.0
        # This starts object following without object following nodes.
        # It starts the pose to pose so that we get planning calls every few seconds
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_test_include_dir,
                 '/pose_to_pose_with_perception_and_planning.launch.py']),
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

        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['-0.7', '0.3', '0.4', '0', '0', '0', 'world', 'target1_frame']
        ))

        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['-0.7', '-0.3', '0.4', '0', '0', '0', 'world', 'target2_frame']
        ))
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return IsaacManipulatorLoadEstimationTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        monitor_topic_name='pose_estimation/output',
        robot_segmenter_topic_names=['cumotion/camera_1/world_depth'],
        use_sim_time=True,
        message_type=Detection3DArray,
        test_duration_seconds=120,
        output_dir=OUTPUT_DIR,
        # TODO: Replace with ros2 monitor node for accurate latency measurement with
        # no rclpy overhead
        max_latency_in_robot_segmentor_ms=2000,
    )
