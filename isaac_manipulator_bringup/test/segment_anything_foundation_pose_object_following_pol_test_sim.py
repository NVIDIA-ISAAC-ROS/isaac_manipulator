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
"""Tests the entire stack on pose estimation run using Isaac Sim setup."""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_manipulator_ros_python_utils.test_utils import (
    IsaacManipulatorSegmentAnythingPolTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter

import pytest
from sensor_msgs.msg import Image


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'isaac_sim'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = os.path.join(
    ISAAC_ROS_WS,
    'segment_anything_foundation_pose_object_following_pol_test_sim')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    IsaacManipulatorSegmentAnythingPolTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch', 'workflows')
    params = get_params_from_config_file_set_in_env(RUN_TEST)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Trigger SAM detection for object following.
    overide_params = {
        'segmentation_type': 'SEGMENT_ANYTHING',
        'object_detection_type': 'SEGMENT_ANYTHING',
        'pose_estimation_type': 'FOUNDATION_POSE',
        'segment_anything_input_points_topic': 'input_points',
        'segment_anything_input_detections_topic': 'input_detections',
        'workflow_type': 'OBJECT_FOLLOWING',
        'enable_nvblox': 'false',
        'camera_type': 'REALSENSE'
    }
    params.update(overide_params)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        segment_anything_model_path = \
            f'{params["sam_model_repository_paths"][0]}/segment_anything/1/model.onnx'
        if not os.path.exists(segment_anything_model_path):
            raise FileNotFoundError(
                f'Segment Anything model not found at {segment_anything_model_path}')
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir, '/core.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        if params['use_sim_time'] == 'true':
            test_nodes += [SetParameter(name='use_sim_time', value=True)]
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return IsaacManipulatorSegmentAnythingPolTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        publish_rate=2.0,  # Publishes points at a rate of 2 hz.
        num_cycles=10,
        publish_sequence='all',
        initial_delay=1.0,
        use_sim_time=True,
        output_dir=OUTPUT_DIR,
        max_latency_ms=800,  # Should average 150 ms on x86, but this number is same as RTDETR
        monitor_topic_name='/segment_anything/binary_segmentation_mask',
        message_type=Image,
    )
