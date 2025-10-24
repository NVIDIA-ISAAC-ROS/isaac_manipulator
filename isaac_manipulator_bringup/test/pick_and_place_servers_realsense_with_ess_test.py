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
"""Tests the entire stack on pose estimation run using Realsense + ESS. Verifies latency and hz."""

from datetime import datetime
import os

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_manipulator_ros_python_utils.test_utils import (
    PoseEstimationServersPolTest
)
from launch.actions import ExecuteProcess, IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS', None)
if ISAAC_ROS_WS is None:
    raise ValueError('ISAAC_ROS_WS environment variable is not set')
RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', f'{ISAAC_ROS_WS}/ess_pose')
RECORD_ROSBAG = os.environ.get('ISAAC_MANIPULATOR_TEST_RECORD_ROSBAG', '').lower() == 'true'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    PoseEstimationServersPolTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch', 'workflows')
    sensor_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch', 'sensors')

    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Override params to be PICK and PLACE
    override_params = {
        'workflow_type': 'PICK_AND_PLACE',
        'camera_type': 'REALSENSE',
        'num_cameras': '1',
        'enable_dnn_depth_in_realsense': 'true',
        'depth_type': 'ESS_FULL',
        'object_detection_type': 'RTDETR',
        'segmentation_type': 'NONE',
        'manual_mode': 'true'
    }
    params.update(override_params)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir, '/core.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [sensor_include_dir, '/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        if RECORD_ROSBAG:
            # Record all topics input into foundation pose node.
            # Record TF as well.
            # Record TF static.
            ros_bag_folder_path = os.path.join(OUTPUT_DIR, 'rosbag')
            os.makedirs(ros_bag_folder_path, exist_ok=True)
            timestamped_ros_bag_folder_path = os.path.join(
                ros_bag_folder_path,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            test_nodes.append(
                ExecuteProcess(
                    cmd=[
                        'ros2', 'bag', 'record', '--storage', 'mcap',
                        '--output', timestamped_ros_bag_folder_path,
                        '/rosout',
                        '/tf',
                        '/tf_static',
                        '/pose_estimation/output',
                        '/foundation_pose_server/camera_info',
                        '/foundation_pose_server/depth',
                        '/foundation_pose_server/image',
                        '/foundation_pose_aligned_dnn_depth_from_realsense_stereo',
                        '/resized_rgb_camera_info',
                        '/camera_1/infra1/image_rect_raw_drop',
                        '/camera_1/infra2/image_rect_raw_drop',
                        '/camera_1/infra1/camera_info_drop',
                        '/camera_1/infra2/camera_info_drop',
                    ],
                    output='both',
                    on_exit=Shutdown(),
                )
            )
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return PoseEstimationServersPolTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        max_timeout_time_for_action_call=10.0,
        num_perception_requests=10,
        use_sim_time=False,
        output_dir=OUTPUT_DIR,
        identifier='ess_pose',
    )
