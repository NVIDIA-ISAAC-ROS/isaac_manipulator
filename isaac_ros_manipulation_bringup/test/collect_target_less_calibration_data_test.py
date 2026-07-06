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
"""Collects calibration data to calibrate robot with camera without calibration target."""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_ros_manipulation_ros_python_utils.test_utils import (
    CollectTargetLessCalibrationDataTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest
import yaml


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'manual_on_robot'
OUTPUT_DIR = os.path.join(os.environ['ISAAC_ROS_WS'], 'calibration_dataset_targetless')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    CollectTargetLessCalibrationDataTest.generate_namespace()
    isaac_ros_manipulation_bringup_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'launch')

    ur_robotiq_driver_launch = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_ur_driver_utils'),
        'launch', 'ur_robotiq_driver.launch.py')
    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Env test config plus workflow overrides; drivers.launch needs robot_launch_file_path in YAML
    override_params = {
        'workflow_type': 'PICK_AND_PLACE',
        'camera_type': 'REALSENSE',
        'num_cameras': '2',
        'manual_mode': 'true',
        'headless': 'true',
        'robot_launch_file_path': ur_robotiq_driver_launch
    }
    params.update(override_params)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    cumotion_urdf_file_path = ''
    if RUN_TEST:
        node_startup_delay = 12.0
        cumotion_urdf_file_path = params['cumotion_urdf_file_path']
        # drivers.launch.py loads workflow from a file; write merged test config (same as bringup)
        fd, merged_workflow = tempfile.mkstemp(
            suffix='.yaml', prefix='collect_targetless_calib_')
        os.close(fd)
        with open(merged_workflow, 'w', encoding='utf-8') as out:
            yaml.dump(params, out, default_flow_style=False, sort_keys=False)
        # Match workflows.launch.py: sensors, core, then driver routing (drivers.launch.py).
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_bringup_launch_dir, '/sensors/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_bringup_launch_dir, '/workflows/core.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_bringup_launch_dir, '/drivers.launch.py']),
            launch_arguments={'manipulator_workflow_config': merged_workflow}.items()))
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return CollectTargetLessCalibrationDataTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        use_sim_time=False,
        camera_topic_names=[
            '/camera_1/color/image_raw',
            '/camera_2/color/image_raw'
        ],
        camera_info_topic_names=[
            '/camera_1/color/camera_info',
            '/camera_2/color/camera_info'
        ],
        base_frame_name='base',
        joint_topic_name='/joint_states',
        joints_to_query=[
            'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
            'camera_1_color_optical_frame',
            'camera_2_color_optical_frame'
        ],
        urdf_file_path=cumotion_urdf_file_path,
        output_dir=OUTPUT_DIR
    )
