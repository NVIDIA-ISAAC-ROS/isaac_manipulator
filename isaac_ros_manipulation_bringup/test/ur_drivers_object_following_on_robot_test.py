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

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_ros_manipulation_ros_python_utils.test_utils import DriverTest
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest
import yaml


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'


@pytest.mark.rostest
def generate_test_description():
    """Run object-following workflow driver stack on hardware for DriverTest."""
    isaac_ros_manipulation_bringup_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'launch')
    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Default reference stack: Universal Robots + Robotiq 2F (ur_robotiq_gripper.yaml)
    override_params = {
        'workflow_type': 'OBJECT_FOLLOWING',
        'camera_type': 'REALSENSE',
        'headless': 'true',
        'robot_launch_file_path': os.path.join(
            get_package_share_directory('isaac_ros_manipulation_ur_driver_utils'),
            'launch', 'ur_robotiq_driver.launch.py'),
    }
    params.update(override_params)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 12.0
        # drivers.launch.py loads workflow from a YAML file; write merged test config so that
        # override_params (including robot_launch_file_path) are picked up.
        fd, merged_workflow = tempfile.mkstemp(
            suffix='.yaml', prefix='drivers_object_following_')
        os.close(fd)
        with open(merged_workflow, 'w', encoding='utf-8') as out:
            yaml.dump(params, out, default_flow_style=False, sort_keys=False)
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

    return DriverTest.generate_test_description(
        run_test=RUN_TEST,
        use_sim_time=False,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        enable_pick_and_place=False
    )
