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
"""Record a ROS bag for calibration validation."""

from datetime import datetime
import os
import time

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_ros_test import (
    IsaacROSBaseTest
)
from launch.actions import ExecuteProcess, IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.parameter import Parameter


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'manual_on_robot'
ROS_BAG_OUTPUT_DIR = os.environ.get('ROS_BAG_OUTPUT_DIR', None)


class RosBagRecorderWaitNSecondsTest(IsaacROSBaseTest):
    """Record ROS bag for a given duration."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  wait_time_for_rosbag_recording_seconds: float):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._wait_time_for_rosbag_recording_seconds = wait_time_for_rosbag_recording_seconds
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_rosbag_recording(self):
        """Test that the ROS bag is recorded for the given duration."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        time.sleep(self._wait_time_for_rosbag_recording_seconds)

        self.node.get_logger().info(
            f'Successfully recorded ROS bag for'
            f' {self._wait_time_for_rosbag_recording_seconds} seconds'
        )


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    RosBagRecorderWaitNSecondsTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'test', 'include')
    sensor_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch', 'sensors')

    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Override params to be PICK and PLACE
    override_params = {
        'camera_type': 'REALSENSE',
        'num_cameras': '1',
        'enable_dnn_depth_in_realsense': 'true',
        'depth_type': 'ESS_FULL',
    }
    params.update(override_params)
    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:

        if ROS_BAG_OUTPUT_DIR is None:
            raise ValueError('ROS_BAG_OUTPUT_DIR environment variable is not set')

        if not os.path.exists(ROS_BAG_OUTPUT_DIR):
            os.makedirs(ROS_BAG_OUTPUT_DIR)

        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir,
                 '/manipulation_container.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [sensor_include_dir, '/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        timestamped_ros_bag_folder_path = os.path.join(
            ROS_BAG_OUTPUT_DIR,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        test_nodes.append(
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record', '--storage', 'mcap',
                    '--output', timestamped_ros_bag_folder_path,
                    '/rosout',
                    '/tf',
                    '/tf_static',
                    '/camera_1/color/camera_info',
                    '/camera_1/color/image_raw',
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

    return RosBagRecorderWaitNSecondsTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        use_sim_time=False,
        node_startup_delay=node_startup_delay,
        wait_time_for_rosbag_recording_seconds=60.0,
    )
