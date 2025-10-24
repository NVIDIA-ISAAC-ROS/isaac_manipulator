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
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from isaac_manipulator_interfaces.action import Insert
import isaac_manipulator_ros_python_utils as manipulator_utils
from isaac_ros_test import IsaacROSBaseTest
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from tf2_ros import TransformBroadcaster, TransformStamped


# Once we have a way to run this test on CI, we can enable this. Currently user needs to make sure
# to store the model file  in the path that is specified in the test config file.
RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
EXPECTED_FPS = 58.0


@pytest.mark.rostest
def generate_test_description():
    """Generate test description for testing gear assembly inference policy."""
    isaac_manipulator_ur_dnn_policy_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_ur_dnn_policy'),
        'test',
        'include'
    )
    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_manipulator_ur_dnn_policy'),
        'test',
        'config',
        'gear_assembly_test_config.yaml'
    )

    params = manipulator_utils.load_yaml_params(test_yaml_config)
    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 10.0
    test_nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_ur_dnn_policy_include_dir, '/gear_assembly.launch.py']),
        launch_arguments={key: str(value) for key, value in params.items()}.items()))

    if not RUN_TEST:
        # Makes the test pass if we do not want to run on CI
        test_nodes = [
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_transform_publisher',
                output='screen',
                arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
            )
        ]

    return GearAssemblyInferencePolicyTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,  # 10 seconds
        ros_bag_path=params['gear_assembly_ros_bag_folder_path']
    )


class GearAssemblyInferencePolicyTest(IsaacROSBaseTest):
    """Test for gear assembly inference policy."""

    _run_test = True
    _run_test_duration = 300.0
    _ros_bag_path: str = ''

    def test_gear_assembly_inference_policy(self):
        """Test gear assembly inference policy."""
        if not self._run_test:
            return
        # Create publishers
        joint_state_pub = self.node.create_publisher(
            JointState,
            '/gear_assembly/joint_states',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10)
        )

        time.sleep(10)  # Give policy time to load

        # Create TF broadcaster for completion detection
        self.tf_broadcaster = TransformBroadcaster(self.node)

        # Create action client for insert policy
        self.action_client = ActionClient(
            self.node,
            Insert,
            '/gear_assembly/insert_policy')

        # Wait for action server
        self.assertTrue(self.action_client.wait_for_server(timeout_sec=5.0))

        # Create dictionaries to store received messages
        received_messages = {}

        # Create subscribers for target states
        self.create_logging_subscribers(
            [('/gear_assembly/target_joint_states', JointState),
             ('/gear_assembly/target_tcp_state', PoseStamped),
             ('/gear_assembly/gear_insertion_status', Int8)],
            received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10)
        )

        # Create timer callback at 60Hz
        def timer_callback():
            current_time = self.node.get_clock().now()

            # Publish joint states
            joint_state = JointState()
            joint_state.header.stamp = current_time.to_msg()
            joint_state.name = ['finger_joint', 'shoulder_pan_joint',
                                'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            joint_state.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            joint_state_pub.publish(joint_state)

        # Create timer
        self.node.create_timer(1/60.0, timer_callback)

        action_call_sent = False
        check_if_insertion_started = False
        check_if_insertion_completed = False
        start_time = time.time()
        while time.time() - start_time < self._run_test_duration:
            rclpy.spin_once(self.node, timeout_sec=0.1)

            if not action_call_sent:
                # Send action goal to insertion action server
                action_call_sent = True
                goal = Insert.Goal()
                goal.goal_pose = PoseStamped()
                goal.goal_pose.header.stamp = self.node.get_clock().now().to_msg()
                goal.goal_pose.header.frame_id = 'base_link'  # This is base or base link.
                goal.goal_pose.pose.position.x = 0.5
                goal.goal_pose.pose.position.y = 0.0
                goal.goal_pose.pose.position.z = 0.5
                goal.goal_pose.pose.orientation.w = 1.0
                goal.goal_pose.pose.orientation.x = 0.0
                goal.goal_pose.pose.orientation.y = 0.0
                goal.goal_pose.pose.orientation.z = 0.0
                self.action_client.send_goal_async(goal)
                continue

            if not check_if_insertion_started:
                # Check if insertion has started. Look into insertion_status topic.
                insertion_status = received_messages.get(
                    '/gear_assembly/gear_insertion_status', [])
                self.node.get_logger().info(f'Insertion status: {insertion_status}')
                if (
                    len(insertion_status) > 0 and
                    insertion_status[-1].data == manipulator_utils.InsertionState.INSERTING.value
                ):
                    self.node.get_logger().info('Insertion has started.')
                    check_if_insertion_started = True
                    continue

            if check_if_insertion_started:
                # Start publishing TFs that slowly converge to the target pose.
                # Send a TF of end effector getting closer to target pose.
                current_time = self.node.get_clock().now()

                # Calculate distance based on time (gradually converge)
                elapsed = time.time() - (start_time + 5.0)  # Start convergence after 5s
                distance = max(0.001, 0.05 - (elapsed * 0.01))  # Linear convergence

                # Create and publish transform
                t = TransformStamped()
                t.header.stamp = current_time.to_msg()
                t.header.frame_id = 'goal'
                t.child_frame_id = 'end_effector'
                t.transform.translation.x = distance
                t.transform.translation.y = distance
                t.transform.translation.z = distance
                t.transform.rotation.w = 1.0

                # Publish transform
                self.tf_broadcaster.sendTransform(t)

            if not check_if_insertion_completed:
                # Check if insertion has completed. Look into insertion_status topic.
                insertion_status = received_messages.get(
                    '/gear_assembly/gear_insertion_status', [])
                if insertion_status and (insertion_status[-1].data ==
                                         manipulator_utils.InsertionState.COMPLETED.value):
                    self.node.get_logger().info('Insertion has completed.')
                    check_if_insertion_completed = True
                    break

        # Check if we got target messages at 60 hz
        target_joint_state_messages = received_messages.get(
            '/gear_assembly/target_joint_states', [])
        number_of_target_joint_state_messages = len(target_joint_state_messages)

        # Calculate duration between first and last message for more accurate Hz
        if number_of_target_joint_state_messages >= 2:
            first_msg_time = rclpy.time.Time.from_msg(
                target_joint_state_messages[0].header.stamp).nanoseconds / 1e9
            last_msg_time = rclpy.time.Time.from_msg(
                target_joint_state_messages[-1].header.stamp).nanoseconds / 1e9
            actual_duration = last_msg_time - first_msg_time
            output_hz = (number_of_target_joint_state_messages - 1) / actual_duration
        else:
            output_hz = 0

        self.node.get_logger().info(
            f'Received {number_of_target_joint_state_messages} '
            f'target joint state messages')

        self.node.get_logger().info(
            f'Output Hz: {output_hz} (based on actual message timestamps)')

        assert check_if_insertion_completed, 'Insertion did not complete.'
        assert output_hz > EXPECTED_FPS, f'Output Hz is less than {EXPECTED_FPS}: {output_hz}'

        # Clean up rosbag
        if os.path.exists(self._ros_bag_path):
            os.remove(self._ros_bag_path)

    @classmethod
    def generate_test_description(cls, run_test, nodes, node_startup_delay,
                                  ros_bag_path):
        """Generate test description for gear assembly inference policy."""
        cls._run_test = run_test
        cls._ros_bag_path = ros_bag_path

        return IsaacROSBaseTest.generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )
