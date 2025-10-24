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

import time
from typing import List
import pytest
import rclpy

from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand


TIMEOUT = 10  # seconds


class IsaacROSGripperDriverTest(IsaacROSBaseTest):
    """Tests for the gripper driver action server and its output joint states."""

    def setup_clients(self):
        """Create client/sub/pub once and reuse across tests."""
        if hasattr(self, '_clients_ready') and self._clients_ready:
            return

        self._rcvd: List[JointState] = []

        # Subscriber for the gripper output state
        self._gripper_state_sub = self.node.create_subscription(
            JointState, '/isaac_gripper_state', lambda m: self._rcvd.append(m), 10
        )

        # Publisher to tick the driver's isaac_joint_states callback
        self._isaac_joint_states_pub = self.node.create_publisher(
            JointState, '/isaac_joint_states', 10
        )

        # Action client for the gripper command
        self._client = ActionClient(
            self.node, GripperCommand, '/robotiq_gripper_controller/gripper_cmd'
        )

        # Wait for server
        end = time.time() + TIMEOUT
        while time.time() < end:
            if self._client.wait_for_server(timeout_sec=0.1):
                self._clients_ready = True
                return
            rclpy.spin_once(self.node, timeout_sec=0.01)

        raise RuntimeError('GripperCommand action server not available')

    def expected_names(self):
        return [
            'finger_joint',
            'right_outer_knuckle_joint',
            'left_outer_finger_joint',
            'right_outer_finger_joint',
            'left_inner_finger_joint',
            'right_inner_finger_joint',
            'left_inner_finger_pad_joint',
            'right_inner_finger_pad_joint',
        ]

    def expected_positions(self, p: float):
        # Matches set_gripper_ctrl_target in the driver
        return [p, p, 0.0, 0.0, -p, -p, p, p]

    def wait_for_messages(self, min_count: int, timeout_sec: float = TIMEOUT):
        end = time.time() + timeout_sec
        while time.time() < end and len(self._rcvd) < min_count:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return len(self._rcvd) >= min_count

    def publish_tick(self, count: int = 1, delay_sec: float = 0.05):
        """Publish to /isaac_joint_states to trigger the driver's republish."""
        for _ in range(count):
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            # Content doesn't matter; callback only uses it as a tick.
            self._isaac_joint_states_pub.publish(msg)
            time.sleep(delay_sec)

    def send_goal(self, p: float):
        goal = GripperCommand.Goal()
        goal.command.position = p
        goal.command.max_effort = 0.0
        return self._client.send_goal_async(goal)

    def assert_latest_state(self, p: float):
        assert len(self._rcvd) > 0, 'No JointState messages received'
        msg = self._rcvd[-1]
        assert list(msg.name) == self.expected_names()
        exp_pos = self.expected_positions(p)
        assert len(msg.position) == len(exp_pos)
        self.node.get_logger().info(f'msg.position: {msg.position}')
        self.node.get_logger().info(f'exp_pos: {exp_pos}')
        for a, b in zip(msg.position, exp_pos):
            assert abs(a - b) < 1e-6
        assert list(msg.velocity) == []
        assert list(msg.effort) == []

    def test_gripper_command_publishes_expected_joint_state(self):
        """
        Send a GripperCommand goal and verify the published `/isaac_gripper_state`.

        Checks:
        - Topic publishes with expected joint name ordering
        - Positions match mapping logic of set_gripper_ctrl_target
        - Velocity and effort arrays are empty
        """
        self.setup_clients()
        self._rcvd.clear()

        p = 0.4
        send_future = self.send_goal(p)

        # Collect a few messages during execution loop
        assert self.wait_for_messages(min_count=3, timeout_sec=TIMEOUT), \
            'No JointState received during goal execution'
        self.assert_latest_state(p)

        # Ensure the goal is accepted and fetch result (best-effort)
        try:
            rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=2.0)
            if send_future.done() and send_future.result() is not None:
                goal_handle = send_future.result()
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=5.0)
        finally:
            pass

    def test_tick_replication_and_multiple_goals(self):
        """
        Verify driver republishes last target on joint state ticks and tracks multiple goals.

        Steps:
        - Send goal p1 and validate mapping during execution
        - Send goal p2 and validate mapping during execution
        - After completion, publish to /isaac_joint_states and verify the last p2 is re-published
        - Publish additional ticks and verify consistency without any new goals
        """
        self.setup_clients()
        self._rcvd.clear()

        # First goal
        p1 = 0.1
        self.send_goal(p1)
        assert self.wait_for_messages(min_count=2, timeout_sec=TIMEOUT), \
            'No JointState received for first goal'
        self.assert_latest_state(p1)

        # Second goal
        p2 = 0.55
        send_future_2 = self.send_goal(p2)
        assert self.wait_for_messages(min_count=10, timeout_sec=TIMEOUT), \
            'No JointState received for second goal'

        # Wait for the second goal to be accepted and attempt to wait for result
        try:
            rclpy.spin_until_future_complete(self.node, send_future_2, timeout_sec=2.0)
            if send_future_2.done() and send_future_2.result() is not None:
                goal_handle = send_future_2.result()
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=6.0)
        finally:
            pass

        # Clear to focus on republishing after execution via ticks
        self._rcvd.clear()

        # Trigger the driver's isaac_joint_states callback (not in action loop now)
        self.publish_tick(count=100, delay_sec=0.05)
        assert self.wait_for_messages(min_count=1, timeout_sec=TIMEOUT), \
            'No JointState received on tick-triggered republish'
        self.assert_latest_state(p2)

        # Additional ticks should keep publishing the last known position
        last_count = len(self._rcvd)
        self.publish_tick(count=2, delay_sec=0.05)
        assert self.wait_for_messages(min_count=last_count + 1, timeout_sec=TIMEOUT), \
            'No additional JointState received on subsequent ticks'
        self.assert_latest_state(p2)


@pytest.mark.rostest
def generate_test_description():
    """Launch the gripper driver node."""
    gripper_node = Node(
        name='gripper_driver',
        package='isaac_manipulator_isaac_sim_utils',
        executable='isaac_sim_gripper_driver.py',
        namespace=IsaacROSGripperDriverTest.generate_namespace(),
    )
    return IsaacROSGripperDriverTest.generate_test_description([gripper_node])
