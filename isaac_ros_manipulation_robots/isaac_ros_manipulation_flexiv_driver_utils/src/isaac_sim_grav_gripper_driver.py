#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Gripper action server and command combiner for Flexiv Grav gripper in Isaac Sim.

Subscribes to arm-only commands from ros2_control on /rizon_arm_command,
appends the current gripper joint positions, and publishes a single combined
JointState to /rizon_sim_command.  This avoids jitter from two separate
publishers sending partial joint updates to Isaac Sim.

Also exposes a GripperCommand action server so the behavior tree can
open/close the gripper.
"""
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

NUM_PUBLISH_ITERATIONS = 40

GRAV_JOINT_NAMES = [
    'finger_joint',
    'left_inner_knuckle_joint',
    'right_inner_knuckle_joint',
    'right_outer_knuckle_joint',
    'left_outer_finger_joint',
    'right_outer_finger_joint',
]


class IsaacSimGravGripperActionServer(Node):

    def __init__(self):
        super().__init__('isaac_sim_grav_gripper_action_server')

        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/flexiv_gripper_node/gripper_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._combined_pub = self.create_publisher(
            JointState, '/rizon_sim_command', 10)

        self._arm_cmd_sub = self.create_subscription(
            JointState, '/rizon_arm_command',
            self._arm_command_callback, 10)

        self._finger_joint_pos = 0.0
        self._action_in_progress = False

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request for Grav gripper')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _arm_command_callback(self, msg: JointState):
        """Combine arm commands with current gripper state and publish."""
        combined = JointState()
        combined.header = Header()
        combined.header.stamp = self.get_clock().now().to_msg()
        combined.name = list(msg.name) + list(GRAV_JOINT_NAMES)
        combined.position = list(msg.position) + self._build_gripper_positions(
            self._finger_joint_pos)
        combined.velocity = []
        combined.effort = []
        self._combined_pub.publish(combined)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal for Grav gripper')
        self._action_in_progress = True

        desired_position = goal_handle.request.command.position
        self._finger_joint_pos = desired_position

        result = GripperCommand.Result()

        for _ in range(NUM_PUBLISH_ITERATIONS):
            if goal_handle.status in (
                GoalStatus.STATUS_CANCELING,
                GoalStatus.STATUS_CANCELED,
            ):
                self.get_logger().info('Gripper goal cancelled')

                result.reached_goal = False
                goal_handle.canceled(result)
                return result

            time.sleep(0.1)

        result.position = desired_position
        result.reached_goal = True

        self._action_in_progress = False
        self.get_logger().info('Gripper goal executed')
        goal_handle.succeed(result)
        return result

    def _build_gripper_positions(self, finger_angle: float):
        """Map finger_joint angle to all 6 Grav gripper joints."""
        return [
            finger_angle,           # finger_joint
            finger_angle,           # left_inner_knuckle_joint
            finger_angle,           # right_inner_knuckle_joint
            finger_angle,           # right_outer_knuckle_joint
            -finger_angle,          # left_outer_finger_joint (negated)
            -finger_angle,          # right_outer_finger_joint (negated)
        ]


def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimGravGripperActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
