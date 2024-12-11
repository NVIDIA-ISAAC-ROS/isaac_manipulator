#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""This file receives an action call for open/close gripper and forwards request to the Isaac Sim
instance through ROS Topics, Isaac Sim supports ROS2 topics as a means of controlling the robot.
"""
import rclpy
from rclpy.node import Node
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor


MAXIMUM_FINGER_JOINT = 0.623


class IsaacSimGripperActionServer(Node):
    def __init__(self):
        super().__init__('isaac_sim_gripper_action_server')

        # Action server initialization
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publisher for joint states
        self.isaac_sim_gripper_control = self.create_publisher(
            JointState, '/isaac_gripper_state', 10)

        # Variable to hold the finger joint position
        self.finger_joint_pos = 0.0

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Extract the desired gripper position from the goal
        desired_position = goal_handle.request.command.position

        # Publish joint state
        self.publish_joint_state(desired_position)

        # Check if the position is valid for success
        goal_handle.succeed()

        # Create a result message
        result = GripperCommand.Result()
        result.position = desired_position
        result.reached_goal = True

        return result

    def publish_joint_state(self, finger_joint_pos: float):
        """Publishes the joint state to the ROS2 topic."""
        if finger_joint_pos > MAXIMUM_FINGER_JOINT:
            finger_joint_pos = MAXIMUM_FINGER_JOINT

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['finger_joint']
        msg.position = [finger_joint_pos]
        msg.velocity = []
        msg.effort = []

        self.isaac_sim_gripper_control.publish(msg)
        self.get_logger().info(
            f'Publishing topic to set gripper open status to {finger_joint_pos}')


def main(args=None):
    rclpy.init(args=args)

    pick_and_place_orchestrator = IsaacSimGripperActionServer()

    executor = MultiThreadedExecutor()
    executor.add_node(pick_and_place_orchestrator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pick_and_place_orchestrator.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    pick_and_place_orchestrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
