#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This file receives an action call for open/close gripper.

It forwards request to the Isaac Sim instance through ROS topics. Isaac Sim supports ROS 2 topics
as a means of controlling the robot. It adds a logic to make sure all the finger joints of the
gripper stay in desired position to account for uncertainty in physics simulation.
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

# TODO: Make num seeds as 1.
MAXIMUM_FINGER_JOINT = 0.623
NUM_PUBLISH_ITERATIONS = 40


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

        # Create subscriber for isaac_joint_states topic
        self.isaac_joint_states_sub = self.create_subscription(
            JointState, '/isaac_joint_states', self.isaac_joint_states_callback, 10)

        # Variable to hold the finger joint position
        self.finger_joint_pos = 0.0
        # Always perform gripper control to keep gripper in sync.
        self.action_callback_happening = False

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request for gripper')
        return GoalResponse.ACCEPT

    def isaac_joint_states_callback(self, msg: JointState):
        """
        Ticks to publish the joint state to Isaac Sim.

        If the action callback is happening, then we publish cacched joint state. If not
        then, do nothing. This is to make sure gripper fingers dont start to twicth away during arm
        motion.

        Args
        ----
            msg (JointState): The joint state message from Isaac Sim

        Returns
        -------
            None

        """
        # Publish the topic back the Isaac Sim to control the robot to make sure
        # It follows the target positions that were last set.
        if not self.action_callback_happening:
            self.publish_joint_state(self.finger_joint_pos)

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal for gripper')
        self.action_callback_happening = True

        desired_position = goal_handle.request.command.position
        self.finger_joint_pos = desired_position

        result = GripperCommand.Result()

        for _ in range(NUM_PUBLISH_ITERATIONS):
            if goal_handle.status in (
                GoalStatus.STATUS_CANCELING,
                GoalStatus.STATUS_CANCELED
            ):
                self.get_logger().info('Gripper goal cancelled')
                goal_handle.canceled()
                result.reached_goal = False
                return result

            self.publish_joint_state(self.finger_joint_pos)
            # We need to put time.sleep here as rclpy spin cause this node to get blocked.
            time.sleep(0.1)

        goal_handle.succeed()

        result.position = desired_position
        result.reached_goal = True

        self.action_callback_happening = False
        self.get_logger().info('Gripper goal executed')

        return result

    def set_gripper_ctrl_target(self, finger_joint_joint_angle: float):
        """
        Set the gripper control target.

        This sets all the positions of the robotiq gripper in a specific way keeping in mind
        what joints are mimic joint and which are inverse mimic joints.

        Args
        ----
            finger_joint_joint_angle (float): The position of the finger joint,
            capped at max value.

        Returns
        -------
            list: The gripper control target.

        """
        gripper_dof_target = [finger_joint_joint_angle] * 8
        gripper_dof_target[2:4] = [0.0] * 2
        gripper_dof_target[4] *= -1  # ['left_inner_finger_joint']
        gripper_dof_target[5] *= -1  # ['right_inner_finger_joint']

        return gripper_dof_target

    def publish_joint_state(self, finger_joint_pos: float) -> None:
        """
        Publish the joint state to the ROS 2 topic.

        Args
        ----
            finger_joint_pos (float): The position of the finger joint, capped at max value.

        Returns
        -------
            None

        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'finger_joint',

            'right_outer_knuckle_joint',

            'left_outer_finger_joint',
            'right_outer_finger_joint',

            'left_inner_finger_joint',
            'right_inner_finger_joint',

            'left_inner_finger_pad_joint',
            'right_inner_finger_pad_joint',
        ]
        msg.position = self.set_gripper_ctrl_target(finger_joint_pos)
        msg.velocity = []
        msg.effort = []

        self.isaac_sim_gripper_control.publish(msg)


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
