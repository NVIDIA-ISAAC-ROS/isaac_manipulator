#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time

from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class MockGripperCommandActionServer(Node):
    """A simple mock action server that simulates gripper commands."""

    def __init__(self):
        super().__init__('mock_gripper_command_server')

        # Create a callback group for the action server
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the action server with the callback group
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd',
            self.execute_callback,
            callback_group=self._action_cb_group
        )

        self.get_logger().info('Mock GripperCommand action server started')
        self.get_logger().info(
            f'Server is ready to accept requests at '
            f'{self.get_namespace()}/robotiq_gripper_controller/gripper_cmd')

    def execute_callback(self, goal_handle):
        """
        Handle the gripper command action request.

        This mock server simulates gripper motion and returns position/result.

        Returns
        -------
        GripperCommand.Result
            The gripper command result

        """
        position = goal_handle.request.command.position
        max_effort = goal_handle.request.command.max_effort

        self.get_logger().info(
            f'Received gripper command: position={position}, max_effort={max_effort}')

        # Add a short sleep to simulate processing time (2 second)
        self.get_logger().info('Processing gripper command...')
        time.sleep(2.0)
        self.get_logger().info('Finished processing after 2 second')

        # Create the result message
        result = GripperCommand.Result()

        # In this mock server, we'll always succeed and set the position to the requested position
        # In a real system, the gripper might not reach exactly the goal position
        result.position = position
        result.effort = max_effort * 0.8  # Simulate using 80% of max effort
        result.stalled = False  # Indicate gripper didn't stall
        result.reached_goal = True  # Indicate gripper reached the goal

        goal_handle.succeed()
        self.get_logger().info(
            f'Gripper command executed successfully. Position: {result.position}')

        return result


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    mock_server = MockGripperCommandActionServer()

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(mock_server)

    try:
        mock_server.get_logger().info(
            'Starting mock gripper command server. Press Ctrl+C to exit.')
        executor.spin()
    except KeyboardInterrupt:
        mock_server.get_logger().info('Server stopped by user')
    finally:
        # Clean up
        executor.shutdown()
        mock_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
