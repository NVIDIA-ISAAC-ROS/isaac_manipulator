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

from isaac_ros_cumotion_interfaces.action import AttachObject
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class MockAttachObjectActionServer(Node):
    """A simple mock action server that simulates attaching and detaching objects to the robot."""

    def __init__(self):
        super().__init__('mock_attach_object_server')

        # Create a callback group for the action server
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the action server with the callback group
        self._action_server = ActionServer(
            self,
            AttachObject,
            'attach_object',
            self.execute_callback,
            callback_group=self._action_cb_group
        )

        # Keep track of whether an object is currently attached
        self._object_attached = False

        self.get_logger().info('Mock AttachObject action server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/attach_object')

    def execute_callback(self, goal_handle):
        """
        Handle the attach object action request.

        This mock simulates object attachment/detachment by returning success.

        Returns
        -------
        AttachObject.Result
            The result of the attach/detach operation

        """
        attach = goal_handle.request.attach_object

        operation = 'attach' if attach else 'detach'
        self.get_logger().info(f'Received request to {operation} object')

        if attach:
            # If an object config is provided in the request, log it

            fallback_radius = getattr(
                goal_handle.request, 'fallback_radius', 0.15)

            if hasattr(goal_handle.request, 'object_config') and goal_handle.request.object_config:
                self.get_logger().info(
                    f'Object config provided. fallback_radius={fallback_radius}')
            else:
                self.get_logger().info(
                    f'No object config provided, using fallback radius: {fallback_radius}')

        # Add a short sleep to simulate processing time (1 second)
        self.get_logger().info(f'Processing {operation} object request...')
        time.sleep(2.0)
        self.get_logger().info('Finished processing after 2 seconds')

        # Create the result message
        result = AttachObject.Result()

        # Check if the operation is valid based on current state
        if attach and self._object_attached:
            result.outcome = 'ERROR: Object already attached'
            goal_handle.abort()
            self.get_logger().error(result.outcome)
        elif not attach and not self._object_attached:
            result.outcome = 'ERROR: No object attached to detach'
            goal_handle.abort()
            self.get_logger().error(result.outcome)
        else:
            # Operation is valid, update state and return success
            self._object_attached = attach
            result.outcome = f'SUCCESS: Object {operation}ed successfully'
            goal_handle.succeed()
            self.get_logger().info(result.outcome)

        return result


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    mock_server = MockAttachObjectActionServer()

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(mock_server)

    try:
        mock_server.get_logger().info(
            'Starting mock attach object server. Press Ctrl+C to exit.')
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
