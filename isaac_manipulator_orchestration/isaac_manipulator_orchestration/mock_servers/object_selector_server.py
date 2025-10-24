#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from isaac_manipulator_interfaces.action import GetSelectedObject
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class MockObjectSelectionServer(Node):
    """A mock action server that performs object selection."""

    def __init__(self):
        super().__init__('mock_get_selected_object_server')

        # Create a callback group for the action server
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the action server with the callback group
        self._action_server = ActionServer(
            self,
            GetSelectedObject,
            '/get_selected_object',
            self.execute_callback,
            callback_group=self._action_cb_group
        )

        self.get_logger().info('MockObjectSelectionServer action server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/get_selected_object')

    def execute_callback(self, goal_handle):
        """
        Handle the object selection action request.

        This mock server demonstrates different selection policies.

        Returns
        -------
        result object
            The object selection result

        """
        detections = goal_handle.request.detections.detections
        num_detections = len(detections)

        self.get_logger().info(
            f'Received request to select from {num_detections} detections')

        if num_detections == 0:
            self.get_logger().error('No detections provided to select from')
            goal_handle.abort()
            return GetSelectedObject.Result()

        # Create the result message
        result = GetSelectedObject.Result()

        # Select the first detection
        selected_detection = detections[0]

        result.selected_detection = selected_detection

        self.get_logger().info(
            f'Selected detection with id: {selected_detection.id}')

        self.get_logger().info('About to succeed the goal handle')
        goal_handle.succeed()

        return result


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    mock_server = MockObjectSelectionServer()

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(mock_server)

    try:
        mock_server.get_logger().info('Starting mock server. Press Ctrl+C to exit.')
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
