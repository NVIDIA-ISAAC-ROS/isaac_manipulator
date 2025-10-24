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

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node


class MockExecuteTrajectoryServer(Node):
    """Simple mock server for execute_trajectory action server."""

    def __init__(self):
        super().__init__('mock_execute_trajectory_server')
        self._action_server = ActionServer(
            self,
            ExecuteTrajectory,
            'execute_trajectory',
            self.execute_callback
        )
        self.get_logger().info('Mock Execute Trajectory Server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/execute_trajectory')

    def execute_callback(self, goal_handle):
        """Execute the trajectory action."""
        self.get_logger().info('Received execute trajectory request')

        # Simple simulation - just wait a bit
        time.sleep(3.0)

        # Always return success
        result = ExecuteTrajectory.Result()
        result.error_code = MoveItErrorCodes()
        result.error_code.val = MoveItErrorCodes.SUCCESS

        goal_handle.succeed()
        return result


def main():
    rclpy.init()
    server = MockExecuteTrajectoryServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Server stopped by user')
    finally:
        server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
