#!/usr/bin/env python3

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
