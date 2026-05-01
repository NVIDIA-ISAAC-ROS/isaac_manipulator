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

from controller_manager_msgs.srv import SwitchController
import rclpy
from rclpy.node import Node


class MockControllerManagerServer(Node):
    """Mock controller manager server for testing switch_controllers behavior."""

    def __init__(self):
        super().__init__('mock_controller_manager_server')

        # Create the switch_controller service
        self.srv = self.create_service(
            SwitchController,
            '/controller_manager/switch_controller',
            self.switch_controller_callback
        )

        self.get_logger().info('Mock Controller Manager Server started')

    def switch_controller_callback(self, request, response):
        """Handle switch controller requests."""
        self.get_logger().info(
            f'Switch controller request: activate={request.activate_controllers}, '
            f'deactivate={request.deactivate_controllers}'
        )

        # Always return success for simplicity
        response.ok = True

        return response


def main(args=None):
    rclpy.init(args=args)
    mock_server = MockControllerManagerServer()

    try:
        rclpy.spin(mock_server)
    except KeyboardInterrupt:
        mock_server.get_logger().info('Server stopped by user')
    finally:
        mock_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
