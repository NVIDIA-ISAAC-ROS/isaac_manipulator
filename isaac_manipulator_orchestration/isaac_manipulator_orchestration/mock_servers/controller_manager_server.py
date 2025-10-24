#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
        response.message = 'Controllers switched successfully'

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
