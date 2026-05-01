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

from isaac_ros_manipulation_interfaces.srv import AssignNameToObject
import rclpy
from rclpy.node import Node


class MockAssignNameServer(Node):
    """Mock service server for object frame name assignment for pose estimation."""

    def __init__(self):
        super().__init__('assign_name_server')
        self._service = self.create_service(
            AssignNameToObject,
            'assign_name_to_object',
            self.assign_name_to_object_callback
        )
        self.get_logger().info('Mock Assign Name Server started')
        self.get_logger().info(
            'Server is ready to accept name assignment requests at assign_name_to_object')

    def assign_name_to_object_callback(self, request, response):
        """Handle assign name to object service request."""
        self.get_logger().info(
            f'Received name assignment request: object_id={request.object_id}, '
            f'name="{request.name}"')

        # Always return success for any name assignment
        response.result = True
        self.get_logger().info(
            f'Successfully assigned name "{request.name}" to object {request.object_id}')

        return response


def main():
    rclpy.init()
    server = MockAssignNameServer()

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
