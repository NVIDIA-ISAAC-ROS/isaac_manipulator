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

from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene
import rclpy
from rclpy.node import Node


class MockPublishStaticPlanningSceneServer(Node):
    """Mock service server for publishing static planning scene."""

    def __init__(self):
        super().__init__('mock_publish_static_planning_scene_server')
        self._service = self.create_service(
            PublishStaticPlanningScene,
            'publish_static_planning_scene',
            self.publish_static_planning_scene_callback
        )
        self.get_logger().info('Mock Publish Static Planning Scene Server started')
        self.get_logger().info(
            'Server is ready to accept requests at publish_static_planning_scene')

    def publish_static_planning_scene_callback(self, request, response):
        """Handle publish static planning scene service request."""
        scene_file_path = request.scene_file_path

        self.get_logger().info(
            f'Received publish static planning scene request with '
            f'scene_file_path: "{scene_file_path}"')

        # Check if scene file path is provided
        if scene_file_path and scene_file_path.strip():
            # Simulate successful scene publishing with provided file
            response.success = True
            response.status = 0  # Success status
            response.message = (f'Successfully published static planning scene '
                                f'from file: {scene_file_path}')
            self.get_logger().info(response.message)
        else:
            # Simulate case where no scene file is provided
            # This matches the behavior described in the original behavior code
            response.success = False
            response.status = 1  # Warning status - no scene file provided
            response.message = 'No static planning scene file provided'
            self.get_logger().warning(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    server = MockPublishStaticPlanningSceneServer()

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
