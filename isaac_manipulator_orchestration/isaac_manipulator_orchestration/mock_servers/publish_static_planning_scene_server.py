#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
