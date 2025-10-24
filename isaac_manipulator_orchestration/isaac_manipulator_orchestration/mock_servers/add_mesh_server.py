#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_interfaces.srv import AddMeshToObject
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class MockAddMeshToObjectServer(Node):
    """A mock server that handles requests to add mesh files to objects."""

    def __init__(self):
        super().__init__('mock_add_mesh_to_object_server')

        # Create a callback group for the service
        self._service_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the service with the callback group
        self._service = self.create_service(
            AddMeshToObject,
            'add_mesh_to_object',
            self.handle_add_mesh_request,
            callback_group=self._service_cb_group
        )

        # Dictionary to store object_id to mesh_file_path mappings
        self._object_meshes = {}

        self.get_logger().info('Mock AddMeshToObject service server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/add_mesh_to_object')

    def handle_add_mesh_request(self, request, response):
        """
        Handle the add mesh to object service request.

        Returns
        -------
        response object
            The populated response object

        """
        self.get_logger().info(
            f'Received request to add meshes to {len(request.object_ids)} objects')

        # Initialize response
        response.success = True
        response.message = 'All mesh assignments successful'
        response.failed_ids = []

        # Check if lists have the same length
        if len(request.object_ids) != len(request.mesh_file_paths):
            response.success = False
            response.message = "Error: Number of object IDs and mesh file paths don't match"
            response.failed_ids = request.object_ids
            return response

        # Process each object-mesh pair
        for i, obj_id in enumerate(request.object_ids):
            mesh_path = request.mesh_file_paths[i]

            try:
                # In a real implementation, this would verify the mesh file exists
                # and associate it with the object in a physics/visualization system
                self._object_meshes[obj_id] = mesh_path
                self.get_logger().info(
                    f'Added mesh {mesh_path} to object {obj_id}')
            except Exception as e:
                self.get_logger().error(
                    f'Failed to add mesh to object {obj_id}: {str(e)}')
                response.success = False
                response.failed_ids.append(obj_id)

        # Update message if there were failures
        if len(response.failed_ids) > 0:
            response.message = f'Failed to assign meshes to {len(response.failed_ids)} objects'

        # Log the current mappings
        self.get_logger().info(
            f'Current object-mesh mappings: {self._object_meshes}')

        return response


def main(args=None):
    rclpy.init(args=args)

    # Create the service server node
    mock_server = MockAddMeshToObjectServer()

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
