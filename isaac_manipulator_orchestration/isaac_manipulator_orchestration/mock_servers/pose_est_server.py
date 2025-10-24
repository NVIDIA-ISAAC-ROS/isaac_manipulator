#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import random
import time

from geometry_msgs.msg import Pose, TransformStamped
from isaac_manipulator_interfaces.action import GetObjectPose
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
import tf2_ros


class MockGetObjectPoseActionServer(Node):
    """A simple mock action server that provides object poses and publishes transforms."""

    def __init__(self):
        super().__init__('mock_get_object_pose_server')

        # Create a callback group for the action server
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the action server with the callback group
        self._action_server = ActionServer(
            self,
            GetObjectPose,
            '/get_object_pose',
            self.execute_callback,
            callback_group=self._action_cb_group
        )

        # Create static transform broadcaster
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Track which object IDs have already been published as static transforms
        self._published_object_ids = set()

        # Publish the static transform from base_link to camera_1_optical_frame
        self._publish_transform(
            target_frame='camera_1_color_optical_frame',
            source_frame='base_link')

        self.get_logger().info('Mock GetObjectPose action server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/get_object_pose')

    def execute_callback(self, goal_handle):
        """
        Handle the pose estimation action request.

        This mock server generates fake pose data for the requested object.

        Returns
        -------
        result object
            The pose estimation result

        """
        object_id = goal_handle.request.object_id
        self.get_logger().info(
            f'Received request for object_id: {object_id}')

        # Add a short sleep to simulate processing time (1 second)
        self.get_logger().info('Processing request...')
        time.sleep(1.0)
        self.get_logger().info('Finished processing after 1 second')

        # Create the result message
        result = GetObjectPose.Result()

        # Create a simple fixed pose
        object_pose = Pose()

        # Random position with some variation (in meters)
        object_pose.position.x = 0.4 + random.uniform(-1.0, 1.0)
        object_pose.position.y = 0.0 + random.uniform(-1.0, 1.0)
        object_pose.position.z = -0.25

        # Simple fixed orientation (upright)
        object_pose.orientation.x = 0.0
        object_pose.orientation.y = 0.0
        object_pose.orientation.z = 0.0
        object_pose.orientation.w = 1.0

        # Set the result
        result.object_pose = object_pose

        # Publish the transform for this object
        self._publish_object_transform(object_id, object_pose)

        self.get_logger().info('About to succeed the goal handle')
        goal_handle.succeed()
        self.get_logger().info(
            f'Returning fixed pose for object {object_id}')

        return result

    def _publish_object_transform(self, object_id: int, pose: Pose):
        """Publish a static transform for the object."""
        if object_id not in self._published_object_ids:
            transform_stamped = TransformStamped()
            transform_stamped.header = Header()
            transform_stamped.header.stamp = self.get_clock().now().to_msg()
            transform_stamped.header.frame_id = 'camera_1_color_optical_frame'  # Parent frame
            # Child frame
            transform_stamped.child_frame_id = f'object_{object_id}'

            # Set the transform from the pose
            transform_stamped.transform.translation.x = pose.position.x
            transform_stamped.transform.translation.y = pose.position.y
            transform_stamped.transform.translation.z = pose.position.z
            transform_stamped.transform.rotation.x = pose.orientation.x
            transform_stamped.transform.rotation.y = pose.orientation.y
            transform_stamped.transform.rotation.z = pose.orientation.z
            transform_stamped.transform.rotation.w = pose.orientation.w

            # Publish the static transform
            self.tf_static_broadcaster.sendTransform(transform_stamped)

            self.get_logger().info(
                f'Published static transform for object_{object_id}')

            # Add the object ID to the set of published object IDs
            self._published_object_ids.add(object_id)

    def _publish_transform(self, target_frame: str, source_frame: str):
        """Publish a static transform between target_frame and source_frame."""
        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = source_frame  # Parent frame
        transform_stamped.child_frame_id = target_frame  # Child frame

        # Set a reasonable camera position relative to base_link
        transform_stamped.transform.translation.x = -0.5
        transform_stamped.transform.translation.y = -1.0
        transform_stamped.transform.translation.z = 0.3

        transform_stamped.transform.rotation.x = 0.0
        transform_stamped.transform.rotation.y = 0.0
        transform_stamped.transform.rotation.z = 0.7071067811865476
        transform_stamped.transform.rotation.w = 0.7071067811865476

        # Publish the static transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)

        self.get_logger().info(
            f'Published static transform from {source_frame} to {target_frame}')


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    mock_server = MockGetObjectPoseActionServer()

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(mock_server)

    try:
        mock_server.get_logger().info('Starting mock server. Press Ctrl+C to exit.')
        executor.spin()  # Use the executor instead of rclpy.spin()
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
