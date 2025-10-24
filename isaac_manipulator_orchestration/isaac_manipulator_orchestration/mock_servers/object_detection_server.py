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

from isaac_manipulator_interfaces.action import GetObjects
from isaac_manipulator_interfaces.msg import ObjectInfo
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, Detection2D, ObjectHypothesis, ObjectHypothesisWithPose


class MockGetObjectsActionServer(Node):
    """A simple mock action server that provides object detection results."""

    def __init__(self):
        super().__init__('mock_get_objects_server')

        # Create a callback group for the action server
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # Create the action server with the callback group
        self._action_server = ActionServer(
            self,
            GetObjects,
            '/get_objects',
            self.execute_callback,
            callback_group=self._action_cb_group
        )

        self.get_logger().info('Mock GetObjects action server started')
        self.get_logger().info(
            f'Server is ready to accept requests at {self.get_namespace()}/get_objects')

    def _create_mock_object(self, object_id, class_id, center_x, center_y,
                            size_x, size_y, confidence_score, timestamp):
        """
        Create a mock ObjectInfo with detection data.

        Parameters
        ----------
        object_id : int
            Unique identifier for the object
        class_id : str
            Class identifier for the object type
        center_x : float
            X coordinate of bounding box center
        center_y : float
            Y coordinate of bounding box center
        size_x : float
            Width of bounding box
        size_y : float
            Height of bounding box
        confidence_score : float
            Detection confidence score (0.0 to 1.0)
        timestamp : builtin_interfaces.msg.Time
            Timestamp for the detection header

        Returns
        -------
        ObjectInfo
            Configured object with detection data

        """
        obj = ObjectInfo()
        obj.object_id = object_id

        # Set up 2D detection with bounding box
        obj.detection_2d = Detection2D()
        obj.detection_2d.header = Header()
        obj.detection_2d.header.stamp = timestamp
        obj.detection_2d.header.frame_id = 'camera'

        # Create bounding box
        obj.detection_2d.bbox = BoundingBox2D()
        obj.detection_2d.bbox.center.position.x = center_x
        obj.detection_2d.bbox.center.position.y = center_y
        obj.detection_2d.bbox.size_x = size_x
        obj.detection_2d.bbox.size_y = size_y

        # Add results with class ID and confidence score
        results = [ObjectHypothesisWithPose()]
        results[0].hypothesis = ObjectHypothesis()
        results[0].hypothesis.class_id = class_id
        results[0].hypothesis.score = confidence_score
        obj.detection_2d.results = results

        return obj

    def execute_callback(self, goal_handle):
        """
        Handle the object detection action request.

        This mock server generates fake object detection data for testing.

        Returns
        -------
        GetObjects.Result
            A result containing a list of detected objects

        """
        self.get_logger().info('Received object detection request')

        # Add a short sleep to simulate processing time (1 second)
        self.get_logger().info('Processing request...')
        time.sleep(1.0)
        self.get_logger().info('Finished processing after 1 second')

        # Create the result message
        result = GetObjects.Result()

        # Get current time for header stamps
        now = self.get_clock().now().to_msg()

        # Create standard mock objects for testing

        # Object 1: mac_and_cheese (class_id: '22')
        obj1 = self._create_mock_object(
            object_id=1,
            class_id='22',  # mac_and_cheese
            center_x=320.0,
            center_y=240.0,
            size_x=120.0,
            size_y=100.0,
            confidence_score=0.92,
            timestamp=now
        )
        result.objects.append(obj1)

        # Object 2: soup_can (class_id: '3')
        obj2 = self._create_mock_object(
            object_id=2,
            class_id='3',  # soup_can
            center_x=480.0,
            center_y=180.0,
            size_x=80.0,
            size_y=140.0,
            confidence_score=0.87,
            timestamp=now
        )
        result.objects.append(obj2)

        # Object 3: cereal_box (class_id: '5')
        obj3 = self._create_mock_object(
            object_id=3,
            class_id='5',  # cereal_box
            center_x=200.0,
            center_y=300.0,
            size_x=90.0,
            size_y=110.0,
            confidence_score=0.85,
            timestamp=now
        )
        result.objects.append(obj3)

        # Object 4: bottle (class_id: '8')
        obj4 = self._create_mock_object(
            object_id=4,
            class_id='8',  # bottle
            center_x=150.0,
            center_y=200.0,
            size_x=60.0,
            size_y=120.0,
            confidence_score=0.90,
            timestamp=now
        )
        result.objects.append(obj4)

        self.get_logger().info('About to succeed the goal handle')
        goal_handle.succeed()
        self.get_logger().info(
            f'Returning {len(result.objects)} detected objects')

        return result


def main(args=None):
    rclpy.init(args=args)

    # Create the action server node
    mock_server = MockGetObjectsActionServer()

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
