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

import os
import pathlib
import time

from cv_bridge import CvBridge
from isaac_manipulator_interfaces.action import SegmentAnything
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, Point2D


TIMEOUT = 10  # seconds
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


@pytest.mark.rostest
def generate_test_description():
    segment_anything_node = ComposableNode(
        name='segment_anything_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::SegmentAnythingServer',
        namespace=IsaacROSSegmentAnythingTest.generate_namespace(),
        parameters=[{
            'is_sam2': False,
            'action_name': 'segment_anything',
            'in_img_topic_name': 'image_color',
            'out_img_topic_name': 'segment_anything_server/image_color',
            'in_camera_info_topic_name': 'camera_info',
            'out_camera_info_topic_name': 'segment_anything_server/camera_info',
            'in_segmentation_mask_topic_name': 'segmentation_mask',
            'in_detections_topic_name': 'detections',
            'out_point_topic_name': 'segment_anything_server/point',
            'input_qos': 'SENSOR_DATA',
            'result_and_output_qos': 'DEFAULT',
            'log_level': 'DEBUG'
        }]
    )

    return IsaacROSSegmentAnythingTest.generate_test_description([
        ComposableNodeContainer(
            name='segment_anything_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[segment_anything_node],
            namespace=IsaacROSSegmentAnythingTest.generate_namespace(),
            output='screen',
        )
    ])


class IsaacROSSegmentAnythingTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_segment_anything_action(self):
        """
        Test SegmentAnything action with point hint.

        1. Publish a dummy image and camera info
        2. Publish dummy segmentation mask and detections
        3. Send SegmentAnything action with an initial hint point
        4. Verify output detection and segmentation mask
        """
        received_messages = {}

        # Generate namespace lookups for all topics
        self.generate_namespace_lookup([
            'image_color',
            'camera_info',
            'segmentation_mask',
            'detections',
            'segment_anything_server/point',
            'segment_anything_server/image_color',
        ])

        # Create publishers
        image_pub = self.node.create_publisher(
            Image, self.namespaces['image_color'],
            qos_profile=10)

        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'],
            qos_profile=10)

        segmentation_mask_pub = self.node.create_publisher(
            Image, self.namespaces['segmentation_mask'],
            qos_profile=10)

        detections_pub = self.node.create_publisher(
            Detection2DArray, self.namespaces['detections'],
            qos_profile=10)

        # Create subscribers
        subs = self.create_logging_subscribers(
            [
                ('segment_anything_server/point', Point2D),
                ('segment_anything_server/image_color', Image),
            ], received_messages,
            accept_multiple_messages=True,
            qos_profile=10
        )

        # Create action client
        action_client = ActionClient(
            self.node,
            SegmentAnything,
            f'{self.generate_namespace()}/segment_anything',
            callback_group=ReentrantCallbackGroup())

        try:
            # Wait for action server
            self.assertTrue(
                action_client.wait_for_server(timeout_sec=TIMEOUT),
                'Action server not available')

            # Create and publish dummy image
            cv_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
            image_msg = CvBridge().cv2_to_imgmsg(cv_image)
            image_msg.header.stamp = self.node.get_clock().now().to_msg()

            # Create and publish dummy camera info
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = image_msg.header.stamp

            # Create and publish dummy segmentation mask
            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            mask_msg = CvBridge().cv2_to_imgmsg(mask)
            mask_msg.header.stamp = image_msg.header.stamp

            # Create and publish dummy detections
            detections_msg = Detection2DArray()
            # Add dummy detection
            detection_msg = Detection2D()
            detection_msg.bbox.center.position.x = IMAGE_WIDTH/2
            detection_msg.bbox.center.position.y = IMAGE_HEIGHT/2
            detection_msg.bbox.size_x = 10.0
            detection_msg.bbox.size_y = 10.0
            detections_msg.detections.append(detection_msg)
            detections_msg.header.stamp = image_msg.header.stamp

            # Send action goal with initial hint
            goal_msg = SegmentAnything.Goal()
            goal_msg.initial_hint_point = Point2D(x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2)
            goal_msg.use_point_hint = True

            future = action_client.send_goal_async(goal_msg)
            start_time = time.time()
            while not future.done() and time.time() - start_time < TIMEOUT:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)

            self.node.get_logger().info(f'Goal handle accepted in'
                                        f'{time.time() - start_time} seconds')

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            # Wait for result
            future = goal_handle.get_result_async()
            while not future.done() and time.time() - start_time < TIMEOUT:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)

            result = future.result()
            self.assertIsNotNone(result)
            # Get the image out of the result and compare it to the original segmentation mask
            image_result = result.result.segmentation_mask
            image_result_cv = CvBridge().imgmsg_to_cv2(image_result)
            np.testing.assert_array_equal(image_result_cv, mask)
            # Wait for output messages
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if all(len(received_messages[key]) > 0 for key in received_messages.keys()):
                    break
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)

            # self.node.get_logger().info(f'Received results: {result}')
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                # Only break if all messages have been received, list of non 0 length
                if all(len(received_messages[key]) > 0 for key in received_messages.keys()):
                    break

            # Verify we received all messages
            self.node.get_logger().info(f'Received point: {received_messages.keys()}')
            for key in received_messages.keys():
                self.node.get_logger().info(f'Received {key}: {len(received_messages[key])}')
            self.assertIn('segment_anything_server/point', received_messages)
            self.assertIn('segment_anything_server/image_color', received_messages)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
            self.node.destroy_publisher(segmentation_mask_pub)
            self.node.destroy_publisher(detections_pub)
            action_client.destroy()
