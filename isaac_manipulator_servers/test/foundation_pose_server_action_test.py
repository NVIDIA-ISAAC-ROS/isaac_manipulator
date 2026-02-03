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
from isaac_manipulator_interfaces.action import EstimatePoseFoundationPose
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection3D, Detection3DArray, ObjectHypothesisWithPose


TIMEOUT = 10  # seconds
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class MockFoundationPoseNode(Node):
    """
    Mock FoundationPoseNode.

    This node is used to test if the parameters are properly set
    when the foundation_pose_server_node sets the parameters.
    """

    def __init__(self):
        super().__init__('mock_foundation_pose_node')
        self.declare_parameter('mesh_file_path', '')
        self.declare_parameter('tf_frame_name', '')


@pytest.mark.rostest
def generate_test_description():
    foundation_pose_server_node = ComposableNode(
        name='foundation_pose_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::FoundationPoseServer',
        namespace=IsaacROSFoundationPoseServerTest.generate_namespace(),
        parameters=[{
            'action_name': 'estimate_pose_foundation_pose',
            'in_img_topic_name': 'image_color',
            'out_img_topic_name': 'foundation_pose_server/image_color',
            'in_camera_info_topic_name': 'camera_info',
            'out_camera_info_topic_name': 'foundation_pose_server/camera_info',
            'in_depth_topic_name': 'depth',
            'out_depth_topic_name': 'foundation_pose_server/depth',
            'out_bbox_topic_name': 'foundation_pose_server/bbox',
            'in_pose_estimate_topic_name': 'poses',
            'out_pose_estimate_topic_name': 'foundation_pose_server/poses',
            'out_segmented_mask_topic_name': 'foundation_pose_server/segmented_mask',
            'foundation_pose_node_name': '/mock_foundation_pose_node',
            'input_qos': 'SENSOR_DATA',
            'result_and_output_qos': 'DEFAULT',
            'log_level': 'DEBUG'
        }]
    )

    return IsaacROSFoundationPoseServerTest.generate_test_description([
        ComposableNodeContainer(
            name='foundation_pose_server_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[foundation_pose_server_node],
            namespace=IsaacROSFoundationPoseServerTest.generate_namespace(),
            output='screen',
        )
    ])


class IsaacROSFoundationPoseServerTest(IsaacROSBaseTest):
    """This test checks the functionality of the `foundation_pose_server` action."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    def set_parameters_callback(self, request, response):
        self.get_logger().info(f'Request parameters: {request.parameters}')
        response.results = []
        for param in request.parameters:
            self.get_logger().info(f'Parameter: {param.name}')
            result = SetParametersResult()
            result.successful = True
            result.reason = 'Parameter set successfully'
            response.results.append(result)

        return response

    def test_foundation_pose_server_action_with_param_service_triggered_by_goal_request(self):
        """
        Test FoundationPoseServer action.

        1. Send request to FoundationPoseServer
        2. Verify output pose estimate
        3. Check if the parameters are set correctly after the goal is accepted
        4. Check if the server was able to publish all the messages to the impementation node
        """
        received_messages = {}

        # Generate namespace lookups for all topics
        self.generate_namespace_lookup([
            'image_color',
            'camera_info',
            'depth',
            'poses',
            'foundation_pose_server/image_color',
            'foundation_pose_server/camera_info',
            'foundation_pose_server/depth',
            'foundation_pose_server/bbox',
            'foundation_pose_server/poses',
        ])

        # Create publishers
        image_pub = self.node.create_publisher(
            Image, self.namespaces['image_color'],
            qos_profile=10)

        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'],
            qos_profile=10)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['depth'],
            qos_profile=10)

        pose_estimate_pub = self.node.create_publisher(
            Detection3DArray, self.namespaces['poses'],
            qos_profile=10)

        # Create subscribers
        subs = self.create_logging_subscribers([
            ('poses', Detection3DArray),
            ('foundation_pose_server/image_color', Image),
            ('foundation_pose_server/camera_info', CameraInfo),
            ('foundation_pose_server/depth', Image),
            ('foundation_pose_server/bbox', Detection2D),
            ('foundation_pose_server/poses', Detection3DArray),
            ], received_messages,
            accept_multiple_messages=True,
            qos_profile=10)

        mock_foundation_pose_node = MockFoundationPoseNode()

        # Check if the set parameters service is available
        self.assertIn(
            '/mock_foundation_pose_node/set_parameters',
            [name for name, _ in mock_foundation_pose_node.get_service_names_and_types()])

        # Create action client
        action_client = ActionClient(
            self.node,
            EstimatePoseFoundationPose,
            f'{self.generate_namespace()}/estimate_pose_foundation_pose',
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

            # Create and publish dummy depth
            depth_msg = Image()
            depth_msg.header.stamp = image_msg.header.stamp
            depth_msg.height = IMAGE_HEIGHT
            depth_msg.width = IMAGE_WIDTH
            depth_msg.encoding = '32FC1'
            depth_msg.is_bigendian = False
            depth_msg.step = IMAGE_WIDTH * 4
            depth_msg.data = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH), dtype=np.int32)

            # Create and publish dummy detection
            detection_msg = Detection2D()
            detection_msg.bbox.center.position.x = IMAGE_WIDTH/2
            detection_msg.bbox.center.position.y = IMAGE_HEIGHT/2
            detection_msg.bbox.size_x = 10.0
            detection_msg.bbox.size_y = 10.0

            # Send action goal
            goal_msg = EstimatePoseFoundationPose.Goal()
            goal_msg.roi = detection_msg
            goal_msg.use_segmentation_mask = False

            # Setting these parameters will trigger the set parameters service
            # of the mock foundation pose node.
            goal_msg.object_frame_name = 'object_frame'
            goal_msg.mesh_file_path = 'test_mesh.obj'

            # Wait for the node under test to subscribe to our publishers
            end_time = time.time() + TIMEOUT
            while time.time() < end_time and (
                image_pub.get_subscription_count() == 0 or
                camera_info_pub.get_subscription_count() == 0 or
                depth_pub.get_subscription_count() == 0
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            self.assertGreater(image_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to image topic in time')
            self.assertGreater(camera_info_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to camera_info topic in time')
            self.assertGreater(depth_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to depth topic in time')

            # Allow subscriber to fully initialize after discovery
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            # Publish messages before sending goal
            depth_pub.publish(depth_msg)
            image_pub.publish(image_msg)
            camera_info_pub.publish(camera_info_msg)

            # Allow messages to be processed
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            future = action_client.send_goal_async(goal_msg)
            start_time = time.time()
            while not future.done() and time.time() - start_time < TIMEOUT:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)
                depth_pub.publish(depth_msg)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)

            self.node.get_logger().info(f'Goal handle accepted in '
                                        f'{time.time() - start_time} seconds')

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            pose_estimate_msg = Detection3DArray()
            pose_estimate_msg.header.stamp = image_msg.header.stamp
            pose_estimate_msg.header.frame_id = 'object_frame'
            pose_estimate_msg.detections.append(Detection3D())
            pose_estimate_msg.detections[0].results.append(ObjectHypothesisWithPose())
            pose_estimate_msg.detections[0].results[0].pose.pose.position.x = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.position.y = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.position.z = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.x = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.y = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.z = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.w = 1.0
            pose_estimate_msg.detections[0].results[0].hypothesis.class_id = 'test_class'
            pose_estimate_msg.detections[0].results[0].hypothesis.score = 1.0

            # Wait for output messages and action result
            future = goal_handle.get_result_async()
            end_time = time.time() + TIMEOUT
            while not future.done() and time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)
                if all(len(received_messages[key]) > 0 for key in received_messages.keys()):
                    break
                depth_pub.publish(depth_msg)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                pose_estimate_pub.publish(pose_estimate_msg)

            # Check if the server was able to publish all the messages to the impementation node
            self.assertIn('foundation_pose_server/image_color', received_messages)
            self.assertIn('foundation_pose_server/camera_info', received_messages)
            self.assertIn('foundation_pose_server/depth', received_messages)
            self.assertIn('foundation_pose_server/bbox', received_messages)
            # Check if the server was able to publish the pose estimate
            self.assertIn('foundation_pose_server/poses', received_messages)

            result = future.result()
            self.assertIsNotNone(result)
            result = result.result
            self.assertEqual(
                result.poses.detections[0].results[0].hypothesis.class_id, 'test_class')
            self.assertEqual(result.poses.detections[0].results[0].hypothesis.score, 1.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.x, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.y, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.z, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.x, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.y, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.z, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.w, 1.0)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(pose_estimate_pub)
            action_client.destroy()
            mock_foundation_pose_node.destroy_node()

    def test_foundation_pose_server_action_with_no_param_service_triggered_by_goal_request(self):
        """
        Test FoundationPoseServer action.

        1. Send request to FoundationPoseServer
        2. Verify output pose estimate
        3. Check if the parameters are unchanged
        4. Check if the server was able to publish all the messages to the impementation node
        """
        received_messages = {}

        # Generate namespace lookups for all topics
        self.generate_namespace_lookup([
            'image_color',
            'camera_info',
            'depth',
            'poses',
            'foundation_pose_server/image_color',
            'foundation_pose_server/camera_info',
            'foundation_pose_server/depth',
            'foundation_pose_server/bbox',
            'foundation_pose_server/poses',
        ])

        # Create publishers
        image_pub = self.node.create_publisher(
            Image, self.namespaces['image_color'],
            qos_profile=10)

        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'],
            qos_profile=10)

        depth_pub = self.node.create_publisher(
            Image, self.namespaces['depth'],
            qos_profile=10)

        pose_estimate_pub = self.node.create_publisher(
            Detection3DArray, self.namespaces['poses'],
            qos_profile=10)

        # Create subscribers
        subs = self.create_logging_subscribers([
            ('poses', Detection3DArray),
            ('foundation_pose_server/image_color', Image),
            ('foundation_pose_server/camera_info', CameraInfo),
            ('foundation_pose_server/depth', Image),
            ('foundation_pose_server/bbox', Detection2D),
            ('foundation_pose_server/poses', Detection3DArray),
            ], received_messages,
            accept_multiple_messages=True,
            qos_profile=10)

        mock_foundation_pose_node = MockFoundationPoseNode()

        # Check if the set parameters service is available
        self.assertIn(
            '/mock_foundation_pose_node/set_parameters',
            [name for name, _ in mock_foundation_pose_node.get_service_names_and_types()])

        # Create action client
        action_client = ActionClient(
            self.node,
            EstimatePoseFoundationPose,
            f'{self.generate_namespace()}/estimate_pose_foundation_pose',
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

            # Create and publish dummy depth
            depth_msg = Image()
            depth_msg.header.stamp = image_msg.header.stamp
            depth_msg.height = IMAGE_HEIGHT
            depth_msg.width = IMAGE_WIDTH
            depth_msg.encoding = '32FC1'
            depth_msg.is_bigendian = False
            depth_msg.step = IMAGE_WIDTH * 4
            depth_msg.data = np.zeros((IMAGE_HEIGHT * IMAGE_WIDTH), dtype=np.int32)

            # Create and publish dummy detection
            detection_msg = Detection2D()
            detection_msg.bbox.center.position.x = IMAGE_WIDTH/2
            detection_msg.bbox.center.position.y = IMAGE_HEIGHT/2
            detection_msg.bbox.size_x = 10.0
            detection_msg.bbox.size_y = 10.0

            # Send action goal
            goal_msg = EstimatePoseFoundationPose.Goal()
            goal_msg.roi = detection_msg
            goal_msg.use_segmentation_mask = False

            # Wait for the node under test to subscribe to our publishers
            end_time = time.time() + TIMEOUT
            while time.time() < end_time and (
                image_pub.get_subscription_count() == 0 or
                camera_info_pub.get_subscription_count() == 0 or
                depth_pub.get_subscription_count() == 0
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            self.assertGreater(image_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to image topic in time')
            self.assertGreater(camera_info_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to camera_info topic in time')
            self.assertGreater(depth_pub.get_subscription_count(), 0,
                               'Node under test did not subscribe to depth topic in time')

            # Allow subscriber to fully initialize after discovery
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            # Publish messages before sending goal
            depth_pub.publish(depth_msg)
            image_pub.publish(image_msg)
            camera_info_pub.publish(camera_info_msg)

            # Allow messages to be processed
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)

            future = action_client.send_goal_async(goal_msg)
            start_time = time.time()
            while not future.done() and time.time() - start_time < TIMEOUT:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)
                depth_pub.publish(depth_msg)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)

            self.node.get_logger().info(f'Goal handle accepted in '
                                        f'{time.time() - start_time} seconds')

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            pose_estimate_msg = Detection3DArray()
            pose_estimate_msg.header.stamp = image_msg.header.stamp
            pose_estimate_msg.header.frame_id = 'object_frame'
            pose_estimate_msg.detections.append(Detection3D())
            pose_estimate_msg.detections[0].results.append(ObjectHypothesisWithPose())
            pose_estimate_msg.detections[0].results[0].pose.pose.position.x = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.position.y = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.position.z = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.x = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.y = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.z = 0.0
            pose_estimate_msg.detections[0].results[0].pose.pose.orientation.w = 1.0
            pose_estimate_msg.detections[0].results[0].hypothesis.class_id = 'test_class'
            pose_estimate_msg.detections[0].results[0].hypothesis.score = 1.0

            # Wait for output messages and action result
            future = goal_handle.get_result_async()
            end_time = time.time() + TIMEOUT
            while not future.done() and time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                rclpy.spin_once(mock_foundation_pose_node, timeout_sec=0.1)
                if all(len(received_messages[key]) > 0 for key in received_messages.keys()):
                    break
                depth_pub.publish(depth_msg)
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                pose_estimate_pub.publish(pose_estimate_msg)

            # Check if the parameters are unchanged
            self.assertEqual(mock_foundation_pose_node.get_parameter('mesh_file_path').value, '')
            self.assertEqual(mock_foundation_pose_node.get_parameter('tf_frame_name').value, '')

            # Check if the server was able to publish all the messages to the impementation node
            self.assertIn('foundation_pose_server/image_color', received_messages)
            self.assertIn('foundation_pose_server/camera_info', received_messages)
            self.assertIn('foundation_pose_server/depth', received_messages)
            self.assertIn('foundation_pose_server/bbox', received_messages)
            # Check if the server was able to publish the pose estimate
            self.assertIn('foundation_pose_server/poses', received_messages)

            result = future.result()
            self.assertIsNotNone(result)
            result = result.result
            self.assertEqual(
                result.poses.detections[0].results[0].hypothesis.class_id, 'test_class')
            self.assertEqual(result.poses.detections[0].results[0].hypothesis.score, 1.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.x, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.y, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.position.z, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.x, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.y, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.z, 0.0)
            self.assertEqual(result.poses.detections[0].results[0].pose.pose.orientation.w, 1.0)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(pose_estimate_pub)
            action_client.destroy()
            mock_foundation_pose_node.destroy_node()
