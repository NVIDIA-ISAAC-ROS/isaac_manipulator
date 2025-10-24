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

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseWithCovariance
from isaac_manipulator_interfaces.action import AddSegmentationMask, GetObjectPose, GetObjects
from isaac_manipulator_interfaces.srv import AddMeshToObject
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D,
                             Detection3DArray, ObjectHypothesisWithPose)


TIMEOUT = 10  # seconds
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


@pytest.mark.rostest
def generate_test_description():
    object_info_node = ComposableNode(
        name='object_info_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectInfoServer',
        namespace=IsaacROSObjectInfoTest.generate_namespace(),
        parameters=[{
            'action_name': 'object_info',
            'object_detection_backend': 'SEGMENT_ANYTHING',  # Add this to use SAM backend
            'segmentation_backend': 'SEGMENT_ANYTHING',
            'pose_estimation_backend': 'FOUNDATION_POSE',    # This is required
            'log_level': 'DEBUG'
        }]
    )

    # Add Foundation Pose server node
    foundation_pose_node = ComposableNode(
        name='foundation_pose_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::FoundationPoseServer',
        namespace=IsaacROSObjectInfoTest.generate_namespace(),
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
            'input_qos': 'DEFAULT',
            'result_and_output_qos': 'DEFAULT',
            'log_level': 'DEBUG'
        }]
    )

    segment_anything_node = ComposableNode(
        name='segment_anything_server',
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::SegmentAnythingServer',
        namespace=IsaacROSObjectInfoTest.generate_namespace(),
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
            'input_qos': 'DEFAULT',
            'result_and_output_qos': 'DEFAULT',
            'log_level': 'DEBUG'
        }]
    )

    return IsaacROSObjectInfoTest.generate_test_description([
        ComposableNodeContainer(
            name='object_info_container',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[object_info_node, segment_anything_node,
                                          foundation_pose_node],
            namespace=IsaacROSObjectInfoTest.generate_namespace(),
            output='screen',
        )
    ])


class IsaacROSObjectInfoTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_object_info_with_sam_backend(self):
        """
        Test object info server with SAM backend.

        1. Create publishers for required topics
        2. Call GetObjects with SAM backend
        3. Call AddSegmentationMask for the detected object
        4. Call GetObjects again to verify object info is populated
        5. Add mesh file path and verify it's correctly populated
        6. Get object pose and verify pose data
        """
        # Create publishers
        image_pub = self.node.create_publisher(
            Image, 'image_color',
            qos_profile=10)

        camera_info_pub = self.node.create_publisher(
            CameraInfo, 'camera_info',
            qos_profile=10)

        segmentation_mask_pub = self.node.create_publisher(
            Image, 'segmentation_mask',
            qos_profile=10)

        detections_pub = self.node.create_publisher(
            Detection2DArray, 'detections',
            qos_profile=10)

        pose_estimate_pub = self.node.create_publisher(
            Detection3DArray, 'poses',
            qos_profile=10)

        # Add to your publishers section
        depth_pub = self.node.create_publisher(
            Image, 'depth',
            qos_profile=10)

        # Create action clients, no namespacing as we hard code name in src.
        get_objects_client = ActionClient(
            self.node,
            GetObjects,
            f'{self.generate_namespace()}/get_objects',
            callback_group=ReentrantCallbackGroup())

        add_segmentation_mask_client = ActionClient(
            self.node,
            AddSegmentationMask,
            f'{self.generate_namespace()}/add_segmentation_mask',
            callback_group=ReentrantCallbackGroup())

        get_object_pose_client = ActionClient(
            self.node,
            GetObjectPose,
            f'{self.generate_namespace()}/get_object_pose',
            callback_group=ReentrantCallbackGroup())

        # Create service client for adding mesh
        add_mesh_client = self.node.create_client(
            AddMeshToObject,
            f'{self.generate_namespace()}/add_mesh_to_object')

        try:
            # Wait for action servers
            self.assertTrue(
                get_objects_client.wait_for_server(timeout_sec=TIMEOUT),
                'GetObjects action server not available')
            self.assertTrue(
                add_segmentation_mask_client.wait_for_server(timeout_sec=TIMEOUT),
                'AddSegmentationMask action server not available')
            self.assertTrue(
                add_mesh_client.wait_for_service(timeout_sec=TIMEOUT),
                'AddMeshToObject service not available')
            self.assertTrue(
                get_object_pose_client.wait_for_server(timeout_sec=TIMEOUT),
                'GetObjectPose action server not available')

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
            detection_msg = Detection2D()
            detection_msg.bbox.center.position.x = IMAGE_WIDTH/2
            detection_msg.bbox.center.position.y = IMAGE_HEIGHT/2
            detection_msg.bbox.size_x = 10.0
            detection_msg.bbox.size_y = 10.0
            detections_msg.detections.append(detection_msg)
            detections_msg.header.stamp = image_msg.header.stamp

            # Create dummy depth image (after your other dummy messages)
            # 1 meter depth
            depth_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32) * 1.0
            depth_msg = CvBridge().cv2_to_imgmsg(depth_image)
            depth_msg.header.stamp = image_msg.header.stamp

            # Step 1: Call GetObjects with SAM backend
            get_objects_goal = GetObjects.Goal()
            get_objects_goal.use_initial_hint = True
            get_objects_goal.initial_hint = Point(x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2, z=0.0)

            self.node.get_logger().info('Sending GetObjects goal')
            future = get_objects_client.send_goal_async(get_objects_goal)

            # Publish messages while waiting for result
            while not future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            result = result_future.result()
            self.assertIsNotNone(result)
            self.assertEqual(len(result.result.objects), 1)
            object_id = result.result.objects[0].object_id

            self.node.get_logger().info(f'Object id from 1st GetObjects: {object_id}')

            # Step 2: Call AddSegmentationMask
            add_segmentation_mask_goal = AddSegmentationMask.Goal()
            add_segmentation_mask_goal.object_id = object_id

            self.node.get_logger().info('Sending AddSegmentationMask goal')
            future = add_segmentation_mask_client.send_goal_async(add_segmentation_mask_goal)

            while not future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            result = result_future.result()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.result.has_segmentation_mask)
            self.assertTrue(result.result.has_segmentation_mask)

            # Step 3: Call GetObjects again to verify object info
            get_objects_goal = GetObjects.Goal()
            get_objects_goal.use_initial_hint = False
            get_objects_goal.initial_hint = Point(x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2, z=0.0)

            self.node.get_logger().info('Sending second GetObjects goal')
            future = get_objects_client.send_goal_async(get_objects_goal)

            while not future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                depth_pub.publish(depth_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            result = result_future.result()
            self.assertIsNotNone(result)
            self.assertEqual(len(result.result.objects), 1)
            object_info = result.result.objects[0]

            for object_msg in result.result.objects:
                object_id = object_msg.object_id
                self.node.get_logger().info(f'Object info: {object_id}')

            # Verify object info is populated correctly
            self.assertEqual(object_info.object_id, object_id)
            self.assertIsNotNone(object_info.detection_2d)
            self.assertIsNotNone(object_info.segmentation_mask)

            # Step 4: Add mesh file path
            mesh_path = str(self.filepath / 'test_mesh.obj')
            # Create an empty file
            with open(mesh_path, 'w') as f:
                f.write('# Test mesh file\n')

            request = AddMeshToObject.Request()
            request.object_ids = [object_id]
            request.mesh_file_paths = [mesh_path]

            self.node.get_logger().info('Calling AddMeshToObject service')
            future = add_mesh_client.call_async(request)
            while not future.done():
                rclpy.spin_once(self.node, timeout_sec=0.1)

            response = future.result()
            self.assertIsNotNone(response)
            self.assertTrue(response.success)
            self.assertEqual(len(response.failed_ids), 0)

            # Step 5: Call GetObjects one last time to verify mesh path
            get_objects_goal = GetObjects.Goal()
            get_objects_goal.use_initial_hint = False  # This will return all current objects
            get_objects_goal.initial_hint = Point(x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2, z=0.0)

            self.node.get_logger().info('Sending final GetObjects goal')
            future = get_objects_client.send_goal_async(get_objects_goal)

            while not future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

            result = result_future.result()
            self.assertIsNotNone(result)
            self.assertEqual(len(result.result.objects), 1)
            object_info = result.result.objects[0]

            # Verify mesh path is populated
            self.assertEqual(object_info.mesh_file_path, mesh_path)

            # Step 6: Call GetObjectPose to get the pose
            get_pose_goal = GetObjectPose.Goal()
            get_pose_goal.object_id = object_id

            self.node.get_logger().info('Sending GetObjectPose goal')
            future = get_object_pose_client.send_goal_async(get_pose_goal)

            # Create and publish dummy pose data
            pose_msg = Detection3DArray()
            pose_msg.header.stamp = image_msg.header.stamp
            detection3d = Detection3D()
            detection3d.results.append(ObjectHypothesisWithPose())
            detection3d.results[0].pose = PoseWithCovariance()
            detection3d.results[0].pose.pose = Pose()
            # Set some dummy pose values
            detection3d.results[0].pose.pose.position.x = 1.0
            detection3d.results[0].pose.pose.position.y = 0.5
            detection3d.results[0].pose.pose.position.z = 0.3
            detection3d.results[0].pose.pose.orientation.w = 1.0  # Identity rotation
            pose_msg.detections.append(detection3d)

            while not future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                pose_estimate_pub.publish(pose_msg)  # Publish pose data
                rclpy.spin_once(self.node, timeout_sec=0.1)

            goal_handle = future.result()
            self.assertIsNotNone(goal_handle)
            self.assertTrue(goal_handle.accepted)

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                image_pub.publish(image_msg)
                camera_info_pub.publish(camera_info_msg)
                segmentation_mask_pub.publish(mask_msg)
                detections_pub.publish(detections_msg)
                pose_estimate_pub.publish(pose_msg)  # Publish pose data
                rclpy.spin_once(self.node, timeout_sec=0.1)

            result = result_future.result()
            self.assertIsNotNone(result)

            # Verify pose data
            self.assertIsNotNone(result.result.object_pose)
            self.assertEqual(result.result.object_pose.position.x, 1.0)
            self.assertEqual(result.result.object_pose.position.y, 0.5)
            self.assertEqual(result.result.object_pose.position.z, 0.3)
            self.assertEqual(result.result.object_pose.orientation.w, 1.0)

            # Cleanup test mesh file
            os.remove(mesh_path)

        finally:
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
            self.node.destroy_publisher(segmentation_mask_pub)
            self.node.destroy_publisher(detections_pub)
            self.node.destroy_publisher(pose_estimate_pub)
            self.node.destroy_publisher(depth_pub)
            get_objects_client.destroy()
            add_segmentation_mask_client.destroy()
            get_object_pose_client.destroy()
            self.node.destroy_client(add_mesh_client)
