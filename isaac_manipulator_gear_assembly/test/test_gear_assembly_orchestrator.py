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
"""Test for Gear Assembly Orchestrator with dummy action servers."""

import random
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Point, Pose, PoseWithCovariance
from isaac_manipulator_interfaces.action import (
    AddSegmentationMask, GearAssembly, GetObjectPose, GetObjects, Insert,
    PickAndPlace
)
from isaac_manipulator_interfaces.msg import ObjectInfo
from isaac_manipulator_interfaces.srv import AddMeshToObject, ClearObjects
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import pytest
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.action import ActionServer
from std_msgs.msg import Header
import tf2_ros
from vision_msgs.msg import Detection2D, Detection3D, ObjectHypothesisWithPose, Point2D, Pose2D


RUN_TEST = True


class GearAssemblyOrchestratorTest(IsaacROSBaseTest):
    """Test class for the gear assembly orchestrator."""

    # Action servers
    def setupServers(self):
        """Set up the action servers."""
        self._get_objects_server = ActionServer(
            self.node, GetObjects, '/get_objects', self._get_objects_callback)
        self._get_object_pose_server = ActionServer(
            self.node, GetObjectPose, '/get_object_pose', self._get_object_pose_callback)
        self._add_segmentation_mask_server = ActionServer(
            self.node, AddSegmentationMask, '/add_segmentation_mask',
            self._add_segmentation_mask_callback)
        self._pick_and_place_server = ActionServer(
            self.node, PickAndPlace, '/pick_and_place', self._pick_and_place_callback)
        self._insert_server = ActionServer(
            self.node, Insert, '/gear_assembly/insert_policy', self._insert_callback)
        self._gripper_server = ActionServer(
            self.node, GripperCommand, '/robotiq_gripper_controller/gripper_cmd',
            self._gripper_callback)

        # Service servers
        self._add_mesh_service = self.node.create_service(
            AddMeshToObject, '/add_mesh_to_object', self._add_mesh_callback)
        self._clear_objects_service = self.node.create_service(
            ClearObjects, '/clear_objects', self._clear_objects_callback)
        self._switch_controller_service = self.node.create_service(
            SwitchController, '/controller_manager/switch_controller',
            self._switch_controller_callback)
        self._set_parameters_service = self.node.create_service(
            SetParameters, '/foundationpose_node/set_parameters',
            self._set_parameters_callback)

        self.point_publisher = self.node.create_publisher(Point, '/input_points_debug', 10)

        # TF2 static broadcaster for dummy transforms
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self._setup_dummy_transforms()

        self.node.get_logger().info('Dummy action servers initialized')

    def _setup_dummy_transforms(self):
        """Set up dummy TF transforms for the test."""
        # Dummy camera transform
        t = self._create_transform('base_link', 'camera_color_optical_frame',
                                   [0.5, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])

        # Dummy gear transforms
        gear_names = ['gear_shaft_large', 'gear_shaft_small', 'gear_shaft_medium']
        positions = [[0.3, 0.2, 0.1], [0.4, 0.3, 0.1], [0.5, 0.4, 0.1]]

        transforms = [t]
        for name, pos in zip(gear_names, positions):
            transforms.append(self._create_transform('base_link',
                                                     name, pos,
                                                     [0.0, 0.0, 0.0, 1.0]))

        self.tf_static_broadcaster.sendTransform(transforms)

    def _create_transform(self, parent, child, translation, rotation):
        """Create a transform stamped message."""
        from geometry_msgs.msg import TransformStamped
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(rotation[0])
        t.transform.rotation.y = float(rotation[1])
        t.transform.rotation.z = float(rotation[2])
        t.transform.rotation.w = float(rotation[3])
        return t

    def _get_objects_callback(self, goal_handle):
        """Mock GetObjects action."""
        self.node.get_logger().info('GetObjects action called')

        # Create a mock object
        obj = ObjectInfo()
        obj.object_id = 1
        obj.detection_2d = Detection2D()
        obj.detection_2d.header = Header()
        obj.detection_2d.header.stamp = self.node.get_clock().now().to_msg()
        obj.detection_2d.header.frame_id = 'base_link'
        center = Pose2D()
        center.position = Point2D()
        center.position.x = 0.4
        center.position.y = 0.3
        center.theta = 0.0  # 2D rotation, typically 0 for object detection
        obj.detection_2d.bbox.center = center
        obj.detection_2d.bbox.size_x = 0.1
        obj.detection_2d.bbox.size_y = 0.1

        detection3d = Detection3D()
        detection3d.results.append(ObjectHypothesisWithPose())
        detection3d.results[0].pose = PoseWithCovariance()
        detection3d.results[0].pose.pose = Pose()
        # Set some dummy pose values
        detection3d.results[0].pose.pose.position.x = 1.0
        detection3d.results[0].pose.pose.position.y = 0.5
        detection3d.results[0].pose.pose.position.z = 0.3
        detection3d.results[0].pose.pose.orientation.w = 1.0  # Identity rotation
        obj.detection_3d = detection3d

        obj.has_segmentation_mask = False
        obj.mesh_file_path = 'tmp'
        obj.name = 'gear_link'

        result = GetObjects.Result()
        result.objects = [obj]

        goal_handle.succeed()
        return result

    def _get_object_pose_callback(self, goal_handle):
        """Mock GetObjectPose action."""
        self.node.get_logger().info(
            f'GetObjectPose action called for object_id: {goal_handle.request.object_id}')

        # Create a mock pose
        pose = Pose()
        pose.position.x = 0.4
        pose.position.y = 0.3
        pose.position.z = 0.15
        pose.orientation.w = 1.0

        result = GetObjectPose.Result()
        result.object_pose = pose

        goal_handle.succeed()
        return result

    def _add_segmentation_mask_callback(self, goal_handle):
        """Mock AddSegmentationMask action."""
        self.node.get_logger().info(
            f'AddSegmentationMask action called for object_id: {goal_handle.request.object_id}')

        result = AddSegmentationMask.Result()
        result.success = True

        goal_handle.succeed()
        return result

    def _pick_and_place_callback(self, goal_handle):
        """Mock PickAndPlace action."""
        self.node.get_logger().info(
            f'PickAndPlace action called for object_id: {goal_handle.request.object_id}')

        result = PickAndPlace.Result()
        result.success = True

        goal_handle.succeed()
        return result

    def _insert_callback(self, goal_handle):
        """Mock Insert action."""
        self.node.get_logger().info('Insert action called')

        result = Insert.Result()
        result.success = True

        goal_handle.succeed()
        return result

    def _gripper_callback(self, goal_handle):
        """Mock GripperCommand action."""
        self.node.get_logger().info(
            f'GripperCommand action called with position: {goal_handle.request.command.position}')

        result = GripperCommand.Result()
        result.position = goal_handle.request.command.position
        result.effort = goal_handle.request.command.max_effort
        result.stalled = False
        result.reached_goal = True

        goal_handle.succeed()
        return result

    def _add_mesh_callback(self, request, response):
        """Mock AddMeshToObject service."""
        self.node.get_logger().info(
            f'AddMeshToObject service called for {len(request.object_ids)} objects')
        response.success = True
        return response

    def _clear_objects_callback(self, request, response):
        """Mock ClearObjects service."""
        self.node.get_logger().info('ClearObjects service called')
        response.count = 1
        return response

    def _switch_controller_callback(self, request, response):
        """Mock SwitchController service."""
        self.node.get_logger().info(
            f'SwitchController service called: activate={request.activate_controllers}, '
            f'deactivate={request.deactivate_controllers}')
        response.ok = True
        return response

    def _set_parameters_callback(self, request, response):
        """Mock SetParameters service."""
        self.node.get_logger().info(
            f'SetParameters service called with {len(request.parameters)} parameters')
        from rcl_interfaces.msg import SetParametersResult
        response.results = [SetParametersResult(successful=True) for _ in request.parameters]
        return response

    def _publish_random_point(self):
        """Publish a random point to trigger the orchestrator."""
        # Generate random point coordinates (simulating user clicks)
        point = Point()
        point.x = random.uniform(100.0, 1500.0)  # Random pixel coordinates
        point.y = random.uniform(100.0, 1000.0)
        point.z = 0.0

        self.point_publisher.publish(point)

    def test_gear_assembly_orchestrator(self):
        """Test that the gear assembly orchestrator processes all 3 gears successfully."""
        self.node.get_logger().info('Starting test for gear assembly orchestrator')
        self.setupServers()

        # Wait for the orchestrator to be ready
        time.sleep(5.0)

        # Create action client for the gear assembly action
        from rclpy.action import ActionClient
        gear_assembly_client = ActionClient(self.node, GearAssembly, '/gear_assembly')

        # Wait for the action server to be available
        self.assertTrue(
            gear_assembly_client.wait_for_server(timeout_sec=30.0),
            'Gear assembly action server not available'
        )

        # Send goal to the gear assembly action
        goal = GearAssembly.Goal()
        goal_future = gear_assembly_client.send_goal_async(goal)

        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=10.0)
        self.assertTrue(goal_future.done(), 'Goal was not accepted in time')

        goal_handle = goal_future.result()
        self.assertTrue(goal_handle.accepted, 'Goal was rejected')

        # Wait for result
        result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=60.0)

        while not result_future.done():
            # Send point topic
            self._publish_random_point()
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertTrue(result_future.done(), 'Action did not complete in time')
        result = result_future.result()

        # Check that the action succeeded
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED, 'Gear assembly action failed')
        self.assertTrue(result.result.success, 'Gear assembly result indicates failure')

    @classmethod
    def generate_test_description(
        cls, nodes, node_startup_delay
    ):
        """Generate test description."""
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Robot Segmentor nodes for testing."""
    GearAssemblyOrchestratorTest.generate_namespace()

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0

    num_cycles = 1
    use_sim_time = False
    mesh_file_paths = ['tmp', 'tmp', 'tmp']
    gripper_close_pos = [0.24, 0.24, 0.24]
    mesh_file_path_for_peg_stand_estimation = ''
    use_ground_truth_pose_estimation = False
    run_rl_inference = True
    wait_for_point_topic = True
    point_topic_name_as_trigger = 'input_points_debug'
    use_joint_space_planner = True
    target_joint_state_for_place_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    offset_for_place_pose = 0.28
    offset_for_insertion_pose = 0.02

    if RUN_TEST:
        node_startup_delay = 12.0

        # Gear assembly orchestrator node
        test_nodes.append(Node(
            package='isaac_manipulator_gear_assembly',
            executable='gear_assembly_orchestrator.py',
            name='gear_assembly_orchestrator',
            output='screen',
            parameters=[{
                'run_test': False,  # We'll call the action directly
                'node_startup_delay': node_startup_delay,
                'num_cycles': num_cycles,
                'use_sim_time': use_sim_time,
                'mesh_file_paths': mesh_file_paths,
                'gripper_close_pos': gripper_close_pos,
                'mesh_file_path_for_peg_stand_estimation': mesh_file_path_for_peg_stand_estimation,
                'use_ground_truth_pose_estimation': use_ground_truth_pose_estimation,
                'run_rl_inference': run_rl_inference,
                'wait_for_point_topic': wait_for_point_topic,
                'point_topic_name_as_trigger': point_topic_name_as_trigger,
                'use_joint_space_planner': use_joint_space_planner,
                'target_joint_state_for_place_pose': target_joint_state_for_place_pose,
                'max_timeout_time_for_action_call': 5.0,
                'offset_for_place_pose': offset_for_place_pose,
                'offset_for_insertion_pose': offset_for_insertion_pose,
            }],
        ))
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base']
        ))
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'base', 'detected_object1']
        ))
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return GearAssemblyOrchestratorTest.generate_test_description(
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
    )
