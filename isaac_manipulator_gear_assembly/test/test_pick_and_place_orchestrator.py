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
"""Test for Pick and Place Orchestrator with dummy action servers."""

import os
import time

from ament_index_python.packages import get_package_share_directory

from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseWithCovariance
from isaac_manipulator_interfaces.action import (
    GetObjectPose, GetObjects, PickAndPlace
)
from isaac_manipulator_interfaces.msg import ObjectInfo
from isaac_manipulator_ros_python_utils import load_yaml_params
from isaac_ros_cumotion_interfaces.action import AttachObject, IKSolution
from isaac_ros_test import IsaacROSBaseTest
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf2_ros
from vision_msgs.msg import Detection2D, Detection3D, ObjectHypothesisWithPose, Point2D, Pose2D


RUN_TEST = True


class PickAndPlaceOrchestratorTest(IsaacROSBaseTest):
    """Test class for the pick and place orchestrator."""

    # Action servers
    def setupServers(self):
        """Set up the action servers."""
        self._get_objects_server = ActionServer(
            self.node, GetObjects, '/get_objects', self._get_objects_callback)
        self._get_object_pose_server = ActionServer(
            self.node, GetObjectPose, '/get_object_pose', self._get_object_pose_callback)
        self._gripper_server = ActionServer(
            self.node, GripperCommand, '/robotiq_gripper_controller/gripper_cmd',
            self._gripper_callback)
        # Action servers required by pick_and_place_orchestrator
        self._attach_object_server = ActionServer(
            self.node, AttachObject, 'attach_object', self._attach_object_callback)

        self._ik_solution_server = ActionServer(
            self.node, IKSolution, '/cumotion/ik', self._ik_solution_callback)

        self.node.get_logger().info('Dummy action servers initialized for Pick & Place')

        # TF2 static broadcaster for dummy transforms
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self._setup_dummy_transforms()

        self.node.get_logger().info('Dummy action servers initialized')

    def _attach_object_callback(self, goal_handle):
        self.node.get_logger().info(
            f'AttachObject action called: attach={goal_handle.request.attach_object}')
        result = AttachObject.Result()
        # It only checks GoalStatus in the orchestrator, so returning an empty result is fine.
        goal_handle.succeed()
        return result

    def _ik_solution_callback(self, goal_handle):
        self.node.get_logger().info('IKSolution action called')

        js = JointState()
        js.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        js.position = [0.0, -1.0, 1.0, 0.0, 1.0, 0.0]
        js.velocity = [0.0] * 6
        js.effort = [0.0] * 6

        result = IKSolution.Result()
        # The orchestrator expects:
        # - result.success to be a non-empty list of bools
        # - result.joint_states to align with that list
        result.success = [True]
        result.joint_states = [js]

        goal_handle.succeed()
        return result

    def _setup_dummy_transforms(self):
        """Set up dummy TF transforms for the test."""
        # Dummy camera transform
        t = self._create_transform('base_link', 'world',
                                   [0.5, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])

        # Dummy gear transforms
        gear_names = ['detected_object1']
        positions = [[0.3, 0.2, 0.1]]

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

    def _pick_and_place_callback(self, goal_handle):
        """Mock PickAndPlace action."""
        self.node.get_logger().info(
            f'PickAndPlace action called for object_id: {goal_handle.request.object_id}')

        result = PickAndPlace.Result()
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

    def _switch_controller_callback(self, request, response):
        """Mock SwitchController service."""
        self.node.get_logger().info(
            f'SwitchController service called: activate={request.activate_controllers}, '
            f'deactivate={request.deactivate_controllers}')
        response.ok = True
        return response

    def test_pick_and_place_orchestrator(self):
        """Test that the pick and place orchestrator processes the object successfully."""
        self.node.get_logger().info('Starting test for pick and place orchestrator')
        self.setupServers()

        # Wait for the orchestrator to be ready
        time.sleep(5.0)

        # Create action client for the pick and place action
        from rclpy.action import ActionClient
        pick_and_place_client = ActionClient(self.node, PickAndPlace, '/pick_and_place')

        # Wait for the action server to be available
        self.assertTrue(
            pick_and_place_client.wait_for_server(timeout_sec=30.0),
            'Pick and place action server not available'
        )

        # Send goal to the pick and place action
        goal = PickAndPlace.Goal()
        goal_future = pick_and_place_client.send_goal_async(goal)

        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=10.0)
        self.assertTrue(goal_future.done(), 'Goal was not accepted in time')

        # If this errors out that means Pick and place action was rejected.
        self.assertTrue(goal_future.result().accepted, 'Goal was rejected')

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
    PickAndPlaceOrchestratorTest.generate_namespace()

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0

    isaac_manipulator_test_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_gear_assembly'),
        'test', 'include')

    # This will test the flow for pick and place orchestrator with sim ground truth disabled.
    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'params',
        'ur10e_robotiq_2f_85_soup_can.yaml'
    )
    params = load_yaml_params(test_yaml_config)

    if RUN_TEST:
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_test_include_dir,
                 '/pick_and_place.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
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

    return PickAndPlaceOrchestratorTest.generate_test_description(
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
    )
