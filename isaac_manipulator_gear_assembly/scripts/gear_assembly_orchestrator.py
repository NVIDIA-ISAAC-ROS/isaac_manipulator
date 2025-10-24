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
"""Orchestration node for Gear Assembly in Isaac Manipulator."""

import os
from threading import Event
import time
from typing import Any, Dict, Iterable, List, Tuple

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from isaac_manipulator_interfaces.action import (
    AddSegmentationMask, GearAssembly, GetObjectPose, GetObjects, Insert,
    PickAndPlace
)
from isaac_manipulator_interfaces.srv import AddMeshToObject, ClearObjects
import isaac_manipulator_ros_python_utils.geometry as geometry_utils
from rcl_interfaces.msg import Parameter as RCLPYParameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as RCLPYNode
from rclpy.qos import QoSProfile
from rclpy.subscription import Subscription
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Header
import tf2_ros


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


class IsaacManipulatorGearAssemblyOrchestrator(RCLPYNode):
    """
    Test for Isaac ROS Gear Assembly POL.

    This test will do object pose estimation using SAM with initial guess.
    It will then update foundation pose to use the mesh.
    It will then do pick and place action on the object.
    It will then do RL insertion action on the object.
    """

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_cycles: int = 10
    _use_sim_time: bool = False
    _mesh_file_paths: List[str] = []
    _run_test: bool = False
    _detection_client: ActionClient = None
    _pose_client: ActionClient = None
    _segmentation_client: ActionClient = None
    _add_mesh_client: ActionClient = None
    _clear_objects_client: ActionClient = None
    _wait_for_point_topic: bool = False
    _point_topic_name_as_trigger: str = ''
    _use_ground_truth_pose_estimation: bool = False
    _verify_pose_estimation_accuracy: bool = False
    _run_rl_inference: bool = False
    _gripper_close_pos: List[float] = None
    _use_joint_space_planner: bool = False
    _target_joint_state_for_place_pose: JointState = None
    _mesh_file_path_for_peg_stand_estimation: str = ''
    _offset_for_place_pose: float = 0.30  # 28 cms
    _offset_for_insertion_pose: float = 0.02  # 2 cms

    # Variables to track object detection backend, segmentation backend and pose estimation backend
    _is_segment_anything_segmentation_enabled: bool = False

    def __init__(self):
        super().__init__('gear_assembly_orchestrator')
        # Declare parameters
        self.declare_parameter('point_topic_name_as_trigger', '')
        self.declare_parameter('wait_for_point_topic', False)
        self.declare_parameter('use_ground_truth_pose_estimation', False)
        self.declare_parameter('verify_pose_estimation_accuracy', False)
        self.declare_parameter('run_rl_inference', False)
        self.declare_parameter('use_joint_space_planner', False)
        self.declare_parameter('target_joint_state_for_place_pose',
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('gripper_close_pos', [0.50, 0.64, 0.52])
        self.declare_parameter('mesh_file_path_for_peg_stand_estimation', '')
        self.declare_parameter('mesh_file_paths', [''])
        self.declare_parameter('camera_prim_name_in_tf', 'camera_color_optical_frame')
        self.declare_parameter('output_dir', ISAAC_ROS_WS)
        self.declare_parameter('node_startup_delay', 0.0)
        self.declare_parameter('is_segment_anything_segmentation_enabled', False)
        self.declare_parameter('initial_hint', False)
        self.declare_parameter('run_test', False)
        self.declare_parameter('max_timeout_time_for_action_call', 10.0)
        self.declare_parameter('num_cycles', 10)
        self.declare_parameter('nodes', [])
        self.declare_parameter('offset_for_place_pose', 0.28)
        self.declare_parameter('offset_for_insertion_pose', 0.0)

        # Then have ONLY ONE assignment per parameter:
        self._point_topic_name_as_trigger = self.get_parameter(
            'point_topic_name_as_trigger').get_parameter_value().string_value
        self._wait_for_point_topic = self.get_parameter(
            'wait_for_point_topic').get_parameter_value().bool_value
        self._use_ground_truth_pose_estimation = self.get_parameter(
            'use_ground_truth_pose_estimation').get_parameter_value().bool_value
        self._verify_pose_estimation_accuracy = self.get_parameter(
            'verify_pose_estimation_accuracy').get_parameter_value().bool_value
        self._run_rl_inference = self.get_parameter(
            'run_rl_inference').get_parameter_value().bool_value
        self._use_joint_space_planner = self.get_parameter(
            'use_joint_space_planner').get_parameter_value().bool_value
        self._target_joint_state_for_place_pose_arr_value = self.get_parameter(
            'target_joint_state_for_place_pose').get_parameter_value().double_array_value
        self._gripper_close_pos = self.get_parameter(
            'gripper_close_pos').get_parameter_value().double_array_value
        self._mesh_file_path_for_peg_stand_estimation = self.get_parameter(
            'mesh_file_path_for_peg_stand_estimation').get_parameter_value().string_value
        self._mesh_file_paths = self.get_parameter(
            'mesh_file_paths').get_parameter_value().string_array_value
        self._camera_prim_name_in_tf = self.get_parameter(
            'camera_prim_name_in_tf').get_parameter_value().string_value
        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._node_startup_delay = self.get_parameter(
            'node_startup_delay').get_parameter_value().double_value
        self._is_segment_anything_segmentation_enabled = self.get_parameter(
            'is_segment_anything_segmentation_enabled').get_parameter_value().bool_value
        self._run_test = self.get_parameter('run_test').get_parameter_value().bool_value
        self._max_timeout_time_for_action_call = self.get_parameter(
            'max_timeout_time_for_action_call').get_parameter_value().double_value
        self._num_cycles = self.get_parameter('num_cycles').get_parameter_value().integer_value
        self._use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self._offset_for_place_pose = self.get_parameter(
            'offset_for_place_pose').get_parameter_value().double_value
        self._offset_for_insertion_pose = self.get_parameter(
            'offset_for_insertion_pose').get_parameter_value().double_value
        self._target_joint_state_for_place_pose = JointState()
        self._target_joint_state_for_place_pose.position = \
            self._target_joint_state_for_place_pose_arr_value
        self._target_joint_state_for_place_pose.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint'
        ]
        self._target_joint_state_for_place_pose.velocity = \
            [0.0] * len(self._target_joint_state_for_place_pose.position)
        self._target_joint_state_for_place_pose.effort = \
            [0.0] * len(self._target_joint_state_for_place_pose.position)

        # Set up clients
        self.setUpClients()

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self.failure_count = 0
        self.total_count = 0

        self._detection_client = ActionClient(self, GetObjects, '/get_objects')
        self._get_objects_goal_handle = None
        self._get_objects_result_received = False
        self._get_objects_result_future = None
        self._get_objects_detected_object_id = None

        self._get_object_pose_client = ActionClient(self, GetObjectPose, '/get_object_pose')
        self._get_object_pose_goal_handle = None
        self._get_object_pose_result_received = False
        self._get_object_pose_result_future = None

        self._segmentation_client = ActionClient(self, AddSegmentationMask,
                                                 '/add_segmentation_mask')
        self._segmentation_goal_handle = None
        self._segmentation_result_received = False
        self._segmentation_result_future = None

        self._add_mesh_client = self.create_client(AddMeshToObject, '/add_mesh_to_object')
        self._add_mesh_goal_handle = None
        self._add_mesh_result_received = False
        self._add_mesh_result_future = None

        self.latest_pose_gotten = None

        self._clear_objects_client = self.create_client(ClearObjects, '/clear_objects')
        self._clear_objects_goal_handle = None
        self._clear_objects_result_received = False
        self._clear_objects_result_future = None

        self.object_pose_estimation_result = None
        self.ground_truth_pose_estimation = None

        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._current_gear = None

        self._joint_state_publisher = self.create_publisher(
            JointState, '/isaac_joint_commands', 10)

        self._gripper_close_pos_publisher = self.create_publisher(
            Float64, '/gripper_close_pos', 10)

        self._switch_sim_controller_to_rl_policy_publisher = self.create_publisher(
            Bool, '/stop_joint_commands', 10)

        self._sim_joint_commands_publisher = self.create_publisher(
            JointState, '/isaac_joint_commands', 10)

        self._gripper_client = ActionClient(
            self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self._gripper_done_event = Event()
        self._gripper_done_result = False

        self.peg_pose = None

        # Also make Action server
        self._action_server = ActionServer(
            self, GearAssembly, '/gear_assembly',
            execute_callback=self.execute_callback)

    def _set_controller_states(self,
                               controllers_to_activate: List[str] = None,
                               controllers_to_deactivate: List[str] = None,
                               timeout_sec: float = 10.0) -> bool:
        """
        Set the state of ROS2 controllers by activating/deactivating them.

        This function uses the controller manager's switch_controller service to
        enable and disable controllers.

        Args
        ----
            node: ROS2 node to use for service calls
            controllers_to_activate: List of controller names to activate
            controllers_to_deactivate: List of controller names to deactivate
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if all controller state changes were successful, False otherwise

        Example
        -------
            # Switch from joint trajectory to impedance control
            success = self._set_controller_states(
                node=self,
                controllers_to_activate=['impedance_controller'],
                controllers_to_deactivate=['scaled_joint_trajectory_controller']
            )

        """
        if SwitchController is None:
            self.get_logger().error(
                'controller_manager_msgs not available. '
                'Install ros-humble-controller-manager-msgs')
            return False

        if controllers_to_activate is None:
            controllers_to_activate = []
        if controllers_to_deactivate is None:
            controllers_to_deactivate = []

        if not controllers_to_activate and not controllers_to_deactivate:
            self.get_logger().warn('No controllers specified to activate or deactivate')
            return True

        # Create service client
        service_name = '/controller_manager/switch_controller'
        client = self.create_client(SwitchController, service_name)

        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Service {service_name} not available '
                                    f'after {timeout_sec} seconds')
            return False

        # Create request
        request = SwitchController.Request()
        request.activate_controllers = controllers_to_activate
        request.deactivate_controllers = controllers_to_deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=timeout_sec).to_msg()

        self.get_logger().info(
            f'Switching controllers - Activating: {controllers_to_activate}, '
            f'Deactivating: {controllers_to_deactivate}')

        # Send request
        try:
            future = client.call_async(request)

            while not future.done():
                time.sleep(0.1)

            if future.result() is None:
                self.get_logger().error('Failed to call controller switch service')
                return False

            result = future.result()
            if result.ok:
                self.get_logger().info('Successfully switched controllers')
                return True
            else:
                self.get_logger().error('Controller switch failed')
                return False

        except Exception as e:
            self.get_logger().error(f'Exception during controller switch: {str(e)}')
            return False
        finally:
            self.destroy_client(client)

    def switch_to_impedance_control(self, timeout_sec: float = 10.0) -> bool:
        """
        Switch from joint trajectory control to impedance control.

        Args
        ----
            node: ROS2 node to use for service calls
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if switch was successful, False otherwise

        """
        return self._set_controller_states(
            controllers_to_activate=['impedance_controller'],
            controllers_to_deactivate=['scaled_joint_trajectory_controller'],
            timeout_sec=timeout_sec
        )

    def switch_to_trajectory_control(self, timeout_sec: float = 10.0) -> bool:
        """
        Switch from impedance control to joint trajectory control.

        Args
        ----
            node: ROS2 node to use for service calls
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if switch was successful, False otherwise

        """
        return self._set_controller_states(
            controllers_to_activate=['scaled_joint_trajectory_controller'],
            controllers_to_deactivate=['impedance_controller'],
            timeout_sec=timeout_sec
        )

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            self.get_logger().error(f'Failed to get transform from tf: {str(e)}')
            return None

    def trigger_get_objects_goal(self, initial_hint: None | Point = None):
        """Send a goal to the object detection action server."""
        self.get_logger().info('Waiting for action server for object detection...')
        if not self._detection_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'Action server for object detection not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        if initial_hint is not None:
            goal_msg.initial_hint = initial_hint
            goal_msg.use_initial_hint = True

        self.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._detection_client.send_goal_async(
            goal_msg, feedback_callback=self.get_objects_feedback_callback)
        send_goal_future.add_done_callback(self.get_objects_goal_response_callback)

    def get_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_objects_goal_handle = future.result()
        if not self._get_objects_goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by the action server.')
        self._get_objects_result_future = self._get_objects_goal_handle.get_result_async()
        self._get_objects_result_future.add_done_callback(self.get_objects_result_callback)

    def get_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        object_ids = []
        if result.objects:
            self.get_logger().info('Objects detected, the objects id are below:')
            for object_obj in result.objects:
                self.get_logger().info(f'Object: {object_obj.object_id}')
                object_ids.append(object_obj.object_id)

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self._get_objects_detected_object_id = result.objects[0].object_id
            self._get_objects_result_received = True
        else:
            self.get_logger().error(f'Action failed with status: {status}')
            self._get_objects_result_received = False

    def trigger_get_object_pose_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.get_logger().info('Waiting for pose estimation action server...')
        if not self._get_object_pose_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Pose estimation action server not available')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._get_object_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.get_object_pose_feedback_callback)
        send_goal_future.add_done_callback(self.get_object_pose_goal_response_callback)

    def get_object_pose_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_object_pose_goal_handle = future.result()
        if not self._get_object_pose_goal_handle.accepted:
            self.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.get_logger().info('Pose estimation goal accepted by the action server.')
        self._get_object_pose_result_future = self._get_object_pose_goal_handle.get_result_async()
        self._get_object_pose_result_future.add_done_callback(self.get_object_pose_result_callback)

    def get_object_pose_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_object_pose_result_callback(self, future):
        """Handle the result from the action server."""
        status = future.result().status
        result = future.result().result
        self.get_logger().info(f'Pose estimation result: {result}')

        # Log result to INFO
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Object pose received for object id')
            self._get_object_pose_result_received = True
            self.latest_pose_gotten = result.object_pose
            # base_T_gear_large_estimated, this is probably Pose
            # Convert it to ROS transform.
            self.object_pose_estimation_result = geometry_utils.ros_pose_to_ros_transform(
                result.object_pose)
        else:
            self.get_logger().error(f'Pose estimation action failed with status: {status}')
            self._get_object_pose_result_received = False

    def trigger_get_segmented_object_goal(self, object_id):
        """Send a goal to the object segmentation action server."""
        self.get_logger().info('Waiting for segmentation action server...')
        if not self._segmentation_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Segmentation action server not available after waiting')
            return

        goal_msg = AddSegmentationMask.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to get segmented object for ID: {object_id}')
        send_goal_future = self._segmentation_client.send_goal_async(
            goal_msg, feedback_callback=self.get_segmented_object_feedback_callback)
        send_goal_future.add_done_callback(self.get_segmented_object_goal_response_callback)

    def get_segmented_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._segmentation_goal_handle = future.result()
        if not self._segmentation_goal_handle.accepted:
            self.get_logger().error('Segmentation goal was rejected by the action server.')
            return

        self.get_logger().info('Segmentation goal accepted by the action server.')
        self._segmentation_result_future = self._segmentation_goal_handle.get_result_async()
        self._segmentation_result_future.add_done_callback(
            self.get_segmented_object_result_callback)

    def get_segmented_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Segmentation Feedback: {feedback_msg.feedback}')

    def get_segmented_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Segmented object received: {result}')
            self._segmentation_result_received = True
        else:
            self.get_logger().error(f'Segmentation action failed with status: {status}')
            self._segmentation_result_received = False

    def trigger_add_mesh_to_object_request(self, object_id, mesh_file_path):
        request = AddMeshToObject.Request()
        request.object_ids = [object_id]
        request.mesh_file_paths = [mesh_file_path]
        future = self._add_mesh_client.call_async(request)
        return future

    def send_clear_objects_request(self, object_ids=None):
        request = ClearObjects.Request()
        if object_ids:
            request.object_ids = object_ids
        future = self._clear_objects_client.call_async(request)
        return future

    def add_mesh_to_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._add_mesh_goal_handle = future.result()
        if not self._add_mesh_goal_handle.accepted:
            self.get_logger().error('Add mesh goal was rejected by the action server.')
            return

        self.get_logger().info('Add mesh goal accepted by the action server.')
        self._add_mesh_result_future = self._add_mesh_goal_handle.get_result_async()
        self._add_mesh_result_future.add_done_callback(self.add_mesh_to_object_result_callback)

    def add_mesh_to_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Add Mesh Feedback: {feedback_msg.feedback}')

    def add_mesh_to_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        self.get_logger().info(f'Add mesh result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Mesh added to object: {result.object_id}')
            self._add_mesh_result_received = True
        else:
            self.get_logger().error(f'Add mesh action failed with status: {status}')
            self._add_mesh_result_received = False

    def clear_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._clear_objects_goal_handle = future.result()
        if not self._clear_objects_goal_handle.accepted:
            self.get_logger().error('Clear objects goal was rejected by the action server.')
            return

        self.get_logger().info('Clear objects goal accepted by the action server.')
        self._clear_objects_result_future = self._clear_objects_goal_handle.get_result_async()
        self._clear_objects_result_future.add_done_callback(self.clear_objects_result_callback)

    def clear_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Clear Objects Feedback: {feedback_msg.feedback}')

    def clear_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        self.get_logger().info(f'Clear objects result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Objects cleared successfully')
            self._clear_objects_result_received = True
        else:
            self.get_logger().error(f'Clear objects action failed with status: {status}')
            self._clear_objects_result_received = False

    def wait_for_result(self, result_var_name: str, timeout: float = 10.0):
        start_time = self.get_clock().now()
        while rclpy.ok():
            # Check the actual instance variable dynamically
            if result_var_name and getattr(self, result_var_name):
                return True

            time_now = self.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                # self.fail(f'Timeout waiting for {result_var_name}')
                self.failure_count += 1
                return False
            self.total_count += 1

            time.sleep(0.1)

    def wait_for_service_result(self, future, timeout=10.0):
        start_time = self.get_clock().now()
        while not future.done():
            time_now = self.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.get_logger().error('Timeout waiting for service result')
                return None
            time.sleep(0.1)
        return future.result()

    def set_gear_mesh_param(self, mesh_file_path: str) -> bool:
        """
        Set the mesh_file_path parameter for the foundationpose_node based on the current gear.

        Args
        ----
            mesh_file_path: Path to the mesh file.

        Returns
        -------
            bool: True if the mesh_file_path parameter was set successfully, False otherwise.

        """
        # Create a parameter client
        param_client = self.create_client(
            SetParameters,
            '/foundationpose_node/set_parameters'
        )

        # Wait for the service to be available
        if not param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Service /foundationpose_node/set_parameters not available'
                ' skipping mesh parameter setting')
            return False

        # Create parameter request
        request = SetParameters.Request()
        parameter = RCLPYParameter()
        parameter.name = 'mesh_file_path'
        parameter.value = ParameterValue(
            string_value=mesh_file_path,
            type=rclpy.Parameter.Type.STRING.value)

        request.parameters = [parameter]

        # Send request
        future = param_client.call_async(request)

        while not future.done():
            time.sleep(0.1)

        if future.result() is None:
            self.get_logger().warn(f'Failed to set mesh_file_path to {mesh_file_path}')
            return False
        self.get_logger().debug(f'Successfully set mesh_file_path to {mesh_file_path}')
        return True

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Pose): The grasp pose to publish.
            parent_frame (str): The parent frame.
            child_frame (str): The child frame.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def publish_pose_on_tf(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """Publish the grasp transform in the world frame."""
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_stamped)
        self.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def wait_for_point_topic_func(self):
        """Wait for a message on the point topic."""
        self.received_messages[self._point_topic_name_as_trigger] = []
        self.get_logger().error('Waiting for point topic input for next step...')
        while True:
            time.sleep(0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.get_logger().info('Point topic received')
                return

    def do_perception_loop(self, initial_hint: Point, mesh_file_path: str):
        # First detect object
        self.trigger_get_objects_goal(initial_hint)

        # Wait for the detection result with a timeout
        self.wait_for_result('_get_objects_result_received',
                             timeout=self._max_timeout_time_for_action_call)
        if not self._get_objects_result_received:
            self.get_logger().error('Object detection action failed')
            self.failure_count += 1
            return False

        self.total_count += 1
        # If segment anything is enabled, we need to get the segmented object first
        if self._is_segment_anything_segmentation_enabled:
            self.trigger_get_segmented_object_goal(self._get_objects_detected_object_id)
            self.wait_for_result('_segmentation_result_received',
                                 timeout=self._max_timeout_time_for_action_call)

        # Now add mesh to object
        if self._is_segment_anything_segmentation_enabled:
            future = self.trigger_add_mesh_to_object_request(
                self._get_objects_detected_object_id, mesh_file_path)
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.get_logger().info(f'Add mesh to object result: {result}')
            if result is None:
                self.get_logger().error('Add mesh to object action failed')
                self.failure_count += 1

                return False
            self.total_count += 1

        # Now get object pose
        self.trigger_get_object_pose_goal(self._get_objects_detected_object_id)
        self.wait_for_result('_get_object_pose_result_received',
                             timeout=self._max_timeout_time_for_action_call)

        self._get_objects_result_received = False
        self._segmentation_result_received = False
        self._add_mesh_result_received = False
        self._get_object_pose_result_received = False
        return True

    def do_perception_for_pose_estimation(self, mesh_file_path: str):
        if self._wait_for_point_topic and self.peg_pose is None:

            initial_hint_for_peg_stand_estimation = None
            initial_hint_for_gear_insertion = None

            self.get_logger().error(
                'Waiting for point topic input for Peg stand estimation...')
            # Wait to get a message on the point topic.
            while True:
                time.sleep(0.1)
                if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                    self.get_logger().info('Point topic received')
                    self.initial_hint_point_msg = \
                        self.received_messages[self._point_topic_name_as_trigger][-1]
                    initial_hint_for_peg_stand_estimation = self.initial_hint_point_msg
                    self.received_messages[self._point_topic_name_as_trigger] = []
                    break

        if self.peg_pose is None:

            # Do perception loop.
            self.get_logger().info(
                'Setting mesh file path for peg stand estimation'
                f': {self._mesh_file_path_for_peg_stand_estimation}')

            self.set_gear_mesh_param(self._mesh_file_path_for_peg_stand_estimation)

            self.get_logger().info('Doing perception loop for peg stand estimation')

            is_peg_stand_estimation_success = self.do_perception_loop(
                initial_hint_for_peg_stand_estimation,
                self._mesh_file_path_for_peg_stand_estimation)
            while not is_peg_stand_estimation_success:
                self.get_logger().error('Peg stand estimation action failed')
                is_peg_stand_estimation_success = self.do_perception_loop(
                    initial_hint_for_peg_stand_estimation,
                    self._mesh_file_path_for_peg_stand_estimation)
                self.get_logger().info('Peg stand estimation action failed, retrying...')

            # This is w.r.t to camera, but we need to convert it to base link.
            self.peg_pose = self.latest_pose_gotten

            self.received_messages[self._point_topic_name_as_trigger] = []
            self.get_logger().info(f'Peg pose: {self.peg_pose}')

            # Now publish this on TF Static with parent farme being
            # camera_frame and child frame being gear_assembly_est_frame.
            self.get_logger().info('Publishing peg pose on TF Static')
            self.publish_pose_on_tf_static(self.peg_pose,
                                           parent_frame=self._camera_prim_name_in_tf,
                                           child_frame='gear_assembly_frame')

        self.received_messages[self._point_topic_name_as_trigger] = []
        self.get_logger().error('Waiting for point topic for gear insertion...')

        # Wait to get a message on the point topic.
        while True:
            time.sleep(0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.get_logger().info('Point topic received')
                self.initial_hint_point_msg = \
                    self.received_messages[self._point_topic_name_as_trigger][-1]
                initial_hint_for_gear_insertion = self.initial_hint_point_msg
                break

        self.get_logger().info(
            f'Setting mesh file path for gear insertion: {mesh_file_path}')
        self.set_gear_mesh_param(mesh_file_path)
        self.get_logger().info('Doing perception loop for gear insertion')
        is_gear_estimation_success = self.do_perception_loop(initial_hint_for_gear_insertion,
                                                             mesh_file_path)
        while not is_gear_estimation_success:
            self.get_logger().info('Gear estimation action failed, retrying...')
            is_gear_estimation_success = self.do_perception_loop(
                initial_hint_for_gear_insertion, mesh_file_path)
            time.sleep(2)

        self.gear_pose = self.latest_pose_gotten

    def update_gripper_close_pos(self, close_gripper_pos: float):
        """Update the gripper close position."""
        self.get_logger().info(f'Updating gripper close position to {close_gripper_pos}')

        for _ in range(10):
            self._gripper_close_pos_publisher.publish(Float64(data=close_gripper_pos))
            time.sleep(0.1)
            self.get_logger().info('Robot has updated its gripper close position')

        self.get_logger().info('Robot has updated its gripper close position')

    def pick_and_place(
        self,
        object_id: int,
        class_id: str = '',
        place_pose: Pose = None,
        gripper_closed_position: float = 0.623,
        use_joint_space_planner: bool = False,
        keep_gripper_closed_after_completion: bool = False,
        only_perform_place: bool = False
    ) -> bool:
        """
        Pick and place an object using the PickAndPlace action.

        Args
        ----
            node: The node to use for the behavior.
            object_id: The ID of the object to pick.
            class_id: The class ID of the object (optional).
            place_pose: The pose where to place the object
            use_joint_space_planner: Whether to use the joint space planner.
            keep_gripper_closed_after_completion: Whether to keep the gripper closed after
                completion.

        Returns
        -------
            True if the behavior was successful, False otherwise.

        """
        pick_and_place_client = ActionClient(self, PickAndPlace, '/pick_and_place')

        # Wait for the action server to be available
        if not pick_and_place_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('PickAndPlace action server not available')
            return False

        # Create the goal
        pick_and_place_goal = PickAndPlace.Goal()
        pick_and_place_goal.object_id = object_id
        pick_and_place_goal.class_id = class_id
        pick_and_place_goal.use_joint_space_planner_for_place_pose = use_joint_space_planner
        pick_and_place_goal.target_joint_state_for_place_pose = \
            self._target_joint_state_for_place_pose
        pick_and_place_goal.gripper_closed_position = gripper_closed_position
        pick_and_place_goal.keep_gripper_closed_after_completion = \
            keep_gripper_closed_after_completion
        pick_and_place_goal.place_pose = place_pose
        pick_and_place_goal.only_perform_place = only_perform_place

        # Send the goal
        self.get_logger().info(f'Sending PickAndPlace goal for object_id: {object_id}')
        goal_future = pick_and_place_client.send_goal_async(pick_and_place_goal)

        # Wait for the goal to be accepted
        while not goal_future.done():
            time.sleep(0.1)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('PickAndPlace goal was rejected')
            return False

        self.get_logger().info('PickAndPlace goal accepted')

        # Wait for the result
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.1)
        result = result_future.result()

        if not result.result.success:
            self.get_logger().error(f'PickAndPlace failed with status: {result.result}')
            return False

        self.get_logger().info(f'PickAndPlace completed successfully: {result.result}')
        return True

    def rl_insertion(
        self,
        peg_pose: Pose
    ) -> bool:
        """
        Insert an object into a peg using the RL insertion action.

        Args
        ----
            peg_pose: The pose of the peg to insert the object into.

        Returns
        -------
            True if the insertion was successful, False otherwise.

        """
        insertion_client = ActionClient(self, Insert, '/gear_assembly/insert_policy')

        if not insertion_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Insertion action server not available')
            return False

        goal = Insert.Goal()
        goal.goal_pose = PoseStamped()
        goal.goal_pose.pose = peg_pose
        goal.goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal.goal_pose.header.frame_id = 'base_link'  # This is base or base link.
        goal_future = insertion_client.send_goal_async(goal)

        while not goal_future.done():
            time.sleep(0.1)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Insertion goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.1)
        self.get_logger().info('Finished with RL insertion.')
        return True

    def _switch_sim_controller_to_rl_policy(self, is_rl_policy: bool):
        """Switch the sim controller to the RL policy."""
        # Just publish this Bool msg to a topic 5 times.
        for _ in range(5):
            self._switch_sim_controller_to_rl_policy_publisher.publish(Bool(data=is_rl_policy))
            time.sleep(0.1)

        self.get_logger().info('Switched sim controller to RL policy')

    def _take_simulation_robot_to_home_position(self):
        """Take the simulation robot to the home position."""
        # Publish a joint topic to sim joint commands topic.
        joint_positions = [
            2.7424,
            -0.9247,
            1.2466,
            -1.8927,
            -1.5708,
            -1.8895
        ]
        # Create JointState message
        joint_state = JointState()
        joint_state.position = joint_positions
        joint_state.velocity = [0.0] * len(joint_positions)
        joint_state.effort = [0.0] * len(joint_positions)

        # Set joint names based on the number of joints (assuming UR10e joint names)
        joint_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.get_logger().info(f'Taking robot to home position: {joint_positions}')

        self.get_logger().info('Publishing joint state to sim joint commands topic')
        self._sim_joint_commands_publisher.publish(joint_state)

        self.get_logger().info('Done publishing joint state to sim joint commands topic')

    def trigger_gripper(self, position: float, max_effort: float):
        """
        Trigger gripper to either open or close.

        Args
        ----
            position (float): Position
            max_effort (float): Max effort

        Returns
        -------
            None

        """
        self._gripper_done_event.clear()
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(position)
        gripper_goal.command.max_effort = float(max_effort)

        gripper_goal_future = self._gripper_client.send_goal_async(gripper_goal)
        gripper_goal_future.add_done_callback(self.gripper_goal_response_cb)

    def gripper_goal_response_cb(self, future):
        """
        Get goal response for Gripper.

        Args
        ----
            future (_type_): Future for goal response

        Returns
        -------
            None

        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for GripperCommand')
            self._gripper_done_result = False
            self._gripper_done_event.set()
            return
        # Cache the goal handle for canceling
        self.get_logger().info('Goal accepted for GripperCommand')
        gripper_result_future = goal_handle.get_result_async()
        gripper_result_future.add_done_callback(self.gripper_goal_result_cb)

    def gripper_goal_result_cb(self, future: Any):
        """
        Get the final result from gripper trigger action call.

        Args
        ----
            future (Any): Future for goal result

        Returns
        -------
            None

        """
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('GripperCommand action server succeeded.')
            self._gripper_done_result = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('GripperCommand action server aborted.')
            self._gripper_done_result = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('GripperCommand action server canceled.')
            self._gripper_done_result = False
        else:
            self.get_logger().error('GripperCommand action server failed.')
            self._gripper_done_result = False

        self._gripper_done_event.set()

    def _open_gripper(self):
        """Open the gripper."""
        self.trigger_gripper(position=0.0, max_effort=10.0)

    def create_logging_subscribers(
        self,
        subscription_requests: Iterable[Tuple[str, Any]],
        received_messages: Dict[str, Iterable],
        use_namespace_lookup: bool = True,
        accept_multiple_messages: bool = False,
        add_received_message_timestamps: bool = False,
        qos_profile: QoSProfile = 10
    ) -> Iterable[Subscription]:
        """
        Create subscribers that log any messages received to the passed-in dictionary.

        Parameters
        ----------
        subscription_requests : Iterable[Tuple[str, Any]]
            List of topic names and topic types to subscribe to.

        received_messages : Dict[str, Iterable]
            Output dictionary mapping topic name to list of messages received

        use_namespace_lookup : bool
            Whether the object's namespace dictionary should be used for topic
            namespace remapping, by default True

        accept_multiple_messages : bool
            Whether the generated subscription callbacks should accept multiple messages,
            by default False

        add_received_message_timestamps : bool
            Whether the generated subscription callbacks should add a timestamp to the messages,
            by default False

        qos_profile : QoSProfile
            What Quality of Service policy to use for all subscribers

        Returns
        -------
        Iterable[Subscription]
            List of subscribers, passing the unsubscribing responsibility to the caller

        """
        received_messages.clear()
        if accept_multiple_messages:
            for topic, _ in subscription_requests:
                received_messages[topic] = []

        def make_callback(topic):
            def callback(msg):
                if accept_multiple_messages:
                    if add_received_message_timestamps:
                        received_messages[topic].append((msg, time.time()))
                    else:
                        received_messages[topic].append(msg)
                else:
                    if topic in received_messages:
                        self.get_logger().error(f'Already received a message on topic {topic}!')
                        raise ValueError(f'Already received a message on topic {topic}!')
                    received_messages[topic] = msg

            return callback

        try:
            subscriptions = [
                self.create_subscription(
                    msg_type,
                    self.namespaces[topic] if use_namespace_lookup else topic,
                    make_callback(topic),
                    qos_profile,
                ) for topic, msg_type in subscription_requests
            ]
        except Exception as e:
            # Silent failures have been observed here. We print and raise to make sure that a
            # trace ends up at the console.
            print('Failed to create subscriptions:')
            print(e)
            raise

        return subscriptions

    def gear_assembly(self) -> bool:
        """Gear assembly function."""
        if self._wait_for_point_topic:
            self.received_messages = {}
            self._subs = self.create_logging_subscribers(
                [(self._point_topic_name_as_trigger, Point)],
                self.received_messages,
                use_namespace_lookup=False,
                accept_multiple_messages=True,
                qos_profile=rclpy.qos.qos_profile_sensor_data)
            self.initial_hint_point_msg = None

        self.get_logger().info('Starting test for manipulator servers POL')
        gears_to_insert = [
            'gear_shaft_large',
            'gear_shaft_small',
            'gear_shaft_medium'
        ]

        gear_ground_truth_sim_names = [
            'gear_large',
            'gear_small',
            'gear_medium'
        ]

        for idx, gear_type in enumerate(gears_to_insert):
            if not self._use_ground_truth_pose_estimation:
                self.get_logger().error('Waiting for user input to trigger perception')
                self.wait_for_point_topic_func()
            self.get_logger().info(f'Triggering assembly for gear type: {gear_type}')

            mesh_file_path_for_gear = self._mesh_file_paths[idx]
            close_gripper_pos = self._gripper_close_pos[idx]

            if self._use_sim_time:
                # We need to do this as we have a control layer for the gripper in Isaac Sim.
                self.update_gripper_close_pos(close_gripper_pos)
                close_gripper_pos = self._gripper_close_pos[idx]
            else:
                # For real robot this doesnt matter, the gripper doesn't matter, it doesnt cause
                # the gear to fly out of the grasp.
                close_gripper_pos = 0.65

            if not self._use_ground_truth_pose_estimation:
                self.do_perception_for_pose_estimation(mesh_file_path_for_gear)

            if self._use_sim_time and self._use_ground_truth_pose_estimation:

                for _ in range(10):
                    time.sleep(0.1)
                # base_T_gear_large_actual, do it w.r.t camera since we get that pose
                # out of the FP node and thats the one the server gives you.
                self.ground_truth_pose_estimation_held_asset = self._get_transform_from_tf(
                    gear_ground_truth_sim_names[idx],
                    'base_link')
                if self.ground_truth_pose_estimation_held_asset is None:
                    self.get_logger().error('Failed to get ground truth pose estimation')
                    return False

                self.gear_pose = self.ground_truth_pose_estimation_held_asset

            if self._use_sim_time and self._verify_pose_estimation_accuracy:
                # Get transformation matrix from ROS2 pose. 4 x 4 matrix.
                object_pose_estimation_transform = \
                    geometry_utils.get_transformation_matrix_from_ros(
                        self.object_pose_estimation_result.translation,
                        self.object_pose_estimation_result.rotation)
                ground_truth_pose_estimation_transform = \
                    geometry_utils.get_transformation_matrix_from_ros(
                        self.ground_truth_pose_estimation.translation,
                        self.ground_truth_pose_estimation.rotation)

                # Now do the check of the diff from ground truth.
                delta_transform = geometry_utils.compute_transformation_delta(
                    object_pose_estimation_transform, ground_truth_pose_estimation_transform)
                translation_norm_delta_cm, translation_ros_delta_mrad = \
                    geometry_utils.compute_delta_translation_rotation(delta_transform)
                self.get_logger().info(
                    f'Translation norm delta: {translation_norm_delta_cm} cm')
                self.get_logger().info(
                    f'Translation delta: {translation_ros_delta_mrad} mrad')

                self.ground_truth_pose_estimation_w_r_t_base = self._get_transform_from_tf(
                    gear_ground_truth_sim_names[idx], 'base_link')
                if self.ground_truth_pose_estimation_w_r_t_base is None:
                    self.get_logger().error('Failed to get ground truth pose estimation')
                    return False
                T_base = geometry_utils.get_transformation_matrix_from_ros(
                    self.ground_truth_pose_estimation_w_r_t_base.translation,
                    self.ground_truth_pose_estimation_w_r_t_base.rotation)

                self.get_logger().info(f'T_base translation in meters: {T_base[:3, 3]}')

            if not self._use_sim_time:
                self.switch_to_trajectory_control()

            if not self._use_ground_truth_pose_estimation:
                self.get_logger().error('Waiting for user input to trigger pick and place')
                self.wait_for_point_topic_func()

            for _ in range(10):
                time.sleep(0.1)

            place_pose_transform = self._get_transform_from_tf(
                gear_type, 'base_link')  # TODO change back to gear assembly
            if place_pose_transform is None:
                self.get_logger().error('Failed to get place pose transform')
                return False

            place_pose = geometry_utils.get_pose_from_transform(place_pose_transform)

            # Rotate place pose by 180 degrees along x to make X face down.
            # Then subtract 15 cm from z axis so that aligns well on top of peg.
            rotated_place_pose = geometry_utils.rotate_pose(place_pose, 180, 'x')
            rotated_place_pose.position.z += self._offset_for_place_pose

            self.get_logger().info(f'Doing gear pickup and place with'
                                   f' offset: {self._offset_for_place_pose}')

            self.publish_pose_on_tf(rotated_place_pose,
                                    parent_frame='base_link',
                                    child_frame='place_pose_static_frame')

            # Also publish detectedobject 1 on TF static
            if not self._use_ground_truth_pose_estimation:
                self.publish_pose_on_tf(self.gear_pose,
                                        parent_frame=self._camera_prim_name_in_tf,
                                        child_frame='detected_object1')
            else:
                gear_pose_in_pose_msg = geometry_utils.get_pose_from_transform(self.gear_pose)

                gear_pose_in_pose_msg.position.z += self._offset_for_insertion_pose
                self.publish_pose_on_tf_static(gear_pose_in_pose_msg,
                                               parent_frame='base_link',
                                               child_frame='detected_object1')
                self.get_logger().info(
                    f'Published detected object 1 on TF static: {gear_pose_in_pose_msg}')
                self._get_objects_detected_object_id = 0
                for _ in range(10):
                    time.sleep(0.1)

            # Now do PickAndHover action on that object using get objects detected id.
            is_place_and_hover_success = self.pick_and_place(
                object_id=self._get_objects_detected_object_id,
                gripper_closed_position=close_gripper_pos,
                place_pose=rotated_place_pose,
                use_joint_space_planner=self._use_joint_space_planner,
                keep_gripper_closed_after_completion=True
            )

            if not is_place_and_hover_success:
                self.get_logger().error('Pick and place action failed')
                self.failure_count += 1
                return False

            for _ in range(10):
                time.sleep(0.1)

            # Use base here because policy sees base.
            peg_pose_transform = self._get_transform_from_tf(
                gear_type, 'base')
            if peg_pose_transform is None:
                self.get_logger().error('Failed to get peg pose transform')
                return False
            peg_pose = geometry_utils.get_pose_from_transform(peg_pose_transform)

            # # Publish this pose for sanity under a new name on tf static
            self.publish_pose_on_tf_static(peg_pose,
                                           parent_frame='base',
                                           child_frame='rl_insertion_pose_frame')
            self.get_logger().info(f'Peg pose: {peg_pose}')

            if self._run_rl_inference:
                if not self._use_sim_time:
                    self.switch_to_impedance_control()
                    # Wait to get a message on the point topic.
                    self.get_logger().error(
                        'Waiting for confirmation to start RL policy...'
                        f'click any point on image. Will try to insert on TF: {gear_type}')
                    self.wait_for_point_topic_func()
                else:
                    self._switch_sim_controller_to_rl_policy(is_rl_policy=True)

                is_rl_insertion_success = self.rl_insertion(
                    peg_pose=peg_pose
                )

                if not is_rl_insertion_success:
                    self.get_logger().error('RL insertion action failed')
                    return False

                if not self._use_sim_time:
                    self.switch_to_trajectory_control()

                    home_pose = rotated_place_pose
                    home_pose.position.z += self._offset_for_place_pose
                    # Now do PickAndHover action on that object using get objects detected id.
                    is_go_to_home_pose_success = self.pick_and_place(
                        object_id=self._get_objects_detected_object_id,
                        gripper_closed_position=close_gripper_pos,
                        place_pose=home_pose,
                        use_joint_space_planner=self._use_joint_space_planner,
                        keep_gripper_closed_after_completion=True,
                        only_perform_place=True
                    )

                self.get_logger().info('RL insertion action completed successfully')

                if not self._use_sim_time and not is_go_to_home_pose_success:
                    self.get_logger().error('Go to home pose failed.')
                    return False

                if self._use_sim_time:
                    # Open gripper.
                    self._open_gripper()
                    self._gripper_done_event.wait(timeout=10.0)

                    self._take_simulation_robot_to_home_position()
                    self.get_logger().info('Waiting for 5 seconds')
                    time.sleep(5.0)
                    self.get_logger().info('Done waiting')
                    # Switching back to cuMotion controller.
                    self._switch_sim_controller_to_rl_policy(is_rl_policy=False)
                    # Reset the gripper done event and result.
                    self._gripper_done_event.clear()
                    self._gripper_done_result = False

            else:
                continue

            # Now clear objects
            future = self.send_clear_objects_request()
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.get_logger().info(f'Clear objects result: {result}')
            if result is None:
                self.get_logger().error('Clear objects action failed')
                self.failure_count += 1
                continue
            self.total_count += 1

            # Reset for next iteration
            self._clear_objects_result_received = False

        self.get_logger().info(f'Failure count: {self.failure_count}')
        if self.failure_count/self.total_count > 0.3:
            self.get_logger().error(f'Pose estimation action failed'
                                    f'{self.failure_count}/{self.total_count} times')
            return False
        return True

    def execute_callback(self, goal_handle) -> GearAssembly.Result:
        """
        Execute the gear assembly action call functionality.

        Args
        ----
            goal_handle: Goal handle for the GearAssembly action

        Returns
        -------
            Insert.Result: Gear assembly result

        """
        self.get_logger().info('Executing gear assembly goal...')
        result = GearAssembly.Result()

        # Call the main function
        success = self.gear_assembly()

        # Set success result
        if success:
            result.success = True
            goal_handle.succeed()
        else:
            result.success = False
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)

    gear_assembly_orchestrator = IsaacManipulatorGearAssemblyOrchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(gear_assembly_orchestrator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        gear_assembly_orchestrator.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    gear_assembly_orchestrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
