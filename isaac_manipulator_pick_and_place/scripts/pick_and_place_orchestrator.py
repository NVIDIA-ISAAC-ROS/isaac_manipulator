#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math
import time
from threading import Event

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseArray, Pose, TransformStamped, Vector3
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from visualization_msgs.msg import Marker

from curobo.types.math import Pose as CuPose
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from end_effector_marker import EndEffectorMarker
from grasp_reader import GraspReader, Transformation
from isaac_ros_cumotion import CumotionGoalSetClient
from isaac_ros_cumotion_interfaces.action import AttachObject
from isaac_manipulator_interfaces.action import GetObjectPose, PickAndPlace

from isaac_manipulator_ros_python_utils.types import (
    ObjectAttachmentShape
)


class PickAndPlaceOrchestrator(Node):

    def __init__(self):
        super().__init__('pick_and_place_orchestrator')
        # Declare parameters
        self.declare_parameter('gripper_collision_links', [''])
        self.declare_parameter('gripper_action_name', '/robotiq_gripper_controller/gripper_cmd')
        self.declare_parameter('time_dilation_factor', 0.2)
        self.declare_parameter('use_ground_truth_pose_from_sim', False)
        self.declare_parameter('publish_grasp_frame', False)
        self.declare_parameter('sleep_time_before_planner_tries_sec', 0.5)
        self.declare_parameter('object_frame_name', 'detected_object1')
        self.declare_parameter('num_planner_tries', 5)
        self.declare_parameter('attach_object_fallback_radius', 0.15)
        self.declare_parameter('grasp_file_path', '')
        self.declare_parameter('grasp_approach_offset_distance', [0.0, 0.0, -0.15])
        self.declare_parameter('retract_offset_distance', [0.0, 0.0, 0.15])
        self.declare_parameter('attach_object_shape', str(ObjectAttachmentShape.CUBOID.value))
        self.declare_parameter('attach_object_mesh_file_path', '')
        self.declare_parameter('attach_object_scale', [0.05, 0.05, 0.1])

        # Whether the grasp approach and retract are in the world frame or the goal frame
        self.declare_parameter('grasp_approach_in_world_frame', False)
        self.declare_parameter('retract_in_world_frame', True)

        # When this parameter is set to True, the place pose of the end effector is taken from the
        # RViz via the interactive marker and place pose in the action request is ignored.
        self.declare_parameter('use_pose_from_rviz', False)
        # The mesh resource URI for the end effector
        mesh_uri = 'package://isaac_manipulator_pick_and_place/meshes/robotiq_2f_85.obj'
        self.declare_parameter('end_effector_mesh_resource_uri', mesh_uri)

        # Extract the parameters
        self._is_running_isaac_sim = \
            self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self._use_ground_truth_pose_from_sim = self.get_parameter(
            'use_ground_truth_pose_from_sim').get_parameter_value().bool_value
        self._publish_grasp_frame = self.get_parameter(
            'publish_grasp_frame').get_parameter_value().bool_value
        self._sleep_time_before_planner_tries_sec = self.get_parameter(
            'sleep_time_before_planner_tries_sec').get_parameter_value().double_value
        self._num_planner_tries_ = self.get_parameter(
            'num_planner_tries').get_parameter_value().integer_value
        self._time_dilation_factor = self.get_parameter(
            'time_dilation_factor').get_parameter_value().double_value
        self._object_frame_name = \
            self.get_parameter('object_frame_name').get_parameter_value().string_value
        self._gripper_collision_links = (
            self.get_parameter('gripper_collision_links').get_parameter_value().string_array_value
        )
        gripper_action_name = (
            self.get_parameter('gripper_action_name').get_parameter_value().string_value
        )
        self._grasp_file_path = self.get_parameter(
            'grasp_file_path').get_parameter_value().string_value
        self._grasp_approach_offset_distance = self.get_parameter(
            'grasp_approach_offset_distance').get_parameter_value().double_array_value
        self._retract_offset_distance = self.get_parameter(
            'retract_offset_distance').get_parameter_value().double_array_value
        self._grasp_approach_in_world_frame = self.get_parameter(
            'grasp_approach_in_world_frame').get_parameter_value().bool_value
        self._retract_in_world_frame = self.get_parameter(
            'retract_in_world_frame').get_parameter_value().bool_value

        self._use_pose_from_rviz = self.get_parameter(
            'use_pose_from_rviz').get_parameter_value().bool_value
        self._end_effector_mesh_resource_uri = self.get_parameter(
            'end_effector_mesh_resource_uri').get_parameter_value().string_value

        self._attach_object_shape = ObjectAttachmentShape(self.get_parameter(
            'attach_object_shape').get_parameter_value().string_value)
        self._attach_object_mesh_file_path = self.get_parameter(
            'attach_object_mesh_file_path').get_parameter_value().string_value
        attach_object_scale_list = self.get_parameter(
            'attach_object_scale').get_parameter_value().double_array_value
        if len(attach_object_scale_list) != 3:
            self.get_logger().error('Received object scale length other than 3!')
            raise ValueError('Excepted object scale to be length 3!')

        self._attach_object_scale = Vector3()
        self._attach_object_scale.x = attach_object_scale_list[0]
        self._attach_object_scale.y = attach_object_scale_list[1]
        self._attach_object_scale.z = attach_object_scale_list[2]

        self._action_server = ActionServer(
            self, PickAndPlace, '/pick_and_place',
            execute_callback=self.execute_callback, cancel_callback=self.cancel_callback,)
        self._get_object_pose_cb_group = MutuallyExclusiveCallbackGroup()
        self._get_object_pose_client = ActionClient(
            self, GetObjectPose, '/get_object_pose', callback_group=self._get_object_pose_cb_group)
        self._get_pose_done_event = Event()
        self._get_pose_done_result = None

        self._planner = CumotionGoalSetClient(node=self)
        self.plan_result = None

        self._object_attach_cb_group = MutuallyExclusiveCallbackGroup()
        self._object_attach_client = ActionClient(
            self, AttachObject, 'attach_object', callback_group=self._object_attach_cb_group)
        self._object_attach_done_event = Event()
        self._object_attach_done_result = False

        self._gripper_cb_group = MutuallyExclusiveCallbackGroup()
        self._gripper_client = ActionClient(
            self, GripperCommand, gripper_action_name, callback_group=self._gripper_cb_group)
        self._gripper_done_event = Event()
        self._gripper_done_result = False

        self._grasp_reader = GraspReader(self._grasp_file_path)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = StaticTransformBroadcaster(self)

        self._client_goal_handles = []

        if self._use_pose_from_rviz:
            future = self._tf_buffer.wait_for_transform_async(
                'base_link', 'gripper_frame', rclpy.time.Time())
            future.add_done_callback(self.initialize_marker)

        self.get_logger().info('Pick and Place Orchestrator has been started.')

    def initialize_marker(self, future):
        self._end_effector_marker = EndEffectorMarker(
                node=self, marker_namespace='end_effector_marker',
                mesh_resource_uri=self._end_effector_mesh_resource_uri)

        if (future.done()):
            transform = self._tf_buffer.lookup_transform(
                'base_link', 'gripper_frame', rclpy.time.Time())
            initial_pose = Pose()
            initial_pose.position.x = transform.transform.translation.x
            initial_pose.position.y = transform.transform.translation.y
            initial_pose.position.z = transform.transform.translation.z
            initial_pose.orientation.w = transform.transform.rotation.w
            initial_pose.orientation.x = transform.transform.rotation.x
            initial_pose.orientation.y = transform.transform.rotation.y
            initial_pose.orientation.z = transform.transform.rotation.z
            self._end_effector_marker.set_pose(initial_pose)

        self.get_logger().info('End effector marker initialized.')

    def publish_grasp_transform(self, grasp_pose: Transformation | Pose,
                                frame_name: str = 'grasp_frame'):
        """Publishes the grasp frame onto TF to visualize where the robot is going to grasp onto
        TF static

        Args:
            grasp_pose (Pose): Grasp pose
            frame_name: Name of TF frame id
        """
        if grasp_pose is None:
            self.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = 'world'
        transform_stamped.child_frame_id = frame_name

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z

        if not isinstance(grasp_pose, Pose):
            transform_stamped.transform.rotation.x = grasp_pose.quaternion.x
            transform_stamped.transform.rotation.y = grasp_pose.quaternion.y
            transform_stamped.transform.rotation.z = grasp_pose.quaternion.z
            transform_stamped.transform.rotation.w = grasp_pose.quaternion.w
        else:
            transform_stamped.transform.rotation.x = grasp_pose.orientation.x
            transform_stamped.transform.rotation.y = grasp_pose.orientation.y
            transform_stamped.transform.rotation.z = grasp_pose.orientation.z
            transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self._tf_broadcaster.sendTransform(transform_stamped)
        self.get_logger().debug(f'Published grasp frame: {transform_stamped.child_frame_id}')

    def wait_for_server(self, action_client: ActionClient, timeout_sec: float = 5.0) -> bool:
        """Wait for the action server to be available.

        Args:
            action_client (ActionClient): Action client
            timeout_sec (float, optional): timeout for client call. Defaults to 5.0.

        Returns:
            bool: Returns if action server is ready for requests
        """
        self.get_logger().info(f'Waiting for {action_client._action_name} action server...')
        action_server_available = action_client.wait_for_server(timeout_sec)
        if not action_server_available:
            self.get_logger().error(
                f'{action_client._action_name} action server is not available. Aborting goal.')
            return False
        return True

    def read_grasp_poses(self) -> PoseArray:
        """Read the grasp poses from a yaml file.

        Returns:
            PoseArray: List of grasps to send to the planner for goal set planning.
        """
        poses_arr = PoseArray()
        try:
            # Get the grasp pose
            grasp_poses = self._grasp_reader.get_pose_for_pick_task(
                world_frame='base_link',
                object_frame_name=self._object_frame_name,
                tf_buffer=self._tf_buffer,
            )

            for i, grasp_pose in enumerate(grasp_poses):
                if self._publish_grasp_frame:
                    self.publish_grasp_transform(grasp_pose, f'grasp_frame_{i}')
                pose = Pose()
                poses_arr.header.frame_id = 'base_link'
                pose.position.x = grasp_pose.position.x
                pose.position.y = grasp_pose.position.y
                pose.position.z = grasp_pose.position.z
                pose.orientation.w = grasp_pose.quaternion.w
                pose.orientation.x = grasp_pose.quaternion.x
                pose.orientation.y = grasp_pose.quaternion.y
                pose.orientation.z = grasp_pose.quaternion.z
                poses_arr.poses.append(pose)

        except Exception as e:
            self.get_logger().error(f'Failed to compute grasp pose: {e}')

        return poses_arr

    def get_plan_grasp(self, goal_pose_array: PoseArray) -> bool:
        """Plan for the list of grasps.

        Args:
            goal_pose_array (PoseArray): List of grasps w.r.t base link

        Returns:
            bool: Returns if a plan for grasp was successful
        """
        # Constuct the contraints axis based on the linear offset distance
        # Index 0-2 is for rotational constraints in x, y, and z respectively
        # Index 3-5 is for linear constraints in x, y, and z respectively
        grasp_approach_path_constraints = [0.1] * 6
        for i, offset in enumerate(self._grasp_approach_offset_distance):
            # If offset is greater than 0.0 do not constraint that axis
            if not math.isclose(offset, 0.0, abs_tol=1e-5):
                grasp_approach_path_constraints[3 + i] = 0.0

        retract_path_constraints = [0.1] * 6
        for i, offset in enumerate(self._retract_offset_distance):
            # If offset is greater than 0.0 do not constraint that axis
            if not math.isclose(offset, 0.0, abs_tol=1e-5):
                retract_path_constraints[3 + i] = 0.0

        self.plan_result = self._planner.move_grasp(
            CuPose.from_list([0, 0, 0, 1, 0, 0, 0]),
            link_name='base_link',
            goal_pose_array=goal_pose_array,
            grasp_approach_offset_distance=self._grasp_approach_offset_distance,
            grasp_approach_path_constraint=grasp_approach_path_constraints,
            retract_offset_distance=self._retract_offset_distance,
            retract_path_constraint=retract_path_constraints,
            grasp_approach_constraint_in_goal_frame=not self._grasp_approach_in_world_frame,
            retract_constraint_in_goal_frame=not self._retract_in_world_frame,
            time_dilation_factor=self._time_dilation_factor,
            disable_collision_links=self._gripper_collision_links,
            update_planning_scene=True,)
        if not self.plan_result.success:
            self.get_logger().error(f'Planning failed. {self.plan_result.message}')
            self.get_logger().error(f'Error Code: {self.plan_result.error_code}')
            return False
        return True

    def get_plan_pose(self, pose: Pose) -> bool:
        """Gets planning pose from cuMotion.

        Args:
            pose (Pose): Pose in ROS format

        Returns:
            bool: Whether planning for pose was successful or not.
        """
        cu_pose = CuPose.from_list([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z])
        self.plan_result = self._planner.move_pose(
            cu_pose,
            link_name='base_link',
            plan_config=MotionGenPlanConfig(time_dilation_factor=self._time_dilation_factor),
            update_planning_scene=True,
            execute=False,
        )
        if not self.plan_result.success:
            self.get_logger().error('Planning failed.')
            return False
        return True

    def trigger_get_object_pose(self, goal_handle):
        """Trigger the initial action call for gettting object pose.

        Args:
            goal_handle (_type_): Goal handle
        """
        self._get_pose_done_event.clear()
        get_object_pose_goal = GetObjectPose.Goal()
        get_object_pose_goal.object_id = goal_handle.request.object_id
        get_object_pose_goal_future = self._get_object_pose_client.send_goal_async(
            get_object_pose_goal)
        get_object_pose_goal_future.add_done_callback(self.get_object_pose_goal_response_cb)

    def get_object_pose_goal_response_cb(self, future):
        """Gets the response from initial acceptance of object pose estimation server.

        Args:
            future (_type_): Future for action call
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for GetObjectPose')
            self._get_pose_done_result = None
            self._get_pose_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for GetObjectPose')
        get_object_pose_result_future = goal_handle.get_result_async()
        get_object_pose_result_future.add_done_callback(self.get_object_pose_goal_result_cb)

    def get_object_pose_goal_result_cb(self, future):
        """Gets the result from the object pose estimation action server.

        Args:
            future (_type_): Future describing the action call status
        """
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._get_pose_done_result = result
            self.get_logger().info('GetObjectPose action server succeeded.')
        elif status == GoalStatus.STATUS_ABORTED:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server aborted.')
        elif status == GoalStatus.STATUS_CANCELED:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server canceled.')
        else:
            self._get_pose_done_result = None
            self.get_logger().error('GetObjectPose action server failed.')
        self._get_pose_done_event.set()

    def trigger_object_attach(self, do_attach: bool):
        """Triggers object attachment or deattachment based on argument variable.

        Args:
            do_attach (bool): Boolean indicating object attachment else deattachment
        """
        self._object_attach_done_event.clear()
        object_attach_goal = AttachObject.Goal()
        object_attach_goal.attach_object = do_attach
        object_attach_goal.fallback_radius = self.get_parameter(
            'attach_object_fallback_radius').get_parameter_value().double_value
        object_attach_goal.object_config = self._create_object_config()

        def feedback(feedback_msg):
            return self.get_logger().info(f'Object Attachment Feedback = \
                                          {feedback_msg.feedback.status}')

        object_attach_goal_future = self._object_attach_client.send_goal_async(
            object_attach_goal, feedback_callback=feedback)
        object_attach_goal_future.add_done_callback(self.object_attach_goal_response_cb)

    def _create_object_config(self) -> Marker:
        """Creates the object attachment config according to the user selection for CUBE or SPHERE

        Returns:
            Marker: Marker object that is used in the client request to object attachment
        """
        marker = Marker()
        gripper_pose_object = self._grasp_reader.get_grasp_pose_object(
            index=self.plan_result.goal_index, tf_buffer=self._tf_buffer)

        pose = Pose()
        pose.position.x = gripper_pose_object.position.x
        pose.position.y = gripper_pose_object.position.y
        pose.position.z = gripper_pose_object.position.z
        pose.orientation.x = gripper_pose_object.quaternion.x
        pose.orientation.y = gripper_pose_object.quaternion.y
        pose.orientation.z = gripper_pose_object.quaternion.z
        pose.orientation.w = gripper_pose_object.quaternion.w
        marker.pose = pose
        marker.scale = self._attach_object_scale

        if self._attach_object_shape == ObjectAttachmentShape.SPHERE:
            marker.type = Marker.SPHERE
        elif self._attach_object_shape == ObjectAttachmentShape.CUBOID:
            marker.type = Marker.CUBE
        elif self._attach_object_shape == ObjectAttachmentShape.CUSTOM_MESH:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = self._attach_object_mesh_file_path
        else:
            self.get_logger().error('Received unknown object type!')

        marker.header.frame_id = 'grasp_frame'
        marker.frame_locked = True
        marker.color.r = 1.0
        marker.color.a = 1.0

        return marker

    def object_attach_goal_response_cb(self, future):
        """Get the object attachment response

        Args:
            future (_type_): Future for object attachment
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for AttachObject')
            self._object_attach_done_result = False
            self._object_attach_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for AttachObject')
        object_attach_result_future = goal_handle.get_result_async()
        object_attach_result_future.add_done_callback(self.object_attach_goal_result_cb)

    def object_attach_goal_result_cb(self, future):
        """Get the object attachment goal result

        Args:
            future (_type_): Future for object attachment
        """
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'{result.outcome}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('AttachObject action server succeeded.')
            self._object_attach_done_result = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('AttachObject action server aborted.')
            self._object_attach_done_result = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('AttachObject action server canceled.')
            self._object_attach_done_result = False
        else:
            self.get_logger().error('AttachObject action server failed.')
            self._object_attach_done_result = False

        self._object_attach_done_event.set()

    def trigger_gripper(self, position: float, max_effort: float):
        """Trigger gripper to either open or close

        Args:
            position (float): Position
            max_effort (float): Max effort
        """
        self._gripper_done_event.clear()
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(position)
        gripper_goal.command.max_effort = float(max_effort)

        gripper_goal_future = self._gripper_client.send_goal_async(gripper_goal)
        gripper_goal_future.add_done_callback(self.gripper_goal_response_cb)

    def gripper_goal_response_cb(self, future):
        """Get goal response for Gripper

        Args:
            future (_type_): Future for goal response
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected for GripperCommand')
            self._gripper_done_result = False
            self._gripper_done_event.set()
            return
        # Cache the goal handle for canceling
        self._client_goal_handles.append(goal_handle)
        self.get_logger().info('Goal accepted for GripperCommand')
        gripper_result_future = goal_handle.get_result_async()
        gripper_result_future.add_done_callback(self.gripper_goal_result_cb)

    def gripper_goal_result_cb(self, future):
        """Get the final result from gripper trigger action call.

        Args:
            future (_type_): Future for goal result
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

    def close_gripper(self, position: float = 0.65, max_effort: float = 10.0) -> bool:
        """Close gripper by triggering an action call.

        Args:
            position (float, optional): Position. Defaults to 0.65.
            max_effort (float, optional): Max effort. Defaults to 10.0.
        Returns:
            bool: bool: Status of whether gripper closed successfully or not
        """
        self.get_logger().info('Closing gripper')
        self.trigger_gripper(position, max_effort)

        self._gripper_done_event.wait()
        if not self._gripper_done_result:
            self.get_logger().error('Failed to close the gripper.')
            return False

        return True

    def open_gripper(self, position: float = 0.0, max_effort: float = 10.0) -> bool:
        """Open gripper by triggering an action call.

        Args:
            position (float, optional): Position. Defaults to 0.65.
            max_effort (float, optional): Max effort. Defaults to 10.0.
        Returns:
            bool: bool: Status of whether gripper opened successfully or not
        """
        self.get_logger().info('Opening gripper')
        self.trigger_gripper(position, max_effort)

        self._gripper_done_event.wait()
        if not self._gripper_done_result:
            self.get_logger().error('Failed to open the gripper.')
            return False

        return True

    def client_cancel_done(self, future):
        """Cancels client call

        Args:
            future (_type_): Future for cancel action call
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().error('Goal failed to cancel')

    def cancel_callback(self, goal_handle) -> CancelResponse:
        """Trigger cancel callback if user has tried to cancel the action call.

        Args:
            goal_handle (_type_): Goal handle

        Returns:
            CancelResponse: Cancel response
        """

        self.get_logger().info('Received cancel request')

        # Cancel all the client goal handles that are executing or canceling
        filtered_client_goal_handles = [
            client_goal_handle for client_goal_handle in self._client_goal_handles
            if client_goal_handle.status == GoalStatus.STATUS_EXECUTING or
            client_goal_handle.status == GoalStatus.STATUS_CANCELING
        ]
        for client_goal_handle in filtered_client_goal_handles:
            client_goal_handle.cancel_goal_async().add_done_callback(self.client_cancel_done)

        # Clear the client goal handles
        self._client_goal_handles.clear()
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle) -> PickAndPlace.Result:
        """Execute the action call functionality

        Args:
            goal_handle (_type_): Goal handle

        Returns:
            _type_: _description_
        """
        self.get_logger().info('Executing goal...')
        result = PickAndPlace.Result()

        # Wait for the get_object_pose action server to be available, do not wait if we use ground
        # truth pose in sim
        if not self._use_ground_truth_pose_from_sim:
            if not self.wait_for_server(self._get_object_pose_client):
                result.success = False
                goal_handle.abort()
                return result

        # Wait for the object attachment/detachment action server to be available
        if not self.wait_for_server(self._object_attach_client):
            result.success = False
            goal_handle.abort()
            return result

        # Wait for the gripper action server to be available
        if not self.wait_for_server(self._gripper_client):
            result.success = False
            goal_handle.abort()
            return result

        # Trigger the get object pose action if sim ground truth is not enabled
        if not self._use_ground_truth_pose_from_sim:
            self.trigger_get_object_pose(goal_handle)
            # Wait for action to be done
            self._get_pose_done_event.wait()
            if self._get_pose_done_result is None:
                self.get_logger().error('Failed to get object pose.')
                result.success = False
                goal_handle.abort()
                return result

        # Trigger the planning for pick phase
        self.get_logger().info('Starting orchestrator for pick and place')
        pick_success = False
        plan_only_retraction = False

        # Open gripper to set it in the right configuration for upcoming task
        if not self.open_gripper():
            result.success = False
            goal_handle.abort()
            return result

        for i in range(self._num_planner_tries_):
            if goal_handle.status == GoalStatus.STATUS_CANCELING or \
               goal_handle.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn('Received request to cancel goal...')
                result.success = False
                goal_handle.abort()
                return result
            self.get_logger().info(f'Executing pick pose {i+1} / {self._num_planner_tries_}')
            if self.get_plan_grasp(self.read_grasp_poses()):
                # Executing grasp trajectory
                self.get_logger().info('Found trajectories.')

                if plan_only_retraction:
                    self.get_logger().info('Re-executing retraction plan')
                    pick_success, _ = self._planner.execute_plan(
                        self.plan_result.planned_trajectory[1])
                    if not pick_success:
                        plan_only_retraction = True
                        time.sleep(self._sleep_time_before_planner_tries_sec)
                        continue
                    pick_success = True
                    break

                pick_success, _ = self._planner.execute_plan(
                    self.plan_result.planned_trajectory[0])
                if not pick_success:
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                if not self.close_gripper():
                    result.success = False
                    goal_handle.abort()
                    return result
                # Executing lift trajectory
                self.get_logger().info('Executing plan')
                pick_success, _ = self._planner.execute_plan(
                    self.plan_result.planned_trajectory[1])
                if not pick_success:
                    plan_only_retraction = True
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                pick_success = True
                break
            else:
                self.get_logger().error('Planning for pick phase failed, trying again')
            self.get_logger().info(
                f'Waiting for {self._sleep_time_before_planner_tries_sec} seconds')
            time.sleep(self._sleep_time_before_planner_tries_sec)

        if not pick_success:
            self.get_logger().error('Planning for pick phase failed.')
            result.success = False
            goal_handle.abort()
            return result

        # Attach object
        self.get_logger().info('Triggering object attachment')
        self.trigger_object_attach(do_attach=True)

        # Wait for action to be done
        self._object_attach_done_event.wait()
        if not self._object_attach_done_result:
            self.get_logger().error('Failed to attach object.')
            result.success = False
            goal_handle.abort()
            return result

        # Clearing the result for the next action (detach object)
        self._object_attach_done_result = False

        # Trigger the planning for drop phase
        self.get_logger().info('Getting place pose')
        place_pose = None
        if self._use_pose_from_rviz:
            place_pose = self._end_effector_marker.get_pose()
        else:
            place_pose = goal_handle.request.place_pose
        # Publish on Rviz for debugging where the robot will place the object
        self.publish_grasp_transform(place_pose, 'place_pose')
        place_success = False
        for i in range(self._num_planner_tries_):
            if goal_handle.status == GoalStatus.STATUS_CANCELING or \
               goal_handle.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn('Received request to cancel goal...')
                result.success = False
                goal_handle.abort()
                return result
            self.get_logger().info(
                f'Executing place pose {i+1} / {self._num_planner_tries_} after '
                f'{self._sleep_time_before_planner_tries_sec} second pause')
            if self.get_plan_pose(place_pose):
                # Executing grasp trajectory
                place_success, _ = self._planner.execute_plan(
                    self.plan_result.planned_trajectory[0])
                if not place_success:
                    time.sleep(self._sleep_time_before_planner_tries_sec)
                    continue
                if not self.open_gripper():
                    result.success = False
                    goal_handle.abort()
                    return result
                place_success = True
                break
            else:
                self.get_logger().error('Planning for drop phase failed, trying again')
            time.sleep(self._sleep_time_before_planner_tries_sec)

        if not place_success:
            self.get_logger().error('Planning for drop phase failed.')
            result.success = False
            goal_handle.abort()
            return result
        # Detach object
        self.get_logger().info('Triggering object detachment')
        self.trigger_object_attach(do_attach=False)
        # Wait for action to be done
        self._object_attach_done_event.wait()
        if not self._object_attach_done_result:
            self.get_logger().error('Failed to detach object.')
            result.success = False
            goal_handle.abort()
            return result

        goal_handle.succeed()
        result.success = True
        return result


def main(args=None):
    rclpy.init(args=args)

    pick_and_place_orchestrator = PickAndPlaceOrchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(pick_and_place_orchestrator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pick_and_place_orchestrator.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    pick_and_place_orchestrator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
