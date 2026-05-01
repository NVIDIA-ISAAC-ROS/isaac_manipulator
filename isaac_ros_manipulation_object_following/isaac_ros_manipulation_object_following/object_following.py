#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from geometry_msgs.msg import Pose, PoseArray
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.action import ExecuteTrajectory as ExecuteTrajectoryAction
from moveit_msgs.msg import MoveItErrorCodes, PlanningScene
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class ObjectFollowingNode(Node):
    """
    Plan and execute motions to follow a detected object using cuMotion directly.

    This node periodically looks up the detected object's pose (grasp frame)
    and uses the cuMotion MotionPlan action server for planning and the MoveIt
    ExecuteTrajectory action server for execution, bypassing MoveIt's
    planning pipeline. Includes stale frame detection and minimum goal
    change thresholds.
    """

    def __init__(self):
        super().__init__('object_following_node')

        self._previous_goal_position = None

        self._world_frame = self.declare_parameter(
            'world_frame', 'base_link').get_parameter_value().string_value

        self._grasp_frame = self.declare_parameter(
            'grasp_frame', 'grasp_frame').get_parameter_value().string_value

        self._grasp_frame_stale_time_threshold = self.declare_parameter(
            'grasp_frame_stale_time_threshold', 30.0).get_parameter_value().double_value

        self._goal_change_position_threshold = self.declare_parameter(
            'goal_change_position_threshold', 0.1).get_parameter_value().double_value

        self._plan_timer_period = self.declare_parameter(
            'plan_timer_period', 2.0).get_parameter_value().double_value

        self._link_name = self.declare_parameter(
            'link_name', 'base_link').get_parameter_value().string_value

        motion_plan_action_name = self.declare_parameter(
            'motion_plan_action_server_name',
            'cumotion/motion_plan').get_parameter_value().string_value

        execute_trajectory_action_name = self.declare_parameter(
            'execute_trajectory_action_server_name',
            'execute_trajectory').get_parameter_value().string_value

        self._time_dilation_factor = self.declare_parameter(
            'time_dilation_factor', 0.2).get_parameter_value().double_value

        self._update_esdf = self.declare_parameter(
            'update_esdf', True).get_parameter_value().bool_value

        plan_cb_group = MutuallyExclusiveCallbackGroup()
        exec_cb_group = MutuallyExclusiveCallbackGroup()

        self._plan_client = ActionClient(
            self, MotionPlan, motion_plan_action_name,
            callback_group=plan_cb_group)

        self._exec_client = ActionClient(
            self, ExecuteTrajectoryAction, execute_trajectory_action_name,
            callback_group=exec_cb_group)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._planning_scene_world = None
        self._planning_scene_sub = self.create_subscription(
            PlanningScene, '/planning_scene',
            self._planning_scene_callback, 10)

        self._planning_in_progress = False

        self.get_logger().info(
            'Waiting for cuMotion motion plan action server at '
            '"%s"...' % motion_plan_action_name)
        self._plan_client.wait_for_server()

        self.get_logger().info(
            'Waiting for execute trajectory action server at '
            '"%s"...' % execute_trajectory_action_name)
        self._exec_client.wait_for_server()

        self.get_logger().info('All action servers are available. Starting planning loop.')
        self.timer = self.create_timer(self._plan_timer_period, self.on_timer)

    def _planning_scene_callback(self, msg):
        self._planning_scene_world = msg.world
        self.get_logger().info(
            'Updated planning scene with %d collision objects' %
            len(msg.world.collision_objects), once=True)

    def _transform_msg_to_pose_msg(self, tf_msg):
        pose = Pose()
        pose.position.x = tf_msg.translation.x
        pose.position.y = tf_msg.translation.y
        pose.position.z = tf_msg.translation.z
        pose.orientation.x = tf_msg.rotation.x
        pose.orientation.y = tf_msg.rotation.y
        pose.orientation.z = tf_msg.rotation.z
        pose.orientation.w = tf_msg.rotation.w
        return pose

    def on_timer(self):
        if self._planning_in_progress:
            return

        try:
            world_frame_pose_grasp_frame = self._tf_buffer.lookup_transform(
                self._world_frame, self._grasp_frame, rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warning(
                'Waiting for object pose transform between %s and %s. %s' % (
                    self._world_frame, self._grasp_frame, ex))
            return

        stale_check_time = (self.get_clock().now() - rclpy.time.Time().from_msg(
            world_frame_pose_grasp_frame.header.stamp)).nanoseconds / 1e9
        if stale_check_time > self._grasp_frame_stale_time_threshold:
            self.get_logger().warn(
                'A new grasp frame has not been received for %.1f seconds.' % (
                    self._grasp_frame_stale_time_threshold,))

        target_pose = self._transform_msg_to_pose_msg(
            world_frame_pose_grasp_frame.transform)
        new_goal = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z,
        ])
        if self._previous_goal_position is not None:
            goal_change_distance = np.linalg.norm(
                self._previous_goal_position - new_goal)
            if goal_change_distance <= self._goal_change_position_threshold:
                self.get_logger().warning(
                    'New goal within %.2f m (%.3f), not setting. Move goal further.' % (
                        self._goal_change_position_threshold, goal_change_distance))
                return

        goal_msg = MotionPlan.Goal()
        goal_msg.goal_pose = PoseArray()
        goal_msg.goal_pose.header.frame_id = self._link_name
        goal_msg.goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal_pose.poses.append(target_pose)
        goal_msg.plan_pose = True
        goal_msg.plan_cspace = False
        goal_msg.plan_grasp = False
        goal_msg.use_current_state = True
        goal_msg.time_dilation_factor = self._time_dilation_factor
        goal_msg.hold_partial_pose = False
        goal_msg.update_esdf = self._update_esdf
        goal_msg.use_planning_scene = True
        if self._planning_scene_world is not None:
            goal_msg.world = self._planning_scene_world
        else:
            self.get_logger().warning(
                'Planning scene not yet received on /planning_scene.')

        self._planning_in_progress = True
        self._pending_goal_position = new_goal
        self.get_logger().debug(
            'Sending goal pose for frame "%s" to cuMotion planner.' % self._grasp_frame)

        send_goal_future = self._plan_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._plan_goal_response_callback)

    def _plan_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Motion plan goal was rejected by server.')
            self._planning_in_progress = False
            return
        self.get_logger().info('Motion plan goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._plan_result_callback)

    def _plan_result_callback(self, future):
        result = future.result().result
        if not result.success:
            self.get_logger().warning('Motion planning failed. Retrying on next iteration.')
            self._planning_in_progress = False
            return
        if len(result.planned_trajectory) == 0:
            self.get_logger().error('Planning succeeded but no trajectory returned.')
            self._planning_in_progress = False
            return
        self.get_logger().info(
            'Motion planning succeeded (planning_time=%.3fs). Executing trajectory...' % (
                result.planning_time,))

        exec_goal = ExecuteTrajectoryAction.Goal()
        exec_goal.trajectory = result.planned_trajectory[0]
        exec_goal.trajectory.joint_trajectory.header = Header()
        exec_goal.trajectory.multi_dof_joint_trajectory.header = Header()
        exec_future = self._exec_client.send_goal_async(exec_goal)
        exec_future.add_done_callback(self._exec_goal_response_callback)

    def _exec_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Execute trajectory goal was rejected.')
            self._planning_in_progress = False
            return
        self.get_logger().info('Execute trajectory goal accepted, waiting for completion...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._exec_result_callback)

    def _exec_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('Trajectory execution succeeded.')
            self._previous_goal_position = self._pending_goal_position
        else:
            self.get_logger().warning(
                'Trajectory execution failed with error code: %s. Retrying.' % (
                    result.error_code.val,))
        self._planning_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    object_following_node = ObjectFollowingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(object_following_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        object_following_node.get_logger().info('KeyboardInterrupt, shutting down.')
    object_following_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
