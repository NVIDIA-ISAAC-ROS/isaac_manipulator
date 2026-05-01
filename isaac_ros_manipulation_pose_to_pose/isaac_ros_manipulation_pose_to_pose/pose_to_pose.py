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
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class PoseToPoseNode(Node):
    """
    Plan and execute motions between target frames using cuMotion directly.

    This node periodically looks up target frame transforms and uses the
    cuMotion MotionPlan action server for planning and the MoveIt
    ExecuteTrajectory action server for execution.
    """

    def __init__(self):
        super().__init__('pose_to_pose_node')

        self._world_frame = self.declare_parameter(
            'world_frame', 'base_link').get_parameter_value().string_value

        self._target_frames = self.declare_parameter(
            'target_frames', ['target1_frame']).get_parameter_value().string_array_value

        self._target_frame_idx = 0

        self._plan_timer_period = self.declare_parameter(
            'plan_timer_period', 0.01).get_parameter_value().double_value

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

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._planning_scene_world = None
        self._planning_scene_sub = self.create_subscription(
            PlanningScene, '/planning_scene',
            self._planning_scene_callback, 10)

        self._planning_in_progress = False

        self.get_logger().info(
            'Waiting for cuMotion motion plan action server at "%s"...' % motion_plan_action_name)
        self._plan_client.wait_for_server()

        self.get_logger().info(
            'Waiting for execute trajectory action server at "%s"...' % (
                execute_trajectory_action_name,))
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

        target_frame = self._target_frames[self._target_frame_idx]
        try:
            world_frame_pose_target_frame = self._tf_buffer.lookup_transform(
                self._world_frame, target_frame,
                self.get_clock().now(), rclpy.duration.Duration(seconds=10.0)
            )
        except TransformException as ex:
            self.get_logger().warning(
                'Waiting for target_frame transform between %s and %s. %s' % (
                    self._world_frame, target_frame, ex))
            return

        target_pose = self._transform_msg_to_pose_msg(
            world_frame_pose_target_frame.transform)

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
        self.get_logger().debug(
            'Sending pose target for frame "%s" to cuMotion planner.' % target_frame)

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
            self._target_frame_idx = (
                (self._target_frame_idx + 1) % len(self._target_frames))
        else:
            self.get_logger().warning(
                'Trajectory execution failed with error code: %s. Retrying.' % (
                    result.error_code.val,))
        self._planning_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    pose_to_pose_node = PoseToPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(pose_to_pose_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pose_to_pose_node.get_logger().info('KeyboardInterrupt, shutting down.')
    pose_to_pose_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
