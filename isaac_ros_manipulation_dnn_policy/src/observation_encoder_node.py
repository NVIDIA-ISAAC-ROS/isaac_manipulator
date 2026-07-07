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

from typing import Callable, List

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from isaac_ros_manipulation_dnn_policy.msg import Inference
import isaac_ros_manipulation_ros_python_utils as utils
import numpy as np
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


def stamp_to_s(stamp: Time) -> float:
    return stamp.sec + utils.ns_to_s(stamp.nanosec)


def get_joint_velocities_fd(current: JointState, previous: JointState) -> List[float]:
    velocities = []

    dt = stamp_to_s(current.header.stamp) - stamp_to_s(previous.header.stamp)
    for curr, prev in zip(current.position, previous.position):
        if dt > 0.0:
            velocities.append((curr - prev) / dt)
        else:
            velocities.append(0.0)

    return velocities


class ObservationEncoderNode(Node):

    def __init__(self):
        super().__init__('observation_encoder_node')

        self.declare_parameter('get_parameter_service', 'inference_node/get_parameters')
        self.declare_parameter('joint_state_age_threshold_ms', 10.0)
        self.declare_parameter('joint_prefix', '')
        self.declare_parameter('quaternion_order', 'wxyz')
        self._joint_prefix = self.get_parameter(
            'joint_prefix').get_parameter_value().string_value
        self._quaternion_to_obs = self._resolve_quaternion_order(
            self.get_parameter('quaternion_order').get_parameter_value().string_value)
        get_parameter_service = self.get_parameter('get_parameter_service')
        self.get_parameter_service = get_parameter_service.get_parameter_value().string_value

        client = self.create_client(GetParameters, self.get_parameter_service)
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"'{self.get_parameter_service}' not available")

        request = GetParameters.Request()
        request.names = [
            'obs_order',
            'arm_joint_names',
            'target_pos_centre',
            'target_pos_range',
            'target_rot_centre',
            'target_rot_range',
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.done() and future.result() is not None:
            response = future.result()
        else:
            raise RuntimeError(f"failed to call service '{self.get_parameter_service}'")

        self.obs_order = response.values[0].string_array_value
        self.arm_joint_names = response.values[1].string_array_value

        target_pos_centre = np.array(response.values[2].double_array_value)
        target_pos_range = np.array(response.values[3].double_array_value)
        target_pos_lower = target_pos_centre - target_pos_range
        target_pos_upper = target_pos_centre + target_pos_range

        target_rot_centre = np.array(
            [utils.deg_to_rad(x) for x in response.values[4].double_array_value])
        target_rot_range = np.array(
            [utils.deg_to_rad(x) for x in response.values[5].double_array_value])
        target_rot_lower = target_rot_centre - target_rot_range
        target_rot_upper = target_rot_centre + target_rot_range

        self.ood_detector = utils.OutOfDistributionDetector(
                 target_position_min=utils.list_to_vector3(target_pos_lower.tolist()),
                 target_position_max=utils.list_to_vector3(target_pos_upper.tolist()),
                 target_rotation_min=utils.list_to_vector3(target_rot_lower.tolist()),
                 target_rotation_max=utils.list_to_vector3(target_rot_upper.tolist()))
        self._target_pos_lower = target_pos_lower.tolist()
        self._target_pos_upper = target_pos_upper.tolist()
        self._target_rot_lower = target_rot_lower.tolist()
        self._target_rot_upper = target_rot_upper.tolist()
        self.joints = {}
        for index, joint in enumerate(self.arm_joint_names):
            self.joints[joint] = index

        self.joint_state = None
        self.joint_state_prev = None
        self._obs_log_count = 0
        self._obs_log_max = 5

        self.create_subscription(
            JointState, 'joint_state', self.joint_state_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_pose_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.publisher = self.create_publisher(
            Inference, 'observation',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self._joint_state_age_threshold = self.get_parameter(
            'joint_state_age_threshold_ms').get_parameter_value().double_value

    @staticmethod
    def _resolve_quaternion_order(order: str) -> Callable:
        """Return a function that converts a Quaternion msg to a list in the requested order."""
        _ORDERS = {
            'wxyz': lambda q: [q.w, q.x, q.y, q.z],
            'xyzw': lambda q: [q.x, q.y, q.z, q.w],
        }
        if order not in _ORDERS:
            raise ValueError(
                f"Invalid quaternion_order '{order}'. Must be one of {list(_ORDERS)}")
        return _ORDERS[order]

    def _strip_prefix(self, name: str) -> str:
        if self._joint_prefix and name.startswith(self._joint_prefix):
            return name[len(self._joint_prefix):]
        return name

    def joint_state_callback(self, joint_state: JointState):
        self.joint_state = joint_state

    def goal_pose_callback(self, goal_pose: PoseStamped):
        if not self.joint_state:
            self.get_logger().warning('waiting for joint state')
            return

        # Find how old the joint state is before pushing into policy.
        current_time = self.get_clock().now().to_msg()
        dt_in_ms = (stamp_to_s(current_time) - stamp_to_s(self.joint_state.header.stamp)) * 1000.0
        if dt_in_ms > self._joint_state_age_threshold:
            self.get_logger().warning(
                f'joint state is {dt_in_ms} ms old (current threshold is '
                f' {self._joint_state_age_threshold} ms), skipping observation')

        obs = []
        for key in self.obs_order:
            if key in ('arm_dof_pos'):
                positions = [0.0] * len(self.joints)
                for name, position in zip(self.joint_state.name, self.joint_state.position):
                    stripped = self._strip_prefix(name)
                    if stripped in self.joints:
                        positions[self.joints[stripped]] = position

                obs += positions

            elif key in ('arm_dof_vel'):
                velocities = [0.0] * len(self.joints)
                for name, velocity in zip(self.joint_state.name, self.joint_state.velocity):
                    stripped = self._strip_prefix(name)
                    if stripped in self.joints:
                        velocities[self.joints[stripped]] = velocity

                obs += velocities

            elif key in ('arm_dof_vel_fd'):
                velocities = [0.0] * len(self.joints)

                if self.joint_state_prev:
                    joint_velocities = get_joint_velocities_fd(
                        self.joint_state, self.joint_state_prev)

                    for name, velocity in zip(self.joint_state.name, joint_velocities):
                        stripped = self._strip_prefix(name)
                        if stripped in self.joints:
                            velocities[self.joints[stripped]] = velocity

                self.joint_state_prev = self.joint_state

                obs += velocities

            elif key in ('target_pos', 'fixed_pos', 'shaft_pos'):
                position = utils.point_to_list(goal_pose.pose.position)

                if not self.ood_detector.target_position_in_distribution(
                        utils.list_to_vector3(position)):
                    self.get_logger().warning('Target position out of distribution.'
                                              f'requested {position} but policy needs Target'
                                              f'Lower bound of position {self._target_pos_lower} '
                                              f'Upper bound of position: {self._target_pos_upper}')

                obs += position

            elif key in ('target_quat', 'fixed_quat', 'shaft_quat'):
                orientation = utils.quaternion_positive(goal_pose.pose.orientation)
                rotation = utils.quaternion_to_rpy(orientation)

                if not self.ood_detector.target_rotation_in_distribution(rotation):
                    self.get_logger().warning(f'Target rotation out of distribution, rotation'
                                              f'requested {rotation} but policy needs Target'
                                              f'Rotation lower bound: {self._target_rot_lower} and'
                                              f'Rotation upper bound: {self._target_rot_upper}')

                obs += self._quaternion_to_obs(orientation)

            else:
                raise ValueError(f"unknown observation key '{key}'")

        # Forward a JointState with stripped prefixes so downstream nodes
        # (e.g. action_decoder) can look up joints by unprefixed name.
        forwarded_js = JointState(header=self.joint_state.header)
        forwarded_js.name = [self._strip_prefix(n) for n in self.joint_state.name]
        forwarded_js.position = list(self.joint_state.position)
        forwarded_js.velocity = list(self.joint_state.velocity)
        forwarded_js.effort = list(self.joint_state.effort)

        matched = sum(1 for n in forwarded_js.name if n in self.joints)
        if matched == 0:
            self.get_logger().error(
                f'No arm joints matched after prefix stripping '
                f'(prefix={self._joint_prefix!r}, '
                f'incoming={self.joint_state.name}, '
                f'expected={list(self.joints.keys())}). '
                f'Skipping observation to prevent unsafe commands.')
            return

        if matched < len(self.joints):
            self.get_logger().warning(
                f'Only {matched}/{len(self.joints)} arm joints matched '
                f'after prefix stripping')

        if self._obs_log_count < self._obs_log_max:
            self._obs_log_count += 1
            self.get_logger().info(
                f'[OBS {self._obs_log_count}] goal_frame={goal_pose.header.frame_id} '
                f'shaft_pos=[{goal_pose.pose.position.x:.6f}, '
                f'{goal_pose.pose.position.y:.6f}, '
                f'{goal_pose.pose.position.z:.6f}] '
                f'shaft_quat=[{goal_pose.pose.orientation.x:.6f}, '
                f'{goal_pose.pose.orientation.y:.6f}, '
                f'{goal_pose.pose.orientation.z:.6f}, '
                f'{goal_pose.pose.orientation.w:.6f}] '
                f'js_topic_names={list(self.joint_state.name)[:3]}... '
                f'arm_pos={[f"{v:.4f}" for v in obs[:7]]} '
                f'arm_vel={[f"{v:.4f}" for v in obs[7:14]]} '
                f'obs_shaft={[f"{v:.6f}" for v in obs[14:21]]}')

        self.publisher.publish(
            Inference(header=goal_pose.header, joint_state=forwarded_js, data=obs))


def main():
    rclpy.init()
    rclpy.spin(ObservationEncoderNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
