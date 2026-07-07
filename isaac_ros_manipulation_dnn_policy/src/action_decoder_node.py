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

import math

from isaac_ros_manipulation_dnn_policy.msg import Inference
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import torch


CONTROLLER_TYPE_STREAMING = 'streaming_joint_state'


class ActionDecoderNode(Node):

    def __init__(self):
        super().__init__('action_decoder_node')

        torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('get_parameter_service', 'inference_node/get_parameters')
        self.declare_parameter('joint_prefix', '')
        self.declare_parameter('controller_name', '')

        get_parameter_service = self.get_parameter('get_parameter_service')
        self.get_parameter_service = get_parameter_service.get_parameter_value().string_value

        self.joint_prefix = self.get_parameter(
            'joint_prefix').get_parameter_value().string_value
        self.controller_name = self.get_parameter(
            'controller_name').get_parameter_value().string_value

        self._direct_output = bool(self.joint_prefix and self.controller_name)

        client = self.create_client(GetParameters, self.get_parameter_service)
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"'{self.get_parameter_service}' not available")

        request = GetParameters.Request()
        request.names = [
            'arm_joint_names',
            'policy_action_space',
            'action_scale_joint_space',
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.done() and future.result() is not None:
            response = future.result()
        else:
            raise RuntimeError(f"failed to call service '{self.get_parameter_service}'")

        self.arm_joint_names = response.values[0].string_array_value
        self.policy_action_space = response.values[1].string_value
        self.action_scale_joint_space = torch.tensor(response.values[2].double_array_value)

        self.joints = {}
        for index, joint in enumerate(self.arm_joint_names):
            self.joints[joint] = index

        self.robot_dof_lower_limits = torch.ones(len(self.arm_joint_names)) * math.pi * -2.0
        self.robot_dof_upper_limits = torch.ones(len(self.arm_joint_names)) * math.pi * 2.0

        if self.policy_action_space != 'joint':
            raise ValueError(f"unsupported policy action space '{self.policy_action_space}'")

        self.create_subscription(
            Inference, 'action', self.callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.target_joint_state_publisher = None
        self.streaming_pub = None

        self.robot_joint_names = [
            f'{self.joint_prefix}{n}' for n in self.arm_joint_names
        ] if self.joint_prefix else list(self.arm_joint_names)

        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        if self._direct_output:
            stream_topic = f'/{self.controller_name}/joint_commands'
            self.streaming_pub = self.create_publisher(
                JointState, stream_topic, ctrl_qos)
            self.get_logger().info(
                f'Streaming mode \u2014 publishing JointState to {stream_topic}')
        else:
            self.target_joint_state_publisher = self.create_publisher(
                JointState, 'target_joint_state',
                QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10))
            self.get_logger().info(
                'Legacy mode \u2014 publishing JointState to target_joint_state')

    def callback(self, msg: Inference):
        joint_positions = [0.0] * len(self.joints)
        matched = 0
        for name, position in zip(msg.joint_state.name, msg.joint_state.position, strict=True):
            if name in self.joints:
                joint_positions[self.joints[name]] = position
                matched += 1

        if matched != len(self.joints):
            self.get_logger().error(
                f'Incomplete joint data in Inference message '
                f'(incoming={list(msg.joint_state.name)}, '
                f'expected={list(self.joints.keys())}, matched={matched}/{len(self.joints)}). '
                f'Dropping to prevent unsafe commands.')
            return

        target_joint_position = torch.tensor(joint_positions).unsqueeze(0) + \
            torch.tensor(msg.data).unsqueeze(0) * self.action_scale_joint_space
        target_joint_position = target_joint_position.clamp(
            self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        positions = target_joint_position.squeeze(0).tolist()

        if self.streaming_pub is not None:
            self.streaming_pub.publish(
                JointState(
                    header=msg.header,
                    name=self.robot_joint_names,
                    position=positions,
                ))
        else:
            self.target_joint_state_publisher.publish(
                JointState(
                    header=msg.header,
                    name=self.arm_joint_names,
                    position=positions,
                )
            )


def main():
    rclpy.init()
    rclpy.spin(ActionDecoderNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
