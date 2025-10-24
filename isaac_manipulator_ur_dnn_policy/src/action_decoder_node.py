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

from isaac_manipulator_ur_dnn_policy.msg import Inference
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import torch


class ActionDecoderNode(Node):

    def __init__(self):
        super().__init__('action_decoder_node')

        # set default torch device
        torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('get_parameter_service', 'inference_node/get_parameters')
        get_parameter_service = self.get_parameter('get_parameter_service')
        self.get_parameter_service = get_parameter_service.get_parameter_value().string_value

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

        self.target_joint_state_publisher = self.create_publisher(
            JointState, 'target_joint_state',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

    def callback(self, msg: Inference):
        joint_positions = [0.0] * len(self.joints)
        for name, position in zip(msg.joint_state.name, msg.joint_state.position):
            if name in self.joints:
                joint_positions[self.joints[name]] = position

        target_joint_position = torch.tensor(joint_positions).unsqueeze(0) + \
            torch.tensor(msg.data).unsqueeze(0) * self.action_scale_joint_space
        target_joint_position = target_joint_position.clamp(
            self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self.target_joint_state_publisher.publish(
            JointState(
                header=msg.header,
                name=self.arm_joint_names,
                position=target_joint_position.squeeze(0).tolist(),
            )
        )


def main():
    rclpy.init()
    rclpy.spin(ActionDecoderNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
