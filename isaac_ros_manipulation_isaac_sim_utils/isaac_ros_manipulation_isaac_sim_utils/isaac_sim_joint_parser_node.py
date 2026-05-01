#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
This file parses the incoming Isaac Sim joint states.

It takes the incoming joint states and converts them to a format that can be consumed by downstream
URDF-based joint state consumers (MoveIt, cuMotion, robot state publisher, etc.).
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class JointParser(Node):

    def __init__(self):
        super().__init__('joint_parser')
        self.subscription = self.create_subscription(
            JointState,
            'isaac_joint_states',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(JointState, 'isaac_parsed_joint_states', 10)

        self.ignore_joints = [
            'left_outer_finger_joint',
            'right_outer_finger_joint',
            'left_inner_finger_pad_joint',
            'right_inner_finger_pad_joint',
            'left_inner_knuckle_joint',
            'left_inner_finger_joint',
            'right_outer_knuckle_joint',
            'right_inner_knuckle_joint',
            'right_inner_finger_joint'
        ]

    def listener_callback(self, msg):
        new_msg = JointState()
        new_msg.header = msg.header

        main_joint_index = None
        for i, name in enumerate(msg.name):
            if name == 'finger_joint':
                main_joint_index = i
                if msg.position[i] > 0.62:
                    msg.position[i] = 0.62
                break

        if main_joint_index is not None:
            main_joint_position = msg.position[main_joint_index]

            for i, name in enumerate(msg.name):
                if name == 'finger_joint':
                    new_msg.name.append(name)
                    new_msg.position.append(main_joint_position)
                elif name not in self.ignore_joints:
                    new_msg.name.append(name)
                    new_msg.position.append(msg.position[i])

            new_msg.velocity = msg.velocity if msg.velocity else [0.0] * len(new_msg.position)
            new_msg.effort = msg.effort if msg.effort else [0.0] * len(new_msg.position)

        self.publisher.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
