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

import copy

from geometry_msgs.msg import Pose, PoseStamped
from isaac_manipulator_interfaces.msg import InsertionRequest
import isaac_manipulator_ros_python_utils as utils
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException


class GoalPosePublisherNode(Node):
    """Node to publish the goal pose to the goal pose topic."""

    def __init__(self):
        super().__init__('goal_pose_publisher_node')

        self.declare_parameter('goal_frame', 'fp_object')
        goal_frame = self.get_parameter('goal_frame')
        self.goal_frame = goal_frame.get_parameter_value().string_value

        self.declare_parameter('world_frame', 'world')
        world_frame = self.get_parameter('world_frame')
        self.world_frame = world_frame.get_parameter_value().string_value

        self.declare_parameter('frequency', 60.0)
        frequency = self.get_parameter('frequency')
        self.frequency = frequency.get_parameter_value().double_value

        self.declare_parameter('enable_publishing_on_trigger', False)
        enable_publishing_on_trigger = self.get_parameter('enable_publishing_on_trigger')
        self.enable_publishing_on_trigger = \
            enable_publishing_on_trigger.get_parameter_value().bool_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pose = None

        self.goal_pose_publisher = self.create_publisher(
            PoseStamped, 'goal_pose',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        if not self.enable_publishing_on_trigger:
            self.create_timer(1.0, self.tf_callback)

        self.create_timer(1.0 / self.frequency, self.timer_callback)

        self.create_subscription(
            InsertionRequest, 'insertion_request_topic', self.insertion_request_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))
        self.triggered_enabled = False

    def tf_callback(self):
        if not self.goal_pose:
            # Lookup and cache goal pose
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame, self.goal_frame, rclpy.time.Time())
                position = utils.vector_to_point(transform.transform.translation)
                orientation = transform.transform.rotation
                self.goal_pose = Pose(position=position, orientation=orientation)
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warning(f'{e}')

    def insertion_request_callback(self, msg: InsertionRequest):
        """
        Cache the goal pose and handle insertion requests.

        It will set the goal pose and enable publishing on trigger. It will also
        handle cancellation requests.

        Args
        ----
            msg: Insertion request message.

        Returns
        -------
            None

        """
        if msg.request_type == InsertionRequest.NEW_INSERTION:
            self.goal_pose = copy.deepcopy(msg.pose.pose)  # Get pose from pose stamped message
            self.triggered_enabled = True
        elif msg.request_type == InsertionRequest.CANCEL_INSERTION:
            self.goal_pose = None
            self.triggered_enabled = False
        else:
            self.get_logger().warning(f'Invalid insertion request type: {msg.request_type}')
        return

    def publish_goal_pose(self):
        """Publish the goal pose to the goal pose topic."""
        if self.goal_pose:
            # Publish cached goal pose
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.world_frame
            msg.pose = self.goal_pose
            self.goal_pose_publisher.publish(msg)
        else:
            self.get_logger().debug('Goal pose is not set')

    def timer_callback(self):
        """Timer callback to publish the goal pose to the goal pose topic."""
        if self.enable_publishing_on_trigger and not self.triggered_enabled:
            self.get_logger().debug('Goal pose publishing is disabled')
            return
        else:
            self.publish_goal_pose()


def main():
    rclpy.init()
    rclpy.spin(GoalPosePublisherNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
