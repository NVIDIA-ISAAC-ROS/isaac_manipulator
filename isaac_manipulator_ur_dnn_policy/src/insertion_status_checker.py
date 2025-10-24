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

import threading

from isaac_manipulator_ros_python_utils.gear_assembly import InsertionState
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int8
import tf2_ros


class InsertionCompletionChecker(Node):
    """Check if insertion is complete based on TF frames."""

    def __init__(self):
        super().__init__('insertion_completion_checker')

        # Declare parameters
        self.declare_parameter('distance_threshold', 0.0130)  # 5mm
        self.declare_parameter('angular_threshold', 3.14)   # 180 degrees
        self.declare_parameter('timeout_seconds', 60.0)     # 60 seconds
        self.declare_parameter('check_frequency', 10.0)     # 10Hz
        self.declare_parameter('end_effector_frame', 'end_effector')
        self.declare_parameter('goal_frame', 'goal')

        # Read parameters
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.angular_threshold = self.get_parameter('angular_threshold').value
        self.timeout_seconds = self.get_parameter('timeout_seconds').value
        self.check_frequency = self.get_parameter('check_frequency').value
        self.end_effector_frame = self.get_parameter('end_effector_frame').value
        self.goal_frame = self.get_parameter('goal_frame').value

        # State tracking
        self.state = InsertionState.IDLE.value
        self.state_lock = threading.Lock()
        self.insertion_start_time = None

        # Set up TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to insertion status
        self.create_subscription(
            Int8,
            'sub_insertion_status',
            self.status_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10)
        )

        # Publisher for completion status
        self.status_publisher = self.create_publisher(
            Int8,
            'pub_insertion_status',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10)
        )

        # Create timer for checking completion
        self.timer = self.create_timer(1.0 / self.check_frequency, self.check_completion)

        self.get_logger().info('Insertion Completion Checker initialized')
        self.get_logger().info(f'Checking distance < {self.distance_threshold}m between '
                               f'{self.end_effector_frame} and {self.goal_frame}')
        self.get_logger().info(f'Timeout set to {self.timeout_seconds} seconds')

    def status_callback(self, msg: Int8):
        """
        Handle status updates from the action server.

        Args
        ----
            msg (Int8): The status message

        Returns
        -------
            None

        """
        new_state = msg.data
        self.get_logger().info(f'Got new state {new_state}')
        with self.state_lock:
            current_state = self.state
        if (
            new_state == InsertionState.INSERTING.value and
            current_state != InsertionState.INSERTING.value
        ):
            # Insertion just started
            with self.state_lock:
                self.state = InsertionState.INSERTING.value
            self.insertion_start_time = self.get_clock().now()
            self.get_logger().info('Insertion started, monitoring for completion')
        elif (
            new_state != InsertionState.INSERTING.value and
            current_state == InsertionState.INSERTING.value
        ):
            # Insertion was manually stopped or completed elsewhere
            with self.state_lock:
                self.state = new_state
            self.get_logger().info(f'Insertion state changed to {InsertionState(new_state).name}')

    def check_completion(self):
        """Check if insertion is complete based on TF frames."""
        with self.state_lock:
            current_state = self.state
        # Only check when in INSERTING state
        if current_state != InsertionState.INSERTING.value:
            return

        # Check for timeout
        current_time = self.get_clock().now()
        elapsed_seconds = (current_time - self.insertion_start_time).nanoseconds / 1e9

        if elapsed_seconds > self.timeout_seconds:
            self.get_logger().warn(f'Insertion timed out after {elapsed_seconds:.1f} seconds')
            self.set_state(InsertionState.FAILED.value)
            return

        # Try to get the transform between end effector and goal
        try:
            parent_frame = self.goal_frame
            child_frame = self.end_effector_frame
            transform = self.tf_buffer.lookup_transform(
                parent_frame,
                child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.01))  # 10ms timeout

            # Calculate distance
            distance = np.linalg.norm([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            # Calculate angle differences (simplified)
            q = transform.transform.rotation
            rotation = R.from_quat([q.x, q.y, q.z, q.w])
            angle = np.linalg.norm(rotation.as_rotvec())

            self.get_logger().info(f'Distance: {distance:.4f}m, Angle: {angle:.4f}rad')

            # Check if we've reached the goal
            if distance < self.distance_threshold and angle < self.angular_threshold:
                self.get_logger().info(
                    f'Insertion complete! Distance: {distance:.4f}m, Angle: {angle:.4f}rad')
                self.set_state(InsertionState.COMPLETED.value)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF error: {e}')
            # Don't fail here, it might just be temporary
            pass

    def set_state(self, new_state: int):
        """Set state and publish status update."""
        with self.state_lock:
            self.state = new_state

        # Publish the status
        status_msg = Int8()
        status_msg.data = new_state
        self.status_publisher.publish(status_msg)


def main():
    rclpy.init()
    node = InsertionCompletionChecker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
