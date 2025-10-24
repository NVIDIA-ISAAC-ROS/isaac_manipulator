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
from threading import Event
import time

from geometry_msgs.msg import PoseStamped
from isaac_manipulator_interfaces.action import Insert
from isaac_manipulator_interfaces.msg import InsertionRequest
from isaac_manipulator_ros_python_utils.gear_assembly import InsertionState
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int8


class InsertPolicyActionServer(Node):
    """Action server for handling insertion policy execution."""

    def __init__(self):
        super().__init__('insert_policy_action_server')

        # Create publisher for insertion request
        self.insertion_req_pub = self.create_publisher(
            InsertionRequest,
            'insertion_request_topic',
            QoSProfile(
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
        )
        self.insertion_status_publisher = self.create_publisher(
            Int8,
            'insertion_status_topic',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
        )

        self.pose: PoseStamped = None

        # Current status of the robot
        self.current_status = InsertionState.IDLE.value
        self.status_lock = threading.Lock()

        # Subscribe to insertion status
        self.create_subscription(
            Int8,
            'insertion_status_topic',
            self.insertion_status_callback,
            10
        )

        # Configure action server with goal acceptance and cancellation handlers
        self._action_server = ActionServer(
            self,
            Insert,
            'insert_policy',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

        self._insertion_done_event = Event()

        self.get_logger().info('Insert Policy Action Server is ready')

    def insertion_status_callback(self, msg: Int8):
        """
        Handle status updates from the robot - ONLY updates status.

        Args
        ----
            msg (Int8): The status message from the robot.

        Returns
        -------
            None

        """
        with self.status_lock:
            old_status = self.current_status
            self.current_status = msg.data  # Msg data is of type Int8

        # Log status changes
        if old_status != self.current_status:
            self.get_logger().info(
                f'Status changed: {InsertionState(old_status).name} -> '
                f'{InsertionState(self.current_status).name}')

            if self.current_status in [
                InsertionState.COMPLETED.value, InsertionState.FAILED.value
            ]:
                # making the done goal handle
                self._insertion_done_event.set()

            # Stop goal pose publishing when leaving INSERTING state
            if (
                old_status == InsertionState.INSERTING.value
                and self.current_status != InsertionState.INSERTING.value
            ):
                self._send_cancel_request()

    def goal_callback(self, goal_request: Insert.Goal) -> GoalResponse:
        """
        Check if we can accept the goal.

        Args
        ----
            goal_request (Insert.Goal): The goal request.

        Returns
        -------
            GoalResponse: The goal response.

        """
        with self.status_lock:
            if self.current_status == InsertionState.INSERTING.value:
                self.get_logger().warn(
                    f'Rejecting goal: Robot is busy with state: '
                    f'{self.current_status}')
                return GoalResponse.REJECT

            self.get_logger().info('Accepting new insertion goal')
            return GoalResponse.ACCEPT

    def _send_cancel_request(self):
        """Send a cancellation request to the insertion policy."""
        insertion_req_msg = InsertionRequest()
        insertion_req_msg.header.stamp = self.get_clock().now().to_msg()
        insertion_req_msg.request_type = InsertionRequest.CANCEL_INSERTION
        self.insertion_req_pub.publish(insertion_req_msg)

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancel request')

        self._send_cancel_request()

        with self.status_lock:
            self.current_status = InsertionState.IDLE.value

        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: Insert.Goal) -> Insert.Result:
        """
        Execute the insertion policy action.

        Args
        ----
            goal_handle (Insert.Goal): The goal handle.

        Returns
        -------
            Insert.Result: The result of the insertion policy action.

        """
        self.get_logger().info('Executing insertion policy goal')

        with self.status_lock:
            self.current_status = InsertionState.INSERTING.value

        # Extract goal pose
        goal_pose = goal_handle.request.goal_pose

        # Create result and feedback messages
        result = Insert.Result()
        feedback_msg = Insert.Feedback()

        # Create a future to track completion
        self._insertion_done_event.clear()

        # Store current goal handle
        request = InsertionRequest()
        request.pose = goal_pose
        request.request_type = InsertionRequest.NEW_INSERTION

        # Publish the goal pose to the insertion request topic
        self.insertion_req_pub.publish(request)
        # Also publish an insertion status to Inserting for other nodes to see
        # what state machine the insertion subtree is in.
        self.insertion_status_publisher.publish(Int8(data=InsertionState.INSERTING.value))
        self.pose = goal_pose

        try:
            while not self._insertion_done_event.is_set():
                time.sleep(0.1)
                self._publish_feedback(goal_handle, feedback_msg)

            self.get_logger().info('Insertion is completed')

            if self.current_status == InsertionState.COMPLETED.value:
                goal_handle.succeed()
                result.success = True
            else:  # FAILED or any other non-success status
                goal_handle.abort()
                result.success = False

        except Exception as e:
            self.get_logger().error(f'Exception during execution: {e}')
            goal_handle.abort()
            result.success = False
            result.message = f'Execution error: {str(e)}'
        finally:
            self._send_cancel_request()

        self.get_logger().info('Return result')

        return result

    def _publish_feedback(self, goal_handle: Insert.Goal, feedback_msg: Insert.Feedback):
        """
        Publish feedback about the current status.

        Args
        ----
            goal_handle (Insert.Goal): The goal handle.
            feedback_msg (Insert.Feedback): The feedback message.

        Returns
        -------
            None

        """
        with self.status_lock:
            current_status = self.current_status

        feedback_msg.status = current_status
        goal_handle.publish_feedback(feedback_msg)

    def destroy_node(self):
        """Clean up resources when destroying the node."""
        self._send_cancel_request()
        super().destroy_node()


def main():
    rclpy.init()
    node = InsertPolicyActionServer()

    # Create a multithreaded executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # Spin using the multithreaded executor
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
