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

import time

from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import RobotTrajectory
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MockMotionPlanServer(Node):
    """Simple mock server for cumotion/motion_plan action server."""

    def __init__(self):
        super().__init__('mock_motion_plan_server')
        self._action_server = ActionServer(
            self,
            MotionPlan,
            'cumotion/motion_plan',
            self.execute_callback
        )
        self.get_logger().info('Mock Motion Plan Server started')
        self.get_logger().info(
            f'Server is ready to accept requests at '
            f'{self.get_namespace()}/cumotion/motion_plan')

    def execute_callback(self, goal_handle):
        """Execute the motion plan action."""
        self.get_logger().info('Received motion plan request')

        # Simulate processing time
        time.sleep(2.0)

        # Create a mock result
        result = MotionPlan.Result()
        result.success = True
        result.error_code.val = 1  # SUCCESS
        result.message = 'Motion plan successful'

        # Create simple dummy trajectories
        result.planned_trajectory = []

        if goal_handle.request.plan_grasp:
            # Return 2 trajectories for grasp: approach (0) and lift (1)
            result.planned_trajectory.append(self._create_dummy_trajectory())
            result.planned_trajectory.append(self._create_dummy_trajectory())

        elif goal_handle.request.plan_pose:
            # Return 1 trajectory for pose planning
            result.planned_trajectory.append(self._create_dummy_trajectory())

        goal_handle.succeed()
        return result

    def _create_dummy_trajectory(self):
        """Create a simple dummy trajectory."""
        traj = RobotTrajectory()
        traj.joint_trajectory = JointTrajectory()
        traj.joint_trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3']

        # Add one simple trajectory point
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0]
        point.velocities = [0.0, 0.0, 0.0]
        point.accelerations = [0.0, 0.0, 0.0]
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        traj.joint_trajectory.points.append(point)

        return traj


def main():
    rclpy.init()
    server = MockMotionPlanServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Server stopped by user')
    finally:
        server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
