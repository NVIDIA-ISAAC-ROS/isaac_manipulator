#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test for Pick and Place Orchestrator with actual planning calls."""
# flake8: noqa: E402

import json
import os
import sys
import time

# Add test directory to path so sibling test modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ament_index_python.packages import get_package_share_directory
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from isaac_ros_manipulation_interfaces.action import (
    GetObjectPose, GetObjects, PickAndPlace
)
from isaac_ros_manipulation_ros_python_utils import load_yaml_params
from isaac_ros_manipulation_ros_python_utils.gear_assembly import (
    parse_joint_state_from_yaml
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource
)
from launch_ros.actions import Node
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
import pytest
import rclpy
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState
import test_pick_and_place_orchestrator as _parent_module

RUN_TEST = True
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')

if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')
PLANNING_CALLS_OUTPUT_DIR = os.path.join(ISAAC_ROS_WS, 'planning_calls_output')


TRANSFORMS = [
    {
        'parent': 'base_link',
        'child': 'wrist_3_link',
        'translation': {
            'x': -1.0144,
            'y': 0.3258,
            'z': 0.2688
        },
        'rotation': {
            'w': -0.0009,
            'x': 0.2108,
            'y': 0.9775,
            'z': 0.0013
        },
        'description': 'Final place position.'
    },
    {
        'parent': 'base_link',
        'child': 'wrist_3_link',
        'translation': {
            'x': -1.11304,
            'y': -0.0820,
            'z': -0.0575
        },
        'rotation': {
            'w': 0.00510531,
            'x': 0.00436991,
            'y': -0.01456639,
            'z': 0.9998489885977783
        },
        'description': 'Pick place position. (detected_object1)'
    },
]


class PlanningCallsTest(_parent_module.PickAndPlaceOrchestratorTest):
    """Test class for pick and place orchestrator with actual planning calls."""

    JOINT_NAMES = [
        'finger_joint',
        'shoulder_pan_joint',
        'wrist_3_joint',
        'wrist_2_joint',
        'wrist_1_joint',
        'elbow_joint',
        'shoulder_lift_joint',
    ]

    INITIAL_JOINT_POSITIONS = [
        0.6858149779735682,
        2.8192882537841797,
        -1.6109302679644983,
        -1.5776174704181116,
        -1.8490525684752406,
        1.1776617209063929,
        -0.9138882917216797,
    ]

    def setupServers(self):
        """Set up servers - use real planning, mock perception and gripper."""
        self._current_joint_positions = list(self.INITIAL_JOINT_POSITIONS)
        self.executed_trajectory_count = 0
        self.output_dir = PLANNING_CALLS_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._joint_state_pub = self.node.create_publisher(
            JointState, '/joint_states', 10)
        self._joint_state_timer = self.node.create_timer(
            0.1, self._publish_joint_states)

        self._get_objects_server = ActionServer(
            self.node, GetObjects, '/get_objects', self._get_objects_callback)
        self._get_object_pose_server = ActionServer(
            self.node, GetObjectPose, '/get_object_pose',
            self._get_object_pose_callback)
        self._gripper_server = ActionServer(
            self.node, GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd',
            self._gripper_callback)
        self._execute_trajectory_server = ActionServer(
            self.node, ExecuteTrajectory, '/execute_trajectory',
            self._execute_trajectory_callback)

        self.node.get_logger().info('Mock perception and gripper servers initialized')

    def _publish_joint_states(self):
        """Publish joint states on a timer."""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = list(self._current_joint_positions)
        msg.velocity = [0.0] * len(self.JOINT_NAMES)
        msg.effort = [0.0] * len(self.JOINT_NAMES)

        self._joint_state_pub.publish(msg)

    def _execute_trajectory_callback(self, goal_handle):
        """Mock ExecuteTrajectory: accept trajectory, update joint states to final position."""
        jt = goal_handle.request.trajectory.joint_trajectory
        if jt.points:
            last_point = jt.points[-1]
            traj_name_to_pos = dict(zip(jt.joint_names, last_point.positions))
            for i, name in enumerate(self.JOINT_NAMES):
                if name in traj_name_to_pos:
                    self._current_joint_positions[i] = traj_name_to_pos[name]
            self.node.get_logger().info(
                f'ExecuteTrajectory: updated joint states from trajectory '
                f'({len(jt.points)} points)')
        else:
            self.node.get_logger().warn('ExecuteTrajectory: received empty trajectory')

        self.executed_trajectory_count += 1
        traj_dict = {
            'joint_names': list(jt.joint_names),
            'points': [
                {
                    'positions': list(pt.positions),
                    'velocities': list(pt.velocities),
                    'accelerations': list(pt.accelerations),
                    'time_from_start': {
                        'sec': pt.time_from_start.sec,
                        'nanosec': pt.time_from_start.nanosec,
                    },
                }
                for pt in jt.points
            ],
        }
        output_path = os.path.join(
            self.output_dir,
            f'trajectory_{self.executed_trajectory_count}.json')
        with open(output_path, 'w') as f:
            json.dump(traj_dict, f, indent=2)

        result = ExecuteTrajectory.Result()
        result.error_code = MoveItErrorCodes()
        result.error_code.val = MoveItErrorCodes.SUCCESS
        goal_handle.succeed()
        return result

    def _ik_solution_callback(self, goal_handle):
        pass

    def _attach_object_callback(self, goal_handle):
        pass

    def test_pick_and_place_orchestrator(self):
        """Test pick and place orchestrator with actual planning calls."""
        self.node.get_logger().info('Starting test for pick and place with real planning')

        time.sleep(10.0)
        self.setupServers()

        from rclpy.action import ActionClient
        pick_and_place_client = ActionClient(self.node, PickAndPlace, '/pick_and_place')

        self.assertTrue(
            pick_and_place_client.wait_for_server(timeout_sec=30.0),
            'Pick and place action server not available'
        )

        place_tf = TRANSFORMS[0]
        place_pose = Pose()
        place_pose.position.x = place_tf['translation']['x']
        place_pose.position.y = place_tf['translation']['y']
        place_pose.position.z = place_tf['translation']['z']
        place_pose.orientation.x = place_tf['rotation']['x']
        place_pose.orientation.y = place_tf['rotation']['y']
        place_pose.orientation.z = place_tf['rotation']['z']
        place_pose.orientation.w = place_tf['rotation']['w']

        model_path = os.path.join(
            ISAAC_ROS_WS, 'isaac_ros_assets', 'isaac_ros_manipulation_dnn_policy')
        target_joint_state = parse_joint_state_from_yaml(
            model_path + '/params/env.yaml', use_sim_time=False)

        goal = PickAndPlace.Goal()
        goal.place_pose = place_pose
        goal.use_joint_space_planner_for_place_pose = True
        goal.target_joint_state_for_place_pose = target_joint_state
        goal_future = pick_and_place_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=10.0)
        self.assertTrue(goal_future.done(), 'Goal was not accepted in time')

        goal_handle = goal_future.result()
        self.assertTrue(goal_handle.accepted, 'Goal was rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=120.0)
        self.assertTrue(result_future.done(), 'Action did not complete in time')

        result = result_future.result()
        self.node.get_logger().info(f'Pick and place result: {result}')
        self.assertTrue(result.result.success, 'Pick and place action failed')

    @classmethod
    def generate_test_description(cls, nodes, node_startup_delay):
        """Generate test description."""
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with actual planning nodes for testing."""
    PlanningCallsTest.generate_namespace()

    test_nodes = []
    node_startup_delay = 1.0

    isaac_ros_manipulation_test_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_gear_assembly'),
        'test', 'include')

    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'params',
        'ur10e_robotiq_2f_140_gear_assembly.yaml'
    )
    params = load_yaml_params(test_yaml_config)

    if RUN_TEST:
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_test_include_dir,
                 '/pick_and_place.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_test_include_dir,
                 '/planning_nodes.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))
        pick_tf = TRANSFORMS[1]
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=[
                str(pick_tf['translation']['x']),
                str(pick_tf['translation']['y']),
                str(pick_tf['translation']['z']),
                str(pick_tf['rotation']['x']),
                str(pick_tf['rotation']['y']),
                str(pick_tf['rotation']['z']),
                str(pick_tf['rotation']['w']),
                'base_link', 'detected_object1',
            ]
        ))
    else:
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return PlanningCallsTest.generate_test_description(
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
    )
