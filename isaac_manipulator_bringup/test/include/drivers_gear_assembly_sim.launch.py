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

from isaac_manipulator_ros_python_utils import (
    CoreConfig, get_isaac_sim_joint_parser_node, get_moveit_group_node,
    get_robot_state_publisher,
    get_visualization_node, start_tool_communication, UrRobotiqDriverConfig
)

from isaac_ros_launch_utils import GroupAction

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)


def launch_setup(context, *args, **kwargs):
    driver_config = UrRobotiqDriverConfig(context)

    manipulator_init_nodes = []
    core_config = CoreConfig(context)

    if driver_config.use_sim_time:
        manipulator_init_nodes.append(
            get_isaac_sim_joint_parser_node(driver_config.use_sim_time))
    else:
        manipulator_init_nodes.append(
            start_tool_communication(driver_config.robot_ip)
        )
    manipulator_init_nodes.append(get_robot_state_publisher(driver_config))
    _, moveit_config = get_moveit_group_node(driver_config)
    if core_config.enable_rviz_visualization:
        manipulator_init_nodes.append(
            get_visualization_node(
                core_config=core_config,
                moveit_config=moveit_config
            )
        )
    return manipulator_init_nodes


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'log_level',
            description='Log level of the container.',
            choices=['debug', 'info', 'warn', 'error']
        ),
        DeclareLaunchArgument(
            'ur_type',
            description='Type/series of used UR robot.',
            choices=['ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e', 'ur16e', 'ur20', 'ur30'],
        ),
        DeclareLaunchArgument(
            'controller_spawner_timeout',
            description='Timeout used when spawning controllers.',
        ),
        DeclareLaunchArgument(
            'tf_prefix',
            description='tf_prefix of the joint names, useful for '
                        'multi-robot setup. If changed, also '
                        'joint names in the controllers configuration have to be updated.',
        ),
        DeclareLaunchArgument(
            'runtime_config_package',
            description='Package with the controllers configuration in config folder. '
            'Usually the argument is not set, it enables use of a custom setup.',
        ),
        DeclareLaunchArgument(
            'initial_joint_controller',
            default_value='scaled_joint_trajectory_controller',
            description='Initially loaded robot controller. This is done so multiple controllers '
                        'do not compete for the same resource',
        ),
        DeclareLaunchArgument(
            'gripper_type',
            description='Type of gripper to use with UR robot',
            choices=['robotiq_2f_85', 'robotiq_2f_140'],
        ),
        DeclareLaunchArgument(
            'joint_limits_file_path',
            default_value='joint_limits_file_path',
            description='Joint limits file path',
        ),
        DeclareLaunchArgument(
            'kinematics_file_path',
            description='Kinematics limits file path',
        ),
        DeclareLaunchArgument(
            'moveit_controllers_file_path',
            description='Move it controller config file path',
        ),
        DeclareLaunchArgument(
            'ros2_controllers_file_path',
            description='ROS2 controls controller config file path',
        ),
        DeclareLaunchArgument(
            'robot_ip', description='IP address by which the robot can be reached. Not needed'
            'for sim-based setup.'
        ),
        DeclareLaunchArgument(
            'workflow_type',
            choices=['POSE_TO_POSE', 'PICK_AND_PLACE', 'OBJECT_FOLLOWING', 'GEAR_ASSEMBLY'],
            description='Type of workflow to run the sim based manipulator pipeline on',
        )
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )

    return LaunchDescription(launch_args + [group_action])
