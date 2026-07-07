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

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    isaac_ros_ws = os.getenv('ISAAC_ROS_WS')
    if not isaac_ros_ws:
        raise ValueError('ISAAC_ROS_WS environment variable is not set')

    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    alpha = float(LaunchConfiguration('alpha').perform(context))
    record = LaunchConfiguration('record').perform(context)
    ros_bag_folder_path = LaunchConfiguration('ros_bag_folder_path').perform(context)
    use_sim_time = LaunchConfiguration(
        'use_sim_time').perform(context).lower() in ('true', '1', 'yes')
    robot_sn = LaunchConfiguration('robot_sn').perform(context)
    controller_name = LaunchConfiguration('controller_name').perform(context)
    input_joint_states_topic = LaunchConfiguration(
        'input_joint_states_topic').perform(context)
    input_goal_pose_topic = LaunchConfiguration(
        'input_goal_pose_topic').perform(context)
    joint_state_age_threshold_ms = float(
        LaunchConfiguration('joint_state_age_threshold_ms').perform(context))

    joint_prefix = f'{robot_sn}_'

    nodes = []

    nodes.append(
        Node(
            name='observation_encoder_node',
            package='isaac_ros_manipulation_dnn_policy',
            executable='observation_encoder_node.py',
            remappings=[
                ('goal_pose', input_goal_pose_topic),
                ('joint_state', input_joint_states_topic),
            ],
            parameters=[{
                'joint_state_age_threshold_ms': joint_state_age_threshold_ms,
                'joint_prefix': joint_prefix,
                'use_sim_time': use_sim_time,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        Node(
            name='inference_node',
            package='isaac_ros_manipulation_dnn_policy',
            executable='inference_node.py',
            parameters=[{
                'checkpoint': checkpoint,
                'alpha': alpha,
                'use_sim_time': use_sim_time,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        Node(
            name='action_decoder_node',
            package='isaac_ros_manipulation_dnn_policy',
            executable='action_decoder_node.py',
            parameters=[{
                'joint_prefix': joint_prefix,
                'controller_name': controller_name,
                'use_sim_time': use_sim_time,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    if record.lower() in ('true', '1', 'yes'):
        target_topic = f'/{controller_name}/joint_commands'
        nodes.append(
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record', '--storage', 'mcap',
                    '--output', ros_bag_folder_path,
                    '/rosout',
                    '/tf',
                    '/tf_static',
                    input_goal_pose_topic,
                    input_joint_states_topic,
                    target_topic,
                    '/observation',
                    '/action',
                ],
                output='both',
                on_exit=Shutdown(),
            )
        )

    return nodes


def generate_launch_description():
    isaac_ros_ws = os.getenv('ISAAC_ROS_WS', '')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    recording_folder_path = f'{isaac_ros_ws}/inference_recordings_{timestamp}'

    launch_args = [
        DeclareLaunchArgument(
            'checkpoint',
            description='Path to .pt model checkpoint file',
        ),
        DeclareLaunchArgument(
            'alpha',
            description='Alpha for exponential moving average (1.0 = no smoothing)',
            default_value='1.0',
        ),
        DeclareLaunchArgument(
            'robot_sn',
            description='Robot serial number (e.g., Rizon4s-062839)',
        ),
        DeclareLaunchArgument(
            'controller_name',
            description='Name of the ros2_control controller for policy output',
            default_value='streaming_position_controller',
        ),
        DeclareLaunchArgument(
            'input_joint_states_topic',
            description='Topic where Flexiv robot publishes joint states',
            default_value='/flexiv_arm/joint_states',
        ),
        DeclareLaunchArgument(
            'input_goal_pose_topic',
            default_value='/goal_pose',
            description='Topic providing the goal pose for the observation encoder',
        ),
        DeclareLaunchArgument(
            'joint_state_age_threshold_ms',
            description='Max age (ms) of joint state before skipping observation',
            default_value='100.0',
        ),
        DeclareLaunchArgument(
            'record',
            description='Record data to rosbag',
            default_value='False',
        ),
        DeclareLaunchArgument(
            'ros_bag_folder_path',
            description='Path to the recording folder',
            default_value=recording_folder_path,
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            description='Use sim time',
            default_value='False',
        ),
    ]

    return LaunchDescription(
        launch_args + [OpaqueFunction(function=launch_setup)])
