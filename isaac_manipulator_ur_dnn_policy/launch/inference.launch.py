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

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the workspace path.
    isaac_ros_ws = os.getenv('ISAAC_ROS_WS')
    if not isaac_ros_ws:
        raise ValueError('ISAAC_ROS_WS environment variable is not set')

    # Find a unique timestamped folder name of format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    recording_folder_path = f'{isaac_ros_ws}/inference_recordings_{timestamp}'
    launch_args = [
        DeclareLaunchArgument(
          'checkpoint',
          description='path to .pth model file',
          default_value='isaac_ros_assets/isaac_manipulator_ur_dnn_policy/model_gripper_140.pt',
        ),
        DeclareLaunchArgument(
          'alpha',
          description='alpha value for exponential moving average',
          default_value='1.0',
        ),
        DeclareLaunchArgument(
          'record',
          description='record data to rosbag',
          default_value='False',
        ),
        DeclareLaunchArgument(
          'ros_bag_folder_path',
          description='Path to the recording folder, make sure it does not pre-exist',
          default_value=recording_folder_path,
        ),
        DeclareLaunchArgument(
          'target_joint_positions',
          description='Topic to publish target joint positions to',
          default_value='/target_joint_positions',
        ),
        DeclareLaunchArgument(
          'input_joint_states',
          description='Topic to publish input joint positions to',
          default_value='/joint_states',
        ),
        DeclareLaunchArgument(
          'input_goal_pose_topic',
          description='Topic to publish input goal pose to',
          default_value='/goal_pose',
        ),
        DeclareLaunchArgument(
          'use_sim_time',
          description='Use sim time',
          default_value='False',
        ),
    ]

    checkpoint = LaunchConfiguration('checkpoint')
    alpha = LaunchConfiguration('alpha')
    record = LaunchConfiguration('record')
    ros_bag_folder_path = LaunchConfiguration('ros_bag_folder_path')
    target_joint_positions = LaunchConfiguration('target_joint_positions')
    input_joint_states = LaunchConfiguration('input_joint_states')
    input_goal_pose_topic = LaunchConfiguration('input_goal_pose_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    nodes = []

    nodes.append(
        Node(
            name='observation_encoder_node',
            package='isaac_manipulator_ur_dnn_policy',
            executable='observation_encoder_node.py',
            remappings=[
                ('goal_pose', input_goal_pose_topic),
                ('joint_state', input_joint_states),
            ],
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        Node(
            name='inference_node',
            package='isaac_manipulator_ur_dnn_policy',
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
            package='isaac_manipulator_ur_dnn_policy',
            executable='action_decoder_node.py',
            remappings=[
                ('target_joint_state', target_joint_positions),
            ],
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        ExecuteProcess(
            condition=IfCondition(record),
            cmd=[
                'ros2', 'bag', 'record', '--storage', 'mcap',
                '--output', ros_bag_folder_path,
                '/rosout',
                '/tf',
                '/tf_static',
                input_goal_pose_topic,
                input_joint_states,
                target_joint_positions,
                '/observation',
                '/action',
            ],
            output='both',
            on_exit=Shutdown(),
        )
    )

    return LaunchDescription(launch_args + nodes)
