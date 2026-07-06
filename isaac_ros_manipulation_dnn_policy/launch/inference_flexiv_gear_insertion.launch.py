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

GEAR_SHAFT_OFFSETS = {
    'gear_small': [0.076125, 0.0, 0.0],
    'gear_medium': [0.030375, 0.0, 0.0],
    'gear_large': [-0.045375, 0.0, 0.0],
}


def launch_setup(context, *args, **kwargs):
    isaac_ros_ws = os.getenv('ISAAC_ROS_WS')
    if not isaac_ros_ws:
        raise ValueError('ISAAC_ROS_WS environment variable is not set')

    # Resolve launch configurations
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    alpha = float(LaunchConfiguration('alpha').perform(context))
    gear_type = LaunchConfiguration('gear_type').perform(context)
    record = LaunchConfiguration('record').perform(context)
    ros_bag_folder_path = LaunchConfiguration('ros_bag_folder_path').perform(context)
    use_sim_time = LaunchConfiguration(
        'use_sim_time').perform(context).lower() in ('true', '1', 'yes')
    robot_sn = LaunchConfiguration('robot_sn').perform(context)
    controller_name = LaunchConfiguration('controller_name').perform(context)
    input_joint_states_topic = LaunchConfiguration(
        'input_joint_states_topic').perform(context)
    goal_frame = LaunchConfiguration('goal_frame').perform(context)
    world_frame = LaunchConfiguration('world_frame').perform(context)
    goal_pose_frequency = float(
        LaunchConfiguration('goal_pose_frequency').perform(context))
    joint_state_age_threshold_ms = float(
        LaunchConfiguration('joint_state_age_threshold_ms').perform(context))
    enable_goal_pose_publisher = LaunchConfiguration(
        'enable_goal_pose_publisher').perform(context).lower() in ('true', '1', 'yes')

    # Resolve gear shaft offset
    if gear_type not in GEAR_SHAFT_OFFSETS:
        raise ValueError(
            f'Unknown gear_type {gear_type}. '
            f'Must be one of: {list(GEAR_SHAFT_OFFSETS.keys())}')

    gear_shaft_offset = GEAR_SHAFT_OFFSETS[gear_type] if enable_goal_pose_publisher else None

    joint_prefix = f'{robot_sn}_'

    # Topic names
    flexiv_joint_states_topic = input_joint_states_topic
    target_joint_positions_topic = f'/{controller_name}/joint_commands'
    goal_pose_topic = LaunchConfiguration(
        'input_goal_pose_topic').perform(context)

    nodes = []

    # 1. Goal Pose Publisher - Looks up gear base TF and publishes as goal pose
    #    Skipped when enable_goal_pose_publisher:=False (external source provides /goal_pose)
    if enable_goal_pose_publisher:
        nodes.append(
            Node(
                name='goal_pose_publisher_node',
                package='isaac_ros_manipulation_dnn_policy',
                executable='goal_pose_publisher_node.py',
                remappings=[
                    ('goal_pose', goal_pose_topic),
                ],
                parameters=[{
                    'goal_frame': goal_frame,
                    'world_frame': world_frame,
                    'frequency': goal_pose_frequency,
                    'enable_publishing_on_trigger': False,
                    'use_sim_time': use_sim_time,
                }],
                output='both',
                on_exit=Shutdown(),
            )
        )

    # 2. Observation Encoder - Encodes joint states + gear shaft pose
    #    joint_prefix strips the robot SN prefix from joint names so the
    #    policy sees generic joint1..joint7.  Empty prefix is a no-op (UR).
    obs_encoder_params = {
        'joint_state_age_threshold_ms': joint_state_age_threshold_ms,
        'joint_prefix': joint_prefix,
        'quaternion_order': 'xyzw',
        'use_sim_time': use_sim_time,
    }
    if gear_shaft_offset is not None:
        obs_encoder_params['gear_shaft_offset'] = gear_shaft_offset
    nodes.append(
        Node(
            name='observation_encoder_node',
            package='isaac_ros_manipulation_dnn_policy',
            executable='observation_encoder_node.py',
            remappings=[
                ('goal_pose', goal_pose_topic),
                ('joint_state', flexiv_joint_states_topic),
            ],
            parameters=[obs_encoder_params],
            output='both',
            on_exit=Shutdown(),
        )
    )

    # 3. Inference Node - Runs the DNN policy (ActorCriticRecurrent / LSTM)
    nodes.append(
        Node(
            name='inference_node',
            package='isaac_ros_manipulation_dnn_policy',
            executable='inference_node.py',
            parameters=[{
                'checkpoint': checkpoint,
                'alpha': alpha,
                'use_sim_time': use_sim_time,
                'step_sleep_s': float(LaunchConfiguration('step_sleep_s').perform(context)),
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    # 4. Action Decoder - Decodes policy output and publishes directly to controller
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

    # 5. Optional: Rosbag recording
    if record.lower() in ('true', '1', 'yes'):
        nodes.append(
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'record', '--storage', 'mcap',
                    '--output', ros_bag_folder_path,
                    '/rosout',
                    '/tf',
                    '/tf_static',
                    goal_pose_topic,
                    flexiv_joint_states_topic,
                    target_joint_positions_topic,
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
            'step_sleep_s',
            description='Sleep time in seconds between inference steps (0.0 = no sleep)',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'gear_type',
            description='Gear type: gear_small, gear_medium, or gear_large',
            default_value='gear_large',
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
        DeclareLaunchArgument(
            'robot_sn',
            description='Robot serial number (e.g., Rizon4s-062939)',
        ),
        DeclareLaunchArgument(
            'controller_name',
            description='Name of the ros2_control controller for policy output. '
                        'Available: streaming_position_controller',
            default_value='streaming_position_controller',
        ),
        DeclareLaunchArgument(
            'input_joint_states_topic',
            description='Topic where Flexiv robot publishes joint states',
            default_value='/flexiv_arm/joint_states',
        ),
        DeclareLaunchArgument(
            'goal_frame',
            description='TF frame name for the gear base',
            default_value='gear_base',
        ),
        DeclareLaunchArgument(
            'world_frame',
            description='World / robot base frame for TF lookups',
            default_value='world',
        ),
        DeclareLaunchArgument(
            'goal_pose_frequency',
            description='Rate (Hz) at which goal pose is published',
            default_value='60.0',
        ),
        DeclareLaunchArgument(
            'joint_state_age_threshold_ms',
            description='Max age (ms) of joint state before skipping observation',
            default_value='100.0',
        ),
        DeclareLaunchArgument(
            'enable_goal_pose_publisher',
            default_value='False',
            description='Enable built-in goal pose publisher. Set to False when using '
                        'an external source (e.g., gear_assembly_pose_estimation.launch.py) '
                        'that already publishes /goal_pose with shaft position.',
        ),
        DeclareLaunchArgument(
            'input_goal_pose_topic',
            default_value='/goal_pose',
            description='Topic providing the goal pose for the observation encoder.',
        ),
    ]

    return LaunchDescription(
        launch_args + [OpaqueFunction(function=launch_setup)])
