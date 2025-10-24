#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, GroupAction,
    OpaqueFunction, Shutdown
)
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    behavior_tree_config_file = str(context.perform_substitution(
        LaunchConfiguration('behavior_tree_config_file')))
    blackboard_config_file = str(context.perform_substitution(
        LaunchConfiguration('blackboard_config_file')))
    log_level = str(context.perform_substitution(LaunchConfiguration('log_level')))
    print_ascii_tree = str(context.perform_substitution(
        LaunchConfiguration('print_ascii_tree'))).lower() == 'true'
    manual_mode = str(context.perform_substitution(
        LaunchConfiguration('manual_mode'))).lower() == 'true'

    cmd_args = [
        'ros2', 'run', 'isaac_manipulator_pick_and_place', 'multi_object_pick_and_place',
        '--behavior_tree_config_file', behavior_tree_config_file,
        '--blackboard_config_file', blackboard_config_file,
        '--log-level', log_level,
    ]

    if print_ascii_tree:
        cmd_args.append('--print-ascii-tree')
    if manual_mode:
        cmd_args.append('--manual-mode')

    return [
        ExecuteProcess(
            cmd=cmd_args,
            output='screen',
            prefix=['xterm -hold -e'] if manual_mode else None,
            on_exit=Shutdown(),
        )
    ]


def generate_launch_description():
    """Launch file for the orchestration script."""
    launch_args = [
        DeclareLaunchArgument(
            'behavior_tree_config_file',
            description='Path to the behavior tree configuration file',
            default_value='multi_object_pick_and_place_behavior_tree_params.yaml'
        ),
        DeclareLaunchArgument(
            'blackboard_config_file',
            description='Path to the blackboard configuration file',
            default_value='multi_object_pick_and_place_blackboard_params.yaml',
        ),
        DeclareLaunchArgument(
            'print_ascii_tree',
            description='Print the ASCII tree',
            default_value='False',
        ),
        DeclareLaunchArgument(
            'manual_mode',
            description='Manual mode',
            default_value='False',
        ),
        DeclareLaunchArgument(
            'log_level',
            description='Log level',
            default_value='info',
        ),
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )

    return LaunchDescription(launch_args + [
        group_action,
    ])
