# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'world_frame',
            default_value='base_link',
            description='The world frame of the robot'),
        DeclareLaunchArgument(
            'target_frames',
            default_value='["target1_frame", "target2_frame"]',
            description='The list of target frames that the robot should plan towards'),
        DeclareLaunchArgument(
            'plan_timer_period',
            default_value='0.01',
            description='The time in seconds for which the goal should request a plan'),
        DeclareLaunchArgument(
            'planner_group_name',
            default_value='ur_manipulator',
            description='The MoveIt group name that the planner should plan for'),
        DeclareLaunchArgument(
            'pipeline_id',
            default_value='isaac_ros_cumotion',
            description='The MoveIt pipeline ID to use'),
        DeclareLaunchArgument(
            'planner_id',
            default_value='cuMotion',
            description='The MoveIt planner ID to use'),
        DeclareLaunchArgument(
            'end_effector_link',
            default_value='wrist_3_link',
            description='The name of the end effector link for planning'),
    ]

    world_frame = LaunchConfiguration('world_frame')
    target_frames = LaunchConfiguration('target_frames')
    plan_timer_period = LaunchConfiguration('plan_timer_period')
    planner_group_name = LaunchConfiguration('planner_group_name')
    pipeline_id = LaunchConfiguration('pipeline_id')
    planner_id = LaunchConfiguration('planner_id')
    end_effector_link = LaunchConfiguration('end_effector_link')

    pose_to_pose_node = Node(
        package='isaac_ros_moveit_goal_setter',
        namespace='',
        executable='pose_to_pose_node',
        name='pose_to_pose_node',
        parameters=[{
            'world_frame': world_frame,
            'target_frames': target_frames,
            'plan_timer_period': plan_timer_period,
            'planner_group_name': planner_group_name,
            'pipeline_id': pipeline_id,
            'planner_id': planner_id,
            'end_effector_link': end_effector_link,
        }],
        output='screen',
        on_exit=Shutdown()
    )

    return launch.LaunchDescription(launch_args + [pose_to_pose_node])
