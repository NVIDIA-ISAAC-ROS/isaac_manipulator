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
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    launch_args = [
        DeclareLaunchArgument(
            'world_frame',
            default_value='base_link',
            description='The world frame of the robot'),
        DeclareLaunchArgument(
            'grasp_frame',
            default_value='grasp_frame',
            description='The grasp frame (goal) that the robot should target'),
        DeclareLaunchArgument(
            'grasp_frame_stale_time_threshold',
            default_value='30.0',
            description='The duration until a grasp frame (goal) not updating is considered stale'),
        DeclareLaunchArgument(
            'goal_change_position_threshold',
            default_value='0.1',
            description='The minimum amount that the goal must move to be targeted'),
        DeclareLaunchArgument(
            'plan_timer_period',
            default_value='0.5',
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
    grasp_frame = LaunchConfiguration('grasp_frame')
    grasp_frame_stale_time_threshold = LaunchConfiguration('grasp_frame_stale_time_threshold')
    goal_change_position_threshold = LaunchConfiguration('goal_change_position_threshold')
    plan_timer_period = LaunchConfiguration('plan_timer_period')
    planner_group_name = LaunchConfiguration('planner_group_name')
    pipeline_id = LaunchConfiguration('pipeline_id')
    planner_id = LaunchConfiguration('planner_id')
    end_effector_link = LaunchConfiguration('end_effector_link')

    goal_init_node = Node(
        package='isaac_ros_moveit_goal_setter',
        namespace='',
        executable='goal_initializer_node',
        name='goal_init_node',
        parameters=[{
            'world_frame': world_frame,
            'grasp_frame': grasp_frame,
            'grasp_frame_stale_time_threshold': grasp_frame_stale_time_threshold,
            'goal_change_position_threshold': goal_change_position_threshold,
            'plan_timer_period': plan_timer_period,
            'planner_group_name': planner_group_name,
            'pipeline_id': pipeline_id,
            'planner_id': planner_id,
            'end_effector_link': end_effector_link,
        }],
        output='screen'
    )

    return (launch.LaunchDescription(launch_args + [goal_init_node]))
