# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
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

import os
from typing import List, Union

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_ros_python_utils.config import (
    GearAssemblyConfig, ObjectFollowingConfig, OrchestrationConfig,
    PoseToPoseConfig, WorkflowType
)

from launch.actions import (
    ExecuteProcess, IncludeLaunchDescription, Shutdown
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def get_pick_and_place_orchestrator(workflow_config: GearAssemblyConfig) -> List[Node]:
    """
    Get the pick and place orchestrator.

    Args
    ----
        workflow_config (WorkflowConfig) : WorkflowConfig that dictates Isaac Sim workflows

    Returns
    -------
        List[Node] : ROS node for Pick and place

    """
    if (
        workflow_config.use_ground_truth_pose_in_sim and
        workflow_config.workflow_type != WorkflowType.GEAR_ASSEMBLY
    ):
        object_frame_name = workflow_config.sim_gt_asset_frame_id
    else:
        object_frame_name = 'detected_object1'
    joint_states_topic = '/joint_states'
    # Robot- and gripper-specific values live on ``driver_config``; workflow-level
    # knobs (retries, scales, attachments, etc.) stay on ``workflow_config``.
    driver_config = workflow_config.driver_config
    pick_and_place_orchestrator_node = Node(
        package='isaac_ros_manipulation_gear_assembly',
        executable='pick_and_place_orchestrator.py',
        name='pick_and_place_orchestrator',
        parameters=[{
            'grasp_file_path': workflow_config.grasps_file_path,
            'isaac_sim': workflow_config.use_sim_time,
            'use_sim_time': workflow_config.use_sim_time,
            'time_dilation_factor': float(workflow_config.time_dilation_factor),
            'retract_offset_distance': [0.0, 0.0, 0.15],
            'object_frame_name': object_frame_name,
            'use_ground_truth_pose_from_sim': workflow_config.use_ground_truth_pose_in_sim,
            'sleep_time_before_planner_tries_sec': workflow_config.pick_and_place_retry_wait_time,
            'num_planner_tries': workflow_config.pick_and_place_planner_retries,
            'publish_grasp_frame': True,
            'gripper_collision_links': driver_config.gripper_collision_links,
            'attach_object_shape': workflow_config.object_attachment_type.value,
            'attach_object_scale': workflow_config.object_attachment_scale,
            'attach_object_mesh_file_path': workflow_config.attach_object_mesh_file_path,
            'grasp_approach_in_world_frame': False,
            'retract_in_world_frame': True,
            'use_pose_from_rviz': workflow_config.use_pose_from_rviz,
            'end_effector_mesh_resource_uri': workflow_config.end_effector_mesh_resource_uri,
            'joint_states_topic': joint_states_topic,
            'move_to_home_pose_after_place': workflow_config.move_to_home_pose_after_place,
            'home_pose': workflow_config.home_pose,
            'seed_state_for_ik_solver_for_joint_space_planner':
                workflow_config.seed_state_for_ik_solver_for_joint_space_planner,
            'base_frame': driver_config.base_frame,
            'gripper_frame': driver_config.gripper_frame,
            'grasp_frame': driver_config.grasp_frame,
            'gripper_action_name': driver_config.gripper_action_name,
            'gripper_open_position': driver_config.gripper_open_position,
            'gripper_close_position': driver_config.gripper_close_position,
            'gripper_settle_time_sec': driver_config.gripper_settle_time_sec,
            'joint_names': driver_config.joint_names,
            'joint_limits_file_path': driver_config.joint_limits_file_path,
        }],
        output='screen',
        on_exit=Shutdown(),
    )
    return [pick_and_place_orchestrator_node]


def get_pose_to_pose(workflow_config: PoseToPoseConfig) -> List[Node]:
    """
    Get pose to pose nodes.

    Launches the pose-to-pose workflow from isaac_ros_manipulation_pose_to_pose
    via isaac_ros_manipulation_bringup (pose_to_pose.launch.py).

    Args
    ----
        workflow_config (PoseToPoseConfig): Configuration for pose to pose workflow

    Returns
    -------
        List[Node]: List of nodes

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )
    pose_to_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/pose_to_pose.launch.py']
        ),
        launch_arguments={
            'time_dilation_factor': str(workflow_config.time_dilation_factor),
        }.items(),
    )
    return [pose_to_pose_launch]


def get_object_following(workflow_config: ObjectFollowingConfig) -> List[Node]:
    """
    Get the node helpful for triggering planning for object following.

    Launches the object-following workflow from
    isaac_ros_manipulation_object_following via isaac_ros_manipulation_bringup
    (object_following.launch.py). Adds offset over detected pose so that the robot gripper
    does not collide with it when following the object.

    Args
    ----
        workflow_config (ObjectFollowingConfig): Configuration for object following workflow

    Returns
    -------
        List[Node]: List of nodes for object following

    """
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch', 'include'
    )

    target_tool_frame = workflow_config.target_tool_frame
    return [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.043', '0.359', '0.065', '0.553',
                       '0.475', '-0.454', '0.513', 'detected_object1',
                       target_tool_frame],
            on_exit=Shutdown(),
        ), IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [launch_files_include_dir, '/object_following.launch.py']
            ),
            launch_arguments={
                'target_tool_frame': target_tool_frame,
                'time_dilation_factor': str(workflow_config.time_dilation_factor),
            }.items()
        )
    ]


def get_multi_object_pick_and_place(
        orchestration_config: OrchestrationConfig) -> List[Union[ExecuteProcess]]:
    """
    Get the multi-object pick and place behavior tree process.

    Parameters
    ----------
    orchestration_config : OrchestrationConfig
        Configuration for the orchestration behavior tree including
        behavior tree config file, blackboard config, and logging settings

    Returns
    -------
    List[Union[ExecuteProcess]]
        ROS process for multi-object pick and place behavior tree

    """
    multi_object_pick_and_place_process = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory('isaac_ros_manipulation_pick_and_place'),
             '/launch/orchestration.launch.py']),
        launch_arguments={
            'behavior_tree_config_file': orchestration_config.behavior_tree_config_file,
            'blackboard_config_file': orchestration_config.blackboard_config_file,
            'print_ascii_tree': orchestration_config.print_ascii_tree,
            'manual_mode': orchestration_config.manual_mode,
            'log_level': orchestration_config.log_level,
            'headless': str(orchestration_config.headless),
            # From bringup `robot_sn` / `tf_prefix`; empty when unused.
            'frame_prefix': orchestration_config.frame_prefix,
        }.items(),
    )
    return [multi_object_pick_and_place_process]
