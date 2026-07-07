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

from enum import Enum
import os
from typing import List

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_ros_python_utils.config import GearAssemblyConfig
from isaac_ros_manipulation_ros_python_utils.manipulator_types import GripperType, RobotType
from launch.actions import (
    IncludeLaunchDescription, Shutdown
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from sensor_msgs.msg import JointState
import yaml


class InsertionState(Enum):
    """
    States for the insertion process.

    Used across observation encoder, action server, and completion checker
    to maintain consistent state definitions.

    """

    IDLE = 0
    INSERTING = 1
    COMPLETED = 2
    FAILED = 3


def get_gear_assembly_nodes(gear_assembly_config: GearAssemblyConfig,
                            use_sim_time: bool) -> List[Node]:
    """
    Get the gear assembly nodes.

    Args
    ----
        gear_assembly_config (GearAssemblyConfig): The gear assembly configuration
        use_sim_time (bool): Whether to use sim time

    Returns
    -------
        List[Node]: The gear assembly nodes

    """
    # This node will look for action client request to move the robot to insert a gripped
    # gear into a gear stand.
    gear_assembly_action_server_node = Node(
        package='isaac_ros_manipulation_dnn_policy',
        executable='insertion_policy_action_server.py',
        name='gear_assembly_action_server',
        namespace='gear_assembly',
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('insertion_request_topic', gear_assembly_config.insertion_request_topic),
            # This is of type Int8 just a state message of type InsertionState (underlying Int8)
            ('insertion_status_topic', gear_assembly_config.insertion_status_topic),
        ],
    )

    gear_assembly_goal_pose_publisher_node = Node(
        package='isaac_ros_manipulation_dnn_policy',
        executable='goal_pose_publisher_node.py',
        name='gear_assembly_goal_pose_publisher',
        namespace='gear_assembly',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_publishing_on_trigger': True,
            'world_frame': gear_assembly_config.driver_config.base_frame,
            'frequency': gear_assembly_config.model_frequency
        }],
        remappings=[
            ('insertion_request_topic', gear_assembly_config.insertion_request_topic),
            ('goal_pose', gear_assembly_config.goal_pose_topic),
        ],
    )

    gear_assembly_completion_checker_node = Node(
        package='isaac_ros_manipulation_dnn_policy',
        executable='insertion_status_checker.py',
        name='gear_assembly_insertion_status_checker',
        namespace='gear_assembly',
        parameters=[{
            'use_sim_time': use_sim_time,
            'goal_frame': 'rl_insertion_pose_frame',
            'end_effector_frame': gear_assembly_config.driver_config.insertion_frame,
            'distance_threshold': 0.0130,
            'angle_threshold': 3.14,
            'timeout_seconds': gear_assembly_config.timeout_for_insertion_action_call
        }],
        remappings=[
            ('sub_insertion_status', gear_assembly_config.insertion_status_topic),
            ('pub_insertion_status', gear_assembly_config.insertion_status_topic),
        ],
    )

    # Check if rosbag already exists, error out if yes.
    if (
        gear_assembly_config.enable_recording and
        os.path.exists(gear_assembly_config.ros_bag_folder_path)
    ):
        raise FileExistsError(
            f'Rosbag already exists at {gear_assembly_config.ros_bag_folder_path}')

    gear_assembly_checkpoint_model_path = gear_assembly_config.model_path + '/' \
        + gear_assembly_config.model_file_name

    is_flexiv = (gear_assembly_config.driver_config.gripper_type == GripperType.GRAV)

    if is_flexiv:
        # Flexiv: use the dedicated inference pipeline with joint remapping,
        # streaming controller output, and enable_goal_pose_publisher=False
        # so the existing gear_assembly_goal_pose_publisher_node feeds /goal_pose.
        inference_launch_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(
                    get_package_share_directory('isaac_ros_manipulation_dnn_policy'),
                    'launch',
                    'inference_flexiv_gear_insertion.launch.py')]),
            launch_arguments={
                'checkpoint': gear_assembly_checkpoint_model_path,
                'ros_bag_folder_path': gear_assembly_config.ros_bag_folder_path,
                'record': 'True' if gear_assembly_config.enable_recording else 'False',
                'use_sim_time': 'True' if use_sim_time else 'False',
                'robot_sn': gear_assembly_config.driver_config.robot_sn,
                'input_joint_states_topic': gear_assembly_config.joint_state_topic,
                'enable_goal_pose_publisher': 'False',
                'input_goal_pose_topic': gear_assembly_config.goal_pose_topic,
                'alpha': str(gear_assembly_config.policy_alpha),
            }.items(),
        )
    else:
        # UR: use the original inference pipeline
        inference_launch_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(
                    get_package_share_directory('isaac_ros_manipulation_dnn_policy'),
                    'launch',
                    'inference.launch.py')]),
            launch_arguments={
                'checkpoint': gear_assembly_checkpoint_model_path,
                'ros_bag_folder_path': gear_assembly_config.ros_bag_folder_path,
                'record': 'True' if gear_assembly_config.enable_recording else 'False',
                'target_joint_positions': gear_assembly_config.target_joint_state_topic,
                'use_sim_time': 'True' if use_sim_time else 'False',
                'input_joint_states': gear_assembly_config.joint_state_topic,
                'input_goal_pose_topic': gear_assembly_config.goal_pose_topic,
            }.items(),
        )

    return [
        inference_launch_node,
        gear_assembly_action_server_node,
        gear_assembly_completion_checker_node,
        gear_assembly_goal_pose_publisher_node
    ]


def parse_joint_state_from_yaml(
    yaml_file_path: str,
    use_sim_time: bool,
    joint_names: List[str],
) -> JointState:
    """
    Parse initial joint positions from a YAML file and return a JointState message.

    Args
    ----
        yaml_file_path (str): Path to the YAML file containing initial_joint_pos
        use_sim_time (bool): Whether to use sim time
        joint_names (List[str]): Joint names for the target robot

    Returns
    -------
        JointState: A JointState message with positions from the YAML file

    Raises
    ------
        FileNotFoundError: If the YAML file doesn't exist
        KeyError: If 'initial_joint_pos' key is not found in the YAML file
        ValueError: If the joint positions list is empty or invalid

    """
    try:
        with open(yaml_file_path, 'r') as file:
            env_yaml = yaml.load(file, Loader=yaml.UnsafeLoader)

        if 'initial_joint_pos' in env_yaml:
            home_target_position = env_yaml['initial_joint_pos']
        else:
            raise ValueError(f'initial_joint_pos not found in {yaml_file_path}')

        if len(home_target_position) != len(joint_names):
            raise ValueError(
                f'initial_joint_pos in {yaml_file_path} has '
                f'{len(home_target_position)} values but robot has '
                f'{len(joint_names)} joints ({joint_names}).')

        added_size = 0
        if use_sim_time:
            added_size = 1

        joint_state = JointState()
        joint_state.position = home_target_position
        joint_state.velocity = [0.0] * (len(home_target_position) + added_size)
        joint_state.effort = [0.0] * (len(home_target_position) + added_size)
        joint_state.name = list(joint_names)

        return joint_state

    except FileNotFoundError:
        raise FileNotFoundError(f'YAML file not found: {yaml_file_path}')
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML file {yaml_file_path}: {e}')


def get_gear_assembly_orchestrator(gear_assembly_config: GearAssemblyConfig) -> List[Node]:
    """
    Get the gear assembly orchestrator node.

    Args
    ----
        gear_assembly_config (GearAssemblyConfig): The gear assembly configuration

    Returns
    -------
        List[Node]: The gear assembly orchestrator node in a list

    """
    gear_mesh_file_paths = [
        gear_assembly_config.gear_large_mesh_file_path,
        gear_assembly_config.gear_small_mesh_file_path,
        gear_assembly_config.gear_medium_mesh_file_path,
    ]

    if gear_assembly_config.use_sim_time:
        camera_tf_name = 'front_stereo_camera_left'
    else:
        camera_tf_name = 'camera_1_color_optical_frame'

    target_joint_state = parse_joint_state_from_yaml(
        gear_assembly_config.model_path + '/params/env.yaml',
        use_sim_time=gear_assembly_config.use_sim_time,
        joint_names=gear_assembly_config.driver_config.joint_names)

    if gear_assembly_config.driver_config.robot_type == RobotType.UR:
        peg_pose_base_frame = 'base'
    else:
        peg_pose_base_frame = gear_assembly_config.driver_config.base_frame

    return [
            Node(
                package='isaac_ros_manipulation_gear_assembly',
                executable='gear_assembly_orchestrator.py',
                name='gear_assembly_orchestrator',
                parameters=[{
                    'use_sim_time': gear_assembly_config.use_sim_time,
                    'wait_for_point_topic': True,
                    'point_topic_name_as_trigger': 'input_points_debug',
                    'max_timeout_time_for_action_call': 10.0,
                    'is_segment_anything_segmentation_enabled': True,
                    # In Sim based we are also using user supplied point.
                    'mesh_file_path_for_peg_stand_estimation':
                        gear_assembly_config.peg_stand_mesh_file_path,
                    'mesh_file_paths': gear_mesh_file_paths,
                    'camera_prim_name_in_tf': camera_tf_name,
                    'use_ground_truth_pose_estimation':
                        gear_assembly_config.gear_assembly_use_ground_truth_pose_in_sim,
                    'verify_pose_estimation_accuracy':
                        gear_assembly_config.verify_pose_estimation_accuracy,
                    'use_joint_space_planner':
                        gear_assembly_config.use_joint_space_planner,
                    'run_rl_inference':
                        gear_assembly_config.run_rl_inference,
                    'target_joint_state_for_place_pose': target_joint_state.position,
                    'output_dir': gear_assembly_config.output_dir,
                    'offset_for_place_pose': gear_assembly_config.offset_for_place_pose,
                    'offset_for_insertion_pose': gear_assembly_config.offset_for_insertion_pose,
                    'base_frame': gear_assembly_config.driver_config.base_frame,
                    'peg_pose_base_frame': peg_pose_base_frame,
                    'gripper_frame': gear_assembly_config.driver_config.gripper_frame,
                    'grasp_frame': gear_assembly_config.driver_config.grasp_frame,
                    'gripper_action_name':
                        gear_assembly_config.driver_config.gripper_action_name,
                    'gripper_open_position':
                        gear_assembly_config.driver_config.gripper_open_position,
                    'gripper_close_position':
                        gear_assembly_config.driver_config.gripper_close_position,
                    'gripper_settle_time_sec':
                        gear_assembly_config.driver_config.gripper_settle_time_sec,
                    'place_pose_z_rotation_deg':
                        gear_assembly_config.place_pose_z_rotation_deg,
                    'post_insertion_gripper_rotation_deg':
                        gear_assembly_config.post_insertion_gripper_rotation_deg,
                    'enable_controller_switching':
                        gear_assembly_config.enable_controller_switching,
                    'joint_names': gear_assembly_config.driver_config.joint_names,
                    'joint_limits_file_path':
                        gear_assembly_config.driver_config.joint_limits_file_path,
                }],
                output='screen',
                on_exit=Shutdown(),
            )
    ]
