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

from typing import List
import os

from ament_index_python.packages import get_package_share_directory

# flake8: noqa: I100
from isaac_manipulator_ros_python_utils.manipulator_types import (
    GripperType
)
from isaac_manipulator_ros_python_utils.config import (
    UrRobotiqDriverConfig
)

from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

import xacro


def get_robot_description_contents_for_real_robot(workflow_config: UrRobotiqDriverConfig) -> str:
    """
    Get the robot description contents for real robot.

    Args
    ----
        workflow_config (UrRobotiqDriverConfig): Workflow config

    Returns
    -------
        str: Robot description contents

    """
    script_filename = PathJoinSubstitution(
        [FindPackageShare('ur_client_library'), 'resources', 'external_control.urscript']
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare('ur_robot_driver'), 'resources', 'rtde_input_recipe.txt']
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare('ur_robot_driver'), 'resources', 'rtde_output_recipe.txt']
    )

    command = [
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        workflow_config.urdf_path,
        ' ',
        'robot_ip:=',
        workflow_config.robot_ip,
        ' ',
        'name:=',
        workflow_config.ur_type,
        ' ',
        'script_filename:=',
        script_filename,
        ' ',
        'input_recipe_filename:=',
        input_recipe_filename,
        ' ',
        'output_recipe_filename:=',
        output_recipe_filename,
        ' ',
        'ur_type:=',
        workflow_config.ur_type,
        ' ',
        'gripper_type:=',
        workflow_config.gripper_type,
        ' ',
        'grasp_parent_frame:=',
        workflow_config.grasp_parent_frame,
        ' '
    ]

    if (
        workflow_config.ur_calibration_file_path is not None and
        workflow_config.ur_calibration_file_path != ''
    ):
        command.append('kinematics_params_file:=')
        command.append(workflow_config.ur_calibration_file_path)
        command.append(' ')

    return Command(command)


def get_robot_description_contents_for_sim(
    urdf_xacro_file: str,
    ur_type: str,
    use_sim_time: bool,
    gripper_type: str,
    grasp_parent_frame: str,
    robot_ip: str,
    dump_to_file: bool = False,
    output_file: str = None,
) -> str:
    """
    Get robot description contents and optionally dump content to file.

    Args
    ----
        asset_name (str): The asset name for robot description
        ur_type (str): UR Type
        use_sim_time (bool): Use sim time for isaac sim platform
        dump_to_file (bool, optional): Dumps xml to file. Defaults to False.
        output_file (str, optional): Output file path if dumps output is True. Defaults to None.

    Returns
    -------
        str: XML contents of robot model

    """
    initial_positions_file = os.path.join(
        get_package_share_directory('isaac_manipulator_robot_description'),
        'config',
        'initial_positions.yaml'
    )

    mappings = {
        'ur_type': ur_type,
        'name': f'{ur_type}_robot',
        'sim_isaac': 'true' if use_sim_time else 'false',
        'use_fake_hardware': 'true' if use_sim_time else 'false',
        'generate_ros2_control_tag': 'false' if use_sim_time else 'true',
        'gripper_type': gripper_type,
        'grasp_parent_frame': grasp_parent_frame,
        'initial_positions_file': initial_positions_file
    }

    if not use_sim_time:
        script_filename = os.path.join(get_package_share_directory('ur_client_library'),
                                       'resources', 'external_control.urscript')
        input_recipe_filename = os.path.join(get_package_share_directory('ur_robot_driver'),
                                             'resources', 'rtde_input_recipe.txt')
        output_recipe_filename = os.path.join(get_package_share_directory('ur_robot_driver'),
                                              'resources', 'rtde_output_recipe.txt')
        mappings['robot_ip'] = robot_ip
        mappings['input_recipe_filename'] = input_recipe_filename
        mappings['output_recipe_filename'] = output_recipe_filename
        mappings['script_filename'] = script_filename

    # Process the .xacro file to convert it to a URDF string
    xacro_processed = xacro.process_file(
        urdf_xacro_file,
        mappings=mappings
    )
    robot_description = xacro_processed.toxml()

    if dump_to_file and output_file:
        with open(output_file, 'w') as file:
            file.write(robot_description)

    return robot_description

def get_gripper_collision_links(gripper_name: GripperType = GripperType.ROBOTIQ_2F_140
                                ) -> List[str]:
    """
    Get gripper collision linkes to disable during retract and approach object phase.

    Args
    ----
        gripper_name (GripperType, optional): _description_. Defaults to
            GripperType.ROBOTIQ_2F_140.

    Raises
    ------
        NotImplementedError: If gripper name is not supported

    Returns
    -------
        List[str]: List of collision links to ignore while planning the retract and approach phase

    """
    if gripper_name == GripperType.ROBOTIQ_2F_140:
        return [
            'left_outer_knuckle',
            'left_inner_knuckle',
            'left_outer_finger',
            'left_inner_finger',
            'left_inner_finger_pad',
            'right_outer_knuckle',
            'right_inner_knuckle',
            'right_outer_finger',
            'right_inner_finger',
            'right_inner_finger_pad',
        ]
    elif gripper_name == GripperType.ROBOTIQ_2F_85:
        return [
            'robotiq_85_base_link',
            'robotiq_85_left_finger_link',
            'robotiq_85_left_finger_tip_link',
            'robotiq_85_left_inner_knuckle_link',
            'robotiq_85_left_knuckle_link',
            'robotiq_85_right_finger_link',
            'robotiq_85_right_finger_tip_link',
            'robotiq_85_right_inner_knuckle_link',
            'robotiq_85_right_knuckle_link',
        ]
    else:
        raise NotImplementedError(f'Gripper type {gripper_name} not supported')
