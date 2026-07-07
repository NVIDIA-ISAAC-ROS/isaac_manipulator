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

from typing import Dict

from isaac_ros_launch_utils.all_types import LaunchConfiguration
from isaac_ros_manipulation_ros_python_utils.config import DriverConfig
from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    get_str_variable, get_workflow_type
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    GripperType, RobotType, WorkflowType
)
from launch.launch_context import LaunchContext


class UrRobotiqDriverConfig(DriverConfig):
    """Config that tracks all variables needed to perform UR and Robotiq workflows."""

    controller_spawner_timeout: LaunchConfiguration
    ur_type: str
    robot_ip: str
    grasp_parent_frame: str
    log_level: str
    remapped_joint_states: Dict
    workflow_type: WorkflowType
    ur_calibration_file_path: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        if self.robot_type is not RobotType.UR:
            raise ValueError(
                f'UrRobotiqDriverConfig requires robot_type={RobotType.UR}, '
                f'got {self.robot_type}')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.ur_type = get_str_variable(context, 'ur_type')
        self.robot_ip = get_str_variable(context, 'robot_ip')
        if self.use_sim_time and self.gripper_type is GripperType.ROBOTIQ_2F_85:
            raise ValueError(
                f'Gripper type {self.gripper_type} not supported for Isaac sim')

        self.log_level = get_str_variable(context, 'log_level')
        self.ur_calibration_file_path = get_str_variable(context, 'ur_calibration_file_path')
        self.controller_spawner_timeout = LaunchConfiguration('controller_spawner_timeout')

        if self.gripper_type is GripperType.ROBOTIQ_2F_140:
            self.grasp_parent_frame = 'robotiq_base_link'
        elif self.gripper_type is GripperType.ROBOTIQ_2F_85:
            self.grasp_parent_frame = 'robotiq_85_base_link'
        else:
            raise ValueError(
                f'Gripper type {self.gripper_type} not supported for UR robots')

        if self.use_sim_time:
            self.remapped_joint_states = {
                '/joint_states': '/isaac_parsed_joint_states',
                '/controller_manager/robot_description': '/robot_description',
            }
        else:
            self.remapped_joint_states = {}
