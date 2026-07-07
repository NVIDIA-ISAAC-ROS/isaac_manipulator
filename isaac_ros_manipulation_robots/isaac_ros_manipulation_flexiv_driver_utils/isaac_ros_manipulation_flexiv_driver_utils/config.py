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

from typing import Dict, Optional

from isaac_ros_launch_utils.all_types import LaunchConfiguration
from isaac_ros_manipulation_ros_python_utils.config import (
    _get_optional_str, DriverConfig,
)
from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    get_str_variable, get_workflow_type
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    RobotType, WorkflowType,
)

from launch.launch_context import LaunchContext


class FlexivRizonDriverConfig(DriverConfig):
    """
    Config for Flexiv Rizon with Grav gripper workflows.

    Consolidates the launch args consumed by both the sim launch
    (``flexiv_rizon_sim_driver.launch.py``) and the real-robot launch
    (``flexiv_rizon_real_driver.launch.py``). Fields that only make sense
    for one of the two code paths (e.g. ``rdk_control_mode`` on the real
    path, ``workflow_type`` on the sim path) are resolved via
    :func:`_get_optional_str` so each launch file only has to declare the
    args it actually cares about.
    """

    # Always-on fields.
    rizon_type: str
    remapped_joint_states: Dict

    # Sim-only / workflow-level args. Optional: set to '' or None when not
    # declared in the launch file using this config.
    workflow_type: Optional[WorkflowType]
    log_level: str
    controller_spawner_timeout: LaunchConfiguration

    # Real-robot-only args. Strings (not bools) because they are forwarded
    # verbatim into xacro mappings and spawner conditions, both of which
    # expect ``'true'`` / ``'false'`` text.
    rdk_control_mode: str
    load_gripper: str
    gripper_name: str
    load_mounted_ft_sensor: str
    use_fake_hardware: str
    fake_sensor_commands: str
    start_rviz: str
    enable_rviz_visualization: str
    rviz_config_file: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        if self.robot_type is not RobotType.FLEXIV:
            raise ValueError(
                f'FlexivRizonDriverConfig requires robot_type={RobotType.FLEXIV}, '
                f'got {self.robot_type}')
        self.rizon_type = get_str_variable(context, 'rizon_type')

        # Workflow- and logging-level args are only meaningful when this
        # config is instantiated inside a higher-level launch (sim driver
        # or workflow). The real-robot driver launch doesn't declare them.
        workflow_type_str = _get_optional_str(context, 'workflow_type')
        self.workflow_type = (
            get_workflow_type(workflow_type_str) if workflow_type_str else None
        )
        self.log_level = _get_optional_str(context, 'log_level')
        self.controller_spawner_timeout = LaunchConfiguration(
            'controller_spawner_timeout')

        # Real-robot-only launch args (all default to ''). The real driver
        # launch declares every one of these; the sim launch declares none.
        self.rdk_control_mode = _get_optional_str(context, 'rdk_control_mode')
        self.load_gripper = _get_optional_str(context, 'load_gripper')
        self.gripper_name = _get_optional_str(context, 'gripper_name')
        self.load_mounted_ft_sensor = _get_optional_str(
            context, 'load_mounted_ft_sensor')
        self.use_fake_hardware = _get_optional_str(context, 'use_fake_hardware')
        self.fake_sensor_commands = _get_optional_str(
            context, 'fake_sensor_commands')
        self.start_rviz = _get_optional_str(context, 'start_rviz')
        self.enable_rviz_visualization = _get_optional_str(
            context, 'enable_rviz_visualization')
        self.rviz_config_file = _get_optional_str(context, 'rviz_config_file')

        if self.use_sim_time:
            self.remapped_joint_states = {
                '/joint_states': '/rizon_parsed_joint_states',
                '/controller_manager/robot_description': '/robot_description',
            }
        else:
            self.remapped_joint_states = {}
