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

import os

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_flexiv_driver_utils.config import (
    FlexivRizonDriverConfig,
)

import xacro


def get_robot_description_contents_for_sim(
    urdf_xacro_file: str,
    rizon_type: str,
    use_sim_time: bool,
    dump_to_file: bool = False,
    output_file: str = None,
) -> str:
    """Get robot description contents for Flexiv Rizon in Isaac Sim."""
    initial_positions_file = os.path.join(
        get_package_share_directory(
            'isaac_ros_manipulation_flexiv_robot_description'),
        'config',
        'initial_positions.yaml'
    )

    mappings = {
        'rizon_type': rizon_type,
        'sim_isaac': 'true' if use_sim_time else 'false',
        'initial_positions_file': initial_positions_file,
        'robot_sn': '',
        'arm_prefix': '',
    }

    xacro_processed = xacro.process_file(
        urdf_xacro_file,
        mappings=mappings
    )
    robot_description = xacro_processed.toxml()

    if dump_to_file and output_file:
        with open(output_file, 'w') as file:
            file.write(robot_description)

    return robot_description


def get_robot_description_contents_for_real(
    driver_config: FlexivRizonDriverConfig,
) -> str:
    """
    Build the real-robot URDF from the third-party ``flexiv_description`` xacro.

    Mirrors the xacro command the real-robot driver launch previously invoked
    inline (``rizon.urdf.xacro`` from ``flexiv_description`` with nine xacro
    mappings pulled from :class:`FlexivRizonDriverConfig`). Kept out of
    :class:`FlexivDriverUtils` so that both ``robot_state_publisher`` and
    ``ros2_control_node`` can share one eagerly resolved description string.

    Args
    ----
        driver_config (FlexivRizonDriverConfig): Driver config whose real-robot
            fields drive the xacro mappings.

    Returns
    -------
        str: URDF XML string ready to be set as ``robot_description``.

    """
    urdf_xacro_file = os.path.join(
        get_package_share_directory('flexiv_description'),
        'urdf', 'rizon.urdf.xacro',
    )
    mappings = {
        'robot_sn': driver_config.robot_sn,
        'rizon_type': driver_config.rizon_type,
        'rdk_control_mode': driver_config.rdk_control_mode,
        'load_gripper': driver_config.load_gripper,
        'gripper_name': driver_config.gripper_name,
        'load_mounted_ft_sensor': driver_config.load_mounted_ft_sensor,
        'use_fake_hardware': driver_config.use_fake_hardware,
        'fake_sensor_commands': driver_config.fake_sensor_commands,
    }
    return xacro.process_file(urdf_xacro_file, mappings=mappings).toxml()


def get_srdf_contents_for_real(
    driver_config: FlexivRizonDriverConfig,
) -> str:
    """
    Build the real-robot SRDF from the third-party ``flexiv_moveit_config`` xacro.

    Counterpart to :func:`get_robot_description_contents_for_real` for the
    semantic description. Uses the same ``robot_sn`` / gripper / FT-sensor
    mappings the real launch's inline xacro command consumed.

    Args
    ----
        driver_config (FlexivRizonDriverConfig): Driver config whose real-robot
            fields drive the xacro mappings.

    Returns
    -------
        str: SRDF XML string ready to be set as
        ``robot_description_semantic``.

    """
    srdf_xacro_file = os.path.join(
        get_package_share_directory('flexiv_moveit_config'),
        'srdf', 'rizon.srdf.xacro',
    )
    mappings = {
        'robot_sn': driver_config.robot_sn,
        'load_gripper': driver_config.load_gripper,
        'load_mounted_ft_sensor': driver_config.load_mounted_ft_sensor,
    }
    return xacro.process_file(srdf_xacro_file, mappings=mappings).toxml()
