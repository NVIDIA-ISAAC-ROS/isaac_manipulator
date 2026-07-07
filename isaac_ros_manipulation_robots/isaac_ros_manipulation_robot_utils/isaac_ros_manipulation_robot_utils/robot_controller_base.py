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

"""
Abstract base class describing the launch-time contract a custom robot must satisfy.

Each robot family that plugs into Isaac Manipulator (UR, Flexiv, and any future
vendor) ships a ``*_driver_utils`` package whose top-level driver launch files
build up three things:

1. a ``robot_state_publisher`` node that broadcasts TF and ``robot_description``,
2. a MoveIt ``move_group`` node (plus the matching ``MoveItConfigsBuilder``),
3. the ``ros2_control`` node(s) and controller spawners that drive the robot.

``RobotControllerBase`` formalises that contract as three abstract methods. To
bring your own robot, subclass :class:`RobotControllerBase`, implement the three
``@abstractmethod`` hooks below, and expose module-level trampolines that the
driver launch files can import (see ``URDriverUtils`` / ``FlexivDriverUtils`` as
reference implementations).
"""

import abc
from typing import Any, List, Tuple

from isaac_ros_manipulation_ros_python_utils.config import DriverConfig
from launch_ros.actions import Node


class RobotControllerBase(abc.ABC):
    """
    Launch-time contract every robot driver implementation must satisfy.

    Subclasses receive a fully-populated :class:`DriverConfig` (or a vendor
    subclass of it) and are expected to translate that config into the ROS 2
    nodes required to bring the robot up. All abstract methods take no extra
    arguments on purpose: everything a subclass needs (URDF/SRDF paths, MoveIt
    config files, sim vs real flag, frame prefixes, joint names, gripper
    parameters, etc.) is reachable via ``self.driver_config``.
    """

    driver_config: DriverConfig

    def __init__(self, driver_config: DriverConfig):
        self.driver_config = driver_config

    @abc.abstractmethod
    def get_robot_state_publisher(self) -> Node:
        """
        Build the ``robot_state_publisher`` node for this robot.

        The returned node must publish the TF tree and the ``robot_description``
        parameter derived from ``self.driver_config.urdf_path`` (and any
        vendor-specific xacro args). In simulation it typically remaps
        ``/joint_states`` to the sim-parsed topic exposed by the driver_config.

        Returns
        -------
            Node: Configured ``robot_state_publisher`` node.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_moveit_group_node(self) -> Tuple[Node, Any]:
        """
        Build the MoveIt ``move_group`` node and its matching config bundle.

        Implementations typically construct a ``MoveItConfigsBuilder`` from the
        SRDF / kinematics / joint-limits / moveit-controllers paths on
        ``self.driver_config``, register the cuMotion planning pipeline, and
        return both the ``move_group`` node and the builder so downstream nodes
        (e.g. RViz) can reuse the same robot description.

        Returns
        -------
            Tuple[Node, Any]: The ``move_group`` node and the MoveIt config
            bundle (a ``MoveItConfigsBuilder`` or its dict form) that produced
            it.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_robot_control_nodes(self) -> List[Node]:
        """
        Build the ``ros2_control`` node(s) and controller spawners.

        Typical implementations return a ``ros2_control_node`` parametrised
        with ``self.driver_config.ros2_controllers_file_path`` plus one spawner
        per controller (joint trajectory controller, joint state broadcaster,
        gripper controller, ...). Whether the real vs simulated control stack
        is selected is up to the implementation; the sim flag is available as
        ``self.driver_config.use_sim_time``.

        Returns
        -------
            List[Node]: Nodes to launch for ``ros2_control`` and its spawners.

        """
        raise NotImplementedError
