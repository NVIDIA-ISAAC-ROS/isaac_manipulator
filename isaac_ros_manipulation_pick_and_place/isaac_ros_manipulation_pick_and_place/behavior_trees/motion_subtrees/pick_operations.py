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
Pick operation subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
pick operations including grasping and lifting motions with controller checks.
Located in behavior_trees.motion_subtrees.pick_operations.
"""

from isaac_ros_manipulation_orchestration.behaviors.motion_behaviors import (
    CloseGripper,
    ExecuteTrajectory,
    OpenGripper,
    PlanToGrasp,
    ReadGraspPoses,
    SwitchControllers
)
from isaac_ros_manipulation_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
import py_trees


def create_execute_grasp_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the execute grasp subtree with retry switch controller.

    Tree structure:
    Execute Grasp Subtree              (Sequence | memory: True)
    ├─ Retry Activate Arm Controller   (Decorator)
    │   └─ Activate Arm Controller     (Behaviour)
    ├─ Execute Approach                (Behaviour | Action)
    └─ Execute Grasp                   (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The execute grasp subtree

    """
    # Execute Grasp Subtree (Sequence)
    execute_grasp_subtree = py_trees.composites.Sequence(
        name='Execute Grasp Subtree', memory=True)

    # Switch Controllers behavior
    switch_controllers_arm_config = behavior_config_initializer.get_arm_controllers_config()
    switch_controllers_arm = SwitchControllers(
        name='Activate Arm Controller',
        controllers_to_activate=switch_controllers_arm_config.controllers_to_activate,
        controllers_to_deactivate=switch_controllers_arm_config.controllers_to_deactivate,
        strictness=switch_controllers_arm_config.strictness
    )

    # Retry Switch Controllers (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_switch_controllers_arm = py_trees.decorators.Retry(
        name='Retry Activate Arm Controller',
        child=switch_controllers_arm,
        num_failures=retry_config.max_controller_retries
    )

    # Execute Approach (Action) - index 0 for approach trajectory
    execute_trajectory_config = behavior_config_initializer.get_execute_trajectory_config()
    execute_approach = ExecuteTrajectory(
        name='Execute Approach',
        action_server_name=execute_trajectory_config.action_server_name,
        index=0
    )

    # Execute Grasp (Action) - index 1 for grasp trajectory
    execute_grasp = ExecuteTrajectory(
        name='Execute Grasp',
        action_server_name=execute_trajectory_config.action_server_name,
        index=1
    )

    # Add children to execute grasp subtree
    execute_grasp_subtree.add_children([
        retry_switch_controllers_arm,
        execute_approach,
        execute_grasp
    ])

    return execute_grasp_subtree


def create_execute_lift_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the execute lift subtree with retry switch controller.

    Tree structure:
    Execute Lift Subtree               (Sequence | memory: True)
    ├─ Retry Activate Arm Controller   (Decorator)
    │   └─ Activate Arm Controller     (Behaviour)
    └─ Execute Lift                    (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The execute lift subtree

    """
    # Execute Lift Subtree (Sequence)
    execute_lift_subtree = py_trees.composites.Sequence(
        name='Execute Lift Subtree', memory=True)

    # Switch Controllers behavior
    switch_controllers_arm_config = behavior_config_initializer.get_arm_controllers_config()
    switch_controllers_arm = SwitchControllers(
        name='Activate Arm Controller',
        controllers_to_activate=switch_controllers_arm_config.controllers_to_activate,
        controllers_to_deactivate=switch_controllers_arm_config.controllers_to_deactivate,
        strictness=switch_controllers_arm_config.strictness
    )

    # Retry Switch Controllers (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_switch_controllers_arm = py_trees.decorators.Retry(
        name='Retry Activate Arm Controller',
        child=switch_controllers_arm,
        num_failures=retry_config.max_controller_retries
    )

    # Execute Lift (Action) - index 2 for lift trajectory
    execute_trajectory_config = behavior_config_initializer.get_execute_trajectory_config()
    execute_lift = ExecuteTrajectory(
        name='Execute Lift',
        action_server_name=execute_trajectory_config.action_server_name,
        index=2
    )

    # Add children to execute lift subtree
    execute_lift_subtree.add_children([
        retry_switch_controllers_arm,
        execute_lift
    ])

    return execute_lift_subtree


def create_pick_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the pick subtree for grasping and lifting an object.

    Tree structure:
    Pick                              (Sequence | memory: True)
    ├─ Open Gripper Subtree           (Sequence | memory: True)
    │   ├─ Retry Activate Tool Controller (Decorator)
    │   │   └─ Activate Tool Controller (Behaviour | SwitchControllers)
    │   └─ Open Gripper               (Behaviour | Action)
    ├─ Read Grasp Poses               (Behaviour)
    ├─ Retry Plan To Grasp            (Decorator)
    │   └─ Plan To Grasp              (Behaviour | Action)
    ├─ Execute Grasp Subtree          (Sequence | memory: True)
    │   └─ [See create_execute_grasp_subtree]
    ├─ Close Gripper Subtree          (Sequence | memory: True)
    │   ├─ Retry Activate Tool Controller (Decorator)
    │   │   └─ Activate Tool Controller (Behaviour | SwitchControllers)
    │   └─ Close Gripper              (Behaviour | Action)
    └─ Execute Lift Subtree           (Sequence | memory: True)
        └─ [See create_execute_lift_subtree]

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The pick subtree

    """
    # Create pick sequence
    pick = py_trees.composites.Sequence(name='Pick', memory=True)

    # Open Gripper Subtree
    open_gripper_subtree = py_trees.composites.Sequence(
        name='Open Gripper Subtree', memory=True)

    # Switch Controllers behavior for open gripper
    switch_controllers_tool_config = behavior_config_initializer.get_tool_controllers_config()
    switch_controllers_tool_open = SwitchControllers(
        name='Activate Tool Controller',
        controllers_to_activate=switch_controllers_tool_config.controllers_to_activate,
        controllers_to_deactivate=switch_controllers_tool_config.controllers_to_deactivate,
        strictness=switch_controllers_tool_config.strictness
    )

    # Retry Switch Controllers for open gripper
    retry_config = behavior_config_initializer.get_retry_config()
    retry_switch_controllers_tool_open = py_trees.decorators.Retry(
        name='Retry Activate Tool Controller',
        child=switch_controllers_tool_open,
        num_failures=retry_config.max_controller_retries
    )

    # Open Gripper (Action)
    open_gripper_config = behavior_config_initializer.get_open_gripper_config()
    open_gripper = OpenGripper(
        name='Open Gripper',
        gripper_action_name=open_gripper_config.gripper_action_name,
        open_position=open_gripper_config.open_position,
        max_effort=open_gripper_config.max_effort
    )

    # Add children to open gripper subtree
    open_gripper_subtree.add_children([
        retry_switch_controllers_tool_open,
        open_gripper
    ])

    # Read Grasp Poses (Behaviour)
    read_grasp_config = behavior_config_initializer.get_read_grasp_poses_config()

    read_grasp_poses = ReadGraspPoses(
        name='Read Grasp Poses',
        publish_grasp_poses=read_grasp_config.publish_grasp_poses
    )

    # Plan To Grasp (Behaviour)
    plan_grasp_config = behavior_config_initializer.get_plan_to_grasp_config()
    plan_to_grasp = PlanToGrasp(
        name='Plan To Grasp',
        action_server_name=plan_grasp_config.action_server_name,
        link_name=plan_grasp_config.link_name,
        grasp_approach_offset_distance=plan_grasp_config.grasp_approach_offset_distance,
        grasp_translation_path_deviation_limit=(
            plan_grasp_config.grasp_translation_path_deviation_limit
        ),
        grasp_translation_terminal_deviation_limit=(
            plan_grasp_config.grasp_translation_terminal_deviation_limit
        ),
        grasp_enable_orientation_path_axis_constraint=(
            plan_grasp_config.grasp_enable_orientation_path_axis_constraint
        ),
        grasp_orientation_terminal_deviation_limit=(
            plan_grasp_config.grasp_orientation_terminal_deviation_limit
        ),
        grasp_orientation_path_axis_deviation_limit=(
            plan_grasp_config.grasp_orientation_path_axis_deviation_limit
        ),
        retract_offset_distance=plan_grasp_config.retract_offset_distance,
        retract_translation_path_deviation_limit=(
            plan_grasp_config.retract_translation_path_deviation_limit
        ),
        retract_translation_terminal_deviation_limit=(
            plan_grasp_config.retract_translation_terminal_deviation_limit
        ),
        retract_enable_orientation_path_axis_constraint=(
            plan_grasp_config.retract_enable_orientation_path_axis_constraint
        ),
        retract_orientation_terminal_deviation_limit=(
            plan_grasp_config.retract_orientation_terminal_deviation_limit
        ),
        retract_orientation_path_axis_deviation_limit=(
            plan_grasp_config.retract_orientation_path_axis_deviation_limit
        ),
        grasp_approach_constraint_in_goal_frame=(
            plan_grasp_config.grasp_approach_constraint_in_goal_frame
        ),
        retract_constraint_in_goal_frame=plan_grasp_config.retract_constraint_in_goal_frame,
        time_dilation_factor=plan_grasp_config.time_dilation_factor,
        update_planning_scene=plan_grasp_config.update_planning_scene,
        world_frame=plan_grasp_config.world_frame,
        enable_aabb_clearing=plan_grasp_config.enable_aabb_clearing,
        esdf_clearing_padding=plan_grasp_config.esdf_clearing_padding,
        aabb_clearing_shape=plan_grasp_config.aabb_clearing_shape,
        aabb_clearing_shape_scale=plan_grasp_config.aabb_clearing_shape_scale
    )

    # Retry Plan To Grasp (Retry Decorator)
    retry_plan_to_grasp = py_trees.decorators.Retry(
        name='Retry Plan To Grasp',
        child=plan_to_grasp,
        num_failures=retry_config.max_planning_retries
    )

    # Get execute grasp subtree
    execute_grasp_subtree = create_execute_grasp_subtree(
        behavior_config_initializer)

    # Close Gripper Subtree
    close_gripper_subtree = py_trees.composites.Sequence(
        name='Close Gripper Subtree', memory=True)

    # Switch Controllers behavior for close gripper
    switch_controllers_tool_close = SwitchControllers(
        name='Activate Tool Controller',
        controllers_to_activate=switch_controllers_tool_config.controllers_to_activate,
        controllers_to_deactivate=switch_controllers_tool_config.controllers_to_deactivate,
        strictness=switch_controllers_tool_config.strictness
    )

    # Retry Switch Controllers for close gripper
    retry_switch_controllers_tool_close = py_trees.decorators.Retry(
        name='Retry Activate Tool Controller',
        child=switch_controllers_tool_close,
        num_failures=retry_config.max_controller_retries
    )

    # Close Gripper (Action)
    close_gripper_config = behavior_config_initializer.get_close_gripper_config()
    close_gripper = CloseGripper(
        name='Close Gripper',
        gripper_action_name=close_gripper_config.gripper_action_name,
        close_position=close_gripper_config.close_position,
        max_effort=close_gripper_config.max_effort
    )

    # Add children to close gripper subtree
    close_gripper_subtree.add_children([
        retry_switch_controllers_tool_close,
        close_gripper
    ])

    # Get execute lift subtree
    execute_lift_subtree = create_execute_lift_subtree(
        behavior_config_initializer)

    # Add children to pick sequence
    pick.add_children([
        open_gripper_subtree,
        read_grasp_poses,
        retry_plan_to_grasp,
        execute_grasp_subtree,
        close_gripper_subtree,
        execute_lift_subtree
    ])

    return pick
