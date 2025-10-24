# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Pick operation subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
pick operations including grasping and lifting motions with controller checks.
Located in behavior_trees.motion_subtrees.pick_operations.
"""

from isaac_manipulator_orchestration.behaviors.motion_behaviors import (
    CloseGripper,
    ExecuteTrajectory,
    OpenGripper,
    PlanToGrasp,
    ReadGraspPoses,
    SwitchControllers
)
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
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

    # Execute Grasp (Action) - index 0 for grasp trajectory
    execute_trajectory_config = behavior_config_initializer.get_execute_trajectory_config()
    execute_grasp = ExecuteTrajectory(
        name='Execute Grasp',
        action_server_name=execute_trajectory_config.action_server_name,
        index=0
    )

    # Add children to execute grasp subtree
    execute_grasp_subtree.add_children([
        retry_switch_controllers_arm,
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

    # Execute Lift (Action) - index 1 for lift trajectory
    execute_trajectory_config = behavior_config_initializer.get_execute_trajectory_config()
    execute_lift = ExecuteTrajectory(
        name='Execute Lift',
        action_server_name=execute_trajectory_config.action_server_name,
        index=1
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
        grasp_approach_path_constraint=plan_grasp_config.grasp_approach_path_constraint,
        retract_offset_distance=plan_grasp_config.retract_offset_distance,
        retract_path_constraint=plan_grasp_config.retract_path_constraint,
        grasp_approach_constraint_in_goal_frame=(
            plan_grasp_config.grasp_approach_constraint_in_goal_frame
        ),
        retract_constraint_in_goal_frame=plan_grasp_config.retract_constraint_in_goal_frame,
        time_dilation_factor=plan_grasp_config.time_dilation_factor,
        disable_collision_links=plan_grasp_config.disable_collision_links,
        update_planning_scene=plan_grasp_config.update_planning_scene,
        world_frame=plan_grasp_config.world_frame,
        enable_aabb_clearing=plan_grasp_config.enable_aabb_clearing,
        esdf_clearing_padding=plan_grasp_config.esdf_clearing_padding
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
