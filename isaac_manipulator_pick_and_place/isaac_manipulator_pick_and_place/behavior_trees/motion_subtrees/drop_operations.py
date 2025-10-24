# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Drop operation subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
drop operations including planning to drop pose and executing drop motion.
Located in behavior_trees.motion_subtrees.drop_operations.
"""

from isaac_manipulator_orchestration.behaviors.motion_behaviors import (
    ExecuteTrajectory,
    OpenGripper,
    PlanToPose,
    ReadDropPose,
    SwitchControllers,
    UpdateDropPoseToHome,
)
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
import py_trees


def create_plan_to_drop_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Selector:
    """
    Create the plan to drop subtree with fallback to home pose.

    Tree structure:
    Plan To Drop Subtree              (Selector | memory: True)
    ├─ Retry Plan To Drop Pose        (Decorator)
    │   └─ Plan To Drop Pose          (Behaviour | Action)
    └─ Plan Recovery Drop Pose        (Sequence | memory: True)
        ├─ Abort Drop                 (Behaviour | sets abort_motion=True, returns SUCCESS)
        ├─ Update Drop Pose To Home   (Behaviour | sets goal_pose=home_pose, returns SUCCESS)
        └─ Plan To Home Pose          (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Selector
        The plan to drop subtree

    """
    # Plan To Drop Subtree
    plan_to_drop_subtree = py_trees.composites.Selector(
        name='Plan To Drop Subtree', memory=True)

    # Plan To Drop Pose (using PlanToPose)
    plan_pose_config = behavior_config_initializer.get_plan_to_pose_config()
    plan_to_drop_pose = PlanToPose(
        name='Plan To Drop Pose',
        action_server_name=plan_pose_config.action_server_name,
        link_name=plan_pose_config.link_name,
        time_dilation_factor=plan_pose_config.time_dilation_factor,
        update_planning_scene=plan_pose_config.update_planning_scene,
        disable_collision_links=plan_pose_config.disable_collision_links,
        aabb_clearing_shape=plan_pose_config.aabb_clearing_shape,
        aabb_clearing_shape_scale=plan_pose_config.aabb_clearing_shape_scale,
        enable_aabb_clearing=plan_pose_config.enable_aabb_clearing,
        esdf_clearing_padding=plan_pose_config.esdf_clearing_padding
    )

    # Retry Plan To Drop Pose (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_plan_to_drop_pose = py_trees.decorators.Retry(
        name='Retry Plan To Drop Pose',
        child=plan_to_drop_pose,
        num_failures=retry_config.max_planning_retries
    )

    # Plan Recovery Drop Pose
    plan_recovery_drop_pose = py_trees.composites.Sequence(
        name='Plan Recovery Drop Pose', memory=True)

    # Abort Drop using SetBlackboardVariable
    abort_drop = py_trees.behaviours.SetBlackboardVariable(
        name='Abort Drop',
        variable_name='abort_motion',
        variable_value=True,
        overwrite=True
    )

    # Update Drop Pose To Home
    update_drop_pose_to_home = UpdateDropPoseToHome(
        name='Update Drop Pose To Home')

    # Plan To Home Pose (using PlanToPose)
    plan_to_home_pose = PlanToPose(
        name='Plan To Home Pose',
        action_server_name=plan_pose_config.action_server_name,
        link_name=plan_pose_config.link_name,
        time_dilation_factor=plan_pose_config.time_dilation_factor,
        update_planning_scene=plan_pose_config.update_planning_scene,
        disable_collision_links=plan_pose_config.disable_collision_links,
        aabb_clearing_shape=plan_pose_config.aabb_clearing_shape,
        aabb_clearing_shape_scale=plan_pose_config.aabb_clearing_shape_scale,
        enable_aabb_clearing=plan_pose_config.enable_aabb_clearing,
        esdf_clearing_padding=plan_pose_config.esdf_clearing_padding
    )

    # Add children to plan recovery drop pose
    plan_recovery_drop_pose.add_children([
        abort_drop,
        update_drop_pose_to_home,
        plan_to_home_pose
    ])

    # Add children to plan to drop subtree
    plan_to_drop_subtree.add_children([
        retry_plan_to_drop_pose,
        plan_recovery_drop_pose
    ])

    return plan_to_drop_subtree


def create_execute_drop_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the execute drop subtree with retry switch controller.

    Tree structure:
    Execute Drop Subtree               (Sequence | memory: True)
    ├─ Retry Activate Arm Controller   (Decorator)
    │   └─ Activate Arm Controller     (Behaviour)
    └─ Execute Drop                    (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The execute drop subtree

    """
    # Execute Drop Subtree (Sequence)
    execute_drop_subtree = py_trees.composites.Sequence(
        name='Execute Drop Subtree', memory=True)

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

    # Execute Drop (Action) - index 0 for drop trajectory
    execute_trajectory_config = behavior_config_initializer.get_execute_trajectory_config()
    execute_drop = ExecuteTrajectory(
        name='Execute Drop',
        action_server_name=execute_trajectory_config.action_server_name,
        index=0
    )

    # Add children to execute drop subtree
    execute_drop_subtree.add_children([
        retry_switch_controllers_arm,
        execute_drop
    ])

    return execute_drop_subtree


def create_drop_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the complete drop subtree workflow.

    Tree structure:
    Drop                              (Sequence | memory: True)
    ├─ Refresh Drop Pose              (Behaviour| ReadDropPose - gets latest drop pose)
    ├─ Plan To Drop Subtree           (Selector | memory: True)
    │   └─ [See create_plan_to_drop_subtree]
    ├─ Execute Drop Subtree           (Sequence | memory: True)
    │   └─ [See create_execute_drop_subtree]
    └─ Open Gripper Subtree           (Sequence | memory: True)
        ├─ Retry Activate Tool Controller (Decorator)
        │   └─ Activate Tool Controller (Behaviour | SwitchControllers)
        └─ Open Gripper               (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The complete drop subtree

    """
    # Create drop sequence
    drop = py_trees.composites.Sequence(name='Drop', memory=True)

    # Refresh Drop Pose - gets the latest drop pose before planning
    # (in case user modified it during pick phase)
    refresh_drop_pose = ReadDropPose(name='Refresh Drop Pose')

    # Get plan to drop subtree
    plan_to_drop_subtree = create_plan_to_drop_subtree(
        behavior_config_initializer)

    # Get execute drop subtree
    execute_drop_subtree = create_execute_drop_subtree(
        behavior_config_initializer)

    # Open Gripper Subtree
    open_gripper_subtree = py_trees.composites.Sequence(
        name='Open Gripper Subtree', memory=True)

    # Switch Controllers behavior for open gripper
    switch_controllers_tool_config = behavior_config_initializer.get_tool_controllers_config()
    switch_controllers_tool = SwitchControllers(
        name='Activate Tool Controller',
        controllers_to_activate=switch_controllers_tool_config.controllers_to_activate,
        controllers_to_deactivate=switch_controllers_tool_config.controllers_to_deactivate,
        strictness=switch_controllers_tool_config.strictness
    )

    # Retry Switch Controllers for open gripper
    retry_config = behavior_config_initializer.get_retry_config()
    retry_switch_controllers_tool = py_trees.decorators.Retry(
        name='Retry Activate Tool Controller',
        child=switch_controllers_tool,
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
        retry_switch_controllers_tool,
        open_gripper
    ])

    # Add children to drop sequence
    drop.add_children([
        refresh_drop_pose,
        plan_to_drop_subtree,
        execute_drop_subtree,
        open_gripper_subtree
    ])

    return drop
