# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Motion workflow subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
complete motion workflows including pick and drop operations with fallback handling.
Located in behavior_trees.motion_subtrees.motion_workflows.
"""

from collections import deque
import operator

from isaac_manipulator_orchestration.behaviors.motion_behaviors import (
    AttachObject,
    DetachObject,
    ReadDropPose,
)
from isaac_manipulator_orchestration.behaviors.update_object_status import (
    MarkObjectAsActive,
    MarkObjectAsDone,
    MarkObjectAsFailed
)
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
from isaac_manipulator_pick_and_place.behavior_trees.motion_subtrees.drop_operations import (
    create_drop_subtree
)
from isaac_manipulator_pick_and_place.behavior_trees.motion_subtrees.pick_operations import (
    create_pick_subtree
)
import py_trees


def check_success_condition(blackboard) -> bool:
    """
    Check if motion was successful (not aborted).

    Condition function for EternalGuard in motion subtree.
    Returns True if abort_motion is False, False otherwise.
    Motion is considered successful when abort_motion is False.

    Parameters
    ----------
    blackboard
        The blackboard client with access to abort_motion

    Returns
    -------
    bool
        True if abort_motion is False, False otherwise

    """
    if not blackboard.exists('abort_motion'):
        return True  # Default to success if variable doesn't exist
    return blackboard.abort_motion is False


def create_motion_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the motion subtree by combining pick and drop subtrees.

    Tree structure:
    Motion                            (Sequence | memory: True)
    ├─ Read Drop Pose                 (Behaviour | Guards motion workflow -
    │                                  waits for initial drop pose)
    ├─ Pick                           (Sequence | memory: True)
    │   └─ [See create_pick_subtree from pick_operations]
    ├─ Retry Attach Object            (Decorator)
    │   └─ Attach Object              (Behaviour)
    ├─ Drop                           (Sequence | memory: True)
    │   └─ [See create_drop_subtree from drop_operations]
    ├─ Detach Object                  (Behaviour)
    └─ Success?                       (Decorator | EternalGuard: SUCCESS if abort_motion==False;
                                         blocks child otherwise)
        └─ Mark Object as Done        (Behaviour)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The motion subtree

    """
    # Create motion sequence
    motion = py_trees.composites.Sequence(name='Motion', memory=True)

    # Read Drop Pose (Behaviour) - ensures drop pose is available before motion
    read_drop_pose = ReadDropPose(name='Read Drop Pose')

    # Get pick subtree
    pick_subtree = create_pick_subtree(behavior_config_initializer)

    # Create attach object behavior
    attach_config = behavior_config_initializer.get_attach_object_config()
    attach_object = AttachObject(
        name='Attach Object',
        action_server_name=attach_config.action_server_name,
        fallback_radius=attach_config.fallback_radius,
        shape=attach_config.shape,
        scale=attach_config.scale,
        gripper_frame=attach_config.gripper_frame,
        grasp_frame=attach_config.grasp_frame
    )

    # Retry Attach Object (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_attach_object = py_trees.decorators.Retry(
        name='Retry Attach Object',
        child=attach_object,
        num_failures=retry_config.max_attachment_retries
    )

    # Get drop subtree
    drop_subtree = create_drop_subtree(behavior_config_initializer)

    # Mark Object as Done
    mark_object_as_done = MarkObjectAsDone(name='Mark Object as Done')

    # Create detach object behavior
    detach_config = behavior_config_initializer.get_detach_object_config()
    detach_object = DetachObject(
        name='Detach Object',
        action_server_name=detach_config.action_server_name
    )

    # Success? guard condition using EternalGuard
    success_guard = py_trees.decorators.EternalGuard(
        name='Success?',
        condition=check_success_condition,
        blackboard_keys={'abort_motion'},
        child=mark_object_as_done
    )

    # Add children to motion sequence
    motion.add_children([
        read_drop_pose,
        pick_subtree,
        retry_attach_object,
        drop_subtree,
        detach_object,
        success_guard
    ])

    return motion


def create_motion_with_fallback_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Selector:
    """
    Create the motion subtree with a fallback to handle failures.

    Tree structure:
    Motion with Fallback              (Selector | memory: True)
    ├─ Motion                         (Sequence | memory: True)
    │   └─ [See create_motion_subtree]
    └─ Force-Fail-and-Mark-Failed     (Decorator | SuccessIsFailure)
        └─ Cleanup and Mark Failed    (Sequence | memory: True)
            ├─ Clear Abort Motion Flag (Behaviour)
            └─ Mark Object as Failed  (Behaviour)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Selector
        The motion subtree with fallback handling

    """
    # Motion with Fallback (Selector)
    motion_with_fallback = py_trees.composites.Selector(
        name='Motion with Fallback', memory=True)

    # Get the motion subtree
    motion_subtree = create_motion_subtree(behavior_config_initializer)

    # Create Cleanup and Mark Failed sequence
    cleanup_and_mark_failed = py_trees.composites.Sequence(
        name='Cleanup and Mark Failed', memory=True)

    # Clear Abort Motion Flag (Behaviour)
    clear_abort_motion_flag = py_trees.behaviours.SetBlackboardVariable(
        name='Clear Abort Motion Flag',
        variable_name='abort_motion',
        variable_value=False,
        overwrite=True
    )

    # Mark Object as Failed (Behaviour)
    mark_object_as_failed = MarkObjectAsFailed(name='Mark Object as Failed')

    # Add children to cleanup and mark failed sequence
    cleanup_and_mark_failed.add_children([
        clear_abort_motion_flag,
        mark_object_as_failed
    ])

    # Force-Fail-and-Mark-Failed (Decorator - SuccessIsFailure)
    force_fail_and_mark_failed = py_trees.decorators.SuccessIsFailure(
        name='Force-Fail-and-Mark-Failed',
        child=cleanup_and_mark_failed
    )

    # Motion with Fallback children
    motion_with_fallback.add_children([
        motion_subtree,
        force_fail_and_mark_failed
    ])

    return motion_with_fallback


def create_motion_workflow(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the complete motion workflow.

    Tree structure:
    Motion Sequence                   (Sequence | memory: True)
    ├─ Wait for Next ID               (Behaviour | Condition)
    ├─ Mark Active Object             (Behaviour)
    └─ Motion with Fallback           (Selector | memory: True)
        └─ [See create_motion_with_fallback_subtree]

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The complete motion workflow

    """
    # Create motion sequence
    motion_sequence = py_trees.composites.Sequence(
        name='Motion Sequence', memory=True)

    # Create wait for next id check
    wait_for_next_id = py_trees.behaviours.CheckBlackboardVariableValue(
        name='Wait for Next ID',
        check=py_trees.common.ComparisonExpression(
            variable='next_object_id',
            value=deque(),
            operator=operator.ne
        )
    )

    # Get motion with fallback subtree
    motion_with_fallback_subtree = create_motion_with_fallback_subtree(behavior_config_initializer)

    # Add children to motion sequence
    motion_sequence.add_children([
        wait_for_next_id,
        MarkObjectAsActive(name='Mark Active Object'),
        motion_with_fallback_subtree
    ])

    # Return the motion sequence
    return motion_sequence
