# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Pose estimation operation subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
pose estimation operations including object pose calculation and queue management.
Located in behavior_trees.perception_subtrees.pose_estimation_operations.
"""

from isaac_manipulator_orchestration.behaviors.perception_behaviors import (
    AssignObjectName,
    MeshAssigner,
    ObjectSelector,
    PoseEstimation
)
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
import py_trees

# Import check_detection_available from detection_operations module
from .detection_operations import check_detection_available


def check_object_queue_empty(blackboard) -> bool:
    """
    Check if the object queue has space.

    Condition function for EternalGuard in pose estimation subtree.
    Returns True if the object queue has space; otherwise, returns False.

    Parameters
    ----------
    blackboard
        The blackboard client with access to next_object_id and max_num_next_object

    Returns
    -------
    bool
        True if len(next_object_id) < max_num_next_object, False otherwise

    """
    if (not blackboard.exists('next_object_id') or
            not blackboard.exists('max_num_next_object')):
        return False
    return len(blackboard.next_object_id) < blackboard.max_num_next_object


def create_pose_estimation_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.decorators.EternalGuard:
    """
    Create the pose estimation subtree for object pose calculation.

    Tree structure:
    Detection Available Guard         (Decorator | EternalGuard: detection available)
    └─ Object Queue Empty Guard       (Decorator | EternalGuard: object queue empty)
        └─ Pose Estimation            (Sequence | memory: True)
            ├─ Object Selector        (Behaviour)
            ├─ Mesh Assigner          (Behaviour)
            ├─ Update Frame Name      (Behaviour)
            └─ Retry Pose Estimation  (Decorator)
                └─ Pose Estimation    (Behaviour | Action)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.decorators.EternalGuard
        The pose estimation subtree with guards

    """
    # Create pose estimation sequence (formerly estimation_sequence)
    pose_estimation = py_trees.composites.Sequence(
        name='Pose Estimation', memory=True)

    # Pose Estimation (Action)
    pose_estimation_config = behavior_config_initializer.get_pose_estimation_config()
    workspace_bounds_config = behavior_config_initializer.get_workspace_bounds_config()
    pose_estimation_action = PoseEstimation(
        name='Pose Estimation',
        action_server_name=pose_estimation_config.action_server_name,
        base_frame_id=pose_estimation_config.base_frame_id,
        camera_frame_id=pose_estimation_config.camera_frame_id,
        workspace_bounds_config=workspace_bounds_config
    )

    # Retry Pose Estimation (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_pose_estimation = py_trees.decorators.Retry(
        name='Retry Pose Estimation',
        child=pose_estimation_action,
        num_failures=retry_config.max_pose_estimation_retries
    )

    # Add children to pose estimation sequence
    object_selector_config = behavior_config_initializer.get_object_selector_config()
    mesh_assigner_config = behavior_config_initializer.get_mesh_assigner_config()
    assign_object_name_config = behavior_config_initializer.get_assign_object_name_config()

    pose_estimation.add_children([
        ObjectSelector(
            name='Object Selector',
            action_server_name=object_selector_config.action_server_name
        ),
        MeshAssigner(
            name='Mesh Assigner',
            service_name=mesh_assigner_config.service_name
        ),
        AssignObjectName(
            name='Assign Object Name',
            service_name=assign_object_name_config.service_name
        ),
        retry_pose_estimation
    ])

    # Create second eternal guard for object queue empty condition
    object_queue_guard = py_trees.decorators.EternalGuard(
        name='Object Queue Empty Guard',
        condition=check_object_queue_empty,
        blackboard_keys={'next_object_id', 'max_num_next_object'},
        child=pose_estimation
    )

    # Create first eternal guard for detection available condition
    detection_guard = py_trees.decorators.EternalGuard(
        name='Detection Available Guard',
        condition=check_detection_available,
        blackboard_keys={'object_info_cache'},
        child=object_queue_guard
    )

    return detection_guard
