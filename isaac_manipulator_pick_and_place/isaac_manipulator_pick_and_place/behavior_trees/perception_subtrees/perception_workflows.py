# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Perception workflow subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
the main perception workflow and drop pose updates.
Located in behavior_trees.perception_subtrees.perception_workflows.
"""

import operator

from isaac_manipulator_orchestration.behaviors.interactive_marker import (
    InteractiveMarker
)
from isaac_manipulator_orchestration.behaviors.perception_behaviors import (
    PublishStaticPlanningSceneBehavior
)
from isaac_manipulator_orchestration.behaviors.update_drop_pose_from_rviz import (
    UpdateDropPoseFromRViz
)
from isaac_manipulator_orchestration.behaviors.update_drop_pose_from_targets import (
    UpdateDropPoseFromTargets
)
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
from isaac_manipulator_pick_and_place.behavior_trees.perception_subtrees import (
    detection_operations,
    pose_estimation_operations,
)
from moveit_msgs.msg import PlanningScene
import py_trees
import py_trees_ros
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def check_use_drop_pose_from_rviz(blackboard) -> bool:
    """
    Check if drop pose should be obtained from RViz interactive marker.

    Condition function for EternalGuard in drop pose input selector.
    Returns True if use_drop_pose_from_rviz is True, False otherwise.

    Parameters
    ----------
    blackboard
        The blackboard client with access to use_drop_pose_from_rviz

    Returns
    -------
    bool
        True if drop pose should come from RViz, False otherwise

    Raises
    ------
    TypeError
        If use_drop_pose_from_rviz exists but is not a boolean value

    """
    if not blackboard.exists('use_drop_pose_from_rviz'):
        return False  # Default to False if parameter doesn't exist

    value = blackboard.use_drop_pose_from_rviz

    # Explicitly check for boolean type to catch configuration errors
    if not isinstance(value, bool):
        raise TypeError(
            f'use_drop_pose_from_rviz must be a boolean, got {type(value).__name__}: {value}. '
            "Check YAML config - ensure value is 'true' or 'false'."
        )

    return value


def create_update_drop_pose_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Parallel:
    """
    Create the update drop pose subtree for updating object drop poses.

    Tree structure:
    Update Drop Pose                   (Parallel | policy: SuccessOnAll, synchronise: False)
    ├─ FailureIsRunning                (Decorator)
    │   └─ RViz Drop Pose Guard        (Decorator | EternalGuard: use_drop_pose_from_rviz)
    │       └─ RViz Interactive Marker (Behaviour)
    └─ FailureIsRunning                (Decorator)
        └─ Detection Available Guard   (Decorator | EternalGuard: detection available)
            └─ Drop Pose Assignment    (Selector | memory: True)
                ├─ RViz Drop Pose Path (Sequence | memory: True)
                │   ├─ Use RViz Mode?  (Condition: use_drop_pose_from_rviz=true)
                │   ├─ RViz Pose Available?(Condition: rviz_drop_pose != None)
                │   └─ Update From RViz(Behaviour: UpdateDropPoseFromRViz)
                └─ Target Poses Path   (Behaviour: UpdateDropPoseFromTargets)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Parallel
        The update drop pose subtree

    """
    # Create update drop pose parallel node
    update_drop_pose = py_trees.composites.Parallel(
        name='Update Drop Pose',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    # Get interactive_marker config to access parameters
    interactive_marker_config = (
        behavior_config_initializer.get_interactive_marker_config())

    # Create RViz interactive marker behavior
    rviz_interactive_marker = InteractiveMarker(
        name='RViz Interactive Marker',
        mesh_resource_uri=interactive_marker_config.mesh_resource_uri,
        reference_frame=interactive_marker_config.reference_frame,
        end_effector_frame=interactive_marker_config.end_effector_frame,
        user_confirmation_timeout=interactive_marker_config.user_confirmation_timeout
    )

    # Create RViz drop pose guard with interactive marker as child
    rviz_drop_pose_guard = py_trees.decorators.EternalGuard(
        name='RViz Drop Pose Guard',
        condition=check_use_drop_pose_from_rviz,
        blackboard_keys={'use_drop_pose_from_rviz'},
        child=rviz_interactive_marker
    )

    # Wrap RViz drop pose guard with FailureIsRunning decorator
    rviz_drop_pose_with_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning',
        child=rviz_drop_pose_guard
    )

    # Create the drop pose assignment selector
    drop_pose_assignment = py_trees.composites.Selector(
        name='Drop Pose Assignment',
        memory=True
    )

    # Create RViz drop pose path (Sequence)
    rviz_drop_pose_path = py_trees.composites.Sequence(
        name='RViz Drop Pose Path',
        memory=True
    )

    # Condition: Check if we should use RViz mode
    use_rviz_condition = py_trees.behaviours.CheckBlackboardVariableValue(
        name='Use RViz Mode?',
        check=py_trees.common.ComparisonExpression(
            variable='use_drop_pose_from_rviz',
            value=True,
            operator=operator.eq
        )
    )

    # Condition: Check if RViz pose is available
    rviz_pose_available_condition = py_trees.behaviours.CheckBlackboardVariableValue(
        name='RViz Pose Available?',
        check=py_trees.common.ComparisonExpression(
            variable='rviz_drop_pose',
            value=None,
            operator=operator.ne
        )
    )

    # Create RViz drop pose updater behavior
    update_from_rviz = UpdateDropPoseFromRViz(name='Update From RViz')

    # Add children to RViz path
    rviz_drop_pose_path.add_children([
        use_rviz_condition,
        rviz_pose_available_condition,
        update_from_rviz
    ])

    # Create target poses updater behavior (fallback)
    update_from_targets = UpdateDropPoseFromTargets(name='Update From Targets')

    # Add children to drop pose assignment selector
    drop_pose_assignment.add_children([
        rviz_drop_pose_path,
        update_from_targets
    ])

    # Create detection available guard with drop pose assignment as child
    detection_guard = py_trees.decorators.EternalGuard(
        name='Detection Available Guard',
        condition=detection_operations.check_detection_available,
        blackboard_keys={'object_info_cache'},
        child=drop_pose_assignment
    )

    # Wrap detection guard with FailureIsRunning decorator
    detection_guard_with_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning',
        child=detection_guard
    )

    # Add children to update drop pose parallel node
    update_drop_pose.add_children([
        rviz_drop_pose_with_decorator,
        detection_guard_with_decorator
    ])

    return update_drop_pose


def create_perception_workflow(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Parallel:
    """
    Create the complete perception workflow.

    Create the complete perception workflow by combining detection, update drop pose,
    and pose estimation subtrees.

    Tree structure:
    Perception Workflow               (Parallel | policy: SuccessOnAll, synchronise: False)
    ├─ Detection                      (Selector | memory: True)
    │   └─ [See create_detection_subtree from detection_operations]
    ├─ Planning Scene Subscriber      (Behaviour | Subscriber)
    ├─ OneShot (Decorator)
    │   └─ Publish Static Planning Scene (Behaviour)
    ├─ FailureIsRunning               (Decorator)
    │   └─ Update Drop Pose           (Parallel | policy: SuccessOnAll, synchronise: False)
    │       └─ [See create_update_drop_pose_subtree]
    └─ FailureIsRunning               (Decorator)
        └─ Detection Available Guard  (Decorator | EternalGuard)
            └─ [See create_pose_estimation_subtree from pose_estimation_operations]

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Parallel
        The complete perception workflow

    """
    # Create perception workflow parallel node
    perception_workflow = py_trees.composites.Parallel(
        name='Perception Workflow',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    # Get detection subtree
    detection_subtree = detection_operations.create_detection_subtree(behavior_config_initializer)

    # Create planning scene subscriber
    planning_scene_subscriber = py_trees_ros.subscribers.ToBlackboard(
        name='Read Planning Scene',
        topic_name='/planning_scene',
        topic_type=PlanningScene,
        blackboard_variables={'planning_scene': None},
        clearing_policy=py_trees.common.ClearingPolicy.NEVER,
        qos_profile=QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
    )

    # Create publish static planning scene behavior with oneshot decorator
    publish_static_planning_scene_config = (
        behavior_config_initializer.get_publish_static_planning_scene_config()
    )
    publish_static_planning_scene_behavior = PublishStaticPlanningSceneBehavior(
        name='Publish Static Planning Scene',
        service_name=publish_static_planning_scene_config.service_name,
        scene_file_path=publish_static_planning_scene_config.scene_file_path
    )
    oneshot_publish_static_planning_scene = py_trees.decorators.OneShot(
        name='OneShot Publish Static Planning Scene',
        child=publish_static_planning_scene_behavior,
        policy=py_trees.common.OneShotPolicy.ON_SUCCESSFUL_COMPLETION
    )

    # Get update drop pose subtree
    update_drop_pose_subtree = create_update_drop_pose_subtree(behavior_config_initializer)

    # Get pose estimation subtree
    pose_estimation_subtree = pose_estimation_operations.create_pose_estimation_subtree(
        behavior_config_initializer)

    # Wrap update_drop_pose_subtree with FailureIsRunning decorator
    update_drop_pose_with_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning',
        child=update_drop_pose_subtree
    )

    # Wrap pose_estimation_subtree with FailureIsRunning decorator
    pose_estimation_with_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning',
        child=pose_estimation_subtree
    )

    # Add children to perception workflow
    perception_workflow.add_children([
        detection_subtree,
        planning_scene_subscriber,
        oneshot_publish_static_planning_scene,
        update_drop_pose_with_decorator,
        pose_estimation_with_decorator
    ])

    # Return the perception workflow without decorator
    return perception_workflow
