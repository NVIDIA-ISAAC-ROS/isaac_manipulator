# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Detection operation subtree creation utilities.

This module provides functionality for creating behavior tree subtrees that handle
detection operations including object detection and stale detection handling.
Located in behavior_trees.perception_subtrees.detection_operations.
"""

import collections
import operator

from isaac_manipulator_interfaces.action import MultiObjectPickAndPlace
from isaac_manipulator_orchestration.behaviors.perception_behaviors import (
    DetectObject,
    FilterDetections,
)
from isaac_manipulator_orchestration.behaviors.report_generation import ReportGeneration
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
import py_trees


def check_detection_available(blackboard) -> bool:
    """
    Check if object detection data is available for processing.

    Condition function for EternalGuard in update drop pose subtree.
    Returns True if detection is available (object_info_cache is not None), False otherwise.
    Detection must be available before we can update drop poses for objects.

    Parameters
    ----------
    blackboard
        The blackboard client with access to object_info_cache

    Returns
    -------
    bool
        True if detection data is available, False otherwise

    """
    return blackboard.object_info_cache is not None


def create_stale_detection_handler(
    timeout_duration: float = 300.0
) -> py_trees.composites.Selector:
    """
    Create a subtree to handle stale detections by either timing out or clearing data.

    Tree structure:
    Stale Detection Handler           (Selector | memory: True)
    ├─ Stale Detection Timeout        (Sequence | memory: True)
    │   ├─ Detection Available?       (Behaviour | Condition)
    │   └─ SuccessIsFailure           (Decorator)
    │       └─ Timer                  (Behaviour | Timeout in seconds)
    └─ Success-is-Failure             (Decorator)
        └─ Wait-and-Clear             (Sequence | memory: True)
            ├─ FailureIsRunning       (Decorator)
            │   └─ Wait for Current Motion Check (Behaviour | Condition)
            ├─ Workflow Status Snapshot (Sequence | memory: True)
            │   ├─ ReportGeneration   (Behaviour)
            │   └─ Set Incomplete Status (Behaviour)
            ├─ Clear Object Info Cache (Behaviour)
            ├─ Clear Selected Object ID (Behaviour)
            └─ Clear Next Object ID   (Behaviour)

    Timer Behavior Transformation:
    Timer alone:        RUNNING (while counting) → SUCCESS (when expires)
    + SuccessIsFailure: RUNNING (while counting) → FAILURE (when expires)

    This creates the desired behavior: detection stays "fresh" (RUNNING) until timeout,
    then becomes "stale" (FAILURE) and triggers the clearing sequence.

    Parameters
    ----------
    timeout_duration : float, optional
        Duration in seconds before detection is considered stale (default: 300.0)

    Returns
    -------
    py_trees.composites.Selector
        The stale detection handler subtree

    """
    # Create stale detection handler selector
    stale_detection_handler = py_trees.composites.Selector(
        name='Stale Detection Handler', memory=True)

    # Return success if object_info_cache exists and is not None, otherwise return failure
    detection_available = py_trees.behaviours.CheckBlackboardVariableValue(
        name='Detection Available?',
        check=py_trees.common.ComparisonExpression(
            variable='object_info_cache',
            value=None,
            operator=operator.ne
        )
    )

    # Create timer for duration in seconds
    timer = py_trees.timers.Timer(
        name='Timer',
        duration=timeout_duration
    )

    # Wrap timer with SuccessIsFailure to convert SUCCESS to FAILURE when timer expires
    success_is_failure_timer = py_trees.decorators.SuccessIsFailure(
        name='SuccessIsFailure',
        child=timer
    )

    # Create sequence for stale detection timeout logic
    stale_detection_timeout = py_trees.composites.Sequence(
        name='Stale Detection Timeout',
        memory=True
    )

    # Add detection check and timer logic to sequence
    stale_detection_timeout.add_children([
        detection_available,
        success_is_failure_timer
    ])

    # Create wait-and-clear sequence
    wait_and_clear = py_trees.composites.Sequence(
        name='Wait-and-Clear', memory=True)

    # Create success-is-failure decorator
    success_is_failure = py_trees.decorators.SuccessIsFailure(
        name='Success-is-Failure',
        child=wait_and_clear
    )

    # Check if motion is complete (active_obj_id == None)
    wait_for_current_motion_check = py_trees.behaviours.CheckBlackboardVariableValue(
        name='Wait for Current Motion Check',
        check=py_trees.common.ComparisonExpression(
            variable='active_obj_id',
            value=None,
            operator=operator.eq
        )
    )

    # Use FailureIsRunning to convert FAILURE to RUNNING when motion is active
    wait_for_current_motion = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning',
        child=wait_for_current_motion_check
    )

    # Clear object_info_cache blackboard variable
    clear_object_info_cache = py_trees.behaviours.SetBlackboardVariable(
        name='Clear Object Info Cache',
        variable_name='object_info_cache',
        variable_value=None,
        overwrite=True
    )

    # Clear selected_object_id blackboard variable
    clear_selected_object_id = py_trees.behaviours.SetBlackboardVariable(
        name='Clear Selected Object ID',
        variable_name='selected_object_id',
        variable_value=None,
        overwrite=True
    )

    # Clear next_object_id blackboard variable
    # Note: lambda is used to create a fresh deque on each execution.
    # Without lambda, the same deque instance would be reused, retaining old values.
    clear_next_object_id = py_trees.behaviours.SetBlackboardVariable(
        name='Clear Next Object ID',
        variable_name='next_object_id',
        variable_value=lambda: collections.deque(),
        overwrite=True
    )

    # Create stale Workflow Status Snapshot sequence to capture incomplete status before clearing
    workflow_status_snapshot = py_trees.composites.Sequence(
        name='Workflow Status Snapshot', memory=True)

    # Add ReportGeneration and SetBlackboardVariable to workflow status snapshot
    report_generation = ReportGeneration(name='ReportGeneration')
    set_incomplete_status = py_trees.behaviours.SetBlackboardVariable(
        name='Set Incomplete Status',
        variable_name='workflow_status',
        variable_value=MultiObjectPickAndPlace.Result.INCOMPLETE,
        overwrite=True
    )

    workflow_status_snapshot.add_children([
        report_generation,
        set_incomplete_status
    ])

    # Add children to wait-and-clear sequence
    wait_and_clear.add_children([
        wait_for_current_motion,
        workflow_status_snapshot,  # Capture status before clearing
        clear_object_info_cache,
        clear_selected_object_id,
        clear_next_object_id
    ])

    # Add children to stale detection handler
    stale_detection_handler.add_children([
        stale_detection_timeout,
        success_is_failure
    ])

    return stale_detection_handler


def create_detection_subtree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Selector:
    """
    Create the detection subtree for object detection.

    Tree structure:
    Detection                         (Selector | memory: True)
    ├─ Stale Detection Handler        (Selector | memory: True)
    │   └─ [See create_stale_detection_handler]
    └─ Detect and Filter              (Sequence | memory: True)
        ├─ Retry Detect Object        (Decorator)
        │   └─ Detect Object          (Behaviour | Action)
        └─ Filter Detections          (Behaviour)

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Selector
        The detection subtree

    """
    # Create detection selector
    detection = py_trees.composites.Selector(name='Detection', memory=True)

    # Get stale detection handler with configured timeout
    stale_detection_config = behavior_config_initializer.get_stale_detection_config()
    stale_detection_handler = create_stale_detection_handler(
        timeout_duration=stale_detection_config.timeout_duration
    )

    # Create sequence for detection and filtering
    detect_and_filter = py_trees.composites.Sequence(
        name='Detect and Filter', memory=True)

    # Detect Object (Action)
    detect_object_config = behavior_config_initializer.get_detect_object_config()
    detect_object = DetectObject(
        name='Detect Object',
        action_server_name=detect_object_config.action_server_name,
        detection_confidence_threshold=detect_object_config.detection_confidence_threshold
    )

    # Retry Detect Object (Retry Decorator)
    retry_config = behavior_config_initializer.get_retry_config()
    retry_detect_object = py_trees.decorators.Retry(
        name='Retry Detect Object',
        child=detect_object,
        num_failures=retry_config.max_detection_retries
    )

    # Filter Detections (Behaviour)
    filter_detections = FilterDetections(name='Filter Detections')

    # Add children to detect and filter sequence
    detect_and_filter.add_children([
        retry_detect_object,
        filter_detections
    ])

    # Add children to detection selector
    detection.add_children([
        stale_detection_handler,
        detect_and_filter
    ])

    return detection
