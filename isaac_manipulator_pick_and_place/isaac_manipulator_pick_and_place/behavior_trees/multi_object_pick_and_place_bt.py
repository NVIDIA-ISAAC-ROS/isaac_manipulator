# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Main behavior tree creation for multi-object pick and place tasks.

This module provides the complete behavior tree structure for coordinating
perception and motion workflows in parallel for multi-object manipulation.
Located in behavior_trees.multi_object_pick_and_place_bt.
"""

from isaac_manipulator_orchestration.behaviors.report_generation import ReportGeneration
from isaac_manipulator_orchestration.behaviors.update_workflow_status import UpdateWorkflowStatus
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
from isaac_manipulator_pick_and_place.behavior_trees.motion_subtrees import (
    create_motion_workflow
)
from isaac_manipulator_pick_and_place.behavior_trees.perception_subtrees import (
    create_perception_workflow
)
import py_trees


def create_workflow_status_snapshot() -> py_trees.composites.Sequence:
    """
    Create the Workflow Status Snapshot sequence for status reporting.

    Tree structure:
    Workflow Status Snapshot          (Sequence | memory: True)
    ├─ ReportGeneration               (Behaviour)
    └─ UpdateWorkflowStatus           (Behaviour)

    Returns
    -------
    py_trees.composites.Sequence
        The Workflow Status Snapshot sequence

    """
    # Create Workflow Status Snapshot sequence
    workflow_status_snapshot = py_trees.composites.Sequence(
        name='Workflow Status Snapshot', memory=True)

    # Create ReportGeneration behavior
    report_generation = ReportGeneration(name='ReportGeneration')

    # Create UpdateWorkflowStatus behavior
    update_workflow_status = UpdateWorkflowStatus(name='UpdateWorkflowStatus')

    # Add children to workflow status snapshot sequence
    workflow_status_snapshot.add_children([
        report_generation,
        update_workflow_status
    ])

    return workflow_status_snapshot


def create_multi_object_pick_and_place_tree(
    behavior_config_initializer: BehaviorTreeConfigInitializer
) -> py_trees.composites.Sequence:
    """
    Create the complete multi-object pick and place behavior tree.

    Tree structure:
    Multi-Object Pick and Place Task  (Sequence | memory: True)
    └─ Workflow                       (Parallel | policy: SuccessOnAll, synchronise: False)
        ├─ FailureIsRunning (Perception) (Decorator)
        │   └─ Perception Workflow    (Parallel | policy: SuccessOnAll, synchronise: False)
        │       └─ [See create_perception_workflow from perception_subtrees]
        ├─ FailureIsRunning (Motion)  (Decorator)
        │   └─ Motion Sequence        (Sequence | memory: True)
        │       └─ [See create_motion_workflow from motion_subtrees]
        └─ Workflow Status Snapshot   (Sequence | memory: True)
            └─ [See create_workflow_status_snapshot]

    Args
    ----
    behavior_config_initializer : BehaviorTreeConfigInitializer
        Configuration initializer for loading behavior parameters.

    Returns
    -------
    py_trees.composites.Sequence
        The complete multi-object pick and place behavior tree

    """
    # Create root
    root = py_trees.composites.Sequence(
        'Multi-Object Pick and Place Task', memory=True)

    # Create the main workflow parallel node
    workflow = py_trees.composites.Parallel(
        name='Workflow',
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )

    # Get perception workflow
    perception_workflow = create_perception_workflow(behavior_config_initializer)

    # Apply FailureIsRunning decorator to perception workflow
    perception_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning (Perception)',
        child=perception_workflow
    )

    # Get motion workflow
    motion_workflow = create_motion_workflow(behavior_config_initializer)

    # Apply FailureIsRunning decorator to motion workflow
    motion_decorator = py_trees.decorators.FailureIsRunning(
        name='FailureIsRunning (Motion)',
        child=motion_workflow
    )

    # Get workflow status snapshot
    workflow_status_snapshot = create_workflow_status_snapshot()

    # Add children to workflow
    workflow.add_children([
        perception_decorator,
        motion_decorator,
        workflow_status_snapshot
    ])

    # Add workflow to root
    root.add_children([workflow])

    return root
