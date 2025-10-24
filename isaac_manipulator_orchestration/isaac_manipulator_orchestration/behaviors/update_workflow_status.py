# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_interfaces.action import MultiObjectPickAndPlace
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


class UpdateWorkflowStatus(py_trees.behaviour.Behaviour):
    """
    Analyze object statuses and update the overall workflow action result.

    This behavior examines all objects in the object_info_cache and determines
    the overall workflow status based on individual object outcomes:

    - All DONE → MultiObjectPickAndPlace.Result.SUCCESS
    - All FAILED → MultiObjectPickAndPlace.Result.FAILED
    - Mix of DONE/FAILED → MultiObjectPickAndPlace.Result.PARTIAL_SUCCESS
    - Any objects still processing → MultiObjectPickAndPlace.Result.UNKNOWN

    The workflow status is stored in the workflow_status blackboard variable.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='workflow_status', access=py_trees.common.Access.WRITE)

    def update(self):
        """Analyze object statuses and update workflow result."""
        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            # No objects detected - this could be considered a failure or unknown
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.UNKNOWN
            return py_trees.common.Status.SUCCESS

        object_cache = self.blackboard.object_info_cache

        if not object_cache:
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.UNKNOWN
            return py_trees.common.Status.SUCCESS

        # Count final statuses
        done_count = 0
        failed_count = 0
        total_objects = len(object_cache)

        for obj_info in object_cache.values():
            status = obj_info.get('status', ObjectStatus.NOT_READY.value)
            if status == ObjectStatus.DONE.value:
                done_count += 1
            elif status == ObjectStatus.FAILED.value:
                failed_count += 1
            # Any other status means object is still being processed

        # Determine overall workflow status
        if done_count == total_objects:
            # All objects completed successfully
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.SUCCESS
        elif failed_count == total_objects:
            # All objects failed
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.FAILED
        elif done_count + failed_count == total_objects:
            # Mix of success and failure, all objects finished
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.PARTIAL_SUCCESS
        else:
            # Some objects still in progress
            self.blackboard.workflow_status = MultiObjectPickAndPlace.Result.UNKNOWN

        return py_trees.common.Status.SUCCESS
