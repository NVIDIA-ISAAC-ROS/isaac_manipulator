# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import collections

from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


def _update_object_status_and_feedback(blackboard, status: ObjectStatus, message: str):
    """
    Update object status and provide feedback.

    Parameters
    ----------
    blackboard : py_trees.blackboard.Client
        The blackboard client
    status : ObjectStatus
        The status to set for the object
    message : str
        Message template for feedback

    """
    obj_id = blackboard.active_obj_id
    if (obj_id is not None and
            blackboard.exists('object_info_cache') and
            blackboard.object_info_cache is not None and
            obj_id in blackboard.object_info_cache):
        # Get object information for feedback
        obj_info = blackboard.object_info_cache[obj_id]
        class_id = obj_info.get('class_id', 'Unknown')

        # Update object status
        blackboard.object_info_cache[obj_id]['status'] = status.value

        # Update action feedback queue
        feedback_msg = message.format(obj_id=obj_id, class_id=class_id)
        if not blackboard.exists('workflow_feedback_queue'):
            blackboard.workflow_feedback_queue = collections.deque()
        blackboard.workflow_feedback_queue.append(feedback_msg)

    blackboard.active_obj_id = None


class MarkObjectAsActive(py_trees.behaviour.Behaviour):
    """
    Transition an object from the motion planning queue to active motion status.

    This behavior pops an object ID from the next_object_id queue and sets it as
    the active_obj_id. It also updates the object's status to IN_MOTION in the
    object_info_cache to indicate that motion planning and execution is underway.

    The behavior coordinates multi-object workflows by ensuring only one object
    is actively being manipulated at a time while others wait in the queue.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='next_object_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.WRITE)

    def update(self):
        if not self.blackboard.exists('next_object_id') or not self.blackboard.next_object_id:
            return py_trees.common.Status.FAILURE

        obj_id = self.blackboard.next_object_id.popleft()
        self.blackboard.active_obj_id = obj_id

        # Update status
        if (self.blackboard.exists('object_info_cache') and
                self.blackboard.object_info_cache is not None and
                obj_id in self.blackboard.object_info_cache):
            self.blackboard.object_info_cache[obj_id]['status'] = \
                ObjectStatus.IN_MOTION.value

        return py_trees.common.Status.SUCCESS


class MarkObjectAsDone(py_trees.behaviour.Behaviour):
    """
    Mark the currently active object as successfully completed.

    This behavior updates the object status to DONE in the object_info_cache,
    clears the active_obj_id, and adds a success message to the workflow_feedback_queue
    for real-time user feedback.

    The behavior indicates successful completion of all manipulation tasks for
    the object (pick, transport, and place operations) and enables the workflow
    to proceed to the next object in multi-object scenarios.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='workflow_feedback_queue', access=py_trees.common.Access.WRITE)

    def update(self):
        message = 'Object {obj_id} (class: {class_id}) completed successfully'
        _update_object_status_and_feedback(self.blackboard, ObjectStatus.DONE, message)
        return py_trees.common.Status.SUCCESS


class MarkObjectAsFailed(py_trees.behaviour.Behaviour):
    """
    Mark the currently active object as failed due to manipulation errors.

    This behavior updates the object status to FAILED in the object_info_cache,
    clears the active_obj_id, and adds a failure message to the workflow_feedback_queue
    for real-time user feedback.

    The behavior handles cases where motion planning, trajectory execution, or
    gripper operations fail, enabling error recovery by preventing repeated
    attempts on problematic objects and allowing workflows to continue with
    remaining objects.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='workflow_feedback_queue', access=py_trees.common.Access.WRITE)

    def update(self):
        message = 'Object {obj_id} (class: {class_id}) failed to complete'
        _update_object_status_and_feedback(self.blackboard, ObjectStatus.FAILED, message)
        return py_trees.common.Status.SUCCESS
