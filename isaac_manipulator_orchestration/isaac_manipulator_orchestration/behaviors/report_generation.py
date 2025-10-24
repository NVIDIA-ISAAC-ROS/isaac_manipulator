# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


class ReportGeneration(py_trees.behaviour.Behaviour):
    """
    Generate a human-readable report of the pick and place operation.

    This behavior reads the object_info_cache and generates a comprehensive
    status report showing the state for all objects in the workflow,
    including whether each object was picked and placed successfully.
    The report is stored in the workflow_summary blackboard variable.

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
            key='workflow_summary', access=py_trees.common.Access.WRITE)

    def update(self):
        """Generate status report from object cache."""
        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            self.blackboard.workflow_summary = 'No objects detected'
            return py_trees.common.Status.SUCCESS

        object_cache = self.blackboard.object_info_cache

        if not object_cache:
            self.blackboard.workflow_summary = 'No objects in cache'
            return py_trees.common.Status.SUCCESS

        # Generate object summaries
        object_summaries = []
        for obj_id, obj_info in object_cache.items():
            class_id = obj_info.get('class_id', 'N/A')
            status = obj_info.get('status', ObjectStatus.NOT_READY.value)

            # Extract drop pose coordinates
            drop_pose = obj_info.get('goal_drop_pose')
            if drop_pose is not None:
                try:
                    x_coord = f'{drop_pose.position.x:.1f}'
                    y_coord = f'{drop_pose.position.y:.1f}'
                    pose_str = f'({x_coord},{y_coord})'
                except AttributeError:
                    pose_str = 'N/A'
            else:
                pose_str = 'N/A'

            # Create compact summary for each object
            obj_summary = f'Object {obj_id} (class: {class_id}) -> {status} at {pose_str}'
            object_summaries.append(obj_summary)

        # Create compact, CLI-friendly report
        if object_summaries:
            objects_part = '; '.join(object_summaries)
            report = f'Object Status Report: {objects_part}. Total objects: {len(object_cache)}'
        else:
            report = 'Object Status Report: No objects processed. Total objects: 0'
        self.blackboard.workflow_summary = report
        return py_trees.common.Status.SUCCESS
