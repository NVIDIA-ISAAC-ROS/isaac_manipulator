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


class UpdateDropPoseFromRViz(py_trees.behaviour.Behaviour):
    """
    Update object_info_cache with drop poses from rviz_drop_pose blackboard variable.

    This behavior assigns drop poses from the RViz interactive marker to objects
    in the object cache. It takes the pose specified via RViz interactive
    marker and assigns it as the goal_drop_pose for objects that are eligible
    for motion planning (status not DONE or FAILED).

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='rviz_drop_pose', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        """Set up the behavior by getting node from kwargs."""
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e
        return True

    def update(self):
        # Check if we have a drop pose to assign
        if (not self.blackboard.exists('rviz_drop_pose') or
                self.blackboard.rviz_drop_pose is None):
            self.node.get_logger().debug(
                f'[{self.name}] Waiting for rviz_drop_pose to be available')
            return py_trees.common.Status.RUNNING

        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            self.node.get_logger().debug(
                f'[{self.name}] Waiting for object_info_cache to be available')
            return py_trees.common.Status.RUNNING

        # Extract the Pose from PoseStamped
        drop_pose = self.blackboard.rviz_drop_pose.pose

        # Update all objects that don't have a drop pose yet
        updated_objects = []
        for obj_id, obj_info in self.blackboard.object_info_cache.items():
            skip_statuses = [ObjectStatus.DONE.value,
                             ObjectStatus.FAILED.value]

            if obj_info.get('status') in skip_statuses:
                continue

            # Store as Pose object
            obj_info['goal_drop_pose'] = drop_pose
            updated_objects.append(obj_id)

        if updated_objects:
            self.node.get_logger().debug(
                f'[{self.name}] Updated drop poses for {len(updated_objects)} '
                f'objects: {updated_objects}')
        else:
            self.node.get_logger().debug(
                f'[{self.name}] No objects available for drop pose update')

        return py_trees.common.Status.SUCCESS
