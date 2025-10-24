# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import py_trees


class ReadDropPose(py_trees.behaviour.Behaviour):
    """
    Read the drop pose from object info cache for the current motion object.

    This behavior retrieves the drop pose for the currently active object
    from the object info cache and places it on the blackboard as the
    goal_drop_pose. This pose can be used by motion planning behaviors to
    determine where to place the object.

    The behavior waits for the drop pose to stabilize (is_drop_pose_updating = False)
    and for the goal_drop_pose to be available in the object cache before proceeding.
    If the drop pose is not yet available, the behavior returns RUNNING to wait
    until it becomes available.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='goal_drop_pose', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='is_drop_pose_updating', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='use_drop_pose_from_rviz', access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        """Set up the behavior by getting node from kwargs."""
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e
        return True

    def update(self):
        """Read drop pose from object info cache for current motion object."""
        # First check if drop pose is currently being updated (only relevant for RViz mode)
        if (self.blackboard.exists('use_drop_pose_from_rviz') and
                self.blackboard.use_drop_pose_from_rviz and
                self.blackboard.exists('is_drop_pose_updating') and
                self.blackboard.is_drop_pose_updating):
            self.node.get_logger().info(f'{self.name}: Waiting for drop pose to stabilize')
            return py_trees.common.Status.RUNNING

        # Check if we have the required blackboard variables
        if (not self.blackboard.exists('active_obj_id') or
                self.blackboard.active_obj_id is None):
            self.node.get_logger().error(
                f'[{self.name}] active_obj_id not available on blackboard')
            return py_trees.common.Status.FAILURE

        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            self.node.get_logger().error(
                f'[{self.name}] object_info_cache not available on blackboard')
            return py_trees.common.Status.FAILURE

        # Get the current motion object ID
        obj_id = self.blackboard.active_obj_id

        # Query the object info cache for this object
        if (self.blackboard.object_info_cache is None or
                obj_id not in self.blackboard.object_info_cache):
            self.node.get_logger().error(
                f'[{self.name}] Object {obj_id} not found in object_info_cache')
            return py_trees.common.Status.FAILURE

        object_info = self.blackboard.object_info_cache[obj_id]

        # Extract the goal_drop_pose from the object info
        if 'goal_drop_pose' not in object_info:
            self.node.get_logger().debug(
                f'{self.name}: Waiting for goal_drop_pose to be available for object {obj_id}')
            return py_trees.common.Status.RUNNING

        # Check if goal_drop_pose is None (not yet assigned)
        if object_info['goal_drop_pose'] is None:
            self.node.get_logger().debug(
                f'{self.name}: Waiting for goal_drop_pose to be assigned for object {obj_id}')
            return py_trees.common.Status.RUNNING

        # Set goal_drop_pose on blackboard (guaranteed to be a Pose object)
        self.blackboard.goal_drop_pose = object_info['goal_drop_pose']
        self.node.get_logger().info(
            f'[{self.name}] Successfully set drop pose for object {obj_id}')

        return py_trees.common.Status.SUCCESS
