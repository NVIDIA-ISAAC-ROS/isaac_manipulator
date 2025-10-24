# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from geometry_msgs.msg import Pose
from isaac_manipulator_interfaces.action import MultiObjectPickAndPlace
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


class UpdateDropPoseFromTargets(py_trees.behaviour.Behaviour):
    """
    Update object_info_cache with drop poses from target_poses based on mode and class_ids.

    This behavior assigns drop poses based on the mode configuration:
    - SINGLE_BIN: Uses the single target pose for all objects
    - MULTI_BIN: Maps target poses to objects based on class_ids

    The behavior only updates objects whose status is not DONE or FAILED.
    Target poses must be geometry_msgs.msg.Pose objects.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='target_poses', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='class_ids', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='mode', access=py_trees.common.Access.READ)
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
        # Check if we have the required blackboard variables
        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            return py_trees.common.Status.RUNNING

        if (not self.blackboard.exists('target_poses') or
                self.blackboard.target_poses is None):
            self.node.get_logger().error(f'[{self.name}] target_poses not available on blackboard')
            return py_trees.common.Status.FAILURE

        if (not self.blackboard.exists('mode') or
                self.blackboard.mode is None):
            self.node.get_logger().error(f'[{self.name}] mode not available on blackboard')
            return py_trees.common.Status.FAILURE

        mode = self.blackboard.mode
        target_poses = self.blackboard.target_poses
        class_ids = self.blackboard.class_ids if self.blackboard.exists('class_ids') else None

        # Validate that target_poses is a list and contains valid Pose objects
        if not isinstance(target_poses, list):
            self.node.get_logger().error(
                f'[{self.name}] target_poses must be a list, got {type(target_poses).__name__}')
            return py_trees.common.Status.FAILURE

        if len(target_poses) == 0:
            self.node.get_logger().debug(
                f'[{self.name}] Waiting for target_poses from action call')
            return py_trees.common.Status.RUNNING

        # Validate that all poses are geometry_msgs.msg.Pose objects
        for i, pose in enumerate(target_poses):
            if not isinstance(pose, Pose):
                self.node.get_logger().error(
                    f'[{self.name}] target_poses[{i}] must be geometry_msgs.msg.Pose, '
                    f'got {type(pose).__name__}')
                return py_trees.common.Status.FAILURE

        if mode == MultiObjectPickAndPlace.Goal.SINGLE_BIN:
            # Single bin mode
            return self._handle_single_bin_mode(target_poses, class_ids)
        elif mode == MultiObjectPickAndPlace.Goal.MULTI_BIN:
            # Multi bin mode
            return self._handle_multi_bin_mode(target_poses, class_ids)
        else:
            self.node.get_logger().error(
                f'[{self.name}] Invalid mode: {mode}. '
                f'Expected {MultiObjectPickAndPlace.Goal.SINGLE_BIN} (SINGLE_BIN) '
                f'or {MultiObjectPickAndPlace.Goal.MULTI_BIN} (MULTI_BIN)')
            return py_trees.common.Status.FAILURE

    def _handle_single_bin_mode(self, target_poses, class_ids):
        """Handle SINGLE_BIN mode."""
        # Validate single bin mode requirements
        if class_ids is not None and len(class_ids) > 0:
            self.node.get_logger().error(
                f'[{self.name}] SINGLE_BIN mode ({MultiObjectPickAndPlace.Goal.SINGLE_BIN}) '
                f'requires class_ids to be empty')
            return py_trees.common.Status.FAILURE

        if len(target_poses) != 1:
            self.node.get_logger().error(
                f'[{self.name}] SINGLE_BIN mode ({MultiObjectPickAndPlace.Goal.SINGLE_BIN}) '
                f'requires exactly one target pose, got {len(target_poses)}')
            return py_trees.common.Status.FAILURE

        # Use the single target pose for all objects
        drop_pose = target_poses[0]
        return self._update_objects_with_pose(drop_pose)

    def _handle_multi_bin_mode(self, target_poses, class_ids):
        """Handle MULTI_BIN mode."""
        # Validate multi bin mode requirements
        if class_ids is None:
            self.node.get_logger().error(
                f'[{self.name}] MULTI_BIN mode ({MultiObjectPickAndPlace.Goal.MULTI_BIN}) '
                f'requires class_ids to be provided')
            return py_trees.common.Status.FAILURE

        if len(class_ids) == 0:
            self.node.get_logger().error(
                f'[{self.name}] MULTI_BIN mode ({MultiObjectPickAndPlace.Goal.MULTI_BIN}) '
                f'requires at least 1 class_id')
            return py_trees.common.Status.FAILURE

        if len(target_poses) != len(class_ids):
            self.node.get_logger().error(
                f'[{self.name}] MULTI_BIN mode ({MultiObjectPickAndPlace.Goal.MULTI_BIN}) '
                f'requires target_poses and class_ids to have same length. '
                f'Got {len(target_poses)} target_poses and {len(class_ids)} class_ids')
            return py_trees.common.Status.FAILURE

        # Validate that class_ids are unique to prevent silent overwrites
        if len(class_ids) != len(set(class_ids)):
            duplicate_ids = [x for x in set(class_ids) if class_ids.count(x) > 1]
            self.node.get_logger().error(
                f'[{self.name}] MULTI_BIN mode ({MultiObjectPickAndPlace.Goal.MULTI_BIN}) '
                f'requires unique class_ids. Found duplicate class_ids: {duplicate_ids}')
            return py_trees.common.Status.FAILURE

        # Create mapping from class_id to target_pose
        class_to_pose_map = dict(zip(class_ids, target_poses))

        # Update objects based on their class_id
        for obj_id, obj_info in self.blackboard.object_info_cache.items():
            skip_statuses = [ObjectStatus.DONE.value, ObjectStatus.FAILED.value]

            if obj_info.get('status') in skip_statuses:
                continue

            # Get the class_id for this object
            obj_class_id = obj_info.get('class_id')
            if obj_class_id is None:
                self.node.get_logger().warning(
                    f'[{self.name}] Object {obj_id} has no class_id, '
                    'skipping drop pose assignment')
                continue

            # Find the corresponding target pose
            if obj_class_id in class_to_pose_map:
                obj_info['goal_drop_pose'] = class_to_pose_map[obj_class_id]
                self.node.get_logger().debug(
                    f'[{self.name}] Assigned drop pose for object {obj_id} '
                    f'with class_id {obj_class_id}')
            else:
                self.node.get_logger().debug(
                    f'[{self.name}] No target pose found for object {obj_id} '
                    f'with class_id {obj_class_id}')

        return py_trees.common.Status.SUCCESS

    def _update_objects_with_pose(self, drop_pose):
        """Update all eligible objects with the given drop pose."""
        for obj_id, obj_info in self.blackboard.object_info_cache.items():
            skip_statuses = [ObjectStatus.DONE.value, ObjectStatus.FAILED.value]

            if obj_info.get('status') in skip_statuses:
                continue

            # Store the drop pose
            obj_info['goal_drop_pose'] = drop_pose

        return py_trees.common.Status.SUCCESS
