# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import collections

from isaac_manipulator_interfaces.action import GetObjectPose
from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_manipulator_ros_python_utils.geometry import (
    get_transformation_matrix_from_ros,
    pose_to_matrix,
)
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import numpy as np
import py_trees
import rclpy
from scipy.spatial.transform import Rotation as R


class PoseEstimation(BaseActionBehavior):
    """
    Estimate pose for the selected object using an action client.

    This behavior calls a pose estimation service to determine the 6DOF pose
    of the selected object in the world frame. Upon successful estimation,
    it updates the object's status to READY_FOR_MOTION and adds the object
    to the motion planning queue.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    base_frame_id : str
        Base frame ID for TF transformations
    camera_frame_id : str
        Camera frame ID for TF transformations
    workspace_bounds_config : object
        Configuration for workspace bounds filtering

    """

    def __init__(self, name: str, action_server_name: str,
                 base_frame_id: str = 'base_link',
                 camera_frame_id: str = 'camera_1_color_optical_frame',
                 workspace_bounds_config=None):
        blackboard_keys = {
            'object_info_cache': py_trees.common.Access.WRITE,
            'selected_object_id': py_trees.common.Access.READ,
            'next_object_id': py_trees.common.Access.WRITE,
            'tf_buffer': py_trees.common.Access.READ
        }
        super().__init__(
            name=name,
            action_type=GetObjectPose,
            action_server_name=action_server_name,
            blackboard_keys=blackboard_keys
        )
        # Frame IDs used for TF lookup
        self.base_frame_id = base_frame_id
        self.camera_frame_id = camera_frame_id
        # Workspace bounds configuration
        self.workspace_bounds_config = workspace_bounds_config

    def setup(self, **kwargs):
        # Call parent's setup to initialize self.node
        super().setup(**kwargs)

    def update(self):
        """
        Drive the pose estimation behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when pose is estimated successfully,
            FAILURE when estimation fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            if self.get_action_state() == BehaviorStatus.FAILED:
                return self._handle_action_failure()
            return py_trees.common.Status.FAILURE

        # Check if we have a valid selected object ID
        selected_id = self.blackboard.selected_object_id
        obj_info_cache = getattr(self.blackboard, 'object_info_cache', None)

        if selected_id is None:
            self.node.get_logger().error('No object selected for pose estimation')
            return py_trees.common.Status.FAILURE

        if obj_info_cache is None:
            self.node.get_logger().error(
                'Object info cache is not available on blackboard')
            return py_trees.common.Status.FAILURE

        if selected_id not in obj_info_cache:
            self.node.get_logger().error(
                f'Selected object ID "{selected_id}" not found in object cache')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the pose estimation process
            self._trigger_get_object_pose(selected_id)
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the pose estimation to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the pose estimation result
            return self._process_pose_result(selected_id)

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _is_in_workspace_bounds(self, position):
        """
        Check if a 3D position lies within the configured workspace bounds.

        Parameters
        ----------
        position : list or np.ndarray
            [x, y, z] coordinates

        Returns
        -------
        bool
            True if position is within bounds or bounds are disabled, False otherwise

        """
        if not self.workspace_bounds_config or not self.workspace_bounds_config.diagonal:
            return True

        try:
            x, y, z = position

            # Compute min/max from diagonal corners
            corner1, corner2 = self.workspace_bounds_config.diagonal
            min_xyz = [min(corner1[i], corner2[i]) for i in range(3)]
            max_xyz = [max(corner1[i], corner2[i]) for i in range(3)]

            return (
                min_xyz[0] <= x <= max_xyz[0] and
                min_xyz[1] <= y <= max_xyz[1] and
                min_xyz[2] <= z <= max_xyz[2]
            )
        except Exception as e:
            self.node.get_logger().warning(
                f'[{self.name}] Error checking workspace bounds: {e}. '
                'Allowing object to proceed.')
            # Fail-open: if any parsing error occurs, do not remove the object
            return True

    def _trigger_get_object_pose(self, object_id):
        """Trigger the get object pose action."""
        self.node.get_logger().info(
            f'[{self.name}] Starting pose estimation for object_id={object_id}')
        get_object_pose_goal = GetObjectPose.Goal()
        get_object_pose_goal.object_id = object_id
        self.send_goal(get_object_pose_goal)

    def _process_pose_result(self, object_id):
        """
        Process the pose estimation result by transforming it to the base frame.

        This method takes the object pose estimated in the camera frame and transforms
        it to the robot's base frame for motion planning. The transformation pipeline:
        1. Get pose from action result (camera frame)
        2. Lookup camera-to-base transform via TF2
        3. Apply transformation: base_pose_object_matrix =
           base_pose_camera_matrix @ camera_pose_object_matrix
        4. Extract position/quaternion and check workspace bounds
        5. Cache for motion planning if within bounds

        Returns
        -------
        py_trees.common.Status
            SUCCESS if pose was processed successfully,
            FAILURE otherwise

        """
        # Check if tf_buffer is available
        if not self.blackboard.exists('tf_buffer') or self.blackboard.tf_buffer is None:
            self.node.get_logger().error(f'[{self.name}] TF buffer is not initialized')
            return py_trees.common.Status.FAILURE

        try:
            # Get object's pose with respect to the camera frame from action result.
            action_result = self.get_action_result()
            camera_pose_object = action_result.object_pose

            self.node.get_logger().info(
                f'[{self.name}] Received camera_pose_object for object_id={object_id}')

            # Convert poses to transformation matrices and transform to base frame
            camera_pose_object_matrix = pose_to_matrix(camera_pose_object)
            base_pose_camera_matrix = self._lookup_transform(
                target_frame=self.base_frame_id,
                source_frame=self.camera_frame_id)
            base_pose_object_matrix = (base_pose_camera_matrix @
                                       camera_pose_object_matrix)  # Matrix multiplication

            # Extract pose components
            position = base_pose_object_matrix[:3, 3]
            rotation_matrix = base_pose_object_matrix[:3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat()

            # Check workspace bounds if configured
            if not self._is_in_workspace_bounds(position):
                self.node.get_logger().info(
                    f'[{self.name}] Object {object_id} at position {position} '
                    'is outside workspace bounds. Removing from cache.')

                # Remove object from cache similar to filter_detections.py
                del self.blackboard.object_info_cache[object_id]
                return py_trees.common.Status.SUCCESS

            # Store the estimated pose
            self._write_estimated_pose_to_cache(object_id, position, quaternion)

            # Update status and enqueue for motion planning
            self.blackboard.object_info_cache[object_id]['status'] = \
                ObjectStatus.READY_FOR_MOTION.value

            if (not self.blackboard.exists('next_object_id') or
                    self.blackboard.next_object_id is None):
                self.node.get_logger().error('next_object_id is not initialized on blackboard')
                self.blackboard.next_object_id = collections.deque()

            self.blackboard.next_object_id.append(object_id)

        except Exception as e:
            self.node.get_logger().error(
                f'[{self.name}] Failed to process pose for object_id={object_id}: {e}')
            return py_trees.common.Status.FAILURE

        self.node.get_logger().info(
            f'[{self.name}] Successfully estimated pose for object_id={object_id}')

        return py_trees.common.Status.SUCCESS

    def _lookup_transform(self, target_frame: str, source_frame: str) -> np.ndarray:
        """
        Lookup transformation between two frames via TF2.

        Parameters
        ----------
        target_frame : str
            Target frame ID for the transformation
        source_frame : str
            Source frame ID for the transformation

        Returns
        -------
        np.ndarray
            4x4 transformation matrix from source to target frame

        Raises
        ------
        RuntimeError
            If TF lookup fails

        """
        try:
            transform_stamped = self.blackboard.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time()
            )
        except Exception as exc:
            raise RuntimeError(
                f'TF lookup failed for {target_frame} -> {source_frame}: {exc}'
            )

        return get_transformation_matrix_from_ros(
            transform_stamped.transform.translation,
            transform_stamped.transform.rotation
        )

    def _write_estimated_pose_to_cache(
        self,
        object_id: str,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """
        Store estimated pose in object cache for motion planning.

        Parameters
        ----------
        object_id : str
            Object identifier
        position : np.ndarray
            3D position [x, y, z] in base frame
        quaternion : np.ndarray
            Rotation quaternion [x, y, z, w] in base frame

        """
        self.blackboard.object_info_cache[object_id]['estimated_pose'] = {
            'position': [float(position[0]), float(position[1]), float(position[2])],
            'orientation': [float(quaternion[0]), float(quaternion[1]),
                            float(quaternion[2]), float(quaternion[3])]
        }

    def _handle_action_failure(self):
        """
        Handle pose estimation failure and update object status.

        Action failures (goal rejected, server errors) mark the object as FAILED,
        making it ineligible for selection until a future pose est. succeeds due to retry logic.
        Non-action failures leave the object eligible for immediate re-selection.

        Returns
        -------
        py_trees.common.Status
            Always FAILURE

        """
        selected_id = self.blackboard.selected_object_id

        # Validate object_info_cache exists
        obj_info_cache = getattr(self.blackboard, 'object_info_cache', None)
        if obj_info_cache is None:
            self.node.get_logger().error(
                'Cannot handle action failure: object_info_cache not available')
            return py_trees.common.Status.FAILURE

        # Validate selected_id and ensure it's in cache
        if selected_id is None or selected_id not in obj_info_cache:
            self.node.get_logger().error(
                f'Cannot handle action failure: invalid object {selected_id}')
            return py_trees.common.Status.FAILURE

        self.node.get_logger().error(
            f'[{self.name}] Pose estimation failed for object_id={selected_id}')

        self.blackboard.object_info_cache[selected_id]['status'] = ObjectStatus.FAILED.value

        return py_trees.common.Status.FAILURE
