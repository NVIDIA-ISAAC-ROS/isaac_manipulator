# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.behaviors.motion_behaviors.motion_utils import (
    create_object_attachment_config
)
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_ros_cumotion_interfaces.action import AttachObject as AttachObjectAction
import py_trees
import rclpy


class AttachObject(BaseActionBehavior):
    """
    Attach an object to the robot gripper using action client.

    This behavior sends an attach object request to the action server. It
    updates the robot's collision geometry with an attached object. The
    object can be represented as a primitive shape or custom mesh.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    fallback_radius : float
        Fallback radius for object attachment when other methods fail
    shape : str
        Shape of the object ('SPHERE', 'CUBOID', 'CUSTOM_MESH')
    scale : List[float]
        Scale of the object as [x, y, z] values
    gripper_frame : str, optional
        The frame w.r.t to which the poses were recorded using Grasp editor.
        Defaults to 'gripper_frame'
    grasp_frame : str, optional
        The grasp frame that is at an offset w.r.t to the gripper frame such
        that the TF aligns between the gripper finger tips. Defaults to 'grasp_frame'

    """

    def __init__(self,
                 name: str,
                 action_server_name: str,
                 fallback_radius: float,
                 shape: str,
                 scale: List[float],
                 gripper_frame: str = 'gripper_frame',
                 grasp_frame: str = 'grasp_frame'
                 ):
        blackboard_keys = {
            'object_info_cache': py_trees.common.Access.READ,
            'active_obj_id': py_trees.common.Access.READ,
            'grasp_reader_manager': py_trees.common.Access.READ,
            'selected_grasp_pose_idx': py_trees.common.Access.READ,
            'tf_buffer': py_trees.common.Access.READ,
            'mesh_file_paths': py_trees.common.Access.READ,
        }
        super().__init__(
            name=name,
            action_type=AttachObjectAction,
            action_server_name=action_server_name,
            blackboard_keys=blackboard_keys
        )

        self.fallback_radius = fallback_radius
        self.shape = shape
        self.scale = scale
        self.gripper_frame = gripper_frame
        self.grasp_frame = grasp_frame

    def update(self):
        """
        Drive the object attachment behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when object is attached successfully,
            FAILURE when attachment fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.node.get_logger().error('Object attachment failed')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            if self.blackboard.active_obj_id is None:
                self.node.get_logger().error('No active object ID found in blackboard')
                return py_trees.common.Status.FAILURE

            if (self.blackboard.object_info_cache is None or
                    self.blackboard.active_obj_id not in self.blackboard.object_info_cache):
                self.node.get_logger().error(
                    f'Object ID {self.blackboard.active_obj_id} not found in object_info_cache')
                return py_trees.common.Status.FAILURE

            object_info = self.blackboard.object_info_cache[self.blackboard.active_obj_id]
            object_class_id = object_info.get('class_id')
            if object_class_id is None:
                self.node.get_logger().error(
                    f"class_id doesn't exist for object ID "
                    f'{self.blackboard.active_obj_id}')
                return py_trees.common.Status.FAILURE

            mesh_file_path = self.blackboard.mesh_file_paths.get(object_class_id)
            if mesh_file_path is None:
                self.node.get_logger().error(
                    f'No mesh file found for class ID {object_class_id}')
                return py_trees.common.Status.FAILURE

            # Start the object attachment process
            self._trigger_attach_object(
                object_class_id=object_class_id,
                mesh_file_path=mesh_file_path
            )
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the action to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the attachment result
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _process_result(self):
        """
        Process the object attachment result.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the object was attached successfully,
            FAILURE otherwise

        """
        # Check the outcome from the action result
        outcome = self.get_action_result().outcome

        if 'attached' in outcome.lower():
            obj_id = self.blackboard.active_obj_id
            self.node.get_logger().info(
                f'[{self.name}] Successfully attached object {obj_id}. Result: {outcome}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] Failed to attach object: {outcome}')
        return py_trees.common.Status.FAILURE

    def _trigger_attach_object(self, object_class_id: str, mesh_file_path: str):
        """Trigger the action call for attaching the object."""
        # Create the goal message
        goal = AttachObjectAction.Goal()
        goal.attach_object = True

        # Use instance variable for fallback radius
        goal.fallback_radius = float(self.fallback_radius)

        grasp_reader = self.blackboard.grasp_reader_manager.get_grasp_reader(object_class_id)

        # Check if transform is available between gripper_frame and grasp_frame before proceeding
        try:
            transform_available = self.blackboard.tf_buffer.can_transform(
                self.gripper_frame, self.grasp_frame, rclpy.time.Time())
            if not transform_available:
                self.node.get_logger().error(
                    f'[{self.name}] Transform from {self.gripper_frame} to {self.grasp_frame} '
                    f'not available for object_id={self.blackboard.active_obj_id}')
                self.set_action_failed()
                return py_trees.common.Status.FAILURE
            else:
                self.node.get_logger().debug(
                    f'[{self.name}] Transform from {self.gripper_frame} to {self.grasp_frame} '
                    f'is available for object_id={self.blackboard.active_obj_id}')
        except Exception as e:
            self.node.get_logger().error(
                f'[{self.name}] Error checking transform availability from '
                f'{self.gripper_frame} to {self.grasp_frame} for '
                f'object_id={self.blackboard.active_obj_id}: {e}')
            self.set_action_failed()
            return py_trees.common.Status.FAILURE

        grasp_pose_object = grasp_reader.get_grasp_pose_object(
            index=self.blackboard.selected_grasp_pose_idx,
            tf_buffer=self.blackboard.tf_buffer,
            gripper_frame=self.gripper_frame,
            grasp_frame=self.grasp_frame
        )

        gripper_pose_object = {
            'position': [
                grasp_pose_object.position.x,
                grasp_pose_object.position.y,
                grasp_pose_object.position.z
            ],
            'orientation': [
                grasp_pose_object.quaternion.x,
                grasp_pose_object.quaternion.y,
                grasp_pose_object.quaternion.z,
                grasp_pose_object.quaternion.w
            ]
        }

        # Create the object attachment configuration
        goal.object_config = create_object_attachment_config(
            gripper_pose_object=gripper_pose_object,
            shape=self.shape,
            scale=self.scale,
            mesh_file_path=mesh_file_path
        )

        self.node.get_logger().info('Attaching object')
        self.send_goal(goal)
        return py_trees.common.Status.RUNNING
