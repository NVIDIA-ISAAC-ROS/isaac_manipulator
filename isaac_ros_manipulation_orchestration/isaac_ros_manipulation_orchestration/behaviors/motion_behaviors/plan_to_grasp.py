# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from geometry_msgs.msg import Pose, Vector3
from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_manipulation_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees


class PlanToGrasp(BaseActionBehavior):
    """
    Plan grasp for the active object.

    This behavior generates motion plans for grasping objects, including
    approach, grasp, and retract trajectories. It uses the grasp poses
    from the blackboard and creates collision-aware motion plans with
    configurable constraints and offsets.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    link_name : str
        Name of the robot link to plan for
    grasp_approach_offset_distance : list[float]
        3D offset distance for grasp approach [x, y, z]
    grasp_translation_path_deviation_limit : float
        Max deviation (m) from the straight-line translation path during grasp segment planning.
        Set to a negative value to disable translation path constraints (terminal target only).
    grasp_translation_terminal_deviation_limit : float
        Max deviation (m) from the translation target at termination during grasp segment planning.
        Set to a negative value to use the cuMotion default (typically 0.0 when a path constraint
        is enabled).
    grasp_enable_orientation_path_axis_constraint : bool
        Whether to enable a path axis constraint (roll-free) during the grasp segment.
    grasp_orientation_terminal_deviation_limit : float
        Max deviation (rad) from the terminal orientation target during the grasp segment.
        Set to a negative value to use the cuMotion default.
    grasp_orientation_path_axis_deviation_limit : float
        Max deviation (rad) for the path axis constraint during the grasp segment.
        Set to a negative value to use the cuMotion default.
    retract_offset_distance : list[float]
        3D offset distance for retract motion [x, y, z]
    retract_translation_path_deviation_limit : float
        Max deviation (m) from the straight-line translation path during retract segment planning.
        Set to a negative value to disable translation path constraints (terminal target only).
    retract_translation_terminal_deviation_limit : float
        Max deviation (m) from the translation target at termination during retract segment
        planning. Set to a negative value to use the cuMotion default.
    retract_enable_orientation_path_axis_constraint : bool
        Whether to enable a path axis constraint (roll-free) during the retract segment.
    retract_orientation_terminal_deviation_limit : float
        Max deviation (rad) from the terminal orientation target during the retract segment.
        Set to a negative value to use the cuMotion default.
    retract_orientation_path_axis_deviation_limit : float
        Max deviation (rad) for the path axis constraint during the retract segment.
        Set to a negative value to use the cuMotion default.
    grasp_approach_constraint_in_goal_frame : bool
        Whether grasp approach constraints are in goal frame
    retract_constraint_in_goal_frame : bool
        Whether retract constraints are in goal frame
    time_dilation_factor : float
        Factor to dilate trajectory execution time
    update_planning_scene : bool
        Whether to use the current planning scene for collision checking
    world_frame : str
        Name of the world frame for planning
    enable_aabb_clearing : bool
        Whether to enable AABB clearing during planning
    esdf_clearing_padding : list[float]
        3D padding for ESDF clearing [x, y, z]
    aabb_clearing_shape : str
        Shape for AABB clearing ('SPHERE', 'CUBOID', or 'CUSTOM_MESH')
    aabb_clearing_shape_scale : list[float]
        Scale factors for AABB clearing shape [x, y, z]

    """

    def __init__(self,
                 name: str,
                 action_server_name: str,
                 link_name: str,
                 grasp_approach_offset_distance: list[float],
                 grasp_translation_path_deviation_limit: float,
                 grasp_translation_terminal_deviation_limit: float,
                 grasp_enable_orientation_path_axis_constraint: bool,
                 grasp_orientation_terminal_deviation_limit: float,
                 grasp_orientation_path_axis_deviation_limit: float,
                 retract_offset_distance: list[float],
                 retract_translation_path_deviation_limit: float,
                 retract_translation_terminal_deviation_limit: float,
                 retract_enable_orientation_path_axis_constraint: bool,
                 retract_orientation_terminal_deviation_limit: float,
                 retract_orientation_path_axis_deviation_limit: float,
                 grasp_approach_constraint_in_goal_frame: bool,
                 retract_constraint_in_goal_frame: bool,
                 time_dilation_factor: float,
                 update_planning_scene: bool,
                 world_frame: str,
                 enable_aabb_clearing: bool,
                 esdf_clearing_padding: list[float],
                 aabb_clearing_shape: str,
                 aabb_clearing_shape_scale: list[float]):
        super().__init__(
            name=name,
            action_type=MotionPlan,
            action_server_name=action_server_name,
            blackboard_keys={
                'planning_scene': py_trees.common.Access.READ,
                'active_obj_id': py_trees.common.Access.READ,
                'object_info_cache': py_trees.common.Access.READ,
                'grasp_poses': py_trees.common.Access.READ,
                'trajectory': py_trees.common.Access.WRITE,
                'selected_grasp_pose_idx': py_trees.common.Access.WRITE,
                'mesh_file_paths': py_trees.common.Access.READ,
            })

        self.link_name = link_name
        self.grasp_approach_offset_distance = grasp_approach_offset_distance
        self.grasp_translation_path_deviation_limit = grasp_translation_path_deviation_limit
        self.grasp_translation_terminal_deviation_limit = (
            grasp_translation_terminal_deviation_limit
        )
        self.grasp_enable_orientation_path_axis_constraint = (
            grasp_enable_orientation_path_axis_constraint
        )
        self.grasp_orientation_terminal_deviation_limit = (
            grasp_orientation_terminal_deviation_limit
        )
        self.grasp_orientation_path_axis_deviation_limit = (
            grasp_orientation_path_axis_deviation_limit
        )
        self.retract_offset_distance = retract_offset_distance
        self.retract_translation_path_deviation_limit = retract_translation_path_deviation_limit
        self.retract_translation_terminal_deviation_limit = (
            retract_translation_terminal_deviation_limit
        )
        self.retract_enable_orientation_path_axis_constraint = (
            retract_enable_orientation_path_axis_constraint
        )
        self.retract_orientation_terminal_deviation_limit = (
            retract_orientation_terminal_deviation_limit
        )
        self.retract_orientation_path_axis_deviation_limit = (
            retract_orientation_path_axis_deviation_limit
        )
        self.grasp_approach_constraint_in_goal_frame = grasp_approach_constraint_in_goal_frame
        self.retract_constraint_in_goal_frame = retract_constraint_in_goal_frame
        self.time_dilation_factor = time_dilation_factor
        self.update_planning_scene = update_planning_scene
        self.world_frame = world_frame
        self.enable_aabb_clearing = enable_aabb_clearing
        self.esdf_clearing_padding = esdf_clearing_padding
        self.aabb_clearing_shape = aabb_clearing_shape
        self.aabb_clearing_shape_scale = aabb_clearing_shape_scale

    def update(self):
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the motion plan process
            if not self._trigger_plan_grasp():
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the motion plan to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process motion plan results
            return self._process_motion_plan_results()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_plan_grasp(self):
        """Trigger motion plan action."""
        # Validate active_obj_id in object cache
        if (not self.blackboard.exists('active_obj_id') or
                self.blackboard.active_obj_id is None):
            self.node.get_logger().error('active_obj_id not available on blackboard')
            return False

        if (not self.blackboard.exists('object_info_cache') or
                self.blackboard.object_info_cache is None):
            self.node.get_logger().error('object_info_cache not available on blackboard')
            return False

        if (self.blackboard.object_info_cache is None or
                self.blackboard.active_obj_id not in self.blackboard.object_info_cache):
            self.node.get_logger().error(
                f'active object ID "{self.blackboard.active_obj_id}"'
                f'not found in object cache')
            return False

        object_info = self.blackboard.object_info_cache[self.blackboard.active_obj_id]
        object_class_id = object_info.get('class_id')
        if object_class_id is None:
            self.node.get_logger().error(
                f"class_id doesn't exist for object ID "
                f'{self.blackboard.active_obj_id}')
            return False

        # Validate grasp_poses before proceeding
        if (not self.blackboard.exists('grasp_poses') or
                self.blackboard.grasp_poses is None):
            self.node.get_logger().error('grasp_poses not available on blackboard')
            return False

        if (not hasattr(self.blackboard.grasp_poses, 'poses') or
                len(self.blackboard.grasp_poses.poses) == 0):
            self.node.get_logger().error(
                'grasp_poses is empty - no poses available for planning')
            return False

        mesh_file_path = self.blackboard.mesh_file_paths.get(object_class_id)
        if mesh_file_path is None:
            self.node.get_logger().error(
                f'No mesh file found for class ID {object_class_id}')
            return False

        # Create a goal message for the motion plan
        self.goal_msg = MotionPlan.Goal()
        self.goal_msg.goal_pose = self.blackboard.grasp_poses
        # Set the frame_id for the goal pose
        self.goal_msg.goal_pose.header.frame_id = self.link_name
        self.goal_msg.goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        self.goal_msg.use_current_state = True
        self.goal_msg.use_planning_scene = self.update_planning_scene
        self.goal_msg.plan_cspace = False
        self.goal_msg.plan_pose = False
        self.goal_msg.time_dilation_factor = self.time_dilation_factor
        self.goal_msg.hold_partial_pose = False
        #  Keeping the orientation identity
        grasp_offset_pose = Pose()
        grasp_offset_pose.position.x = self.grasp_approach_offset_distance[0]
        grasp_offset_pose.position.y = self.grasp_approach_offset_distance[1]
        grasp_offset_pose.position.z = self.grasp_approach_offset_distance[2]
        self.goal_msg.grasp_offset_pose = grasp_offset_pose
        self.goal_msg.grasp_translation_path_deviation_limit = (
            float(self.grasp_translation_path_deviation_limit)
        )
        self.goal_msg.grasp_translation_terminal_deviation_limit = (
            float(self.grasp_translation_terminal_deviation_limit)
        )
        self.goal_msg.grasp_enable_orientation_path_axis_constraint = bool(
            self.grasp_enable_orientation_path_axis_constraint
        )
        self.goal_msg.grasp_orientation_terminal_deviation_limit = (
            float(self.grasp_orientation_terminal_deviation_limit)
        )
        self.goal_msg.grasp_orientation_path_axis_deviation_limit = (
            float(self.grasp_orientation_path_axis_deviation_limit)
        )
        #  Keeping the orientation identity
        retract_offset_pose = Pose()
        retract_offset_pose.position.x = self.retract_offset_distance[0]
        retract_offset_pose.position.y = self.retract_offset_distance[1]
        retract_offset_pose.position.z = self.retract_offset_distance[2]
        self.goal_msg.retract_offset_pose = retract_offset_pose
        self.goal_msg.retract_translation_path_deviation_limit = (
            float(self.retract_translation_path_deviation_limit)
        )
        self.goal_msg.retract_translation_terminal_deviation_limit = (
            float(self.retract_translation_terminal_deviation_limit)
        )
        self.goal_msg.retract_enable_orientation_path_axis_constraint = bool(
            self.retract_enable_orientation_path_axis_constraint
        )
        self.goal_msg.retract_orientation_terminal_deviation_limit = (
            float(self.retract_orientation_terminal_deviation_limit)
        )
        self.goal_msg.retract_orientation_path_axis_deviation_limit = (
            float(self.retract_orientation_path_axis_deviation_limit)
        )
        self.goal_msg.grasp_approach_constraint_in_goal_frame = \
            self.grasp_approach_constraint_in_goal_frame
        self.goal_msg.retract_constraint_in_goal_frame = \
            self.retract_constraint_in_goal_frame
        self.goal_msg.object_frame = self.blackboard.object_info_cache[
            self.blackboard.active_obj_id].get('object_frame_name')

        # Get and log object frame information for debugging
        active_obj_id = self.blackboard.active_obj_id
        obj_info = self.blackboard.object_info_cache[active_obj_id]
        object_frame_name = obj_info.get('object_frame_name')

        self.node.get_logger().info(
            f'[{self.name}] Object info for active_obj_id={active_obj_id}:')
        self.node.get_logger().info(f'[{self.name}]   object_frame_name: {object_frame_name}')
        self.node.get_logger().info(f'[{self.name}]   status: {obj_info.get("status", "N/A")}')
        self.node.get_logger().info(f'[{self.name}]   class_id: {obj_info.get("class_id", "N/A")}')

        self.goal_msg.world_frame = self.world_frame
        self.goal_msg.mesh_resource = mesh_file_path
        self.goal_msg.update_esdf = True
        self.goal_msg.enable_aabb_clearing = self.enable_aabb_clearing
        self.goal_msg.clear_esdf = self.enable_aabb_clearing
        self.goal_msg.object_esdf_clearing_padding = self.esdf_clearing_padding

        # Set AABB clearing shape and scale
        self.goal_msg.object_shape = self.aabb_clearing_shape

        # Convert Python list to Vector3 for object_scale
        object_scale = Vector3()
        object_scale.x = self.aabb_clearing_shape_scale[0]
        object_scale.y = self.aabb_clearing_shape_scale[1]
        object_scale.z = self.aabb_clearing_shape_scale[2]
        self.goal_msg.object_scale = object_scale

        self.goal_msg.plan_grasp = True
        self.goal_msg.plan_approach_to_grasp = True
        self.goal_msg.plan_grasp_to_retract = True

        if self.update_planning_scene:
            if (self.blackboard.exists('planning_scene') and
                    self.blackboard.planning_scene):
                self.goal_msg.world = self.blackboard.planning_scene.world
                self.node.get_logger().info(
                    f'Including planning scene with '
                    f'{len(self.blackboard.planning_scene.world.collision_objects)} '
                    'collision objects')
            else:
                self.node.get_logger().warning(
                    'Static planning scene from MoveIt is not available.')

        self.send_goal(self.goal_msg)
        self.node.get_logger().info(
            f'[{self.name}] Starting grasp planning for object_id={self.blackboard.active_obj_id}')
        return True

    def _process_motion_plan_results(self):
        """Process the motion plan results and update the blackboard."""
        # Save the trajectory from the action result
        action_result = self.get_action_result()
        if hasattr(action_result, 'success'):
            if not action_result.success:
                self.node.get_logger().error(
                    f'[{self.name}] Failed to plan grasp for '
                    f'object_id={self.blackboard.active_obj_id}')

                # Log all available error information
                if hasattr(action_result, 'error_code'):
                    self.node.get_logger().error(f'Error code: {action_result.error_code}')
                if hasattr(action_result, 'message'):
                    self.node.get_logger().error(f'Error message: {action_result.message}')

                return py_trees.common.Status.FAILURE
            self.blackboard.trajectory = action_result.planned_trajectory
            self.blackboard.selected_grasp_pose_idx = action_result.goal_index
            self.node.get_logger().info(
                f'[{self.name}] Successfully planned grasp for '
                f'object_id={self.blackboard.active_obj_id}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] No trajectory found in action result for '
                f'object_id={self.blackboard.active_obj_id}')
            error_code = getattr(action_result, 'error_code', 'Unknown error code')
            message = getattr(action_result, 'message', 'No error message available')
            self.node.get_logger().error(f'Error code: {error_code}')
            self.node.get_logger().error(f'Error message: {message}')
            return py_trees.common.Status.FAILURE
