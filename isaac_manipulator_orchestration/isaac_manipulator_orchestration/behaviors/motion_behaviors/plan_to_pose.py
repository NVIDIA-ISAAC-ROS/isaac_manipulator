# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from geometry_msgs.msg import PoseArray, Vector3
from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_ros_cumotion_interfaces.action import MotionPlan
import py_trees


class PlanToPose(BaseActionBehavior):
    """
    Plan to a pose.

    This behavior generates motion plans to move the robot to a specified
    target pose. It uses the goal_drop_pose from the blackboard and creates
    collision-aware trajectories with configurable collision clearing and
    planning scene integration.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    link_name : str
        Name of the robot link to plan for
    time_dilation_factor : float
        Factor to dilate trajectory execution time
    update_planning_scene : bool
        Whether to use the current planning scene for collision checking
    disable_collision_links : list[str]
        List of link names to disable collision checking for
    aabb_clearing_shape : str
        Shape for AABB clearing ('SPHERE', 'CUBOID', etc.)
    aabb_clearing_shape_scale : list[float]
        Scale factors for AABB clearing shape [x, y, z]
    enable_aabb_clearing : bool
        Whether to enable AABB clearing during planning
    esdf_clearing_padding : list[float]
        Padding for ESDF clearing [x, y, z]

    """

    def __init__(self,
                 name: str,
                 action_server_name: str,
                 link_name: str,
                 time_dilation_factor: float,
                 update_planning_scene: bool,
                 disable_collision_links: list[str],
                 aabb_clearing_shape: str,
                 aabb_clearing_shape_scale: list[float],
                 enable_aabb_clearing: bool,
                 esdf_clearing_padding: list[float]):
        super().__init__(
            name=name,
            action_type=MotionPlan,
            action_server_name=action_server_name,
            blackboard_keys={
                'active_obj_id': py_trees.common.Access.READ,
                'planning_scene': py_trees.common.Access.READ,
                'goal_drop_pose': py_trees.common.Access.READ,  # Type: Pose
                'trajectory': py_trees.common.Access.WRITE,
                'object_info_cache': py_trees.common.Access.READ,
                'mesh_file_paths': py_trees.common.Access.READ,
            })

        self.link_name = link_name
        self.time_dilation_factor = time_dilation_factor
        self.update_planning_scene = update_planning_scene
        self.disable_collision_links = disable_collision_links
        self.aabb_clearing_shape = aabb_clearing_shape
        self.aabb_clearing_shape_scale = aabb_clearing_shape_scale
        self.enable_aabb_clearing = enable_aabb_clearing
        self.esdf_clearing_padding = esdf_clearing_padding

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the motion plan process
            if not self._trigger_plan_pose():
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

    def _trigger_plan_pose(self):
        """Trigger the motion plan action."""
        # Validate goal_drop_pose before using it
        if (not self.blackboard.exists('goal_drop_pose') or
                self.blackboard.goal_drop_pose is None):
            self.node.get_logger().error('goal_drop_pose not available on blackboard')
            return False

        if (not self.blackboard.exists('active_obj_id') or
                self.blackboard.active_obj_id is None):
            self.node.get_logger().error('active_obj_id not available on blackboard')
            return False

        # Capture object_info_cache once to avoid race condition with perception workflow
        if not self.blackboard.exists('object_info_cache'):
            self.node.get_logger().error('object_info_cache not available on blackboard')
            return False

        object_info_cache = self.blackboard.object_info_cache
        if object_info_cache is None:
            self.node.get_logger().error('object_info_cache is None on blackboard')
            return False

        if self.blackboard.active_obj_id not in object_info_cache:
            self.node.get_logger().error(
                f'active object ID "{self.blackboard.active_obj_id}"'
                f'not found in object cache')
            return False

        object_info = object_info_cache[self.blackboard.active_obj_id]
        object_class_id = object_info.get('class_id')
        if object_class_id is None:
            self.node.get_logger().error(
                f"class_id doesn't exist for object ID "
                f'{self.blackboard.active_obj_id}')
            return False

        # Get mesh file path
        mesh_file_path = self.blackboard.mesh_file_paths.get(object_class_id)
        if mesh_file_path is None:
            self.node.get_logger().error(
                f'No mesh file found for class ID {object_class_id}')
            return False

        self.goal_msg = MotionPlan.Goal()
        self.goal_msg.goal_pose = PoseArray()
        self.goal_msg.goal_pose.header.frame_id = self.link_name
        self.goal_msg.goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        self.goal_msg.goal_pose.poses.append(self.blackboard.goal_drop_pose)
        self.goal_msg.plan_pose = True
        self.goal_msg.use_current_state = True
        self.goal_msg.use_planning_scene = self.update_planning_scene
        self.goal_msg.time_dilation_factor = self.time_dilation_factor
        self.goal_msg.hold_partial_pose = False
        self.goal_msg.disable_collision_links = self.disable_collision_links
        self.goal_msg.mesh_resource = mesh_file_path
        self.goal_msg.object_shape = self.aabb_clearing_shape

        # Convert Python list to Vector3 for object_scale (Vector3 expected)
        object_scale = Vector3()
        object_scale.x = self.aabb_clearing_shape_scale[0]
        object_scale.y = self.aabb_clearing_shape_scale[1]
        object_scale.z = self.aabb_clearing_shape_scale[2]
        self.goal_msg.object_scale = object_scale

        self.goal_msg.enable_aabb_clearing = self.enable_aabb_clearing
        self.goal_msg.object_esdf_clearing_padding = self.esdf_clearing_padding

        if self.update_planning_scene:
            if (self.blackboard.exists('planning_scene') and
                    self.blackboard.planning_scene):
                self.goal_msg.world = self.blackboard.planning_scene.world
                collision_count = len(
                    self.blackboard.planning_scene.world.collision_objects)
                self.node.get_logger().info(
                    f'Including planning scene with {collision_count} '
                    'collision objects')
            else:
                self.node.get_logger().warning(
                    'Static planning scene from MoveIt is not available.')

        self.send_goal(self.goal_msg)
        self.node.get_logger().info(
            f'[{self.name}] Starting planning to pose for '
            f'object_id={self.blackboard.active_obj_id}')
        return True

    def _process_motion_plan_results(self):
        """Process the motion plan results and update the blackboard."""
        # Save the trajectory from the action result
        action_result = self.get_action_result()
        if hasattr(action_result, 'success'):
            if not action_result.success:
                self.node.get_logger().error(
                    f'[{self.name}] Failed to plan pose for '
                    f'object_id={self.blackboard.active_obj_id}')

                # Log all available error information
                if hasattr(action_result, 'error_code'):
                    self.node.get_logger().error(f'Error code: {action_result.error_code}')
                if hasattr(action_result, 'message'):
                    self.node.get_logger().error(f'Error message: {action_result.message}')

                return py_trees.common.Status.FAILURE
            self.blackboard.trajectory = action_result.planned_trajectory
            self.node.get_logger().info(
                f'[{self.name}] Successfully planned pose for '
                f'object_id={self.blackboard.active_obj_id}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] No trajectory found in action result for '
                f'object_id={self.blackboard.active_obj_id}')
            if hasattr(action_result, 'error_code'):
                self.node.get_logger().error(f'Error code: {action_result.error_code}')
            if hasattr(action_result, 'message'):
                self.node.get_logger().error(f'Error message: {action_result.message}')
            return py_trees.common.Status.FAILURE
