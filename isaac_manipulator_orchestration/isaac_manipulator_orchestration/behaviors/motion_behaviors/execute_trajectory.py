# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from moveit_msgs.action import ExecuteTrajectory as ExecuteTrajectoryAction
import py_trees
from std_msgs.msg import Header


class ExecuteTrajectory(BaseActionBehavior):
    """
    Execute a trajectory.

    This behavior executes a specific trajectory from the trajectory list
    stored on the blackboard. The trajectory is selected by index, allowing
    for sequential execution of multiple planned trajectories.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    index : int
        Index of the trajectory to execute from the blackboard trajectory list
        (e.g., 0 for grasp/drop trajectory, 1 for lift trajectory)

    """

    def __init__(self,
                 name: str,
                 action_server_name: str,
                 index: int):
        super().__init__(
            name=name,
            action_type=ExecuteTrajectoryAction,
            action_server_name=action_server_name,
            blackboard_keys={
                'active_obj_id': py_trees.common.Access.READ,
                'object_info_cache': py_trees.common.Access.READ,
                'trajectory': py_trees.common.Access.READ
            })
        # Index specifies which trajectory to execute from the trajectory list
        # on blackboard - e.g., index=0 for grasp/drop, index=1 for lift
        self.index = index

    def update(self):
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the execute trajectory process
            if not self._trigger_execute_trajectory():
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the execute trajectory to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process execute trajectory results
            self.node.get_logger().info(
                f'[{self.name}] Successfully executed trajectory for '
                f'object_id={self.blackboard.active_obj_id}')
            return py_trees.common.Status.SUCCESS

        elif self.get_action_state() == BehaviorStatus.FAILED:
            self.node.get_logger().error(
                f'[{self.name}] Failed to execute trajectory for '
                f'object_id={self.blackboard.active_obj_id}')
            return py_trees.common.Status.FAILURE

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')

        return py_trees.common.Status.FAILURE

    def _trigger_execute_trajectory(self):
        """Trigger the execute trajectory action."""
        # Validate trajectory exists and is accessible
        if not self.blackboard.exists('trajectory') or self.blackboard.trajectory is None:
            self.node.get_logger().error('No trajectory available on blackboard')
            return False

        # Check if trajectory has the required index
        try:
            if len(self.blackboard.trajectory) <= self.index:
                self.node.get_logger().error(
                    f'Trajectory index {self.index} out of bounds. '
                    f'Trajectory length: {len(self.blackboard.trajectory)}')
                return False
        except (TypeError, AttributeError) as e:
            self.node.get_logger().error(f'Error validating trajectory: {e}')
            return False

        # Create a goal message for the motion plan
        self.goal_msg = ExecuteTrajectoryAction.Goal()
        self.goal_msg.trajectory = self.blackboard.trajectory[self.index]

        # Set joint trajectory header
        self.goal_msg.trajectory.joint_trajectory.header = Header()

        # Set multi-DOF joint trajectory header
        self.goal_msg.trajectory.multi_dof_joint_trajectory.header = Header()

        self.send_goal(self.goal_msg)
        self.node.get_logger().info(
            f'[{self.name}] Starting trajectory execution for '
            f'object_id={self.blackboard.active_obj_id}, trajectory_index={self.index}')
        return True
