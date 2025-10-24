# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from control_msgs.action import GripperCommand
from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
import py_trees


class CloseGripper(BaseActionBehavior):
    """
    Close gripper using action client.

    This behavior sends a gripper command to close the gripper to a specified
    position with a maximum effort constraint.

    Parameters
    ----------
    name : str
        Name of the behavior
    gripper_action_name : str
        Name of the gripper action server
    close_position : float
        Target position for gripper closure (gripper-specific units)
    max_effort : float
        Maximum effort/force to apply during closure (gripper-specific units)

    """

    def __init__(self,
                 name: str,
                 gripper_action_name: str,
                 close_position: float,
                 max_effort: float
                 ):
        super().__init__(
            name=name,
            action_type=GripperCommand,
            action_server_name=gripper_action_name
        )

        self.close_position = close_position
        self.max_effort = max_effort

    def update(self):
        """
        Drive the gripper closing behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when gripper closes successfully,
            FAILURE when closing fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.node.get_logger().error('Gripper close failed')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the gripper closing process
            self._trigger_close_gripper()
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the gripper action to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the gripper close result
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_close_gripper(self):
        """Trigger the action call for closing the gripper."""
        gripper_goal = GripperCommand.Goal()

        gripper_goal.command.position = float(self.close_position)
        gripper_goal.command.max_effort = float(self.max_effort)

        self.node.get_logger().info(f'Closing gripper to position {self.close_position}')
        self.send_goal(gripper_goal)

    def _process_result(self):
        """
        Process the gripper close action result.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the gripper was closed successfully,
            FAILURE otherwise

        """
        # Check if we reached the goal position
        current_position = getattr(self.get_action_result(), 'position', None)

        if current_position is not None and \
                current_position <= self.close_position:
            self.node.get_logger().info(
                f'[{self.name}] Successfully closed gripper to position: {current_position}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] Failed to close gripper to desired position')
            return py_trees.common.Status.FAILURE
