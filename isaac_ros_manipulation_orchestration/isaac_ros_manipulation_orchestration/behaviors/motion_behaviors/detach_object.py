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


from isaac_ros_cumotion_interfaces.action import AttachObject as AttachObjectAction
from isaac_ros_manipulation_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees


class DetachObject(BaseActionBehavior):
    """
    Detach an object from the robot gripper using action client.

    This behavior sends a detach object request to the action server to
    remove an attached object from the robot's collision geometry. This is
    typically called after placing an object.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to

    """

    def __init__(self,
                 name: str,
                 action_server_name: str):

        blackboard_keys = {
            'active_obj_id': py_trees.common.Access.READ,
        }
        super().__init__(
            name=name,
            action_type=AttachObjectAction,
            action_server_name=action_server_name,
            blackboard_keys=blackboard_keys
        )

    def update(self):
        """
        Drive the object detachment behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when object is detached successfully,
            FAILURE when detachment fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.node.get_logger().error('Object detachment failed')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            if self.blackboard.active_obj_id is None:
                self.node.get_logger().error('No active object ID found in blackboard')
                return py_trees.common.Status.FAILURE

            # Start the object detachment process
            self._trigger_detach_object()
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the action to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the detachment result
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _process_result(self):
        """
        Process the object detachment result.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the object was detached successfully,
            FAILURE otherwise

        """
        # Check the outcome from the action result
        outcome = self.get_action_result().outcome

        if 'detached' in outcome.lower():
            self.node.get_logger().info(
                f'[{self.name}] Successfully detached object. Result: {outcome}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] Failed to detach object: {outcome}')
        return py_trees.common.Status.FAILURE

    def _trigger_detach_object(self):
        """Trigger the action call for detaching the object."""
        # Create the goal message
        goal = AttachObjectAction.Goal()
        goal.attach_object = False  # Set to False for detachment

        self.node.get_logger().info('Detaching object')
        self.send_goal(goal)
