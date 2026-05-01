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

import py_trees


class UpdateDropPoseToHome(py_trees.behaviour.Behaviour):
    """
    Update the goal_drop_pose blackboard variable with the home_pose value.

    This behavior is used as a recovery mechanism when the primary drop pose
    planning fails. It updates the goal_drop_pose to use the home_pose value,
    allowing the robot to attempt a safe fallback motion to a known good
    configuration.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key='goal_drop_pose', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='home_pose', access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        """Set up the behavior by getting node from kwargs."""
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e
        return True

    def update(self):
        """Update goal_drop_pose with home_pose value for recovery planning."""
        if self.blackboard.exists('home_pose') and self.blackboard.home_pose:
            self.blackboard.goal_drop_pose = self.blackboard.home_pose
            self.node.get_logger().info(
                f'[{self.name}] Updated goal_drop_pose to home_pose for recovery drop')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(f'[{self.name}] home_pose not available on blackboard')
            return py_trees.common.Status.FAILURE
