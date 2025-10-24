# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
