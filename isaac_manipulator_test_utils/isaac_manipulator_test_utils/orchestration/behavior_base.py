# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Base class for behavior testing.

This module provides a lightweight base class and utilities for testing behaviors
without dependencies on external configuration files.
"""
import time
from typing import Callable, Optional

from isaac_ros_test import IsaacROSBaseTest
import py_trees
import py_trees_ros
import rclpy
from rclpy.parameter import Parameter

from .blackboard_utils import setup_test_blackboard


class BehaviorTestBase(IsaacROSBaseTest):
    """Base class for behavior testing with essential utilities."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool, use_sim_time: bool, nodes: list,
                                  node_startup_delay: float):
        """Generate test description with common pattern."""
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        self.node = rclpy.create_node(
            'behavior_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters(
                [Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def assert_test_should_run(self) -> bool:
        """Check if test should run based on RUN_TEST flag."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return False
        return True

    def setup_blackboard(self, **kwargs) -> py_trees.blackboard.Client:
        """Create and configure a blackboard for testing."""
        return setup_test_blackboard(**kwargs)

    def tick_tree_until_complete(
        self,
        behavior: py_trees.behaviour.Behaviour,
        condition: Optional[Callable[[], bool]] = None,
        timeout_seconds: float = 60.0,
        spin_timeout: float = 0.1,
        log_interval: float = 5.0,
        tree_name: str = 'Test Tree'
    ) -> py_trees.common.Status:
        """
        Tick a behavior tree until completion or custom condition is met.

        Args:
            behavior: The behavior to test
            condition: Optional custom condition function that returns True when to stop
            timeout_seconds: Maximum time to wait for completion
            spin_timeout: Timeout for each spin_once call
            log_interval: Interval between progress log messages
            tree_name: Name for the tree root

        Returns
        -------
        py_trees.common.Status
            The final status of the behavior

        Raises
        ------
        AssertionError
            If the behavior times out

        """
        # Create a simple root node
        root = py_trees.composites.Sequence(
            name=tree_name, memory=False)
        root.add_child(behavior)

        # Create a regular behavior tree
        tree = py_trees_ros.trees.BehaviourTree(root=root)

        # Setup the tree with the existing node
        tree.setup(node=self.node)

        try:
            # Start timing for timeout
            start_time = time.time()
            tick_count = 0
            last_log_time = start_time

            self.node.get_logger().info(
                f'Starting tree execution with timeout of {timeout_seconds} seconds')

            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Check if timeout has been reached
                if elapsed_time > timeout_seconds:
                    self.node.get_logger().warn(
                        f'Tree execution timed out after {timeout_seconds} seconds')
                    self.fail(
                        f'Behavior timed out after {timeout_seconds} seconds '
                        f'({tick_count} total ticks)')

                tick_count += 1

                # Tick the tree
                tree.tick()

                # Spin the node to allow ROS message processing
                rclpy.spin_once(self.node, timeout_sec=spin_timeout)

                # Check custom condition if provided
                if condition is not None and condition():
                    self.node.get_logger().info(
                        f'Custom condition met after {tick_count} ticks '
                        f'in {elapsed_time:.2f} seconds')
                    break

                # Check if behavior has completed
                if behavior.status in [py_trees.common.Status.SUCCESS,
                                       py_trees.common.Status.FAILURE]:
                    self.node.get_logger().info(
                        f'Behavior completed with status {behavior.status} '
                        f'after {tick_count} ticks in {elapsed_time:.2f} seconds')
                    break

                # Log progress periodically
                if current_time - last_log_time >= log_interval:
                    self.node.get_logger().info(
                        f'Tree execution in progress... '
                        f'{elapsed_time:.1f}s elapsed, {tick_count} ticks')
                    last_log_time = current_time

        finally:
            # Clean up the tree
            tree.shutdown()

        return behavior.status
