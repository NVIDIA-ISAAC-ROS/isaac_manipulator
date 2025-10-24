# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import logging
import threading
from typing import Optional

from isaac_manipulator_interfaces.action import MultiObjectPickAndPlace
from isaac_manipulator_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
from isaac_manipulator_orchestration.utils.blackboard_initializer import BlackboardInitializer
from isaac_manipulator_pick_and_place.behavior_trees.multi_object_pick_and_place_bt import (
    create_multi_object_pick_and_place_tree,
)
import py_trees
import py_trees_ros
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from tf2_ros import Buffer, TransformListener


class MultiObjectPickPlaceOrchestrator:
    """Orchestrate multi-object pick and place behavior tree execution."""

    def __init__(self,
                 behavior_tree_config_file: str = None,
                 blackboard_config_file: str = None,
                 print_ascii_tree: bool = False,
                 manual_mode: bool = False,
                 log_level: str = None,
                 ):
        """
        Initialize orchestrator with all necessary components.

        Parameters
        ----------
        behavior_tree_config_file : str, optional
            Path to the behavior tree configuration file
        blackboard_config_file : str, optional
            Path to the blackboard configuration file
        print_ascii_tree : bool, optional
            Enable ASCII tree display and iteration counts
        manual_mode : bool, optional
            Enable manual mode for tree ticking
        log_level : str, optional
            Explicit log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        """
        # Determine log level if not explicitly provided
        if log_level is None:
            if manual_mode:
                log_level = 'DEBUG'
            elif print_ascii_tree:
                log_level = 'INFO'
            else:
                log_level = 'WARNING'

        # Set up logging
        self._setup_logging(log_level)

        self.print_ascii_tree = print_ascii_tree
        self.iteration = 0
        self.manual_mode = manual_mode

        # Component references
        self.tree: Optional[py_trees_ros.trees.BehaviourTree] = None
        self.action_server: Optional[ActionServer] = None
        self.thread: Optional[threading.Thread] = None
        self.executor: Optional[rclpy.executors.MultiThreadedExecutor] = None
        self.blackboard = None
        self.buffer = None
        self._shutdown_event = threading.Event()

        # Configuration files
        # raise error if behavior_tree_config_file or blackboard_config_file is not provided
        if behavior_tree_config_file is None or blackboard_config_file is None:
            raise ValueError('Behavior tree and blackboard configuration files must be provided')

        self.behavior_tree_config_file = behavior_tree_config_file
        self.blackboard_config_file = blackboard_config_file

        self.logger.info('Initializing MultiObjectPickPlaceOrchestrator')
        self.logger.debug(f'Behavior tree config: {behavior_tree_config_file}')
        self.logger.debug(f'Blackboard config: {blackboard_config_file}')

        # Initialize all components
        self._initialize_rclpy()
        self._setup_blackboard()
        self._setup_behavior_tree()
        self._setup_tf()
        self._setup_action_server()
        self._setup_executor()
        self.is_orchestrator_busy = False
        self.logger.info('MultiObjectPickPlaceOrchestrator initialization complete')

    def _setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Map string log level to logging constant
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level_obj = level_map.get(log_level.upper(), logging.WARNING)
        self.logger.setLevel(log_level_obj)

        # Only add handler if none exists (to avoid duplicate logs)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)s] [%(asctime)s] [%(name)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _initialize_rclpy(self) -> None:
        """Initialize ROS2."""
        self.logger.debug('Initializing ROS2')
        rclpy.init()

    def _setup_behavior_tree(self) -> None:
        """Set up behavior tree with configuration."""
        self.logger.info('Setting up behavior tree')
        behavior_config_initializer = BehaviorTreeConfigInitializer(
            self.behavior_tree_config_file,
            package_name='isaac_manipulator_pick_and_place'
        )

        # Create the behavior tree
        root = create_multi_object_pick_and_place_tree(behavior_config_initializer)

        # Create the behavior tree with ROS integration
        self.tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=self.print_ascii_tree,
        )

        # Setup the tree so tree.node is available
        # No timeout needed since individual server timeouts handle blocking operations
        self.tree.setup()
        self.logger.debug('Behavior tree setup complete')

    def _setup_blackboard(self) -> None:
        """Initialize blackboard with configuration."""
        self.logger.info('Setting up blackboard')
        blackboard_initializer = BlackboardInitializer(
            blackboard_params_file=self.blackboard_config_file,
            package_name='isaac_manipulator_pick_and_place'
        )
        self.blackboard = blackboard_initializer.initialize_blackboard()

        # Setup retry configuration
        behavior_config_initializer = BehaviorTreeConfigInitializer(
            self.behavior_tree_config_file,
            package_name='isaac_manipulator_pick_and_place'
        )
        retry_config = behavior_config_initializer.get_retry_config()
        blackboard_initializer.setup_retry_configuration(retry_config)

        # Setup server timeout configuration
        server_timeout_config = behavior_config_initializer.get_server_timeout_config()
        blackboard_initializer.setup_server_timeout_configuration(server_timeout_config)

        self.logger.debug('Blackboard setup complete')

    def _setup_tf(self) -> None:
        """Set up TF buffer and listener."""
        self.logger.debug('Setting up TF buffer and listener')
        if not self.tree or not self.tree.node:
            raise RuntimeError('Behavior tree must be setup before TF initialization')

        self.buffer = Buffer()
        self.blackboard.register_key(
            key='tf_buffer',
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.tf_buffer = self.buffer

        self.blackboard.register_key(
            key='tf_listener',
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.tf_listener = TransformListener(self.buffer, self.tree.node)

    def _setup_action_server(self) -> None:
        """Set up action server for multi-object pick and place."""
        self.logger.info('Setting up action server')
        if not self.tree or not self.tree.node:
            raise RuntimeError('Behavior tree must be setup before action server initialization')

        self.action_server = ActionServer(
            node=self.tree.node,
            action_name='multi_object_pick_and_place',
            action_type=MultiObjectPickAndPlace,
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback
        )
        self.logger.debug('Action server setup complete')

    def _setup_executor(self) -> None:
        """Set up executor and start spinning in a separate thread."""
        self.logger.debug('Setting up executor')
        if not self.tree or not self.tree.node:
            raise RuntimeError('Behavior tree must be setup before executor initialization')

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.tree.node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()

    def _validate_goal_data(
        self, goal_mode, goal_target_poses, goal_class_ids
    ) -> bool:
        """
        Validate goal request data.

        Parameters
        ----------
        goal_mode : int
            Mode from goal request
        goal_target_poses : List
            Target poses from goal request
        goal_class_ids : List
            Class IDs from goal request

        Returns
        -------
        bool
            True if valid, False if invalid

        """
        # Validate mode
        valid_modes = [MultiObjectPickAndPlace.Goal.SINGLE_BIN,
                       MultiObjectPickAndPlace.Goal.MULTI_BIN]
        if goal_mode not in valid_modes:
            self.logger.warning(
                f'Invalid mode in goal: {goal_mode}. Must be '
                f'{MultiObjectPickAndPlace.Goal.SINGLE_BIN} or '
                f'{MultiObjectPickAndPlace.Goal.MULTI_BIN}')
            return False

        # Validate based on mode
        if goal_mode == MultiObjectPickAndPlace.Goal.SINGLE_BIN:  # Single bin mode
            # Single bin mode: All objects go to single target pose, class_ids must be empty
            if len(goal_target_poses) != 1:
                self.logger.error(
                    f'goal validation failed: single bin mode requires exactly 1 target pose, '
                    f'got {len(goal_target_poses)}')
                return False
            if len(goal_class_ids) != 0:
                self.logger.error(
                    f'goal validation failed: single bin mode requires empty class_ids, '
                    f'got {len(goal_class_ids)}')
                return False
        elif goal_mode == MultiObjectPickAndPlace.Goal.MULTI_BIN:  # Multi-bin mode
            # Multi-bin mode: Objects distributed by class ID, requires matching arrays
            if len(goal_target_poses) == 0 or len(goal_class_ids) == 0:
                self.logger.error(
                    f'goal validation failed: multi-bin mode requires at least 1 target pose '
                    f'and 1 class_id, got {len(goal_target_poses)} poses and '
                    f'{len(goal_class_ids)} class_ids')
                return False
            if len(goal_target_poses) != len(goal_class_ids):
                self.logger.error(
                    f'goal validation failed: multi-bin mode requires equal poses '
                    f'({len(goal_target_poses)}) and class_ids ({len(goal_class_ids)})')
                return False

            # Validate that class_ids are unique to prevent silent overwrites
            if len(goal_class_ids) != len(set(goal_class_ids)):
                duplicate_ids = [x for x in set(goal_class_ids) if goal_class_ids.count(x) > 1]
                self.logger.error(
                    f'goal validation failed: multi-bin mode requires unique class_ids. '
                    f'Found duplicate class_ids: {duplicate_ids}')
                return False

        return True

    # Action server callback methods
    def execute_callback(self, goal_handle):
        """Handle execute request from action client."""
        self.logger.info('Received execute request')

        result = MultiObjectPickAndPlace.Result()

        # Extract goal data
        goal_mode = goal_handle.request.mode
        goal_target_poses = goal_handle.request.target_poses.poses
        goal_class_ids = goal_handle.request.class_ids

        self.logger.info('Updating blackboard with goal values')
        self.blackboard.target_poses = goal_target_poses
        self.blackboard.class_ids = goal_class_ids
        self.blackboard.mode = goal_mode

        # Initialize feedback object
        feedback = MultiObjectPickAndPlace.Feedback()

        # Initialize workflow status
        workflow_status = MultiObjectPickAndPlace.Result.UNKNOWN

        while rclpy.ok() and not goal_handle.is_cancel_requested:
            # Check for workflow feedback messages and publish if available
            if (self.blackboard.exists('workflow_feedback_queue') and
                    self.blackboard.workflow_feedback_queue):
                # Pop the next feedback message from the queue
                current_feedback_message = self.blackboard.workflow_feedback_queue.popleft()

                # Send feedback message to action client
                feedback.message = current_feedback_message
                goal_handle.publish_feedback(feedback)

            # Get current workflow status from blackboard
            workflow_status = self.blackboard.workflow_status if self.blackboard.exists(
                'workflow_status') else MultiObjectPickAndPlace.Result.UNKNOWN

            # Exit when terminal state reached (feedback messages sent before terminating)
            if workflow_status != MultiObjectPickAndPlace.Result.UNKNOWN:
                break

        # Handle goal cancellation
        if goal_handle.is_cancel_requested:
            self.logger.warning('Goal is being canceled')
            goal_handle.canceled()
            result.workflow_status = MultiObjectPickAndPlace.Result.INCOMPLETE
            result.workflow_summary = 'Goal was canceled'
            # Signal the executor to shutdown
            self._shutdown_event.set()
            return result

        # Return final result using current blackboard values
        workflow_summary = self.blackboard.workflow_summary if self.blackboard.exists(
            'workflow_summary') else 'No status available'
        result.workflow_status = workflow_status
        result.workflow_summary = workflow_summary

        # Handle goal completion based on workflow status
        success_statuses = [MultiObjectPickAndPlace.Result.SUCCESS,
                            MultiObjectPickAndPlace.Result.PARTIAL_SUCCESS]
        failure_statuses = [MultiObjectPickAndPlace.Result.FAILED,
                            MultiObjectPickAndPlace.Result.INCOMPLETE]

        if workflow_status in success_statuses:
            goal_handle.succeed()
        elif workflow_status in failure_statuses:
            goal_handle.abort()
        else:
            goal_handle.abort()

        self.is_orchestrator_busy = False
        return result

    def _goal_callback(self, goal_request):
        """Handle goal request from action client."""
        self.logger.debug('Received goal request')

        # Check if the orchestrator is busy
        if self.is_orchestrator_busy:
            self.logger.warning('Orchestrator is already running, rejecting goal')
            return GoalResponse.REJECT

        # Extract goal data
        goal_mode = goal_request.mode
        goal_target_poses = goal_request.target_poses.poses
        goal_class_ids = goal_request.class_ids

        # Validate goal data
        is_valid = self._validate_goal_data(goal_mode, goal_target_poses, goal_class_ids)
        if not is_valid:
            self.logger.warning('Goal validation failed, rejecting goal')
            return GoalResponse.REJECT

        # Accept the goal
        self.logger.debug(
            f'Accepting goal: mode={goal_mode}, poses={len(goal_target_poses)}, '
            f'class_ids={len(goal_class_ids)}')
        self.is_orchestrator_busy = True
        return GoalResponse.ACCEPT

    def _cancel_callback(self, cancel_request):
        """Handle cancel request from action client."""
        self.logger.info('Received cancellation request')
        return CancelResponse.ACCEPT

    def run(self) -> None:
        """Execute main loop based on debug mode."""
        if self.manual_mode:
            self.logger.info('Starting manual mode execution')
            self._run_debug_mode()
        else:
            self.logger.info('Starting normal mode execution')
            self._run_normal_mode()

    def _run_debug_mode(self) -> None:
        """Run in debug mode - manual ticking with user input."""
        try:
            self.logger.info('Press Enter to tick the tree (Ctrl+C to exit)')
            while not self._shutdown_event.is_set():
                try:
                    input()
                    if self.print_ascii_tree:
                        self.logger.debug(f'Iteration {self.iteration + 1}')
                    self.iteration += 1
                    try:
                        if self.tree:
                            result = self.tree.tick()
                            if self.print_ascii_tree:
                                self.logger.debug(f'Tree tick result: {result}')
                    except Exception as e:
                        self.logger.error(f'Error during tree tick: {e}')
                except KeyboardInterrupt:
                    break

        except KeyboardInterrupt:
            self.logger.info('Shutting down...')
        finally:
            self.shutdown()

    def _run_normal_mode(self) -> None:
        """Run in normal mode - continuous ticking."""
        try:
            # Start ticking the tree with 100Hz
            self.tree.tick_tock(10.0)
            while rclpy.ok() and not self._shutdown_event.is_set():
                # Do nothing as the executor is spinning in the background
                pass

        except KeyboardInterrupt:
            self.logger.info('KeyboardInterrupt, shutting down.\n')
        finally:
            self.logger.debug('Shutting down normal mode')
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self.logger.info('Starting shutdown sequence')
        self._shutdown_event.set()

        # Interrupt and destroy components
        if self.tree:
            self.logger.debug('Interrupting behavior tree')
            self.tree.interrupt()

        if self.tree:
            self.logger.debug('Shutting down behavior tree')
            self.tree.shutdown()

        if self.action_server:
            self.logger.debug('Destroying action server')
            self.action_server.destroy()

        if self.thread and self.thread.is_alive():
            self.logger.debug('Waiting for thread to join')
            self.thread.join(timeout=2.0)  # Add timeout to prevent hanging

        # Clean shutdown sequence
        if self.executor:
            self.logger.debug('Shutting down executor')
            self.executor.shutdown()

        self.logger.info('Shutdown complete')


def main():
    """Execute main entry point for the multi-object pick and place orchestrator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Multi-object pick and place behavior tree')
    parser.add_argument('--print-ascii-tree', action='store_true',
                        help='Enable ASCII tree display and iteration counts on terminal')
    parser.add_argument('--manual-mode', action='store_true',
                        help='Enable manual mode. In this mode, the behavior tree will be ticked'
                             'every time the user presses a key.')
    parser.add_argument('--log-level', type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'], default='info',
                        help='Set explicit log level (overrides verbose/manual flags)')
    parser.add_argument('--behavior_tree_config_file', type=str,
                        default='multi_object_pick_and_place_behavior_tree_params.yaml',
                        help='Path to the behavior tree configuration file')
    parser.add_argument('--blackboard_config_file', type=str,
                        default='multi_object_pick_and_place_blackboard_params.yaml',
                        help='Path to the blackboard configuration file')

    args = parser.parse_args()

    # Create and run the orchestrator
    orchestrator = MultiObjectPickPlaceOrchestrator(
        behavior_tree_config_file=args.behavior_tree_config_file,
        blackboard_config_file=args.blackboard_config_file,
        print_ascii_tree=getattr(args, 'print_ascii_tree', False),
        manual_mode=getattr(args, 'manual_mode', False),
        log_level=getattr(args, 'log_level', None),
    )
    try:
        orchestrator.run()
    except Exception as e:
        # Get logger for main function
        logger = logging.getLogger(__name__)
        logger.error(f'Error running orchestrator: {e}')
        orchestrator.shutdown()


if __name__ == '__main__':
    main()
