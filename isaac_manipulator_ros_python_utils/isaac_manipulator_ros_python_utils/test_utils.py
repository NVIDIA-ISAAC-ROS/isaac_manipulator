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
"""Test utils functions for Isaac Manipulator."""

from datetime import datetime
import json
import os
import pathlib
import re
import subprocess
from threading import Event, Thread
import time
import traceback
from typing import Any, Dict, List, Tuple

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from isaac_manipulator_interfaces.action import (
    AddSegmentationMask, GetObjectPose, GetObjects, Insert, PickAndPlace
)
from isaac_manipulator_interfaces.srv import AddMeshToObject, ClearObjects
from isaac_manipulator_ros_python_utils.config import load_yaml_params
import isaac_manipulator_ros_python_utils.geometry as geometry_utils
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import matplotlib.pyplot as plt
import numpy as np
from rcl_interfaces.msg import Parameter as RCLPYParameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as RCLPYNode
from rclpy.parameter import Parameter
from rclpy.qos import QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float64, Header
import tf2_ros
import yaml


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


def get_params_from_config_file_set_in_env(run_test: bool) -> Dict:
    """
    Get parameters from a YAML file set in the environment variable ISAAC_MANIPULATOR_TEST_CONFIG.

    Args
    ----
        run_test (bool): Whether to run the test, if no, then return empty dict

    Returns
    -------
        Dict: Dictionary of parameters

    Raises
    ------
        FileNotFoundError: If the file is not found
        RuntimeError: If the environment variable is not set

    """
    if not run_test:
        return {}

    env_config_path = os.environ.get('ISAAC_MANIPULATOR_TEST_CONFIG')

    if env_config_path is None:
        raise RuntimeError('ISAAC_MANIPULATOR_TEST_CONFIG environment variable is not set')

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f'File not found at {env_config_path}')

    params = load_yaml_params(env_config_path)

    return params


class QosTestBaseClass(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose integration."""

    _filepath: pathlib.Path = pathlib.Path(os.path.dirname(__file__))
    _excluded_topics: list[str] = []
    _topics_with_sensor_data_qos: list[str] = []
    _check_sensor_data_for_topics_for_only_this_node: dict[str, str] = {}
    _check_sensor_data_for_topics_that_have_publisher: list[str] = []
    _run_test: bool = False

    def test_qos_settings(self):
        """Test that all QoS settings are RELIABLE when object following is enabled."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return
        # Get all available topics
        topics_and_types = self.node.get_topic_names_and_types()
        error_out = False
        should_be_best_effort_but_is_reliable = []
        should_be_reliable_but_is_best_effort = []
        for topic_name, _ in topics_and_types:
            if topic_name in self._excluded_topics:
                continue

            self.node.get_logger().warn(f'Checking topic {topic_name}')
            if topic_name in self._check_sensor_data_for_topics_for_only_this_node:
                qos_profile = self._get_qos_profile(
                    topic_name, self._check_sensor_data_for_topics_for_only_this_node[topic_name])
            elif topic_name in self._check_sensor_data_for_topics_that_have_publisher:
                qos_profile = self._get_qos_profile(
                    topic_name, None, is_publisher=True)
            else:
                qos_profile = self._get_qos_profile(topic_name)
            if topic_name in self._topics_with_sensor_data_qos or \
                    topic_name in self._check_sensor_data_for_topics_that_have_publisher:
                if qos_profile is None:
                    self.node.get_logger().warn(f'Topic {topic_name} has no qos profile')
                    continue
                if qos_profile.reliability != QoSReliabilityPolicy.BEST_EFFORT:
                    should_be_best_effort_but_is_reliable.append(topic_name)
                    error_out = True
                    continue
            else:
                if qos_profile is None:
                    self.node.get_logger().warn(f'Topic {topic_name} has no qos profile')
                    continue
                if qos_profile.reliability != QoSReliabilityPolicy.RELIABLE:
                    should_be_reliable_but_is_best_effort.append(topic_name)
                    error_out = True
                    continue

        if error_out:
            self.fail(f'Error in QOS settings:'
                      f' Should be BEST EFFORT: {should_be_best_effort_but_is_reliable}'
                      f' Should be RELIABLE: {should_be_reliable_but_is_best_effort}')

    def _get_qos_profile(self, topic_name: str, node_name: str = None,
                         is_publisher: bool = False) -> QoSReliabilityPolicy | None:
        """
        Check if a topic is using RELIABLE QoS policy.

        Parameters
        ----------
        topic_name : str
            The name of the topic to check
        node_name : str
            The name of the node to check
        is_publisher : bool
            Whether the topic is a publisher

        Returns
        -------
        QoSReliabilityPolicy | None:
            Returns the QoS policy of the topic, otherwise None

        """
        topic_info = self.node.get_publishers_info_by_topic(topic_name)
        if not topic_info:
            topic_info = self.node.get_subscriptions_info_by_topic(topic_name)
            if not topic_info:
                self.node.get_logger().warn(
                    f'No publishers or subscribers found for topic {topic_name}')
                return None

        qos_profile = None
        for idx, info in enumerate(topic_info):
            if info.node_name != node_name and node_name is not None:
                continue
            if is_publisher:
                if info.endpoint_type != 'PUBLISHER':
                    continue
            self.node.get_logger().warn(f'Topic {topic_name} has info for {idx}: {info}')
            # find out if all topics are using the same qos profile
            if qos_profile is None:
                qos_profile = info.qos_profile
            elif qos_profile.reliability != info.qos_profile.reliability:
                self.node.get_logger().warn(f'Topic {topic_name} has qos profile:'
                                            f'{info.qos_profile} but it should be {qos_profile}')
                self.fail(f'Topic {topic_name} has qos profile:'
                          f'{info.qos_profile} but it should be {qos_profile}')

        return qos_profile


class IsaacROSFoundationPoseQosTest(QosTestBaseClass):
    """Test for Isaac ROS Foundation Pose + RT-DETR QOS settings."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  excluded_topics: list[str],
                                  topics_with_sensor_data_qos: list[str],
                                  check_sensor_data_for_topics_for_only_this_node: dict[str, str],
                                  check_sensor_data_for_topics_that_have_publisher: list[str],
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._excluded_topics = excluded_topics
        cls._topics_with_sensor_data_qos = topics_with_sensor_data_qos
        cls._check_sensor_data_for_topics_for_only_this_node = \
            check_sensor_data_for_topics_for_only_this_node
        cls._check_sensor_data_for_topics_that_have_publisher = \
            check_sensor_data_for_topics_that_have_publisher
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class TopicConnectionsGraphTest(IsaacROSBaseTest):
    """Test for graph structure of Foundation Pose and RT-DETR integration."""

    filepath = pathlib.Path(os.path.dirname(__file__))
    _run_test: bool = False

    def test_graph_structure(self):
        """Test that the graph structure matches the expected structure."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return
        # Get all nodes and their relationships
        actual_graph = self._build_actual_graph()
        expected_graph = self._get_expected_graph()

        # Save the graphs for debugging
        self.node.get_logger().info(f'Actual graph: {actual_graph}')

        # Compare the actual graph with the expected graph
        errors = self._compare_graphs(actual_graph, expected_graph)

        # Output errors in a structured way
        if errors:
            self.fail(f'Graph structure validation failed with {len(errors)} errors')

    def _compare_graphs(self, actual: Dict, expected: Dict) -> List[str]:
        """
        Compare the actual and expected graphs and return a list of errors.

        Args:
            actual (Dict): The actual graph
            expected (Dict): The expected graph

        Returns
        -------
            List[str]: A list of errors

        """
        errors = []

        # Check for missing nodes
        missing_nodes = []
        for expected_node in expected.keys():
            node_name = expected_node.split('/')[-1] if '/' in expected_node else expected_node
            if expected_node not in actual:
                missing_nodes.append(
                    f'Missing node: {node_name} - This node was in the expected graph but'
                    f' was not found in the running system')

        if missing_nodes:
            errors.append('\nMISSING NODES:')
            errors.extend(missing_nodes)

        # Check for missing publishers
        missing_publishers = []
        for node_name, node_data in expected.items():
            if node_name not in actual:
                continue

            for expected_pub in node_data.get('publishes', []):
                expected_topic = expected_pub.get('topic')
                expected_subscribers = expected_pub.get('to', [])

                # Find this topic in actual publishers
                found_topic = False
                for actual_pub in actual[node_name].get('publishes', []):
                    if actual_pub.get('topic') == expected_topic:
                        found_topic = True
                        actual_subscribers = actual_pub.get('to', [])

                        # Check missing subscribers
                        for expected_sub in expected_subscribers:
                            if expected_sub not in actual_subscribers:
                                missing_publishers.append(
                                    f'Missing subscriber: {expected_sub} is not subscribed'
                                    f' to {expected_topic} from {node_name}')

                        break

                if not found_topic:
                    missing_publishers.append(
                        f'Missing publication: {node_name} does not publish to {expected_topic}')

        if missing_publishers:
            errors.append('\nMISSING PUBLISHERS:')
            errors.extend(missing_publishers)

        # Check for missing subscribers
        missing_subscribers = []
        for node_name, node_data in expected.items():
            if node_name not in actual:
                continue

            for expected_sub in node_data.get('subscribes', []):
                expected_topic = expected_sub.get('topic')
                expected_publishers = expected_sub.get('from', [])

                # Find this topic in actual subscribers
                found_topic = False
                for actual_sub in actual[node_name].get('subscribes', []):
                    if actual_sub.get('topic') == expected_topic:
                        found_topic = True
                        actual_publishers = actual_sub.get('from', [])

                        # Check missing publishers
                        for expected_pub in expected_publishers:
                            if expected_pub not in actual_publishers:
                                missing_subscribers.append(
                                    f'Missing publisher: {expected_pub} does not publish to'
                                    f' {expected_topic} for {node_name}')

                        break

                if not found_topic:
                    missing_subscribers.append(
                        f'Missing subscription: {node_name} does not subscribe to'
                        f' {expected_topic}')

        if missing_subscribers:
            errors.append('\nMISSING SUBSCRIBERS:')
            errors.extend(missing_subscribers)

        # Log the comparison results
        if errors:
            self.node.get_logger().error('\nGraph Structure Validation Errors:')
            self.node.get_logger().error('==================================================')
            for error in errors:
                self.node.get_logger().error(error)
        else:
            self.node.get_logger().info(
                'Graph structure validation passed. All expected nodes and connections found.')

        return errors

    def _get_node_names_and_namespaces(self) -> List[Tuple[str, str]]:
        """
        Get node names and namespaces using ros2 CLI commands.

        This function provides similar functionality to the ROS 2 API's
        get_node_names_and_namespaces() but uses subprocess with ros2 CLI
        commands to work around limitations in the ROS 2 introspection API
        for composable nodes.

        Returns
        -------
            List[Tuple[str, str]]: List of tuples containing (node_name, namespace)

        """
        self.node.get_logger().info('Getting node names via ros2 CLI...')

        try:
            # Run ros2 node list to get all node names
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True,
                env=os.environ.copy(),  # Use same environment as test
                timeout=10
            )

            # Check if command was successful
            if result.returncode != 0:
                self.node.get_logger().error(f'ros2 node list failed: {result.stderr}')
                return []

            # Parse the node names into the format expected by ROS 2 API
            node_list = [n.strip() for n in result.stdout.splitlines() if n.strip()]
            self.node.get_logger().info(f'Found {len(node_list)} nodes: {node_list}')

            # Parse node names into (node_name, namespace) tuples
            parsed_nodes = []
            for node_path in node_list:
                # Remove leading '/' if present for processing
                clean_path = node_path[1:] if node_path.startswith('/') else node_path

                # Split the node path into parts
                parts = clean_path.split('/')

                if len(parts) == 1:  # Just a node name, no namespace
                    node_name = parts[0]
                    namespace = '/'  # Root namespace
                else:
                    # Last part is the node name, the rest is the namespace
                    node_name = parts[-1]
                    namespace = '/' + '/'.join(parts[:-1])

                parsed_nodes.append((node_name, namespace))

            self.node.get_logger().info(f'Parsed nodes: {parsed_nodes}')
            return parsed_nodes

        except subprocess.TimeoutExpired:
            self.node.get_logger().error('Timeout while getting node names')
            return []
        except Exception as e:
            self.node.get_logger().error(f'Error getting node names: {str(e)}')
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return []

    def _build_actual_graph(self) -> Dict:
        """
        Build a graph of the actual node connections using optimized CLI commands.

        This version is much faster as it gets all topic information at once
        rather than making separate calls for each node and topic.

        Returns
        -------
            Dict: A dictionary representation of the actual node connections

        """
        # Initialize the graph
        graph = {}

        # Get all nodes in the ROS system
        nodes = self._get_node_names_and_namespaces()
        self.node.get_logger().info(f'Found {len(nodes)} nodes through CLI')

        # Create a dictionary of nodes by their fully qualified name AND simple name
        node_by_fqn = {}
        node_by_simple_name = {}  # NEW: Dictionary to look up by simple name

        for node_name, namespace in nodes:
            fqn = f'{namespace}/{node_name}' if namespace != '/' else f'/{node_name}'

            # Skip this test node and ROS internal nodes
            if (node_name == self.node.get_name() or
                node_name in ['_ros2cli_daemon_0'] or
                'launch_ros_' in node_name or
                    'parameter_bridge' in node_name):
                continue

            node_by_fqn[fqn] = (node_name, namespace)
            node_by_simple_name[node_name] = fqn  # NEW: Map simple name to FQN

            # Initialize node in graph
            graph[fqn] = {
                'publishes': [],
                'subscribes': []
            }

        self.node.get_logger().info(f'Nodes being tracked: {list(graph.keys())}')

        # Get all topics at once without types
        self.node.get_logger().info('Getting all topics at once...')
        try:
            # Run command to get list of all topics WITHOUT types
            topic_result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if topic_result.returncode != 0:
                self.node.get_logger().error(f'ros2 topic list failed: {topic_result.stderr}')
                return graph

            # Parse topics - each line is a topic name
            topics = []
            for line in topic_result.stdout.splitlines():
                topic = line.strip()

                # Skip certain topics
                if (topic.startswith('/parameter') or
                    topic.startswith('/rosout') or
                        '/nitros/_supported_types' in topic):
                    continue

                topics.append(topic)

            self.node.get_logger().info(f'Found {len(topics)} topics to process')
            self.node.get_logger().info(f'Topics: {topics[:10]}...')  # Log first 10 topics

            # Sample a few topics for detailed investigation
            sample_topics = topics[:5]  # Take first 5 topics
            self.node.get_logger().info(f'Investigating sample topics in detail: {sample_topics}')

            # Get info for all topics
            topic_connection_count = 0
            for topic in topics:
                # Get detailed info for this topic - use ONLY the topic name
                try:
                    self.node.get_logger().debug(f'Getting info for topic: {topic}')
                    # Use a shorter timeout since there are many topics
                    info_result = subprocess.run(
                        ['ros2', 'topic', 'info', '-v', topic],
                        capture_output=True,
                        text=True,
                        timeout=3)

                    if info_result.returncode != 0:
                        self.node.get_logger().warning(
                            f'Could not get info for topic {topic}: {info_result.stderr}')
                        continue

                    # For sample topics, log the entire output
                    if topic in sample_topics:
                        self.node.get_logger().info(
                            f'Sample topic {topic} info output:\n{info_result.stdout}')

                    # Parse publisher and subscriber nodes
                    publishers = []
                    subscribers = []

                    # Split the output into sections
                    info_text = info_result.stdout

                    # Use regex to find all "Node name:" and "Node namespace:" pairs
                    # followed by "Endpoint type:"
                    pattern = (
                        r'Node name: (.+?)\n'
                        r'Node namespace: (.+?)[\s\S]+?'
                        r'Endpoint type: (PUBLISHER|SUBSCRIPTION)'
                    )
                    matches = re.finditer(pattern, info_text)

                    for match in matches:
                        simple_node_name = match.group(1).strip()
                        node_namespace = match.group(2).strip()
                        endpoint_type = match.group(3).strip()

                        # Construct the FQN - same as what we have in our graph
                        fqn = f'{node_namespace}/{simple_node_name}' if node_namespace != '/' \
                              else f'/{simple_node_name}'

                        # Check if this node is in our graph
                        if fqn in graph:
                            if endpoint_type == 'PUBLISHER':
                                publishers.append(fqn)
                            elif endpoint_type == 'SUBSCRIPTION':
                                subscribers.append(fqn)
                        elif simple_node_name in node_by_simple_name:
                            # Try using the mapped FQN if simple name is known
                            mapped_fqn = node_by_simple_name[simple_node_name]
                            if endpoint_type == 'PUBLISHER':
                                publishers.append(mapped_fqn)
                            elif endpoint_type == 'SUBSCRIPTION':
                                subscribers.append(mapped_fqn)

                    # Log what we found for this topic
                    if publishers or subscribers:
                        self.node.get_logger().info(
                            f'Topic {topic} has {len(publishers)} publishers '
                            f'and {len(subscribers)} subscribers')
                        self.node.get_logger().info(f'Publishers: {publishers}')
                        self.node.get_logger().info(f'Subscribers: {subscribers}')

                        # Update the graph with this topic's connections
                        for pub_node in publishers:
                            if pub_node in graph:
                                graph[pub_node]['publishes'].append({
                                    'topic': topic,
                                    'to': subscribers
                                })

                        for sub_node in subscribers:
                            if sub_node in graph:
                                graph[sub_node]['subscribes'].append({
                                    'topic': topic,
                                    'from': publishers
                                })
                        topic_connection_count += 1
                except Exception as e:
                    self.node.get_logger().error(f'Error processing topic {topic}: {str(e)}')
                    self.node.get_logger().error(traceback.format_exc())

        except Exception as e:
            self.node.get_logger().error(f'Error building graph: {str(e)}')
            self.node.get_logger().error(traceback.format_exc())

        self.node.get_logger().info(f'Total topics with connections: {topic_connection_count}')
        self.node.get_logger().info(f'Final graph has {len(graph)} nodes')

        # Check each node's connections
        for node, connections in graph.items():
            pub_count = len(connections['publishes'])
            sub_count = len(connections['subscribes'])
            if pub_count > 0 or sub_count > 0:
                self.node.get_logger().info(
                    f'Node {node} has {pub_count} publishers and {sub_count} subscribers')
            else:
                self.node.get_logger().warning(f'Node {node} has NO connections')

        return graph

    def _get_expected_graph(self) -> Dict:
        """
        Define the expected graph structure.

        Returns
        -------
            Dict: A dictionary representation of the expected node connections

        """
        # Create the expected graph with all nodes and connections
        raise NotImplementedError('This function should be implemented by the test class')


class FoundationPoseTopicConnectionsGraphTest(TopicConnectionsGraphTest):
    """Test for Isaac ROS Foundation Pose + RT-DETR topic connections."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class PoseEstimationPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _monitor_topic_name: str = ''
    _max_latency_time: float = 1.5
    _num_messages_to_receive: int = 10
    _use_sim_time: bool = False
    _message_type: Any = None
    _run_test: bool = False

    def test_for_pose_estimation_pol(self):
        """Test that verifies that pose estimation are coming in at a certain rate."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return
        self.node.get_logger().info('Starting test for pose estimation POL')
        received_messages = {}

        subs = self.create_logging_subscribers(
            [(self._monitor_topic_name, self._message_type)], received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True)

        try:
            message_times = []
            start_time = time.time()
            received_msg = False
            while len(message_times) < self._num_messages_to_receive:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if time.time() - start_time > 20:
                    break

                if self._monitor_topic_name in received_messages and \
                   len(received_messages[self._monitor_topic_name]) > 0:
                    self.node.get_logger().info('Received message')
                    received_msg = True
                    # Get current ROS time when message was received
                    current_time = self.node.get_clock().now()

                    # Get timestamp from most recent message
                    msg = received_messages[self._monitor_topic_name][-1]
                    msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

                    time_diff = abs((current_time - msg_time).nanoseconds / 1e9)
                    message_times.append(time_diff)

                    self.node.get_logger().info(
                        f'Message {len(message_times)} received. '
                        f'Time difference: {time_diff:.3f} seconds')
                    received_messages[self._monitor_topic_name] = []

            # Verify all messages were within acceptable time difference
            max_diff = max(message_times)
            self.assertTrue(received_msg, 'No messages received for test duration')
            self.assertLess(
                max_diff, self._max_latency_time,
                f'Maximum time difference ({max_diff:.3f}s) exceeded limit of'
                f' {self._max_latency_time}s')

            self.node.get_logger().info(
                f'Successfully received {self._num_messages_to_receive} messages within'
                f' time constraints')

        finally:
            self.node.destroy_subscription(subs)


class IsaacROSFoundationPoseEstimationPolTest(PoseEstimationPolTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  monitor_topic_name: str,
                                  max_latency_time: float,
                                  num_messages_to_receive: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  message_type: Any,
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._monitor_topic_name = monitor_topic_name
        cls._max_latency_time = max_latency_time
        cls._num_messages_to_receive = num_messages_to_receive
        cls._use_sim_time = use_sim_time
        cls._message_type = message_type

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
            )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class ObjectDetectionActionClient(RCLPYNode):
    """This class will fire off a action call for object detection."""

    def __init__(self):
        super().__init__('test_object_detection_action_client')
        self._action_client = ActionClient(self, GetObjects, '/get_objects')
        self._goal_handle = None
        self.result_received = False
        self.result_future = None
        self.detected_object_id = None

    def send_goal(self):
        """Send a goal to the object detection action server."""
        self.get_logger().info('Waiting for action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        self.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by the action server.')
        self.result_future = self._goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self.detected_object_id = result.objects[0].object_id
            object_ids = []
            for obj in result.objects:
                object_ids.append(obj.object_id)
            self.get_logger().info(f'Objects detected with ids: {object_ids}')
            self.result_received = True
        else:
            self.get_logger().error(f'Action failed with status: {status}')
            self.result_received = False


class ObjectPoseEstimationActionClient(RCLPYNode):
    """This class will fire off an action call for object pose."""

    def __init__(self):
        super().__init__('test_object_pose_estimation_action_client')
        self._action_client = ActionClient(self, GetObjectPose, '/get_object_pose')
        self._goal_handle = None
        self.result_received = False
        self.result_future = None
        self.result = None

    def send_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.get_logger().info('Waiting for pose estimation action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Pose estimation action server not available after waiting')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.get_logger().info('Pose estimation goal accepted by the action server.')
        self.result_future = self._goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Object pose received: {result.object_pose}')
            self.result_received = True
            self.result = result
        else:
            self.get_logger().error(f'Pose estimation action failed with status: {status}')
            self.result_received = False
            self.result_future = None
            self._goal_handle = None
            self.result = None


class PoseEstimationServersPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_perception_requests: int = 10
    _use_sim_time: bool = False
    _run_test: bool = False
    _spin_thread: Thread = None
    _executor: MultiThreadedExecutor = None
    _detection_client: ObjectDetectionActionClient = None
    _pose_client: ObjectPoseEstimationActionClient = None
    _output_dir: str = ''
    _identifier: str = ''

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self._detection_client = ObjectDetectionActionClient()
        self._pose_client = ObjectPoseEstimationActionClient()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._detection_client)
        self._executor.add_node(self._pose_client)

        # Start spinning in a separate thread
        self._spin_thread = Thread(target=self._executor.spin)
        self._spin_thread.start()

    def test_for_pose_estimation_servers_pol(self):
        """Test that verifies that pose estimation are coming in at a certain rate."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setUpClients()

        self.node.get_logger().info('Starting test for pose estimation servers POL')
        failure_count = 0
        time.sleep(10)
        poses = []
        for index in range(self._num_perception_requests):
            time.sleep(1)
            # First detect objects
            self._detection_client.send_goal()

            # Wait for the detection result with a timeout
            start_time = self._detection_client.get_clock().now()
            timed_out = False
            while rclpy.ok() and not self._detection_client.result_received:
                time_node_in_node = self._detection_client.get_clock().now()
                max_timeout = self._max_timeout_time_for_action_call
                if (time_node_in_node - start_time).nanoseconds / 1e9 > max_timeout:
                    if self._allow_failure_on_first_try:
                        self.node.get_logger().warn('Allowing failure on first try')
                        timed_out = True
                        break
                    self.fail('Timeout waiting for object detection result.')
                    break
                rclpy.spin_once(self._detection_client, timeout_sec=0.1)

            if not self._allow_failure_on_first_try:
                self.assertTrue(self._detection_client.result_received,
                                'Object detection action did not return successfully')
            elif timed_out:
                failure_count += 1
                continue

            # Now that we've detected an object, get its pose
            if self._detection_client.detected_object_id is not None:
                self._pose_client.send_goal(self._detection_client.detected_object_id)
                timed_out = False
                # Wait for the pose estimation result with a timeout
                start_time = self._pose_client.get_clock().now()
                while rclpy.ok() and not self._pose_client.result_received:
                    time_node_in_node = self._pose_client.get_clock().now()
                    max_timeout = self._max_timeout_time_for_action_call
                    if (time_node_in_node - start_time).nanoseconds / 1e9 > max_timeout:
                        timed_out = True
                        failure_count += 1
                        break
                    rclpy.spin_once(self._pose_client, timeout_sec=0.1)
                if not self._pose_client.result_received and not timed_out:
                    failure_count += 1
            else:
                pass

            # Save the pose estimation result
            if self._pose_client.result_received:
                pose_estimation_result = self._pose_client.result
                self.node.get_logger().info(f'Pose estimation result: {pose_estimation_result}')
                pose_obj = {
                    'position': {
                        'x': pose_estimation_result.object_pose.position.x,
                        'y': pose_estimation_result.object_pose.position.y,
                        'z': pose_estimation_result.object_pose.position.z,
                    },
                    'orientation': {
                        'x': pose_estimation_result.object_pose.orientation.x,
                        'y': pose_estimation_result.object_pose.orientation.y,
                        'z': pose_estimation_result.object_pose.orientation.z,
                        'w': pose_estimation_result.object_pose.orientation.w,
                    },
                    'frame_id': 'camera_1_color_optical_frame',
                    'pose_name_for_tf': f'{self._identifier}_pose_estimate_{index}',
                }
                poses.append(pose_obj)

            # Reset for next iteration
            self._detection_client.result_received = False
            self._pose_client.result_received = False

        self.node.get_logger().info(f'Failure count: {failure_count}')
        self._executor.shutdown()
        self._spin_thread.join()

        # Save the poses to a file
        with open(os.path.join(self._output_dir, 'poses.json'), 'w') as f:
            json.dump(poses, f)

        if failure_count == 1 and self._allow_failure_on_first_try:
            self.node.get_logger().warn('Allowing failure on first try')
            return

        assert failure_count == 0, f'Pose estimation action failed' \
            f'{failure_count}/{self._num_perception_requests} times'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  num_perception_requests: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  output_dir: str,
                                  identifier: str,
                                  node_startup_delay: float,
                                  allow_failure_on_first_try: bool = False):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._num_perception_requests = num_perception_requests
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._identifier = identifier
        cls._allow_failure_on_first_try = allow_failure_on_first_try

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class IsaacManipulatorLoadEstimationPolTest(IsaacROSBaseTest):
    """Test for measuring Isaac Manipulator performance under system load."""

    DEFAULT_NAMESPACE = ''

    # Performance monitoring settings
    _test_duration_seconds: int  # Run test for 1 minute
    _output_dir: str  # Default output directory
    _monitor_topic_name: str = ''
    _message_type: Any = None
    _run_test: bool = False
    _use_sim_time: bool = False
    _robot_segmenter_topic_names: list[str] = []
    _max_latency_in_robot_segmentor_ms: int = 5000

    def test_performance_under_load(self):
        """Test performance metrics of pose estimation under system load."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting performance monitoring test')
        received_messages = {}

        self._all_topic_names = [self._monitor_topic_name] + self._robot_segmenter_topic_names

        # Data collection structures
        latencies = {topic_name: [] for topic_name in self._all_topic_names}
        timestamps = {topic_name: [] for topic_name in self._all_topic_names}
        msg_counts = dict.fromkeys(self._all_topic_names, 0)
        BIG_NUMBER = 1000000
        last_received_time = dict.fromkeys(self._all_topic_names, BIG_NUMBER)

        # Monitoring parameters
        start_time = time.time()
        latest_time = start_time
        sample_interval = 0.1  # seconds

        # Create tuple for subscriptions for robot segmenter topics
        robot_segmenter_subs = [
            (topic_name, Image) for topic_name in self._robot_segmenter_topic_names]

        # Create subscription to monitor messages
        subs = self.create_logging_subscribers(
            [(self._monitor_topic_name, self._message_type)] + robot_segmenter_subs,
            received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)
        failure_due_to_latency = False
        failure_latency_msg = ''
        failure_due_to_hz = False
        failure_hz_msg = ''
        try:
            # Run test for specified duration
            while time.time() - start_time < self._test_duration_seconds:
                # Spin node to process callbacks
                rclpy.spin_once(self.node, timeout_sec=sample_interval)

                current_ros_time = self.node.get_clock().now()
                elapsed = time.time() - start_time

                # Log progress every 5 seconds
                if time.time() - latest_time >= 5.0:
                    self.node.get_logger().info(
                        f'Test running: {elapsed:.1f}s / {self._test_duration_seconds}s, '
                        f'Messages for pose estimation: {msg_counts[self._monitor_topic_name]}')
                    latest_time = time.time()

                # Process received messages
                for monitor_topic_name in self._all_topic_names:
                    if monitor_topic_name in received_messages and \
                       len(received_messages[monitor_topic_name]) > 0:
                        # Get all new messages since last check
                        new_messages = received_messages[monitor_topic_name]

                        for msg in new_messages:
                            # Get timestamp from message
                            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

                            # Calculate latency
                            latency = abs((current_ros_time - msg_time).nanoseconds / 1e9)

                            # Store metrics
                            latencies[monitor_topic_name].append(latency)
                            timestamps[monitor_topic_name].append(elapsed)
                            msg_counts[monitor_topic_name] += 1
                            latency_in_ms = latency * 1000
                            # This one makes sure topic is still ticking
                            if last_received_time[monitor_topic_name] != BIG_NUMBER and \
                               monitor_topic_name in self._robot_segmenter_topic_names:
                                if latency_in_ms > self._max_latency_in_robot_segmentor_ms:
                                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                                        f'{latency_in_ms} ms, which is greater than' \
                                        f'{self._max_latency_in_robot_segmentor_ms} ms'
                                    self.node.get_logger().error(error_msg)
                                    failure_due_to_hz = True
                                    failure_hz_msg = error_msg

                            last_received_time[monitor_topic_name] = time.time()

                            # This one makes sure latency is within acceptable range
                            if monitor_topic_name in self._robot_segmenter_topic_names:
                                if latency_in_ms > self._max_latency_in_robot_segmentor_ms:
                                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                                        f'{latency_in_ms} ms, which is greater than' \
                                        f'{self._max_latency_in_robot_segmentor_ms} ms'
                                    self.node.get_logger().error(error_msg)
                                    failure_due_to_latency = True
                                    failure_latency_msg = error_msg

                            self.node.get_logger().debug(
                                f'Message {msg_counts[monitor_topic_name]} received.'
                                f'Latency: {latency:.3f} seconds')

                        # Clear processed messages to avoid reprocessing
                        received_messages[monitor_topic_name] = []

            # Generate and save performance visualizations
            self._generate_performance_graphs(latencies, timestamps, msg_counts)

            # Calculate and log statistics
            self._log_performance_statistics(latencies)

            # Also do one final check to make sure topics all ticked
            for monitor_topic_name in self._all_topic_names:
                # Only do it for robot segmenter topics
                if monitor_topic_name not in self._robot_segmenter_topic_names:
                    continue
                if last_received_time[monitor_topic_name] == BIG_NUMBER:
                    self.node.get_logger().error(
                        f'Topic {monitor_topic_name} did not receive any messages')
                    self.fail(f'Topic {monitor_topic_name} did not receive any messages')

                latency_in_ms = (time.time() - last_received_time[monitor_topic_name]) * 1000

                # Have it as 5 times of max latency to make sure topic has not stopped publishing
                if latency_in_ms > 5 * self._max_latency_in_robot_segmentor_ms:
                    error_msg = f'Latency for topic {monitor_topic_name} is' \
                        f'{latency_in_ms} ms, which is greater than' \
                        f'{5 * self._max_latency_in_robot_segmentor_ms} ms'
                    self.node.get_logger().error(error_msg)
                    self.fail(error_msg)

            if failure_due_to_latency:
                self.node.get_logger().error(failure_latency_msg)
                self.fail(failure_latency_msg)

            if failure_due_to_hz:
                self.node.get_logger().error(failure_hz_msg)
                self.fail(failure_hz_msg)

        finally:
            self.node.destroy_subscription(subs)

    def _generate_performance_graphs(self, latencies: dict[str, list[float]],
                                     timestamps: dict[str, list[float]],
                                     msg_counts: dict[str, int]):
        """Generate and save performance visualization graphs."""
        for monitor_topic_name in self._all_topic_names:
            if not latencies[monitor_topic_name]:
                self.node.get_logger().error('No messages received during test')
                return

            # Create output directory if it doesn't exist
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Remove / from monitor_topic_name
            monitor_topic_name_str = monitor_topic_name.replace('/', '_')
            output_file = os.path.join(self._output_dir,
                                       f'performance_{monitor_topic_name_str}_{timestamp_str}.png')

            # Create figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Plot 1: Latency over time
            axs[0].plot(timestamps[monitor_topic_name],
                        latencies[monitor_topic_name], 'b-', linewidth=1)
            axs[0].set_title('Message Latency Over Time')
            axs[0].set_ylabel('Latency (seconds)')
            axs[0].grid(True)

            # Plot 2: Message rate
            # Calculate message rate using rolling window
            if len(timestamps[monitor_topic_name]) > 10:
                window_size = 10
                msg_rates = []
                rate_times = []

                for i in range(window_size, len(timestamps[monitor_topic_name])):
                    time_diff = timestamps[monitor_topic_name][i] - \
                        timestamps[monitor_topic_name][i - window_size]
                    if time_diff > 0:
                        rate = window_size / time_diff
                        msg_rates.append(rate)
                        rate_times.append(timestamps[monitor_topic_name][i])

                axs[1].plot(rate_times, msg_rates, 'g-', linewidth=1)
            else:
                axs[1].text(0.5, 0.5, 'Not enough data for rate calculation',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[1].transAxes)

            axs[1].set_title('Message Rate Over Time')
            axs[1].set_ylabel('Messages per second')
            axs[1].grid(True)

            # Plot 3: Jitter (variation in latency)
            if len(latencies[monitor_topic_name]) > 1:
                # Calculate jitter as the difference between consecutive latencies
                jitter = [abs(latencies[monitor_topic_name][i] -
                              latencies[monitor_topic_name][i-1])
                          for i in range(1, len(latencies[monitor_topic_name]))]
                jitter_times = timestamps[monitor_topic_name][1:]

                axs[2].plot(jitter_times, jitter, 'r-', linewidth=1)
            else:
                axs[2].text(0.5, 0.5, 'Not enough data for jitter calculation',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[2].transAxes)

            axs[2].set_title('Jitter Over Time')
            axs[2].set_ylabel('Jitter (seconds)')
            axs[2].set_xlabel('Time (seconds)')
            axs[2].grid(True)

            # Add overall title
            plt.suptitle('Performance Metrics', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save figure
            plt.savefig(output_file, dpi=300)
            self.node.get_logger().info(f'Performance graphs saved to: {output_file}')

            # Close figure to free memory
            plt.close(fig)

    def _log_performance_statistics(self, latencies: dict[str, list[float]]):
        """Calculate and log performance statistics."""
        for monitor_topic_name in latencies:
            if not latencies[monitor_topic_name]:
                self.node.get_logger().error('No messages received - cannot calculate statistics')
                return

            # Calculate statistics
            avg_latency = np.mean(latencies[monitor_topic_name])
            min_latency = np.min(latencies[monitor_topic_name])
            max_latency = np.max(latencies[monitor_topic_name])
            std_latency = np.std(latencies[monitor_topic_name])

            if len(latencies[monitor_topic_name]) > 1:
                # Calculate jitter statistics
                jitter = [abs(latencies[monitor_topic_name][i] -
                              latencies[monitor_topic_name][i-1])
                          for i in range(1, len(latencies[monitor_topic_name]))]
                avg_jitter = np.mean(jitter)
                max_jitter = np.max(jitter)
            else:
                avg_jitter = max_jitter = 0

            # Log statistics
            self.node.get_logger().info(
                f'\nPerformance Statistics Summary:\n'
                f'  Total messages: {len(latencies[monitor_topic_name])}\n'
                f'  Message rate: '
                f'{len(latencies[monitor_topic_name]) / self._test_duration_seconds:.2f}'
                f' msg/sec\n'
                f'  Latency (mean): {avg_latency:.6f} sec\n'
                f'  Latency (min): {min_latency:.6f} sec\n'
                f'  Latency (max): {max_latency:.6f} sec\n'
                f'  Latency (std dev): {std_latency:.6f} sec\n'
                f'  Jitter (avg): {avg_jitter:.6f} sec\n'
                f'  Jitter (max): {max_jitter:.6f} sec\n'
            )


class IsaacManipulatorLoadEstimationTest(IsaacManipulatorLoadEstimationPolTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  monitor_topic_name: str,
                                  use_sim_time: bool,
                                  test_duration_seconds: int,
                                  output_dir: str,
                                  nodes: list[Node],
                                  message_type: Any,
                                  robot_segmenter_topic_names: list[str],
                                  node_startup_delay: float,
                                  max_latency_in_robot_segmentor_ms: int,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._monitor_topic_name = monitor_topic_name
        cls._use_sim_time = use_sim_time
        cls._message_type = message_type
        cls._test_duration_seconds = test_duration_seconds
        cls._output_dir = output_dir
        cls._robot_segmenter_topic_names = robot_segmenter_topic_names
        cls._max_latency_in_robot_segmentor_ms = max_latency_in_robot_segmentor_ms
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
            )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class IsaacManipulatorRobotSegmentorTest(IsaacROSBaseTest):
    """Test to verify robot segmentor correctness check."""

    DEFAULT_NAMESPACE = ''

    # Variables for robot segmentor correctness check
    _run_test: bool = False
    _use_sim_time: bool = False
    _output_dir: str = ''
    _xrdf_file_path: str = ''
    _xrdf_collision_geometry_name: str = ''
    _intrinsics_topic: str = '/front_stereo_camera/left/camera_info'
    _mask_topic: str = '/cumotion/camera_1/robot_mask'
    _joint_states_topic: str = '/isaac_joint_states'
    _depth_topic: str = '/cumotion/camera_1/world_depth'
    _raw_depth_topic: str = '/camera_1/depth_image'
    _links_to_check: list[str] = []
    _num_samples: int = 1000
    _buffer_distance_for_collision_spheres: float = 0.1
    _depth_image_is_float16: bool = False

    def _mark_invalid_points_in_robot_mask(self) -> np.ndarray:
        """Mark invalid points in robot mask."""
        self.invalid_depth_mask = np.zeros_like(self.raw_depth_image)
        invalid_points = np.where(self.raw_depth_image == 0)
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        invalid_points = np.where(self.raw_depth_image == np.inf)
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        invalid_points = np.where(np.isnan(self.raw_depth_image))
        self.robot_mask[invalid_points] = 1
        self.invalid_depth_mask[invalid_points] = 255

        return self.robot_mask, self.invalid_depth_mask

    def setUpVariables(self):
        """Set up the test environment."""
        self.joint_states = None
        self.tf_buffer = None
        self.tf_listener = None
        self.robot_mask = None
        self.camera_intrinsics = None
        self.robot_collision_spheres = None
        self.received_messages = {}

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # Register logging subscribers
        self.register_logging_subscribers()

        # Load camera intrinsics from YAML file
        self.camera_intrinsics, self.joint_states = self._load_camera_intrinsics_and_joint_states()
        self.robot_mask, self.depth_image, self.raw_depth_image = self._load_mask_and_image()

        # Save depth image
        cv2.imwrite(os.path.join(self._output_dir, 'depth_image.png'), self.depth_image)

        # Save robot mask
        cv2.imwrite(os.path.join(self._output_dir, 'robot_mask.png'), self.robot_mask)
        # Now find all points in raw_depth image that are not valid and mark them 0 in robot_mask
        self.robot_mask, self.invalid_depth_mask = self._mark_invalid_points_in_robot_mask()

        # Save robot mask with invalid points marked
        cv2.imwrite(os.path.join(self._output_dir,
                                 'robot_mask_invalid_points_marked.png'), self.robot_mask)
        cv2.imwrite(os.path.join(self._output_dir,
                                 'invalid_depth_mask.png'), self.invalid_depth_mask)

        # save camera intrinsics and joint states to yaml files
        with open(os.path.join(self._output_dir, 'camera_intrinsics.yaml'), 'w') as f:
            yaml.dump(self.camera_intrinsics, f)
        with open(os.path.join(self._output_dir, 'joint_states.yaml'), 'w') as f:
            yaml.dump(self.joint_states, f)

    def register_logging_subscribers(self):
        """Register logging subscribers for required topics."""
        # Create a logging subscriber for the camera_info topic
        self._subs = self.create_logging_subscribers(
            [(self._intrinsics_topic, CameraInfo),
             (self._mask_topic, Image),
             (self._joint_states_topic, JointState),
             (self._depth_topic, Image),
             (self._raw_depth_topic, Image)],
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

    def destroy_logging_subscribers(self):
        """Destroy the logging subscribers."""
        self.node.destroy_subscription(self._subs)

    def _load_camera_intrinsics_and_joint_states(self
                                                 ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """
        Load camera intrinsics and joint states from camera_info and joint_states topics.

        Returns
        -------
            tuple[dict[str, Any], dict[str, Any]]: The camera intrinsics and joint states.

        Raises
        ------
            Exception: If the camera intrinsics and joint states are not received within the
                timeout.

        """
        try:
            # Wait for camera intrinsics to be received (up to 10 seconds)
            self.node.get_logger().info(
                f'Waiting for camera intrinsics on {self._intrinsics_topic}')
            start_time = time.time()

            while time.time() - start_time < self._timeout_seconds_for_service_call:
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if (self._intrinsics_topic in self.received_messages and
                    self._joint_states_topic in self.received_messages) and \
                    (len(self.received_messages[self._intrinsics_topic]) > 0 and
                        len(self.received_messages[self._joint_states_topic]) > 0):
                    self.node.get_logger().info('Camera intrinsics and joint states received')
                    camera_info_msg = self.received_messages[self._intrinsics_topic][-1]
                    joint_states_msg = self.received_messages[self._joint_states_topic][-1]

                    self.camera_distortion_coefficients = np.array(
                        camera_info_msg.d, dtype=np.float32)
                    self.camera_frame_id = camera_info_msg.header.frame_id
                    # Extract intrinsic parameters from CameraInfo message
                    intrinsics = {
                        'fx': camera_info_msg.k[0],  # Focal length x
                        'fy': camera_info_msg.k[4],  # Focal length y
                        'cx': camera_info_msg.k[2],  # Principal point x
                        'cy': camera_info_msg.k[5],  # Principal point y
                        'width': camera_info_msg.width,
                        'height': camera_info_msg.height,
                        'distortion_model': camera_info_msg.distortion_model,
                        'distortion_coeffs': camera_info_msg.d
                    }

                    self.camera_intrinsics_matrix = np.array(
                        camera_info_msg.k, dtype=np.float32).reshape(3, 3)

                    joint_states = {
                        'position': joint_states_msg.position,
                        'velocity': joint_states_msg.velocity,
                        'effort': joint_states_msg.effort
                    }

                    # Log the intrinsic parameters for debugging
                    self.node.get_logger().info(
                        f'Camera intrinsics: fx={intrinsics["fx"]}, fy={intrinsics["fy"]}, '
                        f'cx={intrinsics["cx"]}, cy={intrinsics["cy"]}'
                    )

                    return intrinsics, joint_states

        except Exception as e:
            self.node.get_logger().error(f'Error loading camera intrinsics: {e}')
            self.fail(f'Error loading camera intrinsics: {e}')

        self.node.get_logger().error('Timed out waiting for camera intrinsics')
        self.fail('Timed out waiting for camera intrinsics')

    def _load_mask_and_image(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the robot mask and depth image from the mask and depth topics.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: The robot mask and depth image.

        Raises
        ------
            Exception: If the mask and depth image are not received within the timeout.

        """
        try:
            # Wait for mask to be received (up to 10 seconds)
            self.node.get_logger().info(
                f'Waiting for mask on {self._mask_topic}')
            start_time = time.time()

            while time.time() - start_time < self._timeout_seconds_for_service_call:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self._mask_topic in self.received_messages and \
                    self._depth_topic in self.received_messages and \
                    self._raw_depth_topic in self.received_messages and \
                    (len(self.received_messages[self._mask_topic]) > 0 and
                        len(self.received_messages[self._depth_topic]) > 0 and
                        len(self.received_messages[self._raw_depth_topic]) > 0):
                    self.node.get_logger().info('Mask and depth image received')
                    mask_msg = self.received_messages[self._mask_topic][-1]
                    image_msg = self.received_messages[self._depth_topic][-1]

                    # Convert mask image to numpy array
                    mask_array = np.frombuffer(mask_msg.data, dtype=np.uint8)
                    mask_array = mask_array.reshape(mask_msg.height, mask_msg.width)

                    # Convert depth image to numpy array
                    if self._depth_image_is_float16:
                        depth_array = np.frombuffer(
                            image_msg.data, dtype=np.float16).astype(np.float32)
                    else:
                        depth_array = np.frombuffer(image_msg.data, dtype=np.float32)

                    depth_array = depth_array.reshape(image_msg.height, image_msg.width)

                    if self._depth_image_is_float16:
                        raw_depth_array = np.frombuffer(
                            self.received_messages[self._raw_depth_topic][-1].data,
                            dtype=np.float16
                        ).astype(np.float32).reshape(
                            self.received_messages[self._raw_depth_topic][-1].height,
                            self.received_messages[self._raw_depth_topic][-1].width
                        )
                    else:
                        raw_depth_array = np.frombuffer(
                            self.received_messages[self._raw_depth_topic][-1].data,
                            dtype=np.float32
                        ).reshape(
                            self.received_messages[self._raw_depth_topic][-1].height,
                            self.received_messages[self._raw_depth_topic][-1].width
                        )

                    # Create a dictionary to store the robot mask
                    return mask_array, depth_array, raw_depth_array

            self.node.get_logger().error('Timed out waiting for mask and depth image')
            self.fail('Timed out waiting for mask and depth image')

        except Exception as e:
            self.node.get_logger().error(f'Error loading mask and depth image: {e}')
            self.fail(f'Error loading mask and depth image: {e}')

    def _load_robot_collision_spheres(self, xrdf_file_path: str) -> dict[str, np.ndarray]:
        """
        Load robot collision spheres from XRDF file.

        Args
        ----
            xrdf_file_path (str): Path to the XRDF file

        Returns
        -------
            dict[str, np.ndarray]: A dictionary of collision spheres for each link

        Raises
        ------
            Exception: If the XRDF file is not found or the collision spheres are not loaded
                correctly.

        """
        with open(xrdf_file_path) as file_p:
            xrdf_dict = yaml.load(file_p, Loader=yaml.SafeLoader)

        # Extract collision spheres from XRDF
        collision_spheres = {}
        spheres = xrdf_dict['geometry'][self._xrdf_collision_geometry_name]['spheres']
        for link_name, link_data in spheres.items():
            if link_name in self._links_to_check:
                all_pts = []
                for sphere in link_data:
                    pts = self._sample_points_from_sphere(
                        np.array(sphere['center']),
                        sphere['radius'] - sphere['radius'] / 2.0,
                    )
                    all_pts.append(pts)
                collision_spheres[link_name] = np.vstack(all_pts)

                print(link_name, collision_spheres[link_name])
        return collision_spheres

    def _project_point_to_image(self, point_3d: np.ndarray,
                                T: np.ndarray,
                                camera_intrinsics: np.ndarray,
                                camera_distortion_coefficients: np.ndarray) -> np.ndarray:
        """
        Project a 3D point to image coordinates.

        Args
        ----
            point_3d (np.ndarray): 3D point to project
            T (np.ndarray): Transformation matrix
            camera_intrinsics (np.ndarray): Camera intrinsics
            camera_distortion_coefficients (np.ndarray): Camera distortion coefficients

        Returns
        -------
            np.ndarray: Projected 2D points

        Raises
        ------
            Exception: If the projection fails

        """
        tvec = T[:3, 3]
        rvec = R.from_matrix(T[:3, :3]).as_rotvec()
        projected_points, _ = cv2.projectPoints(
                    point_3d,
                    rvec,
                    tvec,
                    camera_intrinsics,
                    camera_distortion_coefficients,
                )
        return projected_points.reshape(-1, 2)

    def _sample_points_from_sphere(self, center: np.ndarray,
                                   radius: float) -> np.ndarray:
        """
        Sample points uniformly from the surface of a sphere.

        Args
        ----
            center: np.ndarray of shape (3,) containing the center of the sphere,
                usually this is [0, 0, 0]
            radius: float containing the radius of the sphere
            num_samples: int containing the number of samples to generate

        Returns
        -------
            np.ndarray of shape (num_samples, 3) containing the sampled points

        """
        # Generate random points from a normal distribution (Gaussian sampling)
        points = np.random.normal(0, 1, (self._num_samples, 3))

        # Normalize to create unit vectors (points on a unit sphere)
        points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]

        # Scale by radius and add center
        return points * radius + center

    def _check_point_in_mask(self, points_2d: np.ndarray, mask: np.ndarray
                             ) -> Tuple[bool, np.ndarray]:
        """
        Check if points are masked out (black) in the segmentation mask.

        Args
        ----
            points_2d: np.ndarray of shape (N, 2) containing pixel coordinates (u, v)
            mask: np.ndarray representing the segmentation mask

        Returns
        -------
            tuple[bool, np.ndarray]: True if any point is black (0) in the mask, False otherwise
                and the coordinates of the points that are black.

        """
        y_coords = points_2d[:, 1].astype(int)
        x_coords = points_2d[:, 0].astype(int)

        valid_points = (y_coords >= 0) & (y_coords < mask.shape[0]) & \
                       (x_coords >= 0) & (x_coords < mask.shape[1])
        y_coords = y_coords[valid_points]
        x_coords = x_coords[valid_points]

        # Get mask values at the specified points
        mask_values = mask[y_coords, x_coords]
        zero_mask_indices = np.where(mask_values == 0)[0]

        # Get the corresponding coordinates where mask value is 0
        zero_coords = np.column_stack((x_coords[zero_mask_indices], y_coords[zero_mask_indices])) \
            if len(zero_mask_indices) > 0 else np.array([])

        return np.any(mask_values == 0), zero_coords

    def _transform_point_to_another_frame(self, points: np.ndarray,
                                          from_frame: str,
                                          to_frame: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a point or array of points from one frame to another.

        Args
        ----
            points (np.ndarray): Array of points with shape (N, 3)
            from_frame (str): Source frame
            to_frame (str): Target frame

        Returns
        -------
            tuple[np.ndarray, np.ndarray]: Transformed point(s) in the target frame

        Raises
        ------
            Exception: If the transformation fails

        """
        try:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    # Spin a few times to process messages
                    for _ in range(5):
                        rclpy.spin_once(self.node, timeout_sec=0.1)

                    transform_stamped = self.tf_buffer.lookup_transform(
                        to_frame,
                        from_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        self.node.get_logger().warn(
                            f'Transform lookup attempt {attempt+1}/{max_attempts} failed, '
                            f'retrying...'
                        )
                    else:
                        self.node.get_logger().error(
                            f'All {max_attempts} transform lookup attempts failed: {e}'
                        )
                        self.fail(f'All {max_attempts} transform lookup attempts failed: {e}')

            # Extract translation and rotation from transform
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation

            # Convert to numpy matrix
            translation_array = np.array([translation.x, translation.y, translation.z])
            rotation_array = np.array([rotation.x, rotation.y, rotation.z, rotation.w])

            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(rotation_array).as_matrix()

            # Create transformation matrix (4x4)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation_array

            # Convert to homogeneous coordinates
            homogeneous_points = np.ones((points.shape[0], 4), dtype=np.float32)
            homogeneous_points[:, :3] = points

            # Apply transformation
            transformed_homogeneous = np.dot(homogeneous_points, transform_matrix.T)

            # Convert back to 3D coordinates
            transformed_points = transformed_homogeneous[:, :3]

            return transformed_points, transform_matrix

        except Exception as e:
            self.node.get_logger().error(
                f'Error transforming point from {from_frame} to {to_frame}: {str(e)}')
            self.fail(f'Error transforming point from {from_frame} to {to_frame}: {str(e)}')

    def visualize_sphere_points_in_image(self, points_2d, mask_height, mask_width, output_dir,
                                         output_filename='mask_image_with_pts.png',
                                         side_length=20, color=(0, 255, 0),
                                         add_extension_to_pixels=True) -> np.ndarray:
        """
        Visualize projected sphere points in an image by drawing squares around them.

        Args
        ----
            points_2d (np.ndarray): Array of 2D points with shape (N, 2)
            mask_height (int): Height of the mask image
            mask_width (int): Width of the mask image
            output_dir (str): Directory to save the output image
            output_filename (str, optional): Filename for the output image.
                Defaults to 'mask_image_with_pts.png'.
            side_length (int, optional): Side length of the square around each point.
                Defaults to 20.
            color (tuple, optional): Color for the squares in BGR format. Defaults to (0, 255, 0)
                (green).
            add_extension_to_pixels (bool, optional): Whether to add an extension to the pixels
                of the square. Defaults to True.

        Returns
        -------
            np.ndarray: The image with visualized points

        """
        # Make a copy of the image to avoid modifying the original
        vis_image = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        side_length = side_length // 2
        for i in range(points_2d.shape[0]):
            # Get the point coordinates
            y, x = int(points_2d[i, 1]), int(points_2d[i, 0])

            if y < 0 or y >= vis_image.shape[0] or x < 0 or x >= vis_image.shape[1]:
                continue

            vis_image[y, x] = color

            if add_extension_to_pixels:
                # Draw a square around the point
                cv2.rectangle(vis_image,
                              (x-side_length, y-side_length), (x+side_length, y+side_length),
                              color, 2)

        # Save the image
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, vis_image)
        self.node.get_logger().info(f'Saved visualization image to {output_path}')

        return vis_image

    def test_correctness_of_robot_segmentor(self):
        """
        Test correctness of robot segmentor.

        This function tests the correctness of the robot segmentor by projecting the collision
        spheres onto the image and checking if they are in the segmentation mask.

        Raises
        ------
            Exception: If the robot segmentor is not correct

        """
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting robot segmentor correctness check')
        self.setUpVariables()

        # Construct collision sphere points in wrist_3_link frame
        pts_for_link = np.array([[0, 0, 0]], dtype=np.float32)

        # Load collision spheres from XRDF file
        collision_spheres = self._load_robot_collision_spheres(self._xrdf_file_path)
        all_pts_in_2d = []
        fail_on_exit = False
        # Iterate through each link and its collision spheres
        for link in self._links_to_check:
            pts_for_link = collision_spheres[link]

            # Transform points to camera frame
            _, transform_matrix = self._transform_point_to_another_frame(
                pts_for_link, link, self.camera_frame_id)

            # Project points to image
            points_2d = self._project_point_to_image(pts_for_link,
                                                     transform_matrix,
                                                     self.camera_intrinsics_matrix,
                                                     self.camera_distortion_coefficients)

            all_pts_in_2d.append(points_2d)

            # Check if points are in the segmentation mask
            any_point_outside_mask, zero_coords = self._check_point_in_mask(
                points_2d.reshape(-1, 2), self.robot_mask)
            if any_point_outside_mask:
                # Only fail on exit if points outside the mask are a certain percentage
                if zero_coords.shape[0] > 0.1 * points_2d.shape[0]:
                    fail_on_exit = True

                self.visualize_sphere_points_in_image(
                    zero_coords,
                    self.robot_mask.shape[0],
                    self.robot_mask.shape[1],
                    self._output_dir,
                    color=(0, 0, 255),
                    add_extension_to_pixels=False,
                    output_filename=f'{link}_points_not_in_mask.png'
                )
                self.node.get_logger().error(
                    f'{zero_coords.shape[0]}/{points_2d.shape[0]} points are not in the '
                    f'segmentation mask for {link}')

        # Visualize the projected points on the mask image
        self.visualize_sphere_points_in_image(
            np.vstack(all_pts_in_2d),
            self.robot_mask.shape[0],
            self.robot_mask.shape[1],
            self._output_dir,
            color=(0, 255, 0),
            output_filename='projected_points_on_mask.png',
            add_extension_to_pixels=False
        )

        self.destroy_logging_subscribers()

        if fail_on_exit:
            self.fail('Points are not in the segmentation mask')


class IsaacManipulatorRobotSegmentorCorrectnessCheckTest(IsaacManipulatorRobotSegmentorTest):
    """Test to verify robot segmentor correctness check."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  links_to_check: list[str],
                                  output_dir: str,
                                  xrdf_file_path: str,
                                  intrinsics_topic: str,
                                  mask_topic: str,
                                  joint_states_topic: str,
                                  depth_topic: str,
                                  raw_depth_topic: str,
                                  num_samples: int,
                                  buffer_distance_for_collision_spheres: float,
                                  depth_image_is_float16: bool,
                                  xrdf_collision_geometry_name: str,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        cls._xrdf_file_path = xrdf_file_path
        cls._intrinsics_topic = intrinsics_topic
        cls._mask_topic = mask_topic
        cls._joint_states_topic = joint_states_topic
        cls._depth_topic = depth_topic
        cls._raw_depth_topic = raw_depth_topic
        cls._links_to_check = links_to_check
        cls._num_samples = num_samples
        cls._buffer_distance_for_collision_spheres = buffer_distance_for_collision_spheres
        cls._depth_image_is_float16 = depth_image_is_float16
        cls._xrdf_collision_geometry_name = xrdf_collision_geometry_name
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
            )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class ManipulatorRobotSegmentorPolTest(IsaacROSBaseTest):
    """Test to verify robot segmentor is running at required hz."""

    DEFAULT_NAMESPACE = ''

    _run_test: bool = False
    _use_sim_time: bool = False
    _output_dir: str = ''
    _timeout_seconds_for_service_call: int = 10.0
    _intrinsics_topics: list[str] = []
    _input_topics: list[str] = []
    _output_topics: list[str] = []
    _time_to_run: int = 60
    _max_latency_ms: int = 500

    def register_logging_subscribers(self):
        """Register logging subscribers for required topics."""
        # Create a logging subscriber for the camera_info topic
        self.received_messages = {}
        list_of_subscriptions = []
        for intrinsics_topic in self._intrinsics_topics:
            list_of_subscriptions.append((intrinsics_topic, CameraInfo))
        for input_topic in self._input_topics:
            list_of_subscriptions.append((input_topic, Image))

        for output_topic in self._output_topics:
            list_of_subscriptions.append((output_topic, Image))

        self._subs = self.create_logging_subscribers(
            list_of_subscriptions,
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

    def test_robot_segmentor_pol(self):
        """Test to verify robot segmentor is running at required hz."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.node.get_logger().info('Starting robot segmentor correctness check')
        self.register_logging_subscribers()

        self._topics_to_check = []
        self._topics_to_check += self._intrinsics_topics
        self._topics_to_check += self._input_topics
        self._topics_to_check += self._output_topics
        self._data_dict = {}
        # Iterate through the topics and check if the data is being published at the required hz
        start_time = time.time()
        failure_latency_msg = ''
        if_failure_latency = False
        while True:

            rclpy.spin_once(self.node, timeout_sec=0.1)

            for topic in self._topics_to_check:
                self.node.get_logger().info(f'Checking topic: {topic}')
                self.node.get_logger().info(f'Topic type: {type(self.received_messages[topic])}')
                self.node.get_logger().info(f'Number of messages: '
                                            f'{len(self.received_messages[topic])}')
                # Store data to find out hz of each topic and number of messages received
                if topic not in self._data_dict:
                    self._data_dict[topic] = {
                        'timestamp_ns': [],
                        'num_messages': 0,
                        'latency_ms': []
                    }

                if len(self.received_messages[topic]) == 0:
                    continue

                # Calculate hz of each topic
                current_timestamp = self.received_messages[topic][-1].header.stamp
                # Get timestamp in nanoseconds
                current_timestamp_ns = current_timestamp.sec * 1e9 + current_timestamp.nanosec
                self._data_dict[topic]['timestamp_ns'].append(current_timestamp_ns)
                self._data_dict[topic]['num_messages'] += 1
                current_clock_time = self.node.get_clock().now().to_msg()
                latency_ns = current_clock_time.sec * 1e9 + \
                    current_clock_time.nanosec - current_timestamp_ns
                latency_ms = latency_ns / 1e6
                if topic in self._output_topics:
                    self.node.get_logger().info(f'{topic}: {latency_ms} ms')
                if latency_ms > self._max_latency_ms:
                    failure_latency_msg = f'Latency for topic {topic} is {latency_ms} ms higher' \
                                          f'than max allowed latency of {self._max_latency_ms} ms'
                    if_failure_latency = True

                self._data_dict[topic]['latency_ms'].append(latency_ms)

            # If time elapsed is greater than 10 second, calculate hz of each topic
            if time.time() - start_time > self._time_to_run:
                break

        # Plot latency of each topic, save fig to output_dir, different colors for each topic
        plt.figure()
        for topic in self._topics_to_check:
            plt.plot(self._data_dict[topic]['latency_ms'], label=topic)
        plt.legend()
        plt.savefig(os.path.join(self._output_dir, 'latency_plot_robot_segmentor_pol.png'))
        plt.close()

        # First check if each topic has received messages
        for topic in self._topics_to_check:
            if self._data_dict[topic]['num_messages'] == 0:
                self.node.get_logger().error(f'No messages received for topic: {topic}')
                self.fail(f'No messages received for topic: {topic}')

        # The check avg diff between consecutive timestamps is less than 1/hz of the topic
        for topic in self._topics_to_check:

            if topic not in self._output_topics:
                continue

            # Make sure no timestamp arra is of size 0, if yes then fail
            if len(self._data_dict[topic]['timestamp_ns']) == 0:
                self.node.get_logger().error(f'No timestamps received for topic: {topic}')
                self.fail(f'No timestamps received for topic: {topic}')

            avg_diff = np.mean(np.diff(self._data_dict[topic]['timestamp_ns']))
            avg_diff_ms = avg_diff / 1e6
            if avg_diff_ms > 1000:  # 1 second
                self.node.get_logger().error(f'Avg diff between consecutive timestamps for topic'
                                             f'{topic} is {avg_diff_ms} milliseconds')
                self.fail(f'Avg diff between consecutive timestamps for topic'
                          f'{topic} is {avg_diff_ms} milliseconds')

        # Check if latency is higher for output topics
        if if_failure_latency:
            self.node.get_logger().error(failure_latency_msg)
            self.fail(failure_latency_msg)


class IsaacManipulatorRobotSegmentorPolTest(ManipulatorRobotSegmentorPolTest):
    """Test to verify robot segmentor is running at required hz."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  output_dir: str,
                                  intrinsics_topics: list[str],
                                  input_topics: list[str],
                                  output_topics: list[str],
                                  time_to_run: int,
                                  max_latency_ms: int,
                                  timeout_seconds_for_service_call: int = 10.0):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._timeout_seconds_for_service_call = timeout_seconds_for_service_call
        cls._intrinsics_topics = intrinsics_topics
        cls._input_topics = input_topics
        cls._output_topics = output_topics
        cls._time_to_run = time_to_run
        cls._max_latency_ms = max_latency_ms
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
            )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])


class DisableNvbloxTest(IsaacROSBaseTest):
    """Verify nvblox and robot segmentor nodes are not running when nvblox is disabled."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_nvblox_nodes_not_running(self):
        """Verify nvblox and robot segmentor nodes are not running when nvblox is disabled."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        time.sleep(self._node_startup_delay)

        running_nodes = self.node.get_node_names()
        self.node.get_logger().info(f'Running nodes: {running_nodes}')

        # Check that nvblox node is not running
        nvblox_node = 'nvblox_node'
        self.assertNotIn(
            nvblox_node,
            running_nodes,
            f'Nvblox node {nvblox_node} is running when it should be disabled'
        )

        # Check that robot segmentor node is not running
        robot_segmentor_node = 'robot_segmenter_1'
        self.assertNotIn(
            robot_segmentor_node,
            running_nodes,
            f'Robot segmentor node {robot_segmentor_node} is running when it should be disabled'
        )

        self.node.get_logger().info(
            'Successfully verified nvblox and robot segmentor nodes are not running'
        )


class DriverTest(IsaacROSBaseTest):
    """Verify driver nodes are running."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _enable_pick_and_place: bool = False
    _node_startup_delay: float = 0.0

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  enable_pick_and_place: bool):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._enable_pick_and_place = enable_pick_and_place
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
        )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    def test_driver_nodes_running(self):
        """Verify driver nodes are running."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        time.sleep(self._node_startup_delay)

        running_nodes = self.node.get_node_names()
        self.node.get_logger().info(f'Running nodes: {running_nodes}')

        control_nodes_that_should_be_running = [
            'move_group',
            'moveit',
            'moveit_simple_controller_manager',
            'controller_manager',
            'joint_state_broadcaster',
            'io_and_status_controller',
            'speed_scaling_state_broadcaster',
            'force_torque_sensor_broadcaster',
            'tcp_pose_broadcaster',
            'ur_configuration_controller',
            'scaled_joint_trajectory_controller',
            'impedance_controller',
            'controller_stopper',
            'robot_state_publisher',
        ]

        if self._enable_pick_and_place:
            control_nodes_that_should_be_running.append('robotiq_gripper_controller')
            control_nodes_that_should_be_running.append('robotiq_activation_controller')

        for node in control_nodes_that_should_be_running:
            self.assertIn(
                node,
                running_nodes,
                f'{node} is not running when it should be running'
            )


class TestGearPointPublisher(IsaacROSBaseTest):
    """Node that publishes gear coordinates in round-robin fashion for testing."""

    DEFAULT_NAMESPACE = ''

    _run_test: bool = False
    _use_sim_time: bool = False
    _output_dir: str = ''
    _publish_rate: float = 3.0
    _num_cycles: int = 1000
    _publish_sequence: str = 'all'
    _initial_delay: float = 20.0
    _monitor_topic_name: str = ''
    _message_type: Any = None
    _max_latency_ms: int = 1000

    def setupThingsForSAMTest(self):

        # Initialize gear coordinates, this is hardcoded to be correct for sim cameras
        self.gear_coordinates = [
            # Gear base
            {'x': 1254.0, 'y': 1028.0, 'z': 0.0, 'name': 'gear_base'},
            # Gear large
            {'x': 977.0, 'y': 900.0, 'z': 0.0, 'name': 'gear_large'},
            # Gear medium
            {'x': 896.0, 'y': 1012.0, 'z': 0.0, 'name': 'gear_medium'},
            # Gear small
            {'x': 1105.0, 'y': 1022.0, 'z': 0.0, 'name': 'gear_small'}
        ]

        # Filter by sequence if specified
        if self._publish_sequence != 'all':
            filtered_gears = []
            if self._publish_sequence == 'base':
                filtered_gears.append(self.gear_coordinates[0])
            elif self._publish_sequence == 'large':
                filtered_gears.append(self.gear_coordinates[1])
            elif self._publish_sequence == 'medium':
                filtered_gears.append(self.gear_coordinates[2])
            elif self._publish_sequence == 'small':
                filtered_gears.append(self.gear_coordinates[3])
            elif self._publish_sequence == 'base_large':
                filtered_gears.append(self.gear_coordinates[0])
                filtered_gears.append(self.gear_coordinates[1])
            elif self._publish_sequence == 'base_medium':
                filtered_gears.append(self.gear_coordinates[0])
                filtered_gears.append(self.gear_coordinates[2])
            elif self._publish_sequence == 'base_small':
                filtered_gears.append(self.gear_coordinates[0])
                filtered_gears.append(self.gear_coordinates[3])

            # Use filtered gears if any were selected
            if filtered_gears:
                self.gear_coordinates = filtered_gears

        self.received_messages = {}

        self.subs = self.create_logging_subscribers(
            [(self._monitor_topic_name, self._message_type)], self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True)

    def set_gear_mesh_param(self, gear_name: str) -> bool:
        """
        Set the mesh_file_path parameter for the foundationpose_node based on the current gear.

        Args
        ----
            gear_name: Name of the gear ('gear_base', 'gear_large', etc.)

        Returns
        -------
            bool: True if the mesh_file_path parameter was set successfully, False otherwise.

        """
        mesh_file_name = f'{gear_name}.obj'

        mesh_file_path = f'{ISAAC_ROS_WS}/isaac_ros_assets/' \
                         f'isaac_manipulator_ur_dnn_policy/{gear_name}/{mesh_file_name}'

        # Create a parameter client
        param_client = self.node.create_client(
            SetParameters,
            '/foundationpose_node/set_parameters'
        )

        # Wait for the service to be available
        if not param_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn(
                'Service /foundationpose_node/set_parameters not available'
                ' skipping mesh parameter setting')
            return False

        # Create parameter request
        request = SetParameters.Request()
        parameter = RCLPYParameter()
        parameter.name = 'mesh_file_path'
        parameter.value = ParameterValue(
            string_value=mesh_file_path,
            type=rclpy.Parameter.Type.STRING.value)

        request.parameters = [parameter]

        # Send request
        future = param_client.call_async(request)

        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)

        if future.result() is None:
            self.node.get_logger().warn(f'Failed to set mesh_file_path to {mesh_file_path}')
            return False
        self.node.get_logger().debug(f'Successfully set mesh_file_path to {mesh_file_path}')
        return True

    def test_latency_of_mask_output(self):
        """Publish the next gear coordinate and update counters."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setupThingsForSAMTest()

        test_final_status = False
        last_publish_time = time.time()

        while not test_final_status:

            rclpy.spin_once(self.node, timeout_sec=0.1)

            if time.time() - last_publish_time > 1.0 / self._publish_rate:
                self.run_for_each_iteration()
                last_publish_time = time.time()

            if self.current_cycle >= self._num_cycles and self._num_cycles > 0:
                test_final_status = True

            # Check the latency of the mask output
            if len(self.received_messages[self._monitor_topic_name]) > 0:
                current_timestamp = self.received_messages[
                    self._monitor_topic_name][-1].header.stamp

                # Get timestamp in milliseconds
                current_timestamp_ns = current_timestamp.sec * 1e9 + current_timestamp.nanosec
                current_clock_time = self.node.get_clock().now().to_msg()
                latency_ns = current_clock_time.sec * 1e9 + \
                    current_clock_time.nanosec - current_timestamp_ns
                latency_ms = latency_ns / 1e6

                # Check if latency is higher than max allowed latency
                if latency_ms > self._max_latency_ms:
                    err_msg = f'Latency for topic {self._monitor_topic_name} is {latency_ms}'
                    err_msg += 'milliseconds higher than max allowed latency of'
                    err_msg += f'{self._max_latency_ms} milliseconds'
                    self.node.get_logger().error(err_msg)
                    self.fail(err_msg)
                else:
                    self.node.get_logger().info(
                        f'Latency for topic {self._monitor_topic_name} is {latency_ms} '
                        'milliseconds'
                    )

                # Set received message to empty list for next cycle
                self.received_messages[self._monitor_topic_name] = []

    def run_for_each_iteration(self):
        # Stop if we've gone through all cycles
        if self.current_cycle >= self._num_cycles and self._num_cycles > 0:
            self.node.get_logger().info(
                f'Completed all {self._num_cycles} cycles, stopping publisher')
            return

        # Get current gear
        current_gear = self.gear_coordinates[self.current_gear_idx]

        # Create and publish the current point
        point_msg = Point()
        point_msg.x = current_gear['x']
        point_msg.y = current_gear['y']
        point_msg.z = current_gear['z']

        self.set_gear_mesh_param(current_gear['name'])

        self.point_publisher.publish(point_msg)

        self.node.get_logger().debug(
            f"Published {current_gear['name']} point: "
            f"({current_gear['x']}, {current_gear['y']}, {current_gear['z']}), "
            f"cycle {self.current_cycle + 1}/{self._num_cycles if self._num_cycles > 0 else 'inf'}"
        )

        # Move to next gear
        self.current_gear_idx = (self.current_gear_idx + 1) % len(self.gear_coordinates)

        # If we've gone through all gears, increment cycle counter
        if self.current_gear_idx == 0:
            self.current_cycle += 1
            if self._num_cycles > 0:
                self.node.get_logger().debug(
                    f'Completed cycle {self.current_cycle}/{self._num_cycles}'
                )


class IsaacManipulatorSegmentAnythingPolTest(TestGearPointPublisher):
    """Test to verify SAM is running correctly and outputting masks at required rate."""

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  output_dir: str,
                                  publish_rate: float,
                                  num_cycles: int,
                                  publish_sequence: str,
                                  initial_delay: float,
                                  monitor_topic_name: str,
                                  message_type: Any,
                                  max_latency_ms: int):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._publish_rate = publish_rate
        cls._num_cycles = num_cycles
        cls._publish_sequence = publish_sequence
        cls._initial_delay = initial_delay
        cls._monitor_topic_name = monitor_topic_name
        cls._message_type = message_type
        cls._max_latency_ms = max_latency_ms
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay,
        )

    def setUp(self) -> None:
        """Set up before each test method."""
        # Create a ROS node for tests with use_sim_time=True
        self.node = rclpy.create_node(
            'isaac_ros_base_test_node',
            namespace=self.generate_namespace(),
            )

        if self._use_sim_time:
            self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.point_publisher = self.node.create_publisher(
            Point,
            '/segment_anything/input_points',  # Topic name aligned with SAM input points
            10)

        self.current_gear_idx = 0
        self.current_cycle = 0


class IsaacManipulatorServersPolTest(IsaacROSBaseTest):
    """Test for Isaac ROS Foundation Pose+ RT-Detr pose estimation POL."""

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_cycles: int = 10
    _use_sim_time: bool = False
    _mesh_file_path: str = ''
    _run_test: bool = False
    _detection_client: ActionClient = None
    _pose_client: ActionClient = None
    _segmentation_client: ActionClient = None
    _add_mesh_client: ActionClient = None
    _clear_objects_client: ActionClient = None
    _initial_hint: Dict = None

    # Variables to track object detection backend, segmentaiton backend and pose estimation backend
    _is_segment_anything_object_detection_enabled: bool = False
    _is_segment_anything_segmentation_enabled: bool = False
    _is_rt_detr_object_detection_enabled: bool = False

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self.failure_count = 0
        self.total_count = 0

        self._detection_client = ActionClient(self.node, GetObjects, '/get_objects')
        self._get_objects_goal_handle = None
        self._get_objects_result_received = False
        self._get_objects_result_future = None
        self._get_objects_detected_object_id = None

        self._get_object_pose_client = ActionClient(self.node, GetObjectPose, '/get_object_pose')
        self._get_object_pose_goal_handle = None
        self._get_object_pose_result_received = False
        self._get_object_pose_result_future = None

        self._segmentation_client = ActionClient(self.node, AddSegmentationMask,
                                                 '/add_segmentation_mask')
        self._segmentation_goal_handle = None
        self._segmentation_result_received = False
        self._segmentation_result_future = None

        self._add_mesh_client = self.node.create_client(AddMeshToObject, '/add_mesh_to_object')
        self._add_mesh_goal_handle = None
        self._add_mesh_result_received = False
        self._add_mesh_result_future = None

        self._clear_objects_client = self.node.create_client(ClearObjects, '/clear_objects')
        self._clear_objects_goal_handle = None
        self._clear_objects_result_received = False
        self._clear_objects_result_future = None

        self._current_gear = None
        self._initial_hint_point_msg = Point()
        if self._initial_hint is not None:
            self._initial_hint_point_msg.x = self._initial_hint['x']
            self._initial_hint_point_msg.y = self._initial_hint['y']
            self._initial_hint_point_msg.z = self._initial_hint['z']

    def trigger_get_objects_goal(self, initial_hint: None | Point = None):
        """Send a goal to the object detection action server."""
        self.node.get_logger().info('Waiting for action server for object detection...')
        if not self._detection_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error(
                'Action server for object detection not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        if initial_hint is not None:
            goal_msg.initial_hint = initial_hint
            goal_msg.use_initial_hint = True

        self.node.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._detection_client.send_goal_async(
            goal_msg, feedback_callback=self.get_objects_feedback_callback)
        send_goal_future.add_done_callback(self.get_objects_goal_response_callback)

    def get_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_objects_goal_handle = future.result()
        if not self._get_objects_goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.node.get_logger().info('Goal accepted by the action server.')
        self._get_objects_result_future = self._get_objects_goal_handle.get_result_async()
        self._get_objects_result_future.add_done_callback(self.get_objects_result_callback)

    def get_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        object_ids = []
        if result.objects:
            self.node.get_logger().info('Objects detected, the objects id are below:')
            for object_obj in result.objects:
                self.node.get_logger().info(f'Object: {object_obj.object_id}')
                object_ids.append(object_obj.object_id)

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self._get_objects_detected_object_id = result.objects[0].object_id
            self._get_objects_result_received = True
        else:
            self.node.get_logger().error(f'Action failed with status: {status}')
            self._get_objects_result_received = False

    def trigger_get_object_pose_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.node.get_logger().info('Waiting for pose estimation action server...')
        if not self._get_object_pose_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Pose estimation action server not available')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._get_object_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.get_object_pose_feedback_callback)
        send_goal_future.add_done_callback(self.get_object_pose_goal_response_callback)

    def get_object_pose_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_object_pose_goal_handle = future.result()
        if not self._get_object_pose_goal_handle.accepted:
            self.node.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Pose estimation goal accepted by the action server.')
        self._get_object_pose_result_future = self._get_object_pose_goal_handle.get_result_async()
        self._get_object_pose_result_future.add_done_callback(self.get_object_pose_result_callback)

    def get_object_pose_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_object_pose_result_callback(self, future):
        """Handle the result from the action server."""
        status = future.result().status

        # Log result to INFO
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Object pose received for object id')
            self._get_object_pose_result_received = True
        else:
            self.node.get_logger().error(f'Pose estimation action failed with status: {status}')
            self._get_object_pose_result_received = False

    def trigger_get_segmented_object_goal(self, object_id):
        """Send a goal to the object segmentation action server."""
        self.node.get_logger().info('Waiting for segmentation action server...')
        if not self._segmentation_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Segmentation action server not available after waiting')
            return

        goal_msg = AddSegmentationMask.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to get segmented object for ID: {object_id}')
        send_goal_future = self._segmentation_client.send_goal_async(
            goal_msg, feedback_callback=self.get_segmented_object_feedback_callback)
        send_goal_future.add_done_callback(self.get_segmented_object_goal_response_callback)

    def get_segmented_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._segmentation_goal_handle = future.result()
        if not self._segmentation_goal_handle.accepted:
            self.node.get_logger().error('Segmentation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Segmentation goal accepted by the action server.')
        self._segmentation_result_future = self._segmentation_goal_handle.get_result_async()
        self._segmentation_result_future.add_done_callback(
            self.get_segmented_object_result_callback)

    def get_segmented_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Segmentation Feedback: {feedback_msg.feedback}')

    def get_segmented_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Segmented object received: {result}')
            self._segmentation_result_received = True
        else:
            self.node.get_logger().error(f'Segmentation action failed with status: {status}')
            self._segmentation_result_received = False

    def trigger_add_mesh_to_object_request(self, object_id, mesh_file_path):
        request = AddMeshToObject.Request()
        request.object_ids = [object_id]
        request.mesh_file_paths = [mesh_file_path]
        future = self._add_mesh_client.call_async(request)
        return future

    def send_clear_objects_request(self, object_ids=None):
        request = ClearObjects.Request()
        if object_ids:
            request.object_ids = object_ids
        future = self._clear_objects_client.call_async(request)
        return future

    def add_mesh_to_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._add_mesh_goal_handle = future.result()
        if not self._add_mesh_goal_handle.accepted:
            self.node.get_logger().error('Add mesh goal was rejected by the action server.')
            return

        self.node.get_logger().info('Add mesh goal accepted by the action server.')
        self._add_mesh_result_future = self._add_mesh_goal_handle.get_result_async()
        self._add_mesh_result_future.add_done_callback(self.add_mesh_to_object_result_callback)

    def add_mesh_to_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Add Mesh Feedback: {feedback_msg.feedback}')

    def add_mesh_to_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        self.node.get_logger().info(f'Add mesh result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Mesh added to object: {result.object_id}')
            self._add_mesh_result_received = True
        else:
            self.node.get_logger().error(f'Add mesh action failed with status: {status}')
            self._add_mesh_result_received = False

    def clear_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._clear_objects_goal_handle = future.result()
        if not self._clear_objects_goal_handle.accepted:
            self.node.get_logger().error('Clear objects goal was rejected by the action server.')
            return

        self.node.get_logger().info('Clear objects goal accepted by the action server.')
        self._clear_objects_result_future = self._clear_objects_goal_handle.get_result_async()
        self._clear_objects_result_future.add_done_callback(self.clear_objects_result_callback)

    def clear_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Clear Objects Feedback: {feedback_msg.feedback}')

    def clear_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        self.node.get_logger().info(f'Clear objects result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Objects cleared successfully')
            self._clear_objects_result_received = True
        else:
            self.node.get_logger().error(f'Clear objects action failed with status: {status}')
            self._clear_objects_result_received = False

    def wait_for_result(self, result_var_name: str, timeout: float = 10.0):
        start_time = self.node.get_clock().now()
        while rclpy.ok():
            # Check the actual instance variable dynamically
            if result_var_name and getattr(self, result_var_name):
                return True

            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                # self.fail(f'Timeout waiting for {result_var_name}')
                self.failure_count += 1
                return False
            self.total_count += 1

            rclpy.spin_once(self.node, timeout_sec=0.1)

    def wait_for_service_result(self, future, timeout=10.0):
        start_time = self.node.get_clock().now()
        while not future.done():
            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.node.get_logger().error('Timeout waiting for service result')
                return None
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return future.result()

    def test_for_servers_pol(self):
        """Test that verifies that servers action calls under multiple cycles."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setUpClients()

        self.node.get_logger().info('Starting test for manipulator servers POL')

        time.sleep(10)  # Give servers some additional time to come up

        for _ in range(self._num_cycles):

            initial_hint = None
            if self._is_segment_anything_object_detection_enabled:
                initial_hint = self._initial_hint_point_msg

            # First detect object
            self.trigger_get_objects_goal(initial_hint)

            # Wait for the detection result with a timeout
            self.wait_for_result('_get_objects_result_received',
                                 timeout=self._max_timeout_time_for_action_call)
            if not self._get_objects_result_received:
                self.node.get_logger().error('Object detection action failed')
                self.failure_count += 1
                continue

            self.total_count += 1
            # If segment anything is enabled, we need to get the segmented object first
            if self._is_segment_anything_segmentation_enabled:
                self.trigger_get_segmented_object_goal(self._get_objects_detected_object_id)
                self.wait_for_result('_segmentation_result_received',
                                     timeout=self._max_timeout_time_for_action_call)

            # Now add mesh to object
            if self._is_segment_anything_segmentation_enabled:
                future = self.trigger_add_mesh_to_object_request(
                    self._get_objects_detected_object_id, self._mesh_file_path)
                result = self.wait_for_service_result(
                    future, timeout=self._max_timeout_time_for_action_call)
                self.node.get_logger().info(f'Add mesh to object result: {result}')
                if result is None:
                    self.node.get_logger().error('Add mesh to object action failed')
                    self.failure_count += 1
                    continue
                self.total_count += 1

            # Now get object pose
            self.trigger_get_object_pose_goal(self._get_objects_detected_object_id)
            self.wait_for_result('_get_object_pose_result_received',
                                 timeout=self._max_timeout_time_for_action_call)

            # Now clear objects
            future = self.send_clear_objects_request()
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.node.get_logger().info(f'Clear objects result: {result}')
            if result is None:
                self.node.get_logger().error('Clear objects action failed')
                self.failure_count += 1
                continue
            self.total_count += 1

            # Reset for next iteration
            self._get_objects_result_received = False
            self._segmentation_result_received = False
            self._add_mesh_result_received = False
            self._get_object_pose_result_received = False
            self._clear_objects_result_received = False

        self.node.get_logger().info(f'Failure count: {self.failure_count}')
        assert self.failure_count/self.total_count < 0.3, f'Pose estimation action failed' \
            f'{self.failure_count}/{self.total_count} times'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  num_cycles: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  is_segment_anything_object_detection_enabled: bool,
                                  is_segment_anything_segmentation_enabled: bool,
                                  is_rt_detr_object_detection_enabled: bool,
                                  initial_hint: Dict,
                                  mesh_file_path: str,
                                  output_dir: str,
                                  node_startup_delay: float):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._num_cycles = num_cycles
        cls._use_sim_time = use_sim_time
        cls._is_segment_anything_object_detection_enabled = \
            is_segment_anything_object_detection_enabled
        cls._is_segment_anything_segmentation_enabled = is_segment_anything_segmentation_enabled
        cls._is_rt_detr_object_detection_enabled = is_rt_detr_object_detection_enabled
        cls._initial_hint = initial_hint
        cls._output_dir = output_dir
        cls._mesh_file_path = mesh_file_path

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class IsaacManipulatorGearAssemblyPolTest(IsaacROSBaseTest):
    """
    Test for Isaac ROS Gear Assembly POL.

    This test will do object pose estimation using SAM with initial guess.
    It will then update foundation pose to use the mesh.
    It will then do pick and place action on the object.
    It will then do RL insertion action on the object.
    """

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _num_cycles: int = 10
    _ground_truth_sim_prim_name: str = 'gear_large'
    _use_sim_time: bool = False
    _mesh_file_paths: List[str] = []
    _run_test: bool = False
    _detection_client: ActionClient = None
    _pose_client: ActionClient = None
    _segmentation_client: ActionClient = None
    _add_mesh_client: ActionClient = None
    _clear_objects_client: ActionClient = None
    _initial_hint: Dict = None
    _wait_for_point_topic: bool = False
    _point_topic_name_as_trigger: str = ''
    _use_ground_truth_pose_estimation: bool = False
    _verify_pose_estimation_accuracy: bool = False
    _run_rl_inference: bool = False
    _gripper_close_pos: List[float] = None
    _use_joint_space_planner_api: bool = False
    _target_joint_state_for_place_pose: JointState = None
    _publish_only_on_static_tf: bool = False
    _peg_stand_shaft_offset_for_cumotion: float

    # Variables to track object detection backend, segmentaiton backend and pose estimation backend
    _is_segment_anything_object_detection_enabled: bool = False
    _is_segment_anything_segmentation_enabled: bool = False
    _is_rt_detr_object_detection_enabled: bool = False

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self.failure_count = 0
        self.total_count = 0

        self._detection_client = ActionClient(self.node, GetObjects, '/get_objects')
        self._get_objects_goal_handle = None
        self._get_objects_result_received = False
        self._get_objects_result_future = None
        self._get_objects_detected_object_id = None

        self._get_object_pose_client = ActionClient(self.node, GetObjectPose, '/get_object_pose')
        self._get_object_pose_goal_handle = None
        self._get_object_pose_result_received = False
        self._get_object_pose_result_future = None

        self._segmentation_client = ActionClient(self.node, AddSegmentationMask,
                                                 '/add_segmentation_mask')
        self._segmentation_goal_handle = None
        self._segmentation_result_received = False
        self._segmentation_result_future = None

        self._add_mesh_client = self.node.create_client(AddMeshToObject, '/add_mesh_to_object')
        self._add_mesh_goal_handle = None
        self._add_mesh_result_received = False
        self._add_mesh_result_future = None

        self.latest_pose_gotten = None

        self._clear_objects_client = self.node.create_client(ClearObjects, '/clear_objects')
        self._clear_objects_goal_handle = None
        self._clear_objects_result_received = False
        self._clear_objects_result_future = None

        self.object_pose_estimation_result = None
        self.ground_truth_pose_estimation = None

        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

        self._current_gear = None
        self._initial_hint_point_msg = Point()
        if self._initial_hint is not None:
            self._initial_hint_point_msg.x = self._initial_hint['x']
            self._initial_hint_point_msg.y = self._initial_hint['y']
            self._initial_hint_point_msg.z = self._initial_hint['z']

        self._joint_state_publisher = self.node.create_publisher(
            JointState, '/isaac_joint_commands', 10)

        self._gripper_close_pos_publisher = self.node.create_publisher(
            Float64, '/gripper_close_pos', 10)

        self._switch_sim_controller_to_rl_policy_publisher = self.node.create_publisher(
            Bool, '/stop_joint_commands', 10)

        self._sim_joint_commands_publisher = self.node.create_publisher(
            JointState, '/isaac_joint_commands', 10)

        self._gripper_client = ActionClient(
            self.node, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self._gripper_done_event = Event()
        self._gripper_done_result = False

        self.peg_pose = None

    def _set_controller_states(self,
                               controllers_to_activate: List[str] = None,
                               controllers_to_deactivate: List[str] = None,
                               timeout_sec: float = 10.0) -> bool:
        """
        Set the state of ROS2 controllers by activating/deactivating them.

        This function uses the controller manager's switch_controller service to
        enable and disable controllers.

        Args
        ----
            node: ROS2 node to use for service calls
            controllers_to_activate: List of controller names to activate
            controllers_to_deactivate: List of controller names to deactivate
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if all controller state changes were successful, False otherwise

        Example
        -------
            # Switch from joint trajectory to impedance control
            success = self._set_controller_states(
                node=self.node,
                controllers_to_activate=['impedance_controller'],
                controllers_to_deactivate=['scaled_joint_trajectory_controller']
            )

        """
        if SwitchController is None:
            self.node.get_logger().error(
                'controller_manager_msgs not available. '
                'Install ros-humble-controller-manager-msgs')
            return False

        if controllers_to_activate is None:
            controllers_to_activate = []
        if controllers_to_deactivate is None:
            controllers_to_deactivate = []

        if not controllers_to_activate and not controllers_to_deactivate:
            self.node.get_logger().warn('No controllers specified to activate or deactivate')
            return True

        # Create service client
        service_name = '/controller_manager/switch_controller'
        client = self.node.create_client(SwitchController, service_name)

        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().error(f'Service {service_name} not available '
                                         f'after {timeout_sec} seconds')
            return False

        # Create request
        request = SwitchController.Request()
        request.activate_controllers = controllers_to_activate
        request.deactivate_controllers = controllers_to_deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = rclpy.duration.Duration(seconds=timeout_sec).to_msg()

        self.node.get_logger().info(
            f'Switching controllers - Activating: {controllers_to_activate}, '
            f'Deactivating: {controllers_to_deactivate}')

        # Send request
        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)

            if future.result() is None:
                self.node.get_logger().error('Failed to call controller switch service')
                return False

            result = future.result()
            if result.ok:
                self.node.get_logger().info('Successfully switched controllers')
                return True
            else:
                self.node.get_logger().error('Controller switch failed')
                return False

        except Exception as e:
            self.node.get_logger().error(f'Exception during controller switch: {str(e)}')
            return False
        finally:
            self.node.destroy_client(client)

    def switch_to_impedance_control(self, timeout_sec: float = 10.0) -> bool:
        """
        Switch from joint trajectory control to impedance control.

        Args
        ----
            node: ROS2 node to use for service calls
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if switch was successful, False otherwise

        """
        return self._set_controller_states(
            controllers_to_activate=['impedance_controller'],
            controllers_to_deactivate=['scaled_joint_trajectory_controller'],
            timeout_sec=timeout_sec
        )

    def switch_to_trajectory_control(self, timeout_sec: float = 10.0) -> bool:
        """
        Switch from impedance control to joint trajectory control.

        Args
        ----
            node: ROS2 node to use for service calls
            timeout_sec: Timeout for service call in seconds

        Returns
        -------
            bool: True if switch was successful, False otherwise

        """
        return self._set_controller_states(
            controllers_to_activate=['scaled_joint_trajectory_controller'],
            controllers_to_deactivate=['impedance_controller'],
            timeout_sec=timeout_sec
        )

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            raise e

    def trigger_get_objects_goal(self, initial_hint: None | Point = None):
        """Send a goal to the object detection action server."""
        self.node.get_logger().info('Waiting for action server for object detection...')
        if not self._detection_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error(
                'Action server for object detection not available after waiting')
            return

        goal_msg = GetObjects.Goal()
        if initial_hint is not None:
            goal_msg.initial_hint = initial_hint
            goal_msg.use_initial_hint = True

        self.node.get_logger().info('Sending goal to detect objects...')
        send_goal_future = self._detection_client.send_goal_async(
            goal_msg, feedback_callback=self.get_objects_feedback_callback)
        send_goal_future.add_done_callback(self.get_objects_goal_response_callback)

    def get_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_objects_goal_handle = future.result()
        if not self._get_objects_goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the action server.')
            return

        self.node.get_logger().info('Goal accepted by the action server.')
        self._get_objects_result_future = self._get_objects_goal_handle.get_result_async()
        self._get_objects_result_future.add_done_callback(self.get_objects_result_callback)

    def get_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        object_ids = []
        if result.objects:
            self.node.get_logger().info('Objects detected, the objects id are below:')
            for object_obj in result.objects:
                self.node.get_logger().info(f'Object: {object_obj.object_id}')
                object_ids.append(object_obj.object_id)

        if status == GoalStatus.STATUS_SUCCEEDED and len(result.objects) > 0:
            self._get_objects_detected_object_id = result.objects[0].object_id
            self._get_objects_result_received = True
        else:
            self.node.get_logger().error(f'Action failed with status: {status}')
            self._get_objects_result_received = False

    def trigger_get_object_pose_goal(self, object_id):
        """Send a goal to the object pose estimation action server."""
        self.node.get_logger().info('Waiting for pose estimation action server...')
        if not self._get_object_pose_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Pose estimation action server not available')
            return

        goal_msg = GetObjectPose.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to estimate pose for object ID: {object_id}')
        send_goal_future = self._get_object_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.get_object_pose_feedback_callback)
        send_goal_future.add_done_callback(self.get_object_pose_goal_response_callback)

    def get_object_pose_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._get_object_pose_goal_handle = future.result()
        if not self._get_object_pose_goal_handle.accepted:
            self.node.get_logger().error('Pose estimation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Pose estimation goal accepted by the action server.')
        self._get_object_pose_result_future = self._get_object_pose_goal_handle.get_result_async()
        self._get_object_pose_result_future.add_done_callback(self.get_object_pose_result_callback)

    def get_object_pose_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Pose Estimation Feedback: {feedback_msg.feedback}')

    def get_object_pose_result_callback(self, future):
        """Handle the result from the action server."""
        status = future.result().status
        result = future.result().result
        self.node.get_logger().info(f'Pose estimation result: {result}')

        # Log result to INFO
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Object pose received for object id')
            self._get_object_pose_result_received = True
            self.latest_pose_gotten = result.object_pose
            # base_T_gear_large_estimated, this is probably Pose
            # Convert it to ROS transform.
            self.object_pose_estimation_result = geometry_utils.ros_pose_to_ros_transform(
                result.object_pose)
        else:
            self.node.get_logger().error(f'Pose estimation action failed with status: {status}')
            self._get_object_pose_result_received = False

    def trigger_get_segmented_object_goal(self, object_id):
        """Send a goal to the object segmentation action server."""
        self.node.get_logger().info('Waiting for segmentation action server...')
        if not self._segmentation_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('Segmentation action server not available after waiting')
            return

        goal_msg = AddSegmentationMask.Goal()
        goal_msg.object_id = object_id
        self.node.get_logger().info(f'Sending goal to get segmented object for ID: {object_id}')
        send_goal_future = self._segmentation_client.send_goal_async(
            goal_msg, feedback_callback=self.get_segmented_object_feedback_callback)
        send_goal_future.add_done_callback(self.get_segmented_object_goal_response_callback)

    def get_segmented_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._segmentation_goal_handle = future.result()
        if not self._segmentation_goal_handle.accepted:
            self.node.get_logger().error('Segmentation goal was rejected by the action server.')
            return

        self.node.get_logger().info('Segmentation goal accepted by the action server.')
        self._segmentation_result_future = self._segmentation_goal_handle.get_result_async()
        self._segmentation_result_future.add_done_callback(
            self.get_segmented_object_result_callback)

    def get_segmented_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Segmentation Feedback: {feedback_msg.feedback}')

    def get_segmented_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Segmented object received: {result}')
            self._segmentation_result_received = True
        else:
            self.node.get_logger().error(f'Segmentation action failed with status: {status}')
            self._segmentation_result_received = False

    def trigger_add_mesh_to_object_request(self, object_id, mesh_file_path):
        request = AddMeshToObject.Request()
        request.object_ids = [object_id]
        request.mesh_file_paths = [mesh_file_path]
        future = self._add_mesh_client.call_async(request)
        return future

    def send_clear_objects_request(self, object_ids=None):
        request = ClearObjects.Request()
        if object_ids:
            request.object_ids = object_ids
        future = self._clear_objects_client.call_async(request)
        return future

    def add_mesh_to_object_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._add_mesh_goal_handle = future.result()
        if not self._add_mesh_goal_handle.accepted:
            self.node.get_logger().error('Add mesh goal was rejected by the action server.')
            return

        self.node.get_logger().info('Add mesh goal accepted by the action server.')
        self._add_mesh_result_future = self._add_mesh_goal_handle.get_result_async()
        self._add_mesh_result_future.add_done_callback(self.add_mesh_to_object_result_callback)

    def add_mesh_to_object_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Add Mesh Feedback: {feedback_msg.feedback}')

    def add_mesh_to_object_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        # Log result to INFO
        self.node.get_logger().info(f'Add mesh result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f'Mesh added to object: {result.object_id}')
            self._add_mesh_result_received = True
        else:
            self.node.get_logger().error(f'Add mesh action failed with status: {status}')
            self._add_mesh_result_received = False

    def clear_objects_goal_response_callback(self, future):
        """Return the goal response from action server."""
        self._clear_objects_goal_handle = future.result()
        if not self._clear_objects_goal_handle.accepted:
            self.node.get_logger().error('Clear objects goal was rejected by the action server.')
            return

        self.node.get_logger().info('Clear objects goal accepted by the action server.')
        self._clear_objects_result_future = self._clear_objects_goal_handle.get_result_async()
        self._clear_objects_result_future.add_done_callback(self.clear_objects_result_callback)

    def clear_objects_feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        self.node.get_logger().info(f'Clear Objects Feedback: {feedback_msg.feedback}')

    def clear_objects_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        status = future.result().status

        self.node.get_logger().info(f'Clear objects result: {result}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('Objects cleared successfully')
            self._clear_objects_result_received = True
        else:
            self.node.get_logger().error(f'Clear objects action failed with status: {status}')
            self._clear_objects_result_received = False

    def wait_for_result(self, result_var_name: str, timeout: float = 10.0):
        start_time = self.node.get_clock().now()
        while rclpy.ok():
            # Check the actual instance variable dynamically
            if result_var_name and getattr(self, result_var_name):
                return True

            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                # self.fail(f'Timeout waiting for {result_var_name}')
                self.failure_count += 1
                return False
            self.total_count += 1

            rclpy.spin_once(self.node, timeout_sec=0.1)

    def wait_for_service_result(self, future, timeout=10.0):
        start_time = self.node.get_clock().now()
        while not future.done():
            time_now = self.node.get_clock().now()
            if (time_now - start_time).nanoseconds / 1e9 > timeout:
                self.node.get_logger().error('Timeout waiting for service result')
                return None
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return future.result()

    def set_gear_mesh_param(self, mesh_file_path: str) -> bool:
        """
        Set the mesh_file_path parameter for the foundationpose_node based on the current gear.

        Args
        ----
            mesh_file_path: Path to the mesh file.

        Returns
        -------
            bool: True if the mesh_file_path parameter was set successfully, False otherwise.

        """
        # Create a parameter client
        param_client = self.node.create_client(
            SetParameters,
            '/foundationpose_node/set_parameters'
        )

        # Wait for the service to be available
        if not param_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn(
                'Service /foundationpose_node/set_parameters not available'
                ' skipping mesh parameter setting')
            return False

        # Create parameter request
        request = SetParameters.Request()
        parameter = RCLPYParameter()
        parameter.name = 'mesh_file_path'
        parameter.value = ParameterValue(
            string_value=mesh_file_path,
            type=rclpy.Parameter.Type.STRING.value)

        request.parameters = [parameter]

        # Send request
        future = param_client.call_async(request)

        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)

        if future.result() is None:
            self.node.get_logger().warn(f'Failed to set mesh_file_path to {mesh_file_path}')
            return False
        self.node.get_logger().debug(f'Successfully set mesh_file_path to {mesh_file_path}')
        return True

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Pose): The grasp pose to publish.
            parent_frame (str): The parent frame.
            child_frame (str): The child frame.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.node.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.node.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def publish_pose_on_tf(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """Publish the grasp transform in the world frame."""
        if grasp_pose is None:
            self.node.get_logger().error('Grasp pose should never be None')
            return

        if self._publish_only_on_static_tf:
            return self.publish_pose_on_tf_static(grasp_pose, parent_frame, child_frame)

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_stamped)
        self.node.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def wait_for_point_topic_func(self):
        """Wait for a message on the point topic."""
        self.received_messages[self._point_topic_name_as_trigger] = []
        self.node.get_logger().error('Waiting for point topic input for next step...')
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.node.get_logger().info('Point topic received')
                return

    def do_perception_loop(self, initial_hint: Point, mesh_file_path: str):
        # First detect object
        self.trigger_get_objects_goal(initial_hint)

        # Wait for the detection result with a timeout
        self.wait_for_result('_get_objects_result_received',
                             timeout=self._max_timeout_time_for_action_call)
        if not self._get_objects_result_received:
            self.node.get_logger().error('Object detection action failed')
            self.failure_count += 1
            return False

        self.total_count += 1
        # If segment anything is enabled, we need to get the segmented object first
        if self._is_segment_anything_segmentation_enabled:
            self.trigger_get_segmented_object_goal(self._get_objects_detected_object_id)
            self.wait_for_result('_segmentation_result_received',
                                 timeout=self._max_timeout_time_for_action_call)

        # Now add mesh to object
        if self._is_segment_anything_segmentation_enabled:
            future = self.trigger_add_mesh_to_object_request(
                self._get_objects_detected_object_id, mesh_file_path)
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.node.get_logger().info(f'Add mesh to object result: {result}')
            if result is None:
                self.node.get_logger().error('Add mesh to object action failed')
                self.failure_count += 1

                return False
            self.total_count += 1

        # Now get object pose
        self.trigger_get_object_pose_goal(self._get_objects_detected_object_id)
        self.wait_for_result('_get_object_pose_result_received',
                             timeout=self._max_timeout_time_for_action_call)

        self._get_objects_result_received = False
        self._segmentation_result_received = False
        self._add_mesh_result_received = False
        self._get_object_pose_result_received = False
        return True

    def do_perception_for_pose_estimation(self, mesh_file_path: str):
        if self._wait_for_point_topic and self.peg_pose is None:

            initial_hint_for_peg_stand_estimation = None
            initial_hint_for_gear_insertion = None

            self.node.get_logger().error(
                'Waiting for point topic input for Peg stand estimation...')
            # Wait to get a message on the point topic.
            while True:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                    self.node.get_logger().info('Point topic received')
                    self.initial_hint_point_msg = \
                        self.received_messages[self._point_topic_name_as_trigger][-1]
                    initial_hint_for_peg_stand_estimation = self.initial_hint_point_msg
                    self.received_messages[self._point_topic_name_as_trigger] = []
                    break

        if self.peg_pose is None:

            # Do perception loop.
            self.node.get_logger().info(
                'Setting mesh file path for peg stand estimation'
                f': {self._mesh_file_path_for_peg_stand_estimation}')

            self.set_gear_mesh_param(self._mesh_file_path_for_peg_stand_estimation)

            self.node.get_logger().info('Doing perception loop for peg stand estimation')

            is_peg_stand_estimation_success = self.do_perception_loop(
                initial_hint_for_peg_stand_estimation,
                self._mesh_file_path_for_peg_stand_estimation)
            while not is_peg_stand_estimation_success:
                self.node.get_logger().error('Peg stand estimation action failed')
                is_peg_stand_estimation_success = self.do_perception_loop(
                    initial_hint_for_peg_stand_estimation,
                    self._mesh_file_path_for_peg_stand_estimation)
                self.node.get_logger().info('Peg stand estimation action failed, retrying...')

            # This is w.r.t to camera, but we need to convert it to base link.
            self.peg_pose = self.latest_pose_gotten

            self.received_messages[self._point_topic_name_as_trigger] = []
            self.node.get_logger().info(f'Peg pose: {self.peg_pose}')

            if self.peg_pose is None:
                self.fail('Pose estimation failed for peg stand')

            # Now publish this on TF Static with parent farme being
            # camera_frame and child frame being gear_assembly_est_frame.
            self.node.get_logger().info('Publishing peg pose on TF Static')
            self.publish_pose_on_tf_static(self.peg_pose,
                                           parent_frame=self._camera_prim_name_in_tf,
                                           child_frame='gear_assembly_frame')

        self.received_messages[self._point_topic_name_as_trigger] = []
        self.node.get_logger().error('Waiting for point topic for gear insertion...')

        # Wait to get a message on the point topic.
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.node.get_logger().info('Point topic received')
                self.initial_hint_point_msg = \
                    self.received_messages[self._point_topic_name_as_trigger][-1]
                initial_hint_for_gear_insertion = self.initial_hint_point_msg
                break

        self.node.get_logger().info(
            f'Setting mesh file path for gear insertion: {mesh_file_path}')
        self.set_gear_mesh_param(mesh_file_path)
        self.node.get_logger().info('Doing perception loop for gear insertion')
        is_gear_estimation_success = self.do_perception_loop(initial_hint_for_gear_insertion,
                                                             mesh_file_path)
        while not is_gear_estimation_success:
            self.node.get_logger().info('Gear estimation action failed, retrying...')
            is_gear_estimation_success = self.do_perception_loop(
                initial_hint_for_gear_insertion, mesh_file_path)
            time.sleep(2)

        self.gear_pose = self.latest_pose_gotten

        if self.gear_pose is None:
            self.fail('Pose estimation failed for gear')

    def update_gripper_close_pos(self, close_gripper_pos: float):
        """Update the gripper close position."""
        self.node.get_logger().info(f'Updating gripper close position to {close_gripper_pos}')

        for _ in range(10):
            self._gripper_close_pos_publisher.publish(Float64(data=close_gripper_pos))
            rclpy.spin_once(self.node, timeout_sec=0.1)
            self.node.get_logger().info('Robot has updated its gripper close position')

        self.node.get_logger().info('Robot has updated its gripper close position')

    def pick_and_place(
        self,
        object_id: int,
        class_id: str = '',
        place_pose: Pose = None,
        gripper_closed_position: float = 0.623,
        use_joint_space_planner: bool = False,
        keep_gripper_closed_after_completion: bool = False,
        only_perform_place: bool = False
    ) -> bool:
        """
        Pick and place an object using the PickAndPlace action.

        Args
        ----
            node: The node to use for the behavior.
            object_id: The ID of the object to pick.
            class_id: The class ID of the object (optional).
            place_pose: The pose where to place the object
            use_joint_space_planner: Whether to use the joint space planner.
            keep_gripper_closed_after_completion: Whether to keep the gripper closed after
                completion.

        Returns
        -------
            True if the behavior was successful, False otherwise.

        """
        pick_and_place_client = ActionClient(self.node, PickAndPlace, '/pick_and_place')

        # Wait for the action server to be available
        if not pick_and_place_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error('PickAndPlace action server not available')
            return False

        # Create the goal
        pick_and_place_goal = PickAndPlace.Goal()
        pick_and_place_goal.object_id = object_id
        pick_and_place_goal.class_id = class_id
        pick_and_place_goal.use_joint_space_planner_for_place_pose = use_joint_space_planner
        pick_and_place_goal.target_joint_state_for_place_pose = \
            self._target_joint_state_for_place_pose
        pick_and_place_goal.gripper_closed_position = gripper_closed_position
        pick_and_place_goal.keep_gripper_closed_after_completion = \
            keep_gripper_closed_after_completion
        pick_and_place_goal.place_pose = place_pose
        pick_and_place_goal.only_perform_place = only_perform_place

        # Send the goal
        self.node.get_logger().info(f'Sending PickAndPlace goal for object_id: {object_id}')
        goal_future = pick_and_place_client.send_goal_async(pick_and_place_goal)

        # Wait for the goal to be accepted
        rclpy.spin_until_future_complete(self.node, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error('PickAndPlace goal was rejected')
            return False

        self.node.get_logger().info('PickAndPlace goal accepted')

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result()

        if not result.result.success:
            self.node.get_logger().error(f'PickAndPlace failed with status: {result.result}')
            return False

        self.node.get_logger().info(f'PickAndPlace completed successfully: {result.result}')
        return True

    def rl_insertion(
        self,
        peg_pose: Pose
    ) -> bool:
        """
        Insert an object into a peg using the RL insertion action.

        Args
        ----
            peg_pose: The pose of the peg to insert the object into.

        Returns
        -------
            True if the insertion was successful, False otherwise.

        """
        insertion_client = ActionClient(self.node, Insert, '/gear_assembly/insert_policy')

        if not insertion_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error('Insertion action server not available')
            return False

        goal = Insert.Goal()
        goal.goal_pose = PoseStamped()
        goal.goal_pose.pose = peg_pose
        goal.goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.goal_pose.header.frame_id = 'base_link'  # This is base or base link.
        goal_future = insertion_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self.node, goal_future)

        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error('Insertion goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        return True

    def _switch_sim_controller_to_rl_policy(self, is_rl_policy: bool):
        """Switch the sim controller to the RL policy."""
        # Just publish this Bool msg to a topic 5 times.
        for _ in range(5):
            self._switch_sim_controller_to_rl_policy_publisher.publish(Bool(data=is_rl_policy))
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.node.get_logger().info('Switched sim controller to RL policy')

    def _take_simulation_robot_to_home_position(self):
        """Take the simulation robot to the home position."""
        # Publish a joint topic to sim joint commands topic.
        joint_positions = [
            2.7424,
            -0.9247,
            1.2466,
            -1.8927,
            -1.5708,
            -1.8895
        ]
        # Create JointState message
        joint_state = JointState()
        joint_state.position = joint_positions
        joint_state.velocity = [0.0] * len(joint_positions)
        joint_state.effort = [0.0] * len(joint_positions)

        # Set joint names based on the number of joints (assuming UR10e joint names)
        joint_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.node.get_logger().info(f'Taking robot to home position: {joint_positions}')

        self.node.get_logger().info('Publishing joint state to sim joint commands topic')
        self._sim_joint_commands_publisher.publish(joint_state)

        self.node.get_logger().info('Done publishing joint state to sim joint commands topic')

    def trigger_gripper(self, position: float, max_effort: float):
        """
        Trigger gripper to either open or close.

        Args
        ----
            position (float): Position
            max_effort (float): Max effort

        Returns
        -------
            None

        """
        self._gripper_done_event.clear()
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(position)
        gripper_goal.command.max_effort = float(max_effort)

        gripper_goal_future = self._gripper_client.send_goal_async(gripper_goal)
        gripper_goal_future.add_done_callback(self.gripper_goal_response_cb)

    def gripper_goal_response_cb(self, future):
        """
        Get goal response for Gripper.

        Args
        ----
            future (_type_): Future for goal response

        Returns
        -------
            None

        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error('Goal Rejected for GripperCommand')
            self._gripper_done_result = False
            self._gripper_done_event.set()
            return
        # Cache the goal handle for canceling
        self.node.get_logger().info('Goal accepted for GripperCommand')
        gripper_result_future = goal_handle.get_result_async()
        gripper_result_future.add_done_callback(self.gripper_goal_result_cb)

    def gripper_goal_result_cb(self, future: Any):
        """
        Get the final result from gripper trigger action call.

        Args
        ----
            future (Any): Future for goal result

        Returns
        -------
            None

        """
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info('GripperCommand action server succeeded.')
            self._gripper_done_result = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.node.get_logger().error('GripperCommand action server aborted.')
            self._gripper_done_result = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.node.get_logger().error('GripperCommand action server canceled.')
            self._gripper_done_result = False
        else:
            self.node.get_logger().error('GripperCommand action server failed.')
            self._gripper_done_result = False

        self._gripper_done_event.set()

    def _open_gripper(self):
        """Open the gripper."""
        self.trigger_gripper(position=0.0, max_effort=10.0)

    def test_for_servers_pol(self):
        """Test that verifies that servers action calls under multiple cycles."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setUpClients()

        if self._wait_for_point_topic:
            self.received_messages = {}
            self._subs = self.create_logging_subscribers(
                [(self._point_topic_name_as_trigger, Point)],
                self.received_messages,
                use_namespace_lookup=False,
                accept_multiple_messages=True,
                qos_profile=rclpy.qos.qos_profile_sensor_data)
            self.initial_hint_point_msg = None

        self.node.get_logger().info('Starting test for manipulator servers POL')

        # spin for 50 seconds.
        start_time = time.time()
        while time.time() - start_time < 30:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        gears_to_insert = [
            'gear_shaft_large',
            'gear_shaft_small',
            'gear_shaft_medium'
        ]

        gear_ground_truth_sim_names = [
            'gear_large',
            'gear_small',
            'gear_medium'
        ]

        for idx, gear_type in enumerate(gears_to_insert):
            if not self._use_ground_truth_pose_estimation:
                self.node.get_logger().error('Waiting for user input to trigger perception')
                self.wait_for_point_topic_func()

            mesh_file_path_for_gear = self._mesh_file_paths[idx]
            close_gripper_pos = self._gripper_close_pos[idx]

            if self._use_sim_time:
                # We need to do this as we have a control layer for the gripper in Isaac Sim.
                self.update_gripper_close_pos(close_gripper_pos)
                close_gripper_pos = self._gripper_close_pos[idx]
            else:
                # For real robot this doesnt matter, the gripper doesn't matter, it doesnt cause
                # the gear to fly out of the grasp.
                close_gripper_pos = 0.65

            if not self._use_ground_truth_pose_estimation:
                self.do_perception_for_pose_estimation(mesh_file_path_for_gear)

            if self._use_sim_time and self._use_ground_truth_pose_estimation:

                # Spin a few times
                for _ in range(10):
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                # base_T_gear_large_actual, do it w.r.t camera since we get that pose
                # out of the FP node and thats the one the server gives you.
                self.ground_truth_pose_estimation_held_asset = self._get_transform_from_tf(
                    gear_ground_truth_sim_names[idx],
                    'base_link')

                self.gear_pose = self.ground_truth_pose_estimation_held_asset

            if self._use_sim_time and self._verify_pose_estimation_accuracy:
                # Get transformation matrix from ROS2 pose. 4 x 4 matrix.
                object_pose_estimation_transform = \
                    geometry_utils.get_transformation_matrix_from_ros(
                        self.object_pose_estimation_result.translation,
                        self.object_pose_estimation_result.rotation)
                ground_truth_pose_estimation_transform = \
                    geometry_utils.get_transformation_matrix_from_ros(
                        self.ground_truth_pose_estimation.translation,
                        self.ground_truth_pose_estimation.rotation)

                # Now do the check of the diff from ground truth.
                delta_transform = geometry_utils.compute_transformation_delta(
                    object_pose_estimation_transform, ground_truth_pose_estimation_transform)
                translation_norm_delta_cm, translation_ros_delta_mrad = \
                    geometry_utils.compute_delta_translation_rotation(delta_transform)
                self.node.get_logger().info(
                    f'Translation norm delta: {translation_norm_delta_cm} cm')
                self.node.get_logger().info(
                    f'Translation delta: {translation_ros_delta_mrad} mrad')

                self.ground_truth_pose_estimation_w_r_t_base = self._get_transform_from_tf(
                    gear_ground_truth_sim_names[idx], 'base_link')
                T_base = geometry_utils.get_transformation_matrix_from_ros(
                    self.ground_truth_pose_estimation_w_r_t_base.translation,
                    self.ground_truth_pose_estimation_w_r_t_base.rotation)

                self.node.get_logger().info(f'T_base translation in meters: {T_base[:3, 3]}')

                assert False, 'Print this'

            if not self._use_sim_time:
                self.switch_to_trajectory_control()

            if not self._use_ground_truth_pose_estimation:
                self.node.get_logger().error('Waiting for user input to trigger pick and place')
                self.wait_for_point_topic_func()

            # Spin a few times
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            place_pose_transform = self._get_transform_from_tf(
                gear_type, 'base_link')  # TODO change back to gear assembly

            place_pose = geometry_utils.get_pose_from_transform(place_pose_transform)

            # Rotate place pose by 180 degrees about x to make Z axis face down.
            # Then subtract 15 cm from z axis so that aligns well on top of peg.
            rotated_place_pose = geometry_utils.rotate_pose(place_pose, 180, 'x')
            rotated_place_pose.position.z += self._peg_stand_shaft_offset_for_cumotion

            self.node.get_logger().info('Doing gear pickup and place')

            self.publish_pose_on_tf(rotated_place_pose,
                                    parent_frame='base_link',
                                    child_frame='place_pose_static_frame')

            # Also publish detectedobject 1 on TF static
            if not self._use_ground_truth_pose_estimation:
                self.publish_pose_on_tf(self.gear_pose,
                                        parent_frame=self._camera_prim_name_in_tf,
                                        child_frame='detected_object1')
            else:
                gear_pose_in_pose_msg = geometry_utils.get_pose_from_transform(self.gear_pose)

                gear_pose_in_pose_msg.position.z += 0.02  # Add 2 cms
                self.publish_pose_on_tf(gear_pose_in_pose_msg,
                                        parent_frame='base_link',
                                        child_frame='detected_object1')
                self.node.get_logger().info(
                    f'Published detected object 1 on TF static: {gear_pose_in_pose_msg}')
                self._get_objects_detected_object_id = 0
                for _ in range(10):
                    rclpy.spin_once(self.node, timeout_sec=0.1)

            # Now do PickAndHover action on that object using get objects detected id.
            is_place_and_hover_success = self.pick_and_place(
                object_id=self._get_objects_detected_object_id,
                gripper_closed_position=close_gripper_pos,
                place_pose=rotated_place_pose,
                use_joint_space_planner=self._use_joint_space_planner_api,
                keep_gripper_closed_after_completion=True
            )

            if not is_place_and_hover_success:
                self.node.get_logger().error('Pick and place action failed')
                self.failure_count += 1
                self.fail('Pick and place action failed')

            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            # Use base here because policy sees base.
            peg_pose_transform = self._get_transform_from_tf(
                gear_type, 'base')

            peg_pose = geometry_utils.get_pose_from_transform(peg_pose_transform)

            # # Publish this pose for sanity under a new name on tf static
            self.publish_pose_on_tf_static(peg_pose,
                                           parent_frame='base',
                                           child_frame='rl_insertion_pose_frame')
            self.node.get_logger().info(f'Peg pose: {peg_pose}')

            if self._run_rl_inference:
                if not self._use_sim_time:
                    self.switch_to_impedance_control()
                    # Wait to get a message on the point topic.
                    self.node.get_logger().error(
                        'Waiting for confirmation to start RL policy...'
                        f'click any point on image. Will try to insert on TF: {gear_type}')
                    self.wait_for_point_topic_func()
                else:
                    self._switch_sim_controller_to_rl_policy(is_rl_policy=True)

                is_rl_insertion_success = self.rl_insertion(
                    peg_pose=peg_pose
                )

                if not is_rl_insertion_success:
                    self.node.get_logger().error('RL insertion action failed')
                    self.fail('RL insertion action failed')

                self.node.get_logger().info('RL insertion action completed successfully')
                if self._use_sim_time:
                    # Open gripper.
                    self._open_gripper()
                    self._gripper_done_event.wait(timeout=10.0)

                    self._take_simulation_robot_to_home_position()
                    self.node.get_logger().info('Waiting for 5 seconds')
                    rclpy.spin_once(self.node, timeout_sec=5.0)
                    self.node.get_logger().info('Done waiting')
                    # Switching back to cuMotion controller.
                    self._switch_sim_controller_to_rl_policy(is_rl_policy=False)
                    # Reset the gripper done event and result.
                    self._gripper_done_event.clear()
                    self._gripper_done_result = False

            else:
                continue

            # Now clear objects
            future = self.send_clear_objects_request()
            result = self.wait_for_service_result(
                future, timeout=self._max_timeout_time_for_action_call)
            self.node.get_logger().info(f'Clear objects result: {result}')
            if result is None:
                self.node.get_logger().error('Clear objects action failed')
                self.failure_count += 1
                continue
            self.total_count += 1

            # Reset for next iteration
            self._clear_objects_result_received = False

        self.node.get_logger().info(f'Failure count: {self.failure_count}')
        assert self.failure_count/self.total_count < 0.3, f'Pose estimation action failed' \
            f'{self.failure_count}/{self.total_count} times'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  num_cycles: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  is_segment_anything_object_detection_enabled: bool,
                                  is_segment_anything_segmentation_enabled: bool,
                                  is_rt_detr_object_detection_enabled: bool,
                                  initial_hint: Dict,
                                  mesh_file_paths: List[str],
                                  ground_truth_sim_prim_name: str,
                                  camera_prim_name_in_tf: str,
                                  output_dir: str,
                                  node_startup_delay: float,
                                  peg_stand_shaft_offset_for_cumotion: float,
                                  point_topic_name_as_trigger: str = 'input_points_debug',
                                  wait_for_point_topic: bool = False,
                                  use_ground_truth_pose_estimation: bool = False,
                                  verify_pose_estimation_accuracy: bool = False,
                                  run_rl_inference: bool = False,
                                  publish_only_on_static_tf: bool = False,
                                  use_joint_space_planner_api: bool = False,
                                  target_joint_state_for_place_pose: JointState = None,
                                  gripper_close_pos: List[float] = [0.50, 0.64, 0.52],
                                  mesh_file_path_for_peg_stand_estimation: str = ''):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._num_cycles = num_cycles
        cls._use_sim_time = use_sim_time
        cls._is_segment_anything_object_detection_enabled = \
            is_segment_anything_object_detection_enabled
        cls._is_segment_anything_segmentation_enabled = is_segment_anything_segmentation_enabled
        cls._is_rt_detr_object_detection_enabled = is_rt_detr_object_detection_enabled
        cls._initial_hint = initial_hint
        cls._output_dir = output_dir
        cls._mesh_file_paths = mesh_file_paths
        cls._mesh_file_path_for_peg_stand_estimation = mesh_file_path_for_peg_stand_estimation
        cls._wait_for_point_topic = wait_for_point_topic
        cls._point_topic_name_as_trigger = point_topic_name_as_trigger
        cls._ground_truth_sim_prim_name = ground_truth_sim_prim_name
        cls._camera_prim_name_in_tf = camera_prim_name_in_tf
        cls._use_ground_truth_pose_estimation = use_ground_truth_pose_estimation
        cls._verify_pose_estimation_accuracy = verify_pose_estimation_accuracy
        cls._run_rl_inference = run_rl_inference
        cls._gripper_close_pos = gripper_close_pos
        cls._use_joint_space_planner_api = use_joint_space_planner_api
        cls._target_joint_state_for_place_pose = target_joint_state_for_place_pose
        cls._publish_only_on_static_tf = publish_only_on_static_tf
        cls._peg_stand_shaft_offset_for_cumotion = peg_stand_shaft_offset_for_cumotion

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class IsaacManipulatorGearInsertionPolTest(IsaacROSBaseTest):
    """
    Test for Isaac ROS Gear Insertion POL.

    This test will do RL insertion action on the object.
    """

    DEFAULT_NAMESPACE = ''
    _max_timeout_time_for_action_call: float = 10.0
    _ground_truth_sim_prim_name: str = 'gear_large'
    _use_sim_time: bool = False
    _run_test: bool = False
    _wait_for_point_topic: bool = False
    _point_topic_name_as_trigger: str = ''
    _peg_stand_shaft_offset_for_cumotion: float

    def setUpClients(self) -> None:
        """Set up before each test method."""
        self.failure_count = 0
        self.total_count = 0

        self.latest_pose_gotten = None

        self.object_pose_estimation_result = None
        self.ground_truth_pose_estimation = None

        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)

        self._current_gear = None
        self._initial_hint_point_msg = Point()

        self.peg_pose = None

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            raise e

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Pose): The grasp pose to publish.
            parent_frame (str): The parent frame.
            child_frame (str): The child frame.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.node.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.node.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def wait_for_point_topic_func(self):
        """Wait for a message on the point topic."""
        self.received_messages[self._point_topic_name_as_trigger] = []
        self.node.get_logger().error('Waiting for point topic input for next step...')
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                self.node.get_logger().info('Point topic received')
                return

    def rl_insertion(
        self,
        peg_pose: Pose
    ) -> bool:
        """
        Insert an object into a peg using the RL insertion action.

        Args
        ----
            peg_pose: The pose of the peg to insert the object into.

        Returns
        -------
            True if the insertion was successful, False otherwise.

        """
        insertion_client = ActionClient(self.node, Insert, '/gear_assembly/insert_policy')

        if not insertion_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error('Insertion action server not available')
            return False

        goal = Insert.Goal()
        goal.goal_pose = PoseStamped()
        goal.goal_pose.pose = peg_pose
        goal.goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.goal_pose.header.frame_id = 'base'
        goal_future = insertion_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self.node, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.node.get_logger().error('Insertion goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result()

        self.node.get_logger().info(f'Insertion completed successfully: {result.result}')
        return True

    def test_for_servers_pol(self):
        """Test that verifies that servers action calls under multiple cycles."""
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setUpClients()

        if self._wait_for_point_topic:
            self.received_messages = {}
            self._subs = self.create_logging_subscribers(
                [(self._point_topic_name_as_trigger, Point)],
                self.received_messages,
                use_namespace_lookup=False,
                accept_multiple_messages=True,
                qos_profile=rclpy.qos.qos_profile_sensor_data)
            self.initial_hint_point_msg = None

        self.node.get_logger().info('Starting test for manipulator servers POL')

        # spin for 30 seconds.
        start_time = time.time()
        while time.time() - start_time < 30:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        gears_to_insert = [
            'gear_shaft_large',
        ]

        for _, gear_type in enumerate(gears_to_insert):

            # Spin a few times
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)
            # Use base here because policy sees base.
            peg_pose_transform = self._get_transform_from_tf(
                child_frame=gear_type, parent_frame='base')

            peg_pose = geometry_utils.get_pose_from_transform(peg_pose_transform)

            # # Publish this pose for sanity under a new name on tf static
            self.publish_pose_on_tf_static(peg_pose,
                                           parent_frame='base',
                                           child_frame='rl_insertion_pose_frame')
            self.node.get_logger().info(f'Peg pose: {peg_pose}')

            # Wait to get a message on the point topic.
            self.node.get_logger().error(
                'Waiting for confirmation to start RL policy...'
                f'click any point on image. Will try to insert on TF: {gear_type}')
            self.wait_for_point_topic_func()

            is_rl_insertion_success = self.rl_insertion(
                peg_pose=peg_pose
            )

            if not is_rl_insertion_success:
                self.node.get_logger().error('RL insertion action failed')
                self.fail('RL insertion action failed')

            self.node.get_logger().info('RL insertion action completed successfully')

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  max_timeout_time_for_action_call: int,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  ground_truth_sim_prim_name: str,
                                  output_dir: str,
                                  node_startup_delay: float,
                                  peg_stand_shaft_offset_for_cumotion: float,
                                  point_topic_name_as_trigger: str = 'input_points_debug',
                                  wait_for_point_topic: bool = False):
        cls._run_test = run_test
        cls._max_timeout_time_for_action_call = max_timeout_time_for_action_call
        cls._use_sim_time = use_sim_time
        cls._output_dir = output_dir
        cls._wait_for_point_topic = wait_for_point_topic
        cls._point_topic_name_as_trigger = point_topic_name_as_trigger
        cls._ground_truth_sim_prim_name = ground_truth_sim_prim_name
        cls._peg_stand_shaft_offset_for_cumotion = peg_stand_shaft_offset_for_cumotion

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )


class CollectTargetLessCalibrationDataTest(IsaacROSBaseTest):
    """Test for Collect Target Less Calibration Data."""

    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0
    _camera_topic_names: List[str] = []
    _camera_info_topic_names: List[str] = []
    _base_frame_name: str = ''
    _joints_to_query: List[str] = []
    _urdf_file_path: str = ''
    _output_dir: str = ''
    _joint_topic_name: str = '/joint_states'

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  nodes: list[Node],
                                  use_sim_time: bool,
                                  node_startup_delay: float,
                                  camera_topic_names: List[str],
                                  camera_info_topic_names: List[str],
                                  base_frame_name: str,
                                  joints_to_query: List[str],
                                  urdf_file_path: str,
                                  output_dir: str,
                                  joint_topic_name: str):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._camera_topic_names = camera_topic_names
        cls._camera_info_topic_names = camera_info_topic_names
        cls._base_frame_name = base_frame_name
        cls._joints_to_query = joints_to_query
        cls._urdf_file_path = urdf_file_path
        cls._output_dir = output_dir
        cls._joint_topic_name = joint_topic_name

        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )

    def setUpClients(self) -> None:
        """Set up before each test method."""
        # Setup subscibers to images and camera info topics
        self.received_messages = {}
        topic_name_tuples = []

        self._received_image_topics = {}
        self._received_camera_info_topics = {}
        self._received_joint_transforms = {}

        for camera_topic_name in self._camera_topic_names:
            topic_name_tuples.append((camera_topic_name, Image))
            self._received_image_topics[camera_topic_name] = False

        for camera_info_topic_name in self._camera_info_topic_names:
            topic_name_tuples.append((camera_info_topic_name, CameraInfo))
            self._received_camera_info_topics[camera_info_topic_name] = False

        topic_name_tuples.append((self._joint_topic_name, JointState))

        for joint_name in self._joints_to_query:
            self._received_joint_transforms[joint_name] = False

        self._subs = self.create_logging_subscribers(
            topic_name_tuples,
            self.received_messages,
            use_namespace_lookup=False,
            accept_multiple_messages=True,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        # Set up TF subsriber
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        self._bridge = CvBridge()

    def _save_image_to_output_dir(self, data: Image, output_dir: str, camera_topic_name: str):
        """Save image to output directory."""
        # Save image to output directory
        # Use cvbridge to convert the image to cv2 format
        os.makedirs(output_dir, exist_ok=True)
        cv_image = np.array(self._bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        image_path = os.path.join(output_dir,
                                  camera_topic_name.replace('/', '_') + '.png')
        cv2.imwrite(image_path, cv_image)
        self.node.get_logger().info(f'Saved image to {image_path}')

    def _save_camera_info_to_output_dir(self,
                                        data: CameraInfo,
                                        output_dir: str,
                                        camera_info_topic_name: str):
        """Save camera info to output directory."""

        def camera_info_to_dict(camera_info_msg: CameraInfo) -> Dict[str, Any]:
            """Convert ROS CameraInfo message to dictionary for JSON serialization."""
            return {
                'header': {
                    'frame_id': camera_info_msg.header.frame_id
                },
                'width': camera_info_msg.width,
                'height': camera_info_msg.height,
                'distortion_model': camera_info_msg.distortion_model,
                'D': list(camera_info_msg.d),
                'K': list(camera_info_msg.k),
                'R': list(camera_info_msg.r),
                'P': list(camera_info_msg.p)
            }
        # Save camera info to output directory
        os.makedirs(output_dir, exist_ok=True)
        camera_info_path = os.path.join(output_dir,
                                        camera_info_topic_name.replace('/', '_') + '.json')
        with open(camera_info_path, 'w') as f:
            json.dump(camera_info_to_dict(data), f)
        self.node.get_logger().info(f'Saved camera info to {camera_info_path}')

    def _save_joint_transform_to_output_dir(self,
                                            data: TransformStamped,
                                            output_dir: str,
                                            joint_name: str):
        """Save joint transform to output directory."""
        # Save joint transform to output directory
        os.makedirs(output_dir, exist_ok=True)
        joint_transform_path = os.path.join(output_dir, joint_name + '.json')

        # Convert the transform to a dictionary
        transform_dict = {
            'translation': {
                'x': data.transform.translation.x,
                'y': data.transform.translation.y,
                'z': data.transform.translation.z
            },
            'rotation': {
                'x': data.transform.rotation.x,
                'y': data.transform.rotation.y,
                'z': data.transform.rotation.z,
                'w': data.transform.rotation.w
            },
            'parent_frame': data.header.frame_id,
            'child_frame': data.child_frame_id
        }

        with open(joint_transform_path, 'w') as f:
            json.dump(transform_dict, f)
        self.node.get_logger().info(f'Saved joint transform to {joint_transform_path}')

    def test_run_calibration_data_collection_test(self):

        self.setUpClients()

        # spin the node for 1000 seconds
        joint_angles_saved = False
        start_time = time.time()
        max_timeout_time_for_action_call = 10  # 10 seconds to get the data
        while True:
            if time.time() - start_time > max_timeout_time_for_action_call:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(1)
            self.node.get_logger().info('Spinning the node for 1000 seconds')

            # Now first check for all camera topics to have received messages
            for camera_topic_name in self._camera_topic_names:
                if not self._received_image_topics[camera_topic_name]:
                    # Check if the camera info topic has received a message
                    if (
                        self.received_messages[camera_topic_name] is not None and
                        len(self.received_messages[camera_topic_name]) > 0
                    ):
                        self._received_image_topics[camera_topic_name] = True
                        # Save image in output directory by first checking it to cv2.
                        self._save_image_to_output_dir(
                            data=self.received_messages[camera_topic_name][-1],
                            output_dir=self._output_dir,
                            camera_topic_name=camera_topic_name
                        )
                else:
                    continue

            # Now check for all camera info topics to have received messages
            for camera_info_topic_name in self._camera_info_topic_names:
                if not self._received_camera_info_topics[camera_info_topic_name]:
                    if (
                        self.received_messages[camera_info_topic_name] is not None and
                        len(self.received_messages[camera_info_topic_name]) > 0
                    ):
                        self._received_camera_info_topics[camera_info_topic_name] = True
                        # Save camera info in output directory by first checking it to cv2.
                        self._save_camera_info_to_output_dir(
                            data=self.received_messages[camera_info_topic_name][-1],
                            output_dir=self._output_dir,
                            camera_info_topic_name=camera_info_topic_name
                        )
                else:
                    continue

            if not joint_angles_saved:
                if (
                    self.received_messages[self._joint_topic_name] is not None and
                    len(self.received_messages[self._joint_topic_name]) > 0
                ):
                    joint_state = self.received_messages[self._joint_topic_name][0]
                    json_dict = {
                        'positions': list(joint_state.position),
                        'names': joint_state.name
                    }
                    with open(f'{self._output_dir}/joint_angles.json', 'w') as f:
                        json.dump(json_dict, f)
                    joint_angles_saved = True

            # Now check all joints and check if their transform is available w.r.t base link frame
            for joint_name in self._joints_to_query:
                if self._received_joint_transforms[joint_name]:
                    continue
                # Get the transform from the TF buffer
                try:
                    transform = self._tf_buffer.lookup_transform(
                        self._base_frame_name,
                        joint_name,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                except Exception as e:
                    self.node.get_logger().warn(f'Failed to get transform for {joint_name}: {e}')
                    continue

                if transform is not None:
                    # Save joint transform in output directory by first checking it to cv2.
                    self._save_joint_transform_to_output_dir(
                        data=transform,
                        output_dir=self._output_dir,
                        joint_name=joint_name
                    )
                    self._received_joint_transforms[joint_name] = True


class IsaacManipulatorPoseEstimationErrorPolTest(IsaacManipulatorGearAssemblyPolTest):
    """Test for Isaac ROS Pose Estimation Error POL."""

    def do_perception_for_pose_estimation(self, mesh_file_path: str):
        if self._wait_for_point_topic and self.peg_pose is None:

            initial_hint_for_peg_stand_estimation = None

            self.node.get_logger().error(
                'Waiting for point topic input for Peg stand estimation...')
            # Wait to get a message on the point topic.
            while True:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if len(self.received_messages[self._point_topic_name_as_trigger]) > 0:
                    self.node.get_logger().info('Point topic received')
                    self.initial_hint_point_msg = \
                        self.received_messages[self._point_topic_name_as_trigger][-1]
                    initial_hint_for_peg_stand_estimation = self.initial_hint_point_msg
                    self.received_messages[self._point_topic_name_as_trigger] = []
                    break

        if self.peg_pose is None:

            # Do perception loop.
            self.node.get_logger().info(
                'Setting mesh file path for peg stand estimation'
                f': {self._mesh_file_path_for_peg_stand_estimation}')

            self.set_gear_mesh_param(self._mesh_file_path_for_peg_stand_estimation)

            self.node.get_logger().info('Doing perception loop for peg stand estimation')

            is_peg_stand_estimation_success = self.do_perception_loop(
                initial_hint_for_peg_stand_estimation,
                self._mesh_file_path_for_peg_stand_estimation)
            while not is_peg_stand_estimation_success:
                self.node.get_logger().error('Peg stand estimation action failed')
                is_peg_stand_estimation_success = self.do_perception_loop(
                    initial_hint_for_peg_stand_estimation,
                    self._mesh_file_path_for_peg_stand_estimation)
                self.node.get_logger().info('Peg stand estimation action failed, retrying...')

            # This is w.r.t to camera, but we need to convert it to base link.
            self.peg_pose = self.latest_pose_gotten

            self.received_messages[self._point_topic_name_as_trigger] = []
            self.node.get_logger().info(f'Peg pose: {self.peg_pose}')

            # if self.peg_pose is None:
            #     self.fail('Pose estimation failed for peg stand')
            if self.peg_pose is None:
                self.node.get_logger().error('Pose estimation failed for peg stand')
            else:
                # Now publish this on TF Static with parent farme being
                # camera_frame and child frame being gear_assembly_est_frame.
                self.node.get_logger().info('Publishing peg pose on TF Static')
                self.publish_pose_on_tf_static(self.peg_pose,
                                               parent_frame=self._camera_prim_name_in_tf,
                                               child_frame='gear_assembly_frame')

    def test_for_pose_estimation_error_pol(self):
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setUpClients()
        hang_test_after_completion = True
        if self._wait_for_point_topic:
            self.received_messages = {}
            self._subs = self.create_logging_subscribers(
                [(self._point_topic_name_as_trigger, Point)],
                self.received_messages,
                use_namespace_lookup=False,
                accept_multiple_messages=True,
                qos_profile=rclpy.qos.qos_profile_sensor_data)
            self.initial_hint_point_msg = None

        self.node.get_logger().info('Starting test for manipulator pose estimation error POL')

        # spin for 50 seconds.
        start_time = time.time()
        while time.time() - start_time < 30:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        gears_to_insert = [
            'gear_shaft_large',
        ]

        for idx, gear_type in enumerate(gears_to_insert):
            if not self._use_ground_truth_pose_estimation:
                self.node.get_logger().error('Waiting for user input to trigger perception')
                self.wait_for_point_topic_func()

            mesh_file_path_for_gear = self._mesh_file_paths[idx]
            close_gripper_pos = self._gripper_close_pos[idx]
            close_gripper_pos = 0.65

            if not self._use_ground_truth_pose_estimation:
                self.do_perception_for_pose_estimation(mesh_file_path_for_gear)

            if not self._use_sim_time:
                self.switch_to_trajectory_control()

            if not self._use_ground_truth_pose_estimation:
                self.node.get_logger().error('Waiting for user input to trigger pick and place')
                self.wait_for_point_topic_func()

            # Spin a few times
            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            place_pose_transform = self._get_transform_from_tf(
                gear_type, 'base_link')

            place_pose = geometry_utils.get_pose_from_transform(place_pose_transform)

            # Rotate place pose by 180 degrees about x to make Z face down.
            # Then subtract 15 cm from z axis so that aligns well on top of peg.
            rotated_place_pose = geometry_utils.rotate_pose(place_pose, 180, 'x')
            rotated_place_pose.position.z += self._peg_stand_shaft_offset_for_cumotion

            self.node.get_logger().info('Doing gear pickup and place')

            self.publish_pose_on_tf(rotated_place_pose,
                                    parent_frame='base_link',
                                    child_frame='place_pose_static_frame')

            # Now do PickAndHover action on that object using get objects detected id.
            is_place_and_hover_success = self.pick_and_place(
                object_id=0,
                gripper_closed_position=close_gripper_pos,
                place_pose=rotated_place_pose,
                use_joint_space_planner=self._use_joint_space_planner_api,
                keep_gripper_closed_after_completion=True,
                only_perform_place=True
            )

            if not is_place_and_hover_success:
                self.node.get_logger().error('Pick and place action failed')
                self.failure_count += 1
                self.fail('Pick and place action failed')

            for _ in range(10):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            # Use base here because policy sees base.
            peg_pose_transform = self._get_transform_from_tf(
                gear_type, 'base')

            peg_pose = geometry_utils.get_pose_from_transform(peg_pose_transform)

            # Publish this pose for sanity under a new name on tf static
            self.publish_pose_on_tf_static(peg_pose,
                                           parent_frame='base',
                                           child_frame='rl_insertion_pose_frame')
            self.node.get_logger().info(f'Peg pose: {peg_pose}')

            if hang_test_after_completion:
                time.sleep(1000)


class PoseEstimationDifferentDepthBackendsTest(IsaacROSBaseTest):
    """Test for Isaac ROS Test Hardcoded Poses."""

    DEFAULT_NAMESPACE = ''
    _run_test: bool = False
    _use_sim_time: bool = False
    _node_startup_delay: float = 0.0
    _poses: list[dict] = []
    _hang_test_after_completion: bool = False

    def publish_pose_on_tf_static(self, grasp_pose: Pose, parent_frame: str, child_frame: str):
        """
        Publish the grasp transform in the world frame.

        Args
        ----
            grasp_pose (Pose): The grasp pose to publish.
            parent_frame (str): The parent frame.
            child_frame (str): The child frame.

        Returns
        -------
            None

        """
        if grasp_pose is None:
            self.node.get_logger().error('Grasp pose should never be None')
            return

        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = parent_frame
        transform_stamped.child_frame_id = child_frame

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = grasp_pose.position.x
        transform_stamped.transform.translation.y = grasp_pose.position.y
        transform_stamped.transform.translation.z = grasp_pose.position.z
        transform_stamped.transform.rotation.x = grasp_pose.orientation.x
        transform_stamped.transform.rotation.y = grasp_pose.orientation.y
        transform_stamped.transform.rotation.z = grasp_pose.orientation.z
        transform_stamped.transform.rotation.w = grasp_pose.orientation.w

        # Publish the transform
        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.node.get_logger().info(f'Published pose frame from {parent_frame} to {child_frame}')

    def _get_transform_from_tf(self, child_frame: str, parent_frame: str) -> TransformStamped:
        """Get the transform of a child frame relative to a parent frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame, child_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            return transform.transform
        except Exception as e:
            raise e

    def setup_clients(self):

        # Create tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)

    def test_for_different_depth_backends(self):
        if not self._run_test:
            self.node.get_logger().warn('RUN_TEST is not set to true')
            return

        self.setup_clients()
        self.node.get_logger().info('Starting test for different depth backends')

        for idx in range(len(self._poses)):

            pose_obj = self._poses[idx]
            pose = Pose()
            pose.position.x = pose_obj['position']['x']
            pose.position.y = pose_obj['position']['y']
            pose.position.z = pose_obj['position']['z']
            pose.orientation.x = pose_obj['orientation']['x']
            pose.orientation.y = pose_obj['orientation']['y']
            pose.orientation.z = pose_obj['orientation']['z']
            pose.orientation.w = pose_obj['orientation']['w']

            self.node.get_logger().info(f'Pose: {pose_obj}')

            # First publish on tf static under a name.
            self.publish_pose_on_tf_static(
                pose,
                parent_frame=pose_obj['frame_id'],
                child_frame=pose_obj['pose_name_for_tf']
            )

            # spin a few times.
            for _ in range(100):
                rclpy.spin_once(self.node, timeout_sec=0.1)

        if self._hang_test_after_completion:
            time.sleep(1000)

    @classmethod
    def generate_test_description(cls, run_test: bool,
                                  use_sim_time: bool,
                                  nodes: list[Node],
                                  node_startup_delay: float,
                                  poses: list[dict],
                                  hang_test_after_completion: bool):
        cls._run_test = run_test
        cls._use_sim_time = use_sim_time
        cls._node_startup_delay = node_startup_delay
        cls._poses = poses
        cls._hang_test_after_completion = hang_test_after_completion
        return super().generate_test_description(
            nodes=nodes,
            node_startup_delay=node_startup_delay
        )
