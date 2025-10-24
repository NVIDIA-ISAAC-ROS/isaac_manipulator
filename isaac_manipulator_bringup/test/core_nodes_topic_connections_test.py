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
"""Tests the graph structure of core nodes in Isaac Manipulator."""

import os
from typing import Dict

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    FoundationPoseTopicConnectionsGraphTest, load_yaml_params
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'isaac_sim'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Cumotion, robot segmentor and nvblox nodes for testing."""
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch', 'workflows')
    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'params',
        'sim_launch_params.yaml'
    )
    params = load_yaml_params(test_yaml_config)

    override_params = {
        'workflow_type': 'PICK_AND_PLACE',
        'manual_mode': 'true',
        'enable_nvblox': 'true',
    }

    params.update(override_params)

    node_startup_delay = 1.0
    test_nodes = []

    if RUN_TEST:
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir, '/core.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
    else:
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))
    return CoreNodesTopicConnectionsGraphTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay
    )


class CoreNodesTopicConnectionsGraphTest(FoundationPoseTopicConnectionsGraphTest):
    """Test for graph structure of core nodes."""

    def _get_expected_graph(self) -> Dict:
        """
        Define the expected graph structure for core nodes.

        Returns
        -------
        Dict
            A dictionary representation of the expected node connections

        """
        expected_graph = {}

        # Initialize core nodes
        nodes_to_include = [
            '/cumotion_planner',
            '/robot_segmenter_1',
            '/nvblox_node',
            '/object_attachment',
            '/nitros_bridge_node_depth_1',
        ]

        for node_name in nodes_to_include:
            expected_graph[node_name] = {
                'publishes': [],
                'subscribes': []
            }

        # Add connections for cumotion_planner
        expected_graph['/cumotion_planner']['publishes'].extend([
            {'topic': '/curobo/voxels', 'to': []},
        ])

        expected_graph['/robot_segmenter_1']['publishes'].extend([
            {'topic': '/cumotion/camera_1/world_depth_bridge', 'to': []},
            {'topic': '/cumotion/camera_1/robot_mask_bridge', 'to': []},
        ])

        expected_graph['/nitros_bridge_node_depth_1']['publishes'].extend([
            {'topic': '/cumotion/camera_1/world_depth', 'to': []},
        ])

        # Add connections for nvblox_node
        expected_graph['/nvblox_node']['subscribes'].extend([
            {'topic': '/cumotion/camera_1/world_depth', 'from': ['/nitros_bridge_node_depth_1']},
            {'topic': '/cumotion/camera_1/world_depth/nitros', 'from': []},
            {'topic': '/front_stereo_camera/left/camera_info', 'from': []},
            {'topic': '/front_stereo_camera/left/image_raw', 'from': []},
            {'topic': '/front_stereo_camera/left/image_raw/nitros', 'from': []},
            {'topic': '/pointcloud', 'from': []},
            {'topic': '/pose', 'from': []}
        ])

        # Add connections for object_attachment
        expected_graph['/object_attachment']['subscribes'].extend([
            {'topic': '/cumotion/camera_1/world_depth', 'from': []},
            {'topic': '/front_stereo_camera/left/camera_info', 'from': []}
        ])

        # Add connections for robot_segmenter_1
        expected_graph['/robot_segmenter_1']['subscribes'].extend([
            {'topic': '/front_stereo_camera/left/camera_info', 'from': []},
            {'topic': '/front_stereo_camera/depth/ground_truth', 'from': []}
        ])

        # Log the expected graph nodes
        self.node.get_logger().info('Expected graph nodes:')
        for node_name in expected_graph.keys():
            self.node.get_logger().info(f'  - {node_name}')

        return expected_graph
