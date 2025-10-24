# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test for the DetectObject behavior."""

import os
import time

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_orchestration.behaviors.perception_behaviors.detect_object import (
    DetectObject
)
from isaac_manipulator_test_utils.orchestration.behavior_base import BehaviorTestBase
from isaac_manipulator_test_utils.orchestration.timeouts import (
    BEHAVIOR_TIMEOUT,
    DEFAULT_SERVER_TIMEOUT_CONFIG,
    get_node_startup_delay,
    LOG_INTERVAL,
    SPIN_TIMEOUT
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import py_trees
import pytest


RUN_TEST = True  # For now, always run this simple test


class DetectObjectTest(BehaviorTestBase):
    """Test DetectObject behavior with object detection server."""

    def test_detect_object_behavior(self):
        """Test that the DetectObject behavior successfully detects objects."""
        if not self.assert_test_should_run():
            return

        # Wait for the specified node startup delay
        time.sleep(self._node_startup_delay)

        self.node.get_logger().info(
            'Launch file started successfully, testing DetectObject behavior.')

        # Set up blackboard with empty object cache (will be populated by detection)
        # DetectObject uses BaseActionBehavior - needs server timeout config
        blackboard = self.setup_blackboard(
            object_info_cache={},
            server_timeout_config=DEFAULT_SERVER_TIMEOUT_CONFIG
        )

        # Create behavior with default detection configuration
        detect_object_behavior = DetectObject(
            name='Detect Object',
            action_server_name='/get_objects',
            detection_confidence_threshold=0.5
        )

        status = self.tick_tree_until_complete(
            detect_object_behavior,
            timeout_seconds=BEHAVIOR_TIMEOUT,
            spin_timeout=SPIN_TIMEOUT,
            log_interval=LOG_INTERVAL,
            tree_name='Simple Object Detection'
        )

        # Verify the behavior completed successfully
        self.assertEqual(status, py_trees.common.Status.SUCCESS)

        # Verify objects were detected
        object_info = blackboard.object_info_cache
        self.assertIsNotNone(object_info)
        self.assertIsInstance(object_info, dict)
        self.assertGreater(len(object_info), 0, 'At least one object should be detected')

        self.node.get_logger().info(
            f'DetectObject behavior test completed successfully! '
            f'Detected {len(object_info)} objects.')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for the DetectObject behavior test."""
    # Get the package directory
    test_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_orchestration'),
        'test', 'include'
    )

    # Set up the test nodes
    test_nodes = []

    if RUN_TEST:
        # Include the object detection test launch file
        test_nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(test_include_dir,
                                 'object_detection_test.launch.py')
                )
            )
        )
    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return DetectObjectTest.generate_test_description(
        run_test=RUN_TEST,
        use_sim_time=False,
        nodes=test_nodes,
        node_startup_delay=get_node_startup_delay(has_external_nodes=RUN_TEST)
    )
