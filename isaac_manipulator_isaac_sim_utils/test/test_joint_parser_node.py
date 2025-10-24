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

import time
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import pytest
import rclpy
from sensor_msgs.msg import JointState


TIMEOUT = 10  # seconds


class IsaacROSJointParserTest(IsaacROSBaseTest):
    """This test checks the functionality of the `joint_parser` node."""

    def test_joint_parser_basic_functionality(self):
        """
        Test JointParser node basic functionality.

        1. Send joint states with various joints including ignored ones
        2. Verify that ignored joints are filtered out
        3. Verify that finger_joint is processed correctly (clamped to 0.62)
        4. Verify output format is correct
        """
        received_messages = {}

        # Generate namespace lookups for topics
        self.generate_namespace_lookup([
            'isaac_joint_states',
            'isaac_parsed_joint_states',
        ])

        # Create publisher for input joint states
        joint_state_pub = self.node.create_publisher(
            JointState, self.namespaces['isaac_joint_states'],
            qos_profile=10)

        # Create subscriber for output joint states
        subs = self.create_logging_subscribers(
            [('isaac_parsed_joint_states', JointState)],
            received_messages,
            accept_multiple_messages=True,
            qos_profile=10
        )

        try:
            # Wait for the node to be ready
            time.sleep(1.0)

            # Create test joint state message with various joints
            input_joint_state = JointState()
            input_joint_state.header.stamp = self.node.get_clock().now().to_msg()
            input_joint_state.header.frame_id = 'base_link'

            # Add joints that should be kept
            input_joint_state.name = [
                'joint1',
                'joint2',
                'finger_joint',  # This should be processed specially
                'left_outer_finger_joint',  # This should be ignored
                'right_outer_finger_joint',  # This should be ignored
                'left_inner_finger_pad_joint',  # This should be ignored
                'right_inner_finger_pad_joint',  # This should be ignored
                'left_inner_knuckle_joint',  # This should be ignored
                'left_inner_finger_joint',  # This should be ignored
                'right_outer_knuckle_joint',  # This should be ignored
                'right_inner_knuckle_joint',  # This should be ignored
                'right_inner_finger_joint',  # This should be ignored
                'joint3'  # This should be kept
            ]

            input_joint_state.position = [
                0.1,  # joint1
                0.2,  # joint2
                0.8,  # finger_joint (should be clamped to 0.62)
                0.3,  # left_outer_finger_joint (ignored)
                0.4,  # right_outer_finger_joint (ignored)
                0.5,  # left_inner_finger_pad_joint (ignored)
                0.6,  # right_inner_finger_pad_joint (ignored)
                0.7,  # left_inner_knuckle_joint (ignored)
                0.8,  # left_inner_finger_joint (ignored)
                0.9,  # right_outer_knuckle_joint (ignored)
                1.0,  # right_inner_knuckle_joint (ignored)
                1.1,  # right_inner_finger_joint (ignored)
                0.3   # joint3
            ]

            input_joint_state.velocity = [0.1] * len(input_joint_state.name)
            input_joint_state.effort = [0.2] * len(input_joint_state.name)

            # Publish the joint state
            joint_state_pub.publish(input_joint_state)

            # Wait for the output message
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if (
                    'isaac_parsed_joint_states' in received_messages
                    and len(received_messages['isaac_parsed_joint_states']) > 0
                ):
                    break

            # Verify we received the output message
            self.assertIn('isaac_parsed_joint_states', received_messages)
            self.assertGreater(len(received_messages['isaac_parsed_joint_states']), 0)

            output_joint_state = received_messages['isaac_parsed_joint_states'][0]

            # Verify header is preserved
            self.assertEqual(output_joint_state.header.frame_id, 'base_link')

            # Verify only non-ignored joints are present
            expected_joints = ['joint1', 'joint2', 'finger_joint', 'joint3']
            self.assertEqual(set(output_joint_state.name), set(expected_joints))

            # Verify joint positions are correct
            joint_positions = dict(zip(output_joint_state.name, output_joint_state.position))
            self.assertAlmostEqual(joint_positions['joint1'], 0.1, places=5)
            self.assertAlmostEqual(joint_positions['joint2'], 0.2, places=5)
            self.assertAlmostEqual(joint_positions['joint3'], 0.3, places=5)

            # Verify finger_joint is clamped to 0.62
            self.assertAlmostEqual(joint_positions['finger_joint'], 0.62, places=5)

            # Verify velocity and effort values are preserved for non-ignored joints
            for i, joint_name in enumerate(output_joint_state.name):
                self.assertAlmostEqual(output_joint_state.velocity[i], 0.1, places=5)
                self.assertAlmostEqual(output_joint_state.effort[i], 0.2, places=5)

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(joint_state_pub)

    def test_joint_parser_finger_joint_below_threshold(self):
        """
        Test JointParser node with finger_joint below threshold.

        Verify that finger_joint values below 0.62 are not clamped.
        """
        received_messages = {}

        # Generate namespace lookups for topics
        self.generate_namespace_lookup([
            'isaac_joint_states',
            'isaac_parsed_joint_states',
        ])

        # Create publisher for input joint states
        joint_state_pub = self.node.create_publisher(
            JointState, self.namespaces['isaac_joint_states'],
            qos_profile=10)

        # Create subscriber for output joint states
        subs = self.create_logging_subscribers(
            [('isaac_parsed_joint_states', JointState)],
            received_messages,
            accept_multiple_messages=True,
            qos_profile=10
        )

        try:
            # Wait for the node to be ready
            time.sleep(1.0)

            # Create test joint state message with finger_joint below threshold
            input_joint_state = JointState()
            input_joint_state.header.stamp = self.node.get_clock().now().to_msg()
            input_joint_state.header.frame_id = 'base_link'

            input_joint_state.name = ['finger_joint', 'joint1']
            input_joint_state.position = [0.5, 0.3]  # finger_joint below 0.62 threshold
            input_joint_state.velocity = [0.1, 0.2]
            input_joint_state.effort = [0.3, 0.4]

            # Publish the joint state
            joint_state_pub.publish(input_joint_state)

            # Wait for the output message
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if (
                    'isaac_parsed_joint_states' in received_messages
                    and len(received_messages['isaac_parsed_joint_states']) > 0
                ):
                    break

            # Verify we received the output message
            self.assertIn('isaac_parsed_joint_states', received_messages)
            self.assertGreater(len(received_messages['isaac_parsed_joint_states']), 0)

            output_joint_state = received_messages['isaac_parsed_joint_states'][0]

            # Verify finger_joint is not clamped when below threshold
            joint_positions = dict(zip(output_joint_state.name, output_joint_state.position))
            self.assertAlmostEqual(joint_positions['finger_joint'], 0.5, places=5)
            self.assertAlmostEqual(joint_positions['joint1'], 0.3, places=5)

        finally:
            for sub in subs:
                self.node.destroy_subscription(sub)
            self.node.destroy_publisher(joint_state_pub)


@pytest.mark.rostest
def generate_test_description():
    """Generate test description."""
    joint_parser_node = Node(
        name='joint_parser',
        package='isaac_manipulator_isaac_sim_utils',
        executable='isaac_sim_joint_parser_node.py',
        namespace=IsaacROSJointParserTest.generate_namespace(),
    )

    return IsaacROSJointParserTest.generate_test_description([joint_parser_node])
