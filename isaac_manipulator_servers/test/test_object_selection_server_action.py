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

from isaac_manipulator_interfaces.action import GetSelectedObject
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectSelectionPolicy
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import pytest
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
import rclpy.action
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    object_selection_node = ComposableNode(
        package='isaac_manipulator_servers',
        plugin='nvidia::isaac::manipulation::ObjectSelectionServer',
        name='object_selection_server',
        namespace=ObjectSelectionServerActionTest.generate_namespace(),
        parameters=[{'action_name': 'get_selected_object',
                     'selection_policy': ObjectSelectionPolicy.FIRST.value}]
    )

    object_selection_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='object_selection_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[
            object_selection_node,
        ],
        output='screen'
    )
    return ObjectSelectionServerActionTest.generate_test_description(
        nodes=[object_selection_container],
        node_startup_delay=5.0  # Wait for the container to start
    )


class ObjectSelectionServerActionTest(IsaacROSBaseTest):
    """This test checks the functionality of the `object_selection_server` action."""

    def test_select_first(self):
        """Test that the selection policy is 'first'."""
        service_name = f'{self.generate_namespace()}/object_selection_server/set_parameters'
        set_param_client = self.node.create_client(SetParameters, service_name)
        assert set_param_client.wait_for_service(timeout_sec=1.0)
        request = SetParameters.Request()
        param = Parameter(name='selection_policy')
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = ObjectSelectionPolicy.FIRST.value
        request.parameters = [param]
        future = set_param_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        assert future.result().results[0].successful
        action_client = rclpy.action.ActionClient(
            self.node, GetSelectedObject, 'get_selected_object')
        assert action_client.wait_for_server(timeout_sec=1.0)
        detections = Detection2DArray()
        for score in [0.1, 0.9, 0.5]:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.score = score
            det.results.append(hyp)
            detections.detections.append(det)
        goal_msg = GetSelectedObject.Goal()
        goal_msg.detections = detections
        send_goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_goal_future)
        goal_handle = send_goal_future.result()
        assert goal_handle.accepted
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, get_result_future)
        result = get_result_future.result().result
        assert result is not None
        assert result.selected_detection.results[0].hypothesis.score == 0.1

    def test_select_highest_score(self):
        """Test that the selection policy is 'highest_score'."""
        service_name = (f'{self.generate_namespace()}/object_selection_server/set_parameters')
        set_param_client = self.node.create_client(SetParameters, service_name)
        assert set_param_client.wait_for_service(timeout_sec=1.0)
        request = SetParameters.Request()
        param = Parameter(name='selection_policy')
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = ObjectSelectionPolicy.HIGHEST_SCORE.value
        request.parameters = [param]
        future = set_param_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        assert future.result().results[0].successful
        action_client = rclpy.action.ActionClient(
            self.node, GetSelectedObject, 'get_selected_object')
        assert action_client.wait_for_server(timeout_sec=1.0)
        detections = Detection2DArray()
        for score in [0.1, 0.9, 0.5]:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.score = score
            det.results.append(hyp)
            detections.detections.append(det)
        goal_msg = GetSelectedObject.Goal()
        goal_msg.detections = detections
        send_goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_goal_future)
        goal_handle = send_goal_future.result()
        assert goal_handle.accepted
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, get_result_future)
        result = get_result_future.result().result
        assert result is not None
        assert result.selected_detection.results[0].hypothesis.score == 0.9

    def test_select_random(self):
        """Test that the selection policy is 'random'."""
        service_name = f'{self.generate_namespace()}/object_selection_server/set_parameters'
        set_param_client = self.node.create_client(SetParameters, service_name)
        assert set_param_client.wait_for_service(timeout_sec=1.0)
        request = SetParameters.Request()
        param = Parameter(name='selection_policy')
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = ObjectSelectionPolicy.RANDOM.value
        request.parameters = [param]
        future = set_param_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        assert future.result().results[0].successful
        action_client = rclpy.action.ActionClient(
            self.node, GetSelectedObject, 'get_selected_object')
        assert action_client.wait_for_server(timeout_sec=1.0)
        detections = Detection2DArray()
        for score in [0.1, 0.9, 0.5]:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.score = score
            det.results.append(hyp)
            detections.detections.append(det)
        goal_msg = GetSelectedObject.Goal()
        goal_msg.detections = detections
        send_goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_goal_future)
        goal_handle = send_goal_future.result()
        assert goal_handle.accepted
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, get_result_future)
        result = get_result_future.result().result
        assert result is not None
        expected_scores = {0.1, 0.9, 0.5}
        score = result.selected_detection.results[0].hypothesis.score
        assert score in expected_scores
