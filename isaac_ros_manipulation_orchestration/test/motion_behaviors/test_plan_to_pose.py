# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests that PlanToPose forwards its config into the MotionPlan action goal."""

from geometry_msgs.msg import Pose
from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_manipulation_orchestration.behaviors.motion_behaviors import PlanToPose
from isaac_ros_manipulation_test_utils.orchestration.behavior_base import BehaviorTestBase
from isaac_ros_manipulation_test_utils.orchestration.timeouts import (
    BEHAVIOR_TIMEOUT,
    DEFAULT_SERVER_TIMEOUT_CONFIG,
    get_node_startup_delay,
    LOG_INTERVAL,
    SPIN_TIMEOUT,
)
from launch_ros.actions import Node
from moveit_msgs.msg import RobotTrajectory
import py_trees
import pytest
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup


ACTION_SERVER_NAME = 'cumotion/motion_plan'
TEST_MESH_RESOURCE = '/tmp/plan_to_pose_test_mesh.obj'
TEST_CLASS_ID = '22'
TEST_OBJ_ID = 1

TEST_DROP_POSE_X = 0.5
TEST_DROP_POSE_Y = 0.0
TEST_DROP_POSE_Z = 0.2

TEST_SHAPE = 'SPHERE'
TEST_SHAPE_SCALE = [0.1, 0.1, 0.1]
TEST_ESDF_PADDING = [0.05, 0.05, 0.05]


class PlanToPoseGoalWiringTest(BehaviorTestBase):
    """Verify that PlanToPose populates MotionPlan.Goal from its constructor args."""

    def setUp(self):
        super().setUp()
        self._received_goals = []
        # Run the goal accept and the execute callback on separate threads of
        # the action server so that spin_once can keep ticking the client while
        # execute_callback is preparing the response.
        self._mock_server = ActionServer(
            self.node,
            MotionPlan,
            ACTION_SERVER_NAME,
            execute_callback=self._execute_mock_motion_plan,
            callback_group=ReentrantCallbackGroup(),
        )

    def tearDown(self):
        self._mock_server.destroy()
        super().tearDown()

    def _execute_mock_motion_plan(self, goal_handle):
        """Record the incoming goal and return a successful result immediately."""
        self._received_goals.append(goal_handle.request)
        result = MotionPlan.Result()
        result.success = True
        result.error_code.val = 1  # MoveItErrorCodes.SUCCESS
        result.message = 'Mock motion plan succeeded'
        result.planned_trajectory = [RobotTrajectory()]
        goal_handle.succeed()
        return result

    def _seed_blackboard(self):
        drop_pose = Pose()
        drop_pose.position.x = TEST_DROP_POSE_X
        drop_pose.position.y = TEST_DROP_POSE_Y
        drop_pose.position.z = TEST_DROP_POSE_Z
        drop_pose.orientation.w = 1.0

        object_info_cache = {
            TEST_OBJ_ID: {
                'class_id': TEST_CLASS_ID,
                'object_frame_name': f'object_{TEST_OBJ_ID}',
                'status': 'NOT_READY',
            }
        }
        mesh_file_paths = {TEST_CLASS_ID: TEST_MESH_RESOURCE}

        return self.setup_blackboard(
            goal_drop_pose=drop_pose,
            active_obj_id=TEST_OBJ_ID,
            object_info_cache=object_info_cache,
            mesh_file_paths=mesh_file_paths,
            planning_scene=None,
            trajectory=None,
            server_timeout_config=DEFAULT_SERVER_TIMEOUT_CONFIG,
        )

    def _run_plan_to_pose(self, *, enable_aabb_clearing):
        """Instantiate, tick to completion, return the captured goal message."""
        self._seed_blackboard()

        behavior = PlanToPose(
            name='Plan To Drop Pose',
            action_server_name=ACTION_SERVER_NAME,
            link_name='base_link',
            time_dilation_factor=0.2,
            # update_planning_scene=False so the behavior does not require a
            # populated planning_scene on the blackboard.
            update_planning_scene=False,
            aabb_clearing_shape=TEST_SHAPE,
            aabb_clearing_shape_scale=TEST_SHAPE_SCALE,
            enable_aabb_clearing=enable_aabb_clearing,
            esdf_clearing_padding=TEST_ESDF_PADDING,
        )

        status = self.tick_tree_until_complete(
            behavior,
            timeout_seconds=BEHAVIOR_TIMEOUT,
            spin_timeout=SPIN_TIMEOUT,
            log_interval=LOG_INTERVAL,
            tree_name='Plan To Pose Goal Wiring Test',
        )

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertEqual(
            len(self._received_goals), 1,
            'Expected exactly one MotionPlan goal to reach the mock server')
        return self._received_goals[0]

    def test_plan_to_pose_forwards_aabb_clearing_enabled(self):
        """When clearing is enabled, goal.enable_aabb_clearing and clear_esdf are True."""
        goal = self._run_plan_to_pose(enable_aabb_clearing=True)

        self.assertTrue(goal.enable_aabb_clearing)
        self.assertTrue(goal.clear_esdf)

        # Surrounding fields that define "plan to a single pose with clearing"
        self.assertTrue(goal.plan_pose)
        self.assertFalse(goal.plan_grasp)
        self.assertFalse(goal.plan_cspace)
        self.assertTrue(goal.update_esdf)

        # The pose and clearing geometry must be populated from the blackboard.
        self.assertEqual(len(goal.goal_pose.poses), 1)
        self.assertAlmostEqual(
            goal.goal_pose.poses[0].position.x, TEST_DROP_POSE_X, places=4)
        self.assertAlmostEqual(
            goal.goal_pose.poses[0].position.z, TEST_DROP_POSE_Z, places=4)
        self.assertEqual(goal.object_shape, TEST_SHAPE)
        self.assertAlmostEqual(goal.object_scale.x, TEST_SHAPE_SCALE[0], places=4)
        self.assertAlmostEqual(goal.object_scale.y, TEST_SHAPE_SCALE[1], places=4)
        self.assertAlmostEqual(goal.object_scale.z, TEST_SHAPE_SCALE[2], places=4)
        self.assertEqual(
            [round(p, 4) for p in goal.object_esdf_clearing_padding],
            [round(p, 4) for p in TEST_ESDF_PADDING])
        self.assertEqual(goal.mesh_resource, TEST_MESH_RESOURCE)

    def test_plan_to_pose_forwards_aabb_clearing_disabled(self):
        """When clearing is disabled, goal.enable_aabb_clearing and clear_esdf are False."""
        goal = self._run_plan_to_pose(enable_aabb_clearing=False)

        self.assertFalse(goal.enable_aabb_clearing)
        self.assertFalse(goal.clear_esdf)

        # plan_pose is still the active planning mode — only the clearing flags flip.
        self.assertTrue(goal.plan_pose)
        self.assertFalse(goal.plan_grasp)


@pytest.mark.rostest
def generate_test_description():
    """
    Launch description for the plan-to-pose goal-wiring test.

    The mock MotionPlan action server is created in-process inside ``setUp``.
    A dummy static_transform_publisher is included so that the launch
    description has a live process keeping the launch alive until the tests
    run; without it, launch shuts down as soon as the startup timer fires and
    the tests never get a chance to execute.
    """
    keep_alive = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='plan_to_pose_test_keep_alive',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
    )
    return PlanToPoseGoalWiringTest.generate_test_description(
        run_test=True,
        use_sim_time=False,
        nodes=[keep_alive],
        node_startup_delay=get_node_startup_delay(has_external_nodes=True),
    )
