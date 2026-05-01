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

"""
End-to-end test for the drop-pose planning subtree.

Loads the shipped ``multi_object_pick_and_place_behavior_tree_params.yaml`` via
the real ``BehaviorTreeConfigInitializer``, builds the real drop subtree from
``drop_operations.create_plan_to_drop_subtree``, and verifies that the
``MotionPlan`` action goal reaching cuMotion carries the AABB-clearing
configuration declared in the YAML.
"""

import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_manipulation_orchestration.utils.behavior_tree_config import (
    BehaviorTreeConfigInitializer,
)
from isaac_ros_manipulation_pick_and_place.behavior_trees.motion_subtrees.drop_operations import (
    create_plan_to_drop_subtree,
)
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


PICK_AND_PLACE_PACKAGE = 'isaac_ros_manipulation_pick_and_place'
PARAMS_FILE_NAME = 'multi_object_pick_and_place_behavior_tree_params.yaml'
MOTION_PLAN_ACTION = 'cumotion/motion_plan'
TEST_MESH_RESOURCE = '/tmp/drop_flow_test_mesh.obj'
TEST_CLASS_ID = '22'
TEST_OBJ_ID = 1
TEST_DROP_POSE_X = 0.5
TEST_DROP_POSE_Y = 0.0
TEST_DROP_POSE_Z = 0.2


class DropFlowAabbClearingTest(BehaviorTestBase):
    """End-to-end drop-subtree AABB-clearing goal-wiring test."""

    def setUp(self):
        super().setUp()
        self._received_goals = []
        # Reentrant callback group lets the executor keep spinning the client
        # while the mock's execute_callback prepares the response.
        self._mock_server = ActionServer(
            self.node,
            MotionPlan,
            MOTION_PLAN_ACTION,
            execute_callback=self._execute_mock_motion_plan,
            callback_group=ReentrantCallbackGroup(),
        )

    def tearDown(self):
        self._mock_server.destroy()
        super().tearDown()

    def _execute_mock_motion_plan(self, goal_handle):
        """Record the incoming goal and return a successful trajectory."""
        self._received_goals.append(goal_handle.request)
        result = MotionPlan.Result()
        result.success = True
        result.error_code.val = 1  # MoveItErrorCodes.SUCCESS
        result.message = 'Mock motion plan succeeded'
        result.planned_trajectory = [RobotTrajectory()]
        goal_handle.succeed()
        return result

    def _load_real_config_initializer(self):
        params_path = os.path.join(
            get_package_share_directory(PICK_AND_PLACE_PACKAGE),
            'params',
            PARAMS_FILE_NAME,
        )
        return BehaviorTreeConfigInitializer(
            behavior_tree_params_file=params_path,
            package_name=PICK_AND_PLACE_PACKAGE,
        )

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

        # The recovery branch of the drop subtree is only ticked when the primary
        # planning path fails. The primary succeeds here because the mock server
        # returns SUCCESS immediately, so the home_pose / abort_motion keys below
        # are set only to satisfy key registration and are never read.
        return self.setup_blackboard(
            goal_drop_pose=drop_pose,
            active_obj_id=TEST_OBJ_ID,
            object_info_cache=object_info_cache,
            mesh_file_paths=mesh_file_paths,
            planning_scene=None,
            trajectory=None,
            server_timeout_config=DEFAULT_SERVER_TIMEOUT_CONFIG,
            abort_motion=False,
            home_pose=Pose(),
        )

    def test_drop_subtree_forwards_aabb_clearing_from_yaml(self):
        """
        Real YAML + real drop subtree must produce a goal with clearing enabled.

        The shipped YAML plus the real drop subtree must produce a MotionPlan
        goal with AABB clearing enabled and every clearing field populated
        from the YAML.
        """
        self._seed_blackboard()

        initializer = self._load_real_config_initializer()
        plan_to_pose_config = initializer.get_plan_to_pose_config()

        subtree = create_plan_to_drop_subtree(initializer)

        status = self.tick_tree_until_complete(
            subtree,
            timeout_seconds=BEHAVIOR_TIMEOUT,
            spin_timeout=SPIN_TIMEOUT,
            log_interval=LOG_INTERVAL,
            tree_name='Drop Subtree AABB Clearing Test',
        )

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertEqual(
            len(self._received_goals), 1,
            'Drop subtree must send exactly one MotionPlan goal')

        goal = self._received_goals[0]

        # Core contract for drop-pose planning with cuMotion + nvblox:
        # both flags must be True so cuMotion clears ESDF voxels around the
        # drop pose before IK.
        self.assertTrue(goal.enable_aabb_clearing)
        self.assertTrue(goal.clear_esdf)

        # Surrounding fields that define "plan to a single pose with clearing"
        self.assertTrue(goal.plan_pose)
        self.assertFalse(goal.plan_grasp)
        self.assertFalse(goal.plan_cspace)
        self.assertTrue(goal.update_esdf)

        # The clearing geometry must come from the YAML, not from hard-coded
        # defaults in the behavior constructor.
        self.assertEqual(goal.object_shape, plan_to_pose_config.aabb_clearing_shape)
        self.assertAlmostEqual(
            goal.object_scale.x,
            plan_to_pose_config.aabb_clearing_shape_scale[0], places=4)
        self.assertAlmostEqual(
            goal.object_scale.y,
            plan_to_pose_config.aabb_clearing_shape_scale[1], places=4)
        self.assertAlmostEqual(
            goal.object_scale.z,
            plan_to_pose_config.aabb_clearing_shape_scale[2], places=4)
        self.assertEqual(
            [round(p, 4) for p in goal.object_esdf_clearing_padding],
            [round(p, 4) for p in plan_to_pose_config.esdf_clearing_padding])

        # The drop pose and mesh resource must flow from the blackboard.
        self.assertEqual(len(goal.goal_pose.poses), 1)
        self.assertAlmostEqual(
            goal.goal_pose.poses[0].position.x, TEST_DROP_POSE_X, places=4)
        self.assertAlmostEqual(
            goal.goal_pose.poses[0].position.z, TEST_DROP_POSE_Z, places=4)
        self.assertEqual(goal.mesh_resource, TEST_MESH_RESOURCE)


@pytest.mark.rostest
def generate_test_description():
    """
    Launch description for the drop-flow test.

    The mock MotionPlan action server is created in-process inside ``setUp``.
    A dummy static_transform_publisher keeps the launch description alive until
    the tests run; without it, launch shuts down as soon as the startup timer
    fires and the tests never execute.
    """
    keep_alive = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='drop_flow_test_keep_alive',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
    )
    return DropFlowAabbClearingTest.generate_test_description(
        run_test=True,
        use_sim_time=False,
        nodes=[keep_alive],
        node_startup_delay=get_node_startup_delay(has_external_nodes=True),
    )
