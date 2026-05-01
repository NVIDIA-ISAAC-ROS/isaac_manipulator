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
End-to-end planning test for the drop-pose phase.

Launches the real cuMotion planner alongside a mock nvblox ESDF service that
returns a voxel grid with an obstacle block centered on the target drop pose.
The mock records every clearing request and, if one arrives, removes the
matching voxels from its internal obstacle set so subsequent ESDF responses
report them as free space.

With AABB clearing disabled, the obstacle block survives into cuMotion's
planning world and IK cannot find a collision-free solution at the drop
pose. With AABB clearing enabled, cuMotion forwards the clearing geometry to
the mock, the voxels at the drop pose fall out of the obstacle set, and
planning succeeds.
"""

import math
import os
import threading
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose
from isaac_ros_cumotion_interfaces.action import MotionPlan
import launch
from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
from nvblox_msgs.srv import EsdfAndGradients
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.executors
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


ROBOT_DESCRIPTION_PACKAGE = 'isaac_ros_cumotion_robot_description'
URDF_FILE_NAME = 'ur10e_robotiq_2f_140.urdf'
XRDF_FILE_NAME = 'ur10e_robotiq_2f_140.xrdf'
ESDF_SERVICE_NAME = '/nvblox_node/get_esdf_and_gradient'
STATIC_SCENE_SERVICE_NAME = '/publish_static_planning_scene'
MOTION_PLAN_ACTION = 'cumotion/motion_plan'
JOINT_STATES_TOPIC = '/cumotion_test/joint_states'

# Drop pose used throughout the test. Matches the shape of the example pose
# from the Pick and Place tutorial action goal.
DROP_POSE_X = 0.5
DROP_POSE_Y = 0.3
DROP_POSE_Z = 0.5
DROP_ORI_W = 0.017994
DROP_ORI_X = -0.677772
DROP_ORI_Y = 0.734752
DROP_ORI_Z = 0.020993

# Voxel grid parameters for the mock ESDF. A 1.2m cube centered near the base
# link, large enough to cover the reachable workspace.
VOXEL_SIZE = 0.05
GRID_EXTENT_VOXELS = 24  # 24 * 0.05 = 1.2m per side
GRID_ORIGIN = (-0.4, -0.6, -0.1)  # lower corner of the grid in base_link frame
GRID_FRAME = 'base_link'

# Obstacle volume around the drop pose. We fill a 9x9x9 voxel block (45cm
# cube) centered on the drop pose so the obstacle is large enough to occlude
# the drop pose regardless of minor discrepancies between how the mock and
# cuMotion's SDF grid interpret the flat Float32MultiArray's index ordering.
# Voxels far from the drop pose remain free so cuMotion's start state (home
# configuration) is not in collision and the arm has a feasible approach
# region.
OBSTACLE_HALF_VOXELS = 4  # 4 -> 9x9x9 block (45cm cube)
OBSTACLE_DISTANCE = -0.025  # inside obstacle, half a voxel deep
FREE_DISTANCE = 1.0  # 1m of clearance everywhere else

# Plausible UR10e home configuration. cuMotion needs a start joint state from
# somewhere to plan; we publish this to the test-specific joint_states topic.
UR10E_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
UR10E_JOINT_POSITIONS = [
    0.0,
    -math.pi / 2.0,
    math.pi / 2.0,
    -math.pi / 2.0,
    -math.pi / 2.0,
    0.0,
]


def world_to_voxel(point_xyz):
    ix = int((point_xyz[0] - GRID_ORIGIN[0]) / VOXEL_SIZE)
    iy = int((point_xyz[1] - GRID_ORIGIN[1]) / VOXEL_SIZE)
    iz = int((point_xyz[2] - GRID_ORIGIN[2]) / VOXEL_SIZE)
    return ix, iy, iz


def voxel_to_world(voxel_idx):
    return (
        GRID_ORIGIN[0] + (voxel_idx[0] + 0.5) * VOXEL_SIZE,
        GRID_ORIGIN[1] + (voxel_idx[1] + 0.5) * VOXEL_SIZE,
        GRID_ORIGIN[2] + (voxel_idx[2] + 0.5) * VOXEL_SIZE,
    )


def default_obstructed_voxels():
    """
    Return the initial obstacle voxel set centered at the drop pose.

    Fills a ``(2 * OBSTACLE_HALF_VOXELS + 1) ** 3`` voxel block around
    ``(DROP_POSE_X, DROP_POSE_Y, DROP_POSE_Z)`` and drops any voxels that fall
    outside the grid extent.
    """
    cx, cy, cz = world_to_voxel((DROP_POSE_X, DROP_POSE_Y, DROP_POSE_Z))
    obstructed = set()
    for ox in range(-OBSTACLE_HALF_VOXELS, OBSTACLE_HALF_VOXELS + 1):
        for oy in range(-OBSTACLE_HALF_VOXELS, OBSTACLE_HALF_VOXELS + 1):
            for oz in range(-OBSTACLE_HALF_VOXELS, OBSTACLE_HALF_VOXELS + 1):
                voxel = (cx + ox, cy + oy, cz + oz)
                if all(0 <= v < GRID_EXTENT_VOXELS for v in voxel):
                    obstructed.add(voxel)
    return obstructed


def populate_esdf_response(response, obstructed):
    """Fill the response with an ESDF grid built from the obstacle voxel set."""
    response.header.frame_id = GRID_FRAME
    response.origin_m = Point(
        x=float(GRID_ORIGIN[0]),
        y=float(GRID_ORIGIN[1]),
        z=float(GRID_ORIGIN[2]),
    )
    response.voxel_size_m = float(VOXEL_SIZE)

    array = Float32MultiArray()
    for label, size in (('x', GRID_EXTENT_VOXELS),
                        ('y', GRID_EXTENT_VOXELS),
                        ('z', GRID_EXTENT_VOXELS)):
        dim = MultiArrayDimension()
        dim.label = label
        dim.size = size
        dim.stride = 0  # cuMotion reads size only
        array.layout.dim.append(dim)

    total = GRID_EXTENT_VOXELS ** 3
    data = [FREE_DISTANCE] * total
    for (ix, iy, iz) in obstructed:
        # Linear index: cumotion_lib's SDF grid uses F-order (x fastest, z
        # slowest) when reading from the flat buffer. Using C-order here would
        # cause the mock's "obstacle voxels" to land at different world
        # locations than cuMotion expects, so clearing requests would not
        # match up.
        idx = ix + iy * GRID_EXTENT_VOXELS + iz * GRID_EXTENT_VOXELS * GRID_EXTENT_VOXELS
        data[idx] = OBSTACLE_DISTANCE
    array.data = data

    response.esdf_and_gradients = array
    response.success = True


class MockEsdfServer:
    """
    In-process nvblox ESDF service that tracks clearing requests.

    Runs its own rclpy node and SingleThreadedExecutor on a dedicated thread
    so that it stays responsive while the test fixture is waiting on action
    results from cuMotion. Clearing geometry received from cuMotion is applied
    to an internal obstacle set so that subsequent responses see the cleared
    voxels as free space.
    """

    def __init__(self):
        self._node = rclpy.create_node('mock_nvblox_esdf')
        self._service = self._node.create_service(
            EsdfAndGradients,
            ESDF_SERVICE_NAME,
            self._callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.received_requests = []
        self.obstructed = default_obstructed_voxels()

        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True, name='mock_esdf_spin')
        self._thread.start()

    def reset_obstructed(self):
        self.obstructed = default_obstructed_voxels()

    def shutdown(self):
        self._executor.shutdown()
        self._node.destroy_service(self._service)
        self._node.destroy_node()

    def _callback(self, request, response):
        self.received_requests.append(request)

        for center, radius in zip(
                request.spheres_to_clear_center_m,
                request.spheres_to_clear_radius_m):
            self._clear_sphere(center, radius)

        for minimum, size in zip(
                request.aabbs_to_clear_min_m, request.aabbs_to_clear_size_m):
            self._clear_aabb(minimum, size)

        populate_esdf_response(response, self.obstructed)
        return response

    def _clear_sphere(self, center, radius):
        radius_sq = radius * radius
        to_remove = set()
        for voxel in self.obstructed:
            wx, wy, wz = voxel_to_world(voxel)
            dx = wx - center.x
            dy = wy - center.y
            dz = wz - center.z
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                to_remove.add(voxel)
        self.obstructed -= to_remove

    def _clear_aabb(self, minimum, size):
        to_remove = set()
        for voxel in self.obstructed:
            wx, wy, wz = voxel_to_world(voxel)
            if (minimum.x <= wx <= minimum.x + size.x and
                    minimum.y <= wy <= minimum.y + size.y and
                    minimum.z <= wz <= minimum.z + size.z):
                to_remove.add(voxel)
        self.obstructed -= to_remove


@pytest.mark.rostest
def generate_test_description():
    """Launch description: cuMotion planner + static-planning-scene server in a container."""
    urdf_path = os.path.join(
        get_package_share_directory(ROBOT_DESCRIPTION_PACKAGE), 'urdf', URDF_FILE_NAME)
    xrdf_path = os.path.join(
        get_package_share_directory(ROBOT_DESCRIPTION_PACKAGE), 'xrdf', XRDF_FILE_NAME)

    static_scene = ComposableNode(
        name='static_planning_scene_server',
        package='isaac_ros_cumotion',
        plugin='nvidia::isaac_ros::cumotion::StaticPlanningSceneServer',
        parameters=[{'moveit_collision_objects_scene_file': ''}],
    )

    cumotion_planner = ComposableNode(
        name='cumotion_planner',
        package='isaac_ros_cumotion',
        plugin='nvidia::isaac_ros::cumotion::CumotionPlanner',
        parameters=[{
            'urdf_file_path': urdf_path,
            'xrdf_file_path': xrdf_path,
            'read_esdf_world': True,
            'update_esdf_on_request': True,
            'esdf_service_name': ESDF_SERVICE_NAME,
            'static_planning_scene_service_name': STATIC_SCENE_SERVICE_NAME,
            'static_scene_service_max_wait_attempts': 120,
            'joint_states_topic': JOINT_STATES_TOPIC,
            'add_ground_plane': False,
            'publish_world_collision_spheres': False,
            'publish_self_collision_spheres': False,
            'publish_cumotion_world_as_voxels': False,
        }],
    )

    # Match the production cumotion launch recipe: both components in ONE
    # single-threaded component_container. The container loads static_scene
    # first (quick constructor advertises the service), then cumotion_planner
    # (whose blocking constructor finds the service immediately).
    cumotion_container = ComposableNodeContainer(
        name='cumotion_test_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[static_scene, cumotion_planner],
        output='screen',
    )

    return launch.LaunchDescription([
        cumotion_container,
        TimerAction(period=2.0, actions=[launch_testing.actions.ReadyToTest()]),
    ])


def make_pose(x, y, z, qw, qx, qy, qz):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = float(qw)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    return pose


class CumotionPlanPoseEsdfClearingTest(unittest.TestCase):
    """Launches cuMotion + mock ESDF and drives two MotionPlan goals through it."""

    _test_node = None
    _joint_state_pub = None
    _mock_esdf = None
    _action_client = None
    _joint_state_thread = None
    _joint_state_stop = None

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls._test_node = rclpy.create_node('cumotion_esdf_test_client')

        cls._joint_state_pub = cls._test_node.create_publisher(
            JointState, JOINT_STATES_TOPIC, 10)

        cls._joint_state_stop = threading.Event()
        cls._joint_state_thread = threading.Thread(
            target=cls._joint_state_loop, daemon=True, name='joint_state_pub')
        cls._joint_state_thread.start()

        cls._mock_esdf = MockEsdfServer()

        cls._action_client = ActionClient(
            cls._test_node, MotionPlan, MOTION_PLAN_ACTION)

        deadline = time.time() + 120.0
        while not cls._action_client.wait_for_server(timeout_sec=1.0):
            if time.time() > deadline:
                raise RuntimeError(
                    f'{MOTION_PLAN_ACTION} action server never appeared within 120s')

    @classmethod
    def tearDownClass(cls):
        cls._joint_state_stop.set()
        cls._joint_state_thread.join(timeout=2.0)
        cls._mock_esdf.shutdown()
        cls._test_node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _joint_state_loop(cls):
        while not cls._joint_state_stop.is_set():
            msg = JointState()
            msg.header.stamp = cls._test_node.get_clock().now().to_msg()
            msg.name = UR10E_JOINT_NAMES
            msg.position = UR10E_JOINT_POSITIONS
            cls._joint_state_pub.publish(msg)
            time.sleep(0.05)

    def _send_plan_pose_goal(
            self,
            *,
            enable_aabb_clearing,
            object_shape='CUBOID',
            object_scale_xyz=(0.6, 0.6, 0.6),
            object_padding=(0.05, 0.05, 0.05)):
        goal = MotionPlan.Goal()
        goal.goal_pose.header.frame_id = 'base_link'
        goal.goal_pose.header.stamp = self._test_node.get_clock().now().to_msg()
        goal.goal_pose.poses.append(make_pose(
            DROP_POSE_X, DROP_POSE_Y, DROP_POSE_Z,
            DROP_ORI_W, DROP_ORI_X, DROP_ORI_Y, DROP_ORI_Z))
        goal.plan_pose = True
        goal.plan_grasp = False
        goal.plan_cspace = False
        goal.use_current_state = True
        goal.use_planning_scene = False
        goal.time_dilation_factor = 0.2
        goal.enable_aabb_clearing = enable_aabb_clearing
        goal.clear_esdf = enable_aabb_clearing
        goal.update_esdf = True
        # Default: a CUBOID clearing shape large enough to carve an opening
        # through the 45cm obstacle block. AABB clearing math is
        # indexing-independent which avoids ambiguity about how the flat ESDF
        # array is ordered. Callers can override to test smaller regions.
        goal.object_shape = object_shape
        goal.object_scale.x = float(object_scale_xyz[0])
        goal.object_scale.y = float(object_scale_xyz[1])
        goal.object_scale.z = float(object_scale_xyz[2])
        goal.object_esdf_clearing_padding = list(object_padding)

        goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self._test_node, goal_future, timeout_sec=15.0)
        goal_handle = goal_future.result()
        self.assertIsNotNone(goal_handle, 'MotionPlan goal_handle never arrived')
        self.assertTrue(goal_handle.accepted, 'cuMotion rejected the MotionPlan goal')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self._test_node, result_future, timeout_sec=60.0)
        result_wrapper = result_future.result()
        self.assertIsNotNone(result_wrapper, 'MotionPlan result never arrived')
        return result_wrapper.result

    def test_01_plan_pose_fails_when_drop_pose_is_obstructed_without_clearing(self):
        """Reproduces the NVBug symptom: obstructed drop pose, no clearing -> IK failure."""
        self._mock_esdf.received_requests.clear()
        self._mock_esdf.reset_obstructed()

        result = self._send_plan_pose_goal(enable_aabb_clearing=False)

        self.assertFalse(
            result.success,
            'cuMotion planned through the obstructed drop pose without AABB '
            'clearing; expected IK/collision failure.')
        self.assertGreaterEqual(
            len(self._mock_esdf.received_requests), 1,
            'cuMotion did not call the nvblox ESDF service for plan_pose')
        last_req = self._mock_esdf.received_requests[-1]
        self.assertEqual(
            len(last_req.spheres_to_clear_center_m), 0,
            'Clearing disabled, but spheres were sent to nvblox')
        self.assertEqual(
            len(last_req.aabbs_to_clear_min_m), 0,
            'Clearing disabled, but AABBs were sent to nvblox')

    def test_02_plan_pose_succeeds_when_aabb_clearing_removes_drop_pose_obstacle(self):
        """After clearing removes the obstacle, cuMotion finds a collision-free plan."""
        self._mock_esdf.received_requests.clear()
        self._mock_esdf.reset_obstructed()

        result = self._send_plan_pose_goal(enable_aabb_clearing=True)

        clearing_req = next(
            (req for req in self._mock_esdf.received_requests
             if len(req.spheres_to_clear_center_m) > 0
             or len(req.aabbs_to_clear_min_m) > 0),
            None,
        )
        self.assertIsNotNone(
            clearing_req,
            'cuMotion never forwarded a clearing request to nvblox when AABB '
            'clearing was enabled')
        self.assertTrue(
            result.success,
            'cuMotion failed to plan even after AABB clearing removed the '
            'drop-pose obstacle.')

    def test_03_plan_pose_fails_when_clearing_region_is_smaller_than_obstacle(self):
        """
        Captures QA's "enable_aabb_clearing=True but planning still fails" story.

        Uses the same 45cm obstacle block at the drop pose but asks cuMotion to
        clear only a 10cm SPHERE with no padding. The clearing request IS sent
        to nvblox (so enable_aabb_clearing was honored), but the sphere is too
        small to remove enough voxels around the drop pose and IK still fails.
        This matches the failure mode where users turn on clearing and then
        need to grow aabb_clearing_shape_scale or switch to CUSTOM_MESH before
        planning succeeds.
        """
        self._mock_esdf.received_requests.clear()
        self._mock_esdf.reset_obstructed()

        result = self._send_plan_pose_goal(
            enable_aabb_clearing=True,
            object_shape='SPHERE',
            object_scale_xyz=(0.1, 0.1, 0.1),
            object_padding=(0.0, 0.0, 0.0),
        )

        clearing_req = next(
            (req for req in self._mock_esdf.received_requests
             if len(req.spheres_to_clear_center_m) > 0
             or len(req.aabbs_to_clear_min_m) > 0),
            None,
        )
        self.assertIsNotNone(
            clearing_req,
            'enable_aabb_clearing=True but cuMotion never forwarded a clearing '
            'request to nvblox.')
        self.assertFalse(
            result.success,
            'cuMotion planned through the obstructed drop pose with only a '
            '10cm clearing sphere; expected planning to fail because the '
            'obstacle block extends well beyond that region.')
