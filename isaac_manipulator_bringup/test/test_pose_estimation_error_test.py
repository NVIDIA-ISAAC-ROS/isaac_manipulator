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
"""
This test will make robot go to a pose that is on top of the peg stand.

Then the user can visually estimate what the pose error is of the wrist_3_link w.r.t to the
peg stand shaft of the large gear.
"""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    get_params_from_config_file_set_in_env, parse_joint_state_from_yaml
)
from isaac_manipulator_ros_python_utils.test_utils import (
    IsaacManipulatorPoseEstimationErrorPolTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'manual_on_robot'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = f'/{ISAAC_ROS_WS}/pose_estimation_error_test'
ISAAC_ROS_ASSETS_DIR = f'{ISAAC_ROS_WS}/isaac_ros_assets'

if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Cumotion, perception and nvblox nodes for testing."""
    IsaacManipulatorPoseEstimationErrorPolTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch')
    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Trigger SAM detection for object following.
    override_params = {
        'segmentation_type': 'SEGMENT_ANYTHING',
        'object_detection_type': 'SEGMENT_ANYTHING',
        'pose_estimation_type': 'FOUNDATION_POSE',
        'segment_anything_input_points_topic': 'input_points',
        'segment_anything_input_detections_topic': 'input_detections',
        'workflow_type': 'GEAR_ASSEMBLY',
        'camera_type': 'REALSENSE',
        'num_cameras': 1,
        'foundation_pose_mesh_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                                          '/gear_large/gear_large.obj',
        'attach_object_mesh_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                                        '/gear_large/gear_large.obj',
        'peg_stand_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                               '/gear_base/gear_base.obj'
    }
    params.update(override_params)

    gear_mesh_file_paths = [
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_large/gear_large.obj',
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_small/gear_small.obj',
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_medium/gear_medium.obj',
    ]

    # This is the IK position that cuMotion to try to match closely in planning for place pose.
    target_joint_state = None
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:

        target_joint_state = parse_joint_state_from_yaml(
            params['gear_assembly_model_path'] + '/params/env.yaml',
            use_sim_time=False
        )
        node_startup_delay = 12.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir, '/workflows/core.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir,
                    '/drivers/ur_robotiq_driver.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_manipulator_workflow_bringup_include_dir,
                    '/sensors/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

    else:
        # Makes the test pass if we do not want to run on CI
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link']
        ))

    return IsaacManipulatorPoseEstimationErrorPolTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        # Do a GetObjects -> GetObjectPose -> AddMeshToObject -> PickAndHover -> RLInsertion op.
        num_cycles=1,
        is_segment_anything_object_detection_enabled=True,
        is_segment_anything_segmentation_enabled=True,
        is_rt_detr_object_detection_enabled=False,
        # User supplied initial hint for object (large gear)
        initial_hint={'x': 418.0, 'y': 605.0, 'z': 0.0},  # realsense for large gear.
        wait_for_point_topic=True,
        point_topic_name_as_trigger='input_points_debug',
        mesh_file_paths=gear_mesh_file_paths,
        mesh_file_path_for_peg_stand_estimation=params['peg_stand_file_path'],
        ground_truth_sim_prim_name='gear_large',
        # Dummy name since we dont have ground truth in sim.
        camera_prim_name_in_tf='camera_1_color_optical_frame',
        max_timeout_time_for_action_call=10.0,
        use_joint_space_planner_api=True,
        publish_only_on_static_tf=True,
        use_sim_time=False,
        target_joint_state_for_place_pose=target_joint_state,
        run_rl_inference=True,
        output_dir=OUTPUT_DIR,
        peg_stand_shaft_offset_for_cumotion=0.32
    )
