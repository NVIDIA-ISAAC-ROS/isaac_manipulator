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
Gear Assembly Policy with Isaac Manipulator in simulation.

This test will first run SAM to get the object pose of the gear. Then it will fire off
these action calls:
- Gripper open
- Move to gear (cumotion) -> PickAndHover with gripper closed.
- Insert action call.
- See if insertion was successful.
"""

import os
import shutil

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils import (
    load_yaml_params,
    parse_joint_state_from_yaml
)
from isaac_manipulator_ros_python_utils.test_utils import IsaacManipulatorGearAssemblyPolTest
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'manual_isaac_sim'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = f'/{ISAAC_ROS_WS}/manipulator_gear_assembly_test_sim'
ISAAC_ROS_ASSETS_DIR = f'{ISAAC_ROS_WS}/isaac_ros_assets'

if ISAAC_ROS_WS is None:
    raise RuntimeError('ISAAC_ROS_WS environment variable is not set')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Cumotion, perception and nvblox nodes for testing."""
    IsaacManipulatorGearAssemblyPolTest.generate_namespace()
    isaac_manipulator_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'launch')
    isaac_manipulator_robot_description_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_robot_description'),
        'config')
    test_yaml_config = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'),
        'params',
        'sim_launch_params.yaml'
    )
    params = load_yaml_params(test_yaml_config)

    # Trigger SAM detection for object following.
    override_params = {
        'segmentation_type': 'SEGMENT_ANYTHING',
        'object_detection_type': 'SEGMENT_ANYTHING',
        'pose_estimation_type': 'FOUNDATION_POSE',
        'segment_anything_input_points_topic': 'input_points',
        'segment_anything_input_detections_topic': 'input_detections',
        'workflow_type': 'GEAR_ASSEMBLY',
        # Update grasp file path for the large gear.
        # TODO(kchahal): Add a mechanism to update the grasps inside of behavior tree for
        # differentobjects.
        'grasps_file_path': f'{isaac_manipulator_robot_description_include_dir}/'
                            'robotiq_2f_140_grasps_large_gear.yaml',
        # TODO(kchahal): Add real paths once the location is finalized.
        'foundation_pose_mesh_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                                          '/gear_large/gear_large.obj',
        'attach_object_mesh_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                                        '/gear_large/gear_large.obj',
        'peg_stand_file_path': f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy'
                               '/gear_base/gear_base.obj',
        'gear_assembly_ros_bag_path_for_rl_inference': '/tmp/gear_assembly_rl_inference.bag',
        'use_ground_truth_pose_in_sim': 'true',
        'enable_nvblox': 'true',
        'sim_gt_asset_frame_id': 'detected_object1'
    }

    gear_mesh_file_paths = [
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_large/gear_large.obj',
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_small/gear_small.obj',
        f'{ISAAC_ROS_ASSETS_DIR}/isaac_manipulator_ur_dnn_policy/gear_medium/gear_medium.obj',
    ]

    params.update(override_params)

    target_joint_state = None

    # Remove rosbag if exists first
    if os.path.exists(params['gear_assembly_ros_bag_path_for_rl_inference']):
        # remove even if its a directory
        if os.path.isdir(params['gear_assembly_ros_bag_path_for_rl_inference']):
            shutil.rmtree(params['gear_assembly_ros_bag_path_for_rl_inference'])
        else:
            os.remove(params['gear_assembly_ros_bag_path_for_rl_inference'])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        target_joint_state = parse_joint_state_from_yaml(
            params['gear_assembly_model_path'] + '/params/env.yaml',
            use_sim_time=True)
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

    return IsaacManipulatorGearAssemblyPolTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        # Do a GetObjects -> GetObjectPose -> AddMeshToObject -> PickAndHover -> RLInsertion op.
        num_cycles=1,
        is_segment_anything_object_detection_enabled=True,
        is_segment_anything_segmentation_enabled=True,
        is_rt_detr_object_detection_enabled=False,
        # In Sim based we are also using user supplied point marking to align with real.
        wait_for_point_topic=True,
        point_topic_name_as_trigger='input_points_debug',
        mesh_file_path_for_peg_stand_estimation=params['peg_stand_file_path'],

        # User supplied initial hint for object (large gear)
        initial_hint={'x': 884.0, 'y': 1002.0, 'z': 0.0},
        mesh_file_paths=gear_mesh_file_paths,
        camera_prim_name_in_tf='front_stereo_camera_left',
        peg_stand_shaft_offset_for_cumotion=0.32,  # 0.25 meters for 85 gripper.
        max_timeout_time_for_action_call=10.0,
        ground_truth_sim_prim_name='gear_large',
        use_ground_truth_pose_estimation=True,
        verify_pose_estimation_accuracy=False,
        use_joint_space_planner_api=True,
        run_rl_inference=True,
        use_sim_time=True,
        target_joint_state_for_place_pose=target_joint_state,
        output_dir=OUTPUT_DIR,
    )
