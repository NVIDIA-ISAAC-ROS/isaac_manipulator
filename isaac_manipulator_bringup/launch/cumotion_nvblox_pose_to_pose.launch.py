# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaac_ros_launch_utils.all_types as lut
import isaac_ros_launch_utils as lu

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.types import CameraType, DepthType, TrackingType


def generate_launch_description() -> lut.LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg(
        'camera_type',
        default=CameraType.hawk,
        choices=[str(CameraType.hawk), str(CameraType.realsense)],
        cli=True,
        description='Camera sensor to use for this example')
    args.add_arg(
        'hawk_depth_mode',
        default=DepthType.ess_full,
        choices=DepthType.names(),
        cli=True,
        description='Depth mode for Hawk camera')
    args.add_arg(
        'num_cameras',
        default=1,
        choices=['1', '2'],
        cli=True,
        description='Number of cameras to run for 3d reconstruction')
    args.add_arg(
        'no_robot_mode',
        default=False,
        cli=True,
        description='Whether to run without a robot arm (perception only mode).')
    args.add_arg(
        'setup',
        default='None',
        cli=True,
        description='The name of the setup you are running on (specifying calibration '
                    'workspace bounds and camera ids).'
    )
    args.add_arg(
        'rosbag', 'None', cli=True, description='Path to rosbag (running on sensor if not set).')
    args.add_arg('log_level', 'error', cli=True, choices=['debug', 'info', 'warn', 'error'])
    args.add_arg(
        'disable_esdf_visualizer',
        default=False,
        cli=True,
        description='If true, disables the ESDF visualizer (useful for testing).')
    args.add_arg(
        'enable_test_keepalive',
        default=False,
        cli=True,
        description='If true, enables a keepalive node for testing purposes.')
    args.add_arg(
        'rosbag_loop',
        default=False,
        cli=True,
        description='If true, enables looping playback of rosbag (useful for testing).')
    args.add_arg(
        'disable_cameras',
        default=False,
        cli=True,
        description='If true, disables camera drivers (useful for testing).')
    actions = args.get_launch_actions()

    # Configuration
    run_from_bag = lu.is_valid(args.rosbag)
    is_hawk_camera = lu.is_equal(args.camera_type, str(CameraType.hawk))
    is_realsense_camera = lu.is_equal(args.camera_type, str(CameraType.realsense))

    # Globally set use_sim_time if we're running from bag or sim
    actions.append(lut.SetParameter('use_sim_time', True, condition=lut.IfCondition(run_from_bag)))

    # RealSense driver
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/realsense.launch.py',
            launch_arguments={
                'num_cameras': args.num_cameras,
                'camera_ids_config_name': args.setup
            },
            condition=lut.IfCondition(lut.AndSubstitution(
                is_realsense_camera,
                lut.AndSubstitution(
                    lu.is_not(run_from_bag),
                    lu.is_not(args.disable_cameras)
                )
            )),
        ))

    # Hawk driver
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/hawk.launch.py',
            condition=lut.IfCondition(lut.AndSubstitution(
                is_hawk_camera,
                lut.AndSubstitution(
                    lu.is_not(run_from_bag),
                    lu.is_not(args.disable_cameras)
                )
            )),
        ))

    # Play ros2bag
    actions.append(
        lu.play_rosbag(
            bag_path=args.rosbag,
            loop=args.rosbag_loop,
            condition=lut.IfCondition(lu.is_valid(args.rosbag))
        ))

    # ESS
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/ess.launch.py',
            launch_arguments={
                'ess_mode': args.hawk_depth_mode,
            },
            condition=lut.IfCondition(lut.AndSubstitution(
                is_hawk_camera,
                lu.is_not(args.disable_cameras)
            )),
        ))

    # Cumotion + robot segmenter
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/cumotion.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'num_cameras': args.num_cameras,
                'no_robot_mode': args.no_robot_mode,
                'from_bag': run_from_bag,
                'workspace_bounds_name': args.setup,
                'disable_esdf_visualizer': args.disable_esdf_visualizer,
            },
        ))

    # Nvblox
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/nvblox.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'num_cameras': args.num_cameras,
                'no_robot_mode': args.no_robot_mode,
                'workspace_bounds_name': args.setup
            },
            # Delay startup when running live (waiting on drivers to start)
            delay=lu.if_else_substitution(run_from_bag, '0.0', '5.0')))

    # Goal setter
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/pose_to_pose.launch.py',
            condition=lut.IfCondition(
                lut.AndSubstitution(lu.is_not(args.no_robot_mode), lu.is_not(run_from_bag))),
        ))

    # Calibration
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/static_transforms.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'num_cameras': args.num_cameras,
                'tracking_type': TrackingType.pose_to_pose,
                'calibration_name': args.setup,
                'broadcast_world_base_link': args.no_robot_mode,
            },
        ))

    # Visualization
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/visualization/visualization.launch.py',
            launch_arguments={'camera_type': args.camera_type}))

    # Component container - only create if we actually need composable nodes
    # Create container only for robot mode (not for camera-only or disabled modes)
    create_container = lu.is_not(args.no_robot_mode)

    actions.append(
        lu.component_container(
            constants.MANIPULATOR_CONTAINER_NAME,
            log_level=args.log_level,
            condition=lut.IfCondition(create_container)
        ))

    # Test keepalive node - simple node that stays alive during testing
    actions.append(
        lut.Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='test_keepalive_listener',
            condition=lut.IfCondition(args.enable_test_keepalive),
            output='screen'
        ))

    return lut.LaunchDescription(actions)
