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

from isaac_ros_launch_utils.all_types import (
    LaunchDescription,
    SetParameter,
    IfCondition,
    AndSubstitution
)
import isaac_ros_launch_utils as lu

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.types import (
    CameraType, DepthType, TrackingType, PoseEstimationType
)
from isaac_manipulator_ros_python_utils.launch_utils import (
    get_depth_resolution, get_rgb_resolution
)


def generate_launch_description() -> LaunchDescription:
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
        'rtdetr_class_id',
        default='22',
        cli=True,
        description='Class ID of the object to be detected. The default corresponds to the '
                    'Mac and Cheese box if the SyntheticaDETR v1.0.0 model file is used. '
                    'Refer to the SyntheticaDETR model documentation for additional supported '
                    'objects and their class IDs.',
    )
    args.add_arg(
        'rosbag', 'None', cli=True, description='Path to rosbag (running on sensor if not set).')
    args.add_arg('log_level', 'error', cli=True, choices=['debug', 'info', 'warn', 'error'])
    args.add_arg(
        'pose_estimation_type',
        default=PoseEstimationType.foundationpose,
        choices=PoseEstimationType.names(),
        cli=True,
        description='Pose estimation model to use for this example')
    actions = args.get_launch_actions()

    # Configuration
    run_from_bag = lu.is_valid(args.rosbag)
    is_hawk_camera = lu.is_equal(args.camera_type, str(CameraType.hawk))
    is_realsense_camera = lu.is_equal(args.camera_type, str(CameraType.realsense))
    run_dope = lu.is_equal(args.pose_estimation_type, str(PoseEstimationType.dope))
    run_foundationpose = lu.is_equal(args.pose_estimation_type,
                                     str(PoseEstimationType.foundationpose))
    # When running foundation pose for pose estimation, we also need rtdetr for object detection.
    run_rtdetr = lu.is_equal(args.pose_estimation_type, str(PoseEstimationType.foundationpose))
    image_input_topic = lu.if_else_substitution(is_hawk_camera, 'left/image_rect',
                                                '/camera_1/color/image_raw')
    camera_info_input_topic = lu.if_else_substitution(is_hawk_camera, 'left/camera_info_rect',
                                                      '/camera_1/color/camera_info')
    input_fps = lu.if_else_substitution(is_hawk_camera, '10', '15')
    dropped_fps = lu.if_else_substitution(is_hawk_camera, '8', '13')
    is_object_following = 'True'

    # Image resolutions
    depth_image_width, depth_image_height = get_depth_resolution(args.camera_type,
                                                                 args.hawk_depth_mode)
    rgb_image_width, rgb_image_height = get_rgb_resolution(args.camera_type)
    dope_width, dope_height = get_rgb_resolution(str(CameraType.realsense))

    # Globally set use_sim_time if we're running from bag or sim
    actions.append(SetParameter('use_sim_time', True, condition=IfCondition(run_from_bag)))

    # RealSense driver
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/realsense.launch.py',
            launch_arguments={
                'num_cameras': args.num_cameras,
                'camera_ids_config_name': args.setup
            },
            condition=IfCondition(AndSubstitution(is_realsense_camera, lu.is_not(run_from_bag))),
        ))

    # Hawk driver
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/hawk.launch.py',
            condition=IfCondition(AndSubstitution(is_hawk_camera, lu.is_not(run_from_bag))),
        ))

    # Play ros2bag
    actions.append(
        lu.play_rosbag(bag_path=args.rosbag, condition=IfCondition(lu.is_valid(args.rosbag))))

    # ESS
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/ess.launch.py',
            launch_arguments={
                'ess_mode': args.hawk_depth_mode,
            },
            condition=IfCondition(is_hawk_camera),
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

    # Rtdetr
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/rtdetr.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'image_width': rgb_image_width,
                'image_height': rgb_image_height,
                'image_input_topic': image_input_topic,
                'camera_info_input_topic': camera_info_input_topic,
                'input_fps': input_fps,
                'dropped_fps': dropped_fps,
                'rtdetr_is_object_following': is_object_following
            },
            condition=IfCondition(run_rtdetr),
        ))

    # Foundationpose
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/foundationpose.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'rgb_image_width': rgb_image_width,
                'rgb_image_height': rgb_image_height,
                'depth_image_width': depth_image_width,
                'depth_image_height': depth_image_height,
                'detection2_d_array_topic': 'detections_output',
                'is_object_following': is_object_following,
                'object_class_id': args.rtdetr_class_id
            },
            condition=IfCondition(run_foundationpose),
        ))

    # Dope
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/dope.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'input_image_width': rgb_image_width,
                'input_image_height': rgb_image_height,
                'dope_network_image_width': dope_width,
                'dope_network_image_height': dope_height,
                'image_input_topic': image_input_topic,
                'camera_info_input_topic': camera_info_input_topic,
                'input_fps': input_fps,
                'dropped_fps': dropped_fps
            },
            condition=IfCondition(run_dope)))

    # Goal setter
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/goal.launch.py',
            launch_arguments={
                'grasp_frame': 'goal_frame'
            },
            condition=IfCondition(
                AndSubstitution(lu.is_not(args.no_robot_mode), lu.is_not(run_from_bag))),
        ))

    # Calibration
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/static_transforms.launch.py',
            launch_arguments={
                'camera_type': args.camera_type,
                'tracking_type': TrackingType.follow_object,
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

    # Component container
    actions.append(
        lu.component_container(constants.MANIPULATOR_CONTAINER_NAME, log_level=args.log_level))

    return LaunchDescription(actions)
