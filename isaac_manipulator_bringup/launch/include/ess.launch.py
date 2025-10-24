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

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.launch_utils import get_dnn_stereo_depth_resolution
from isaac_manipulator_ros_python_utils.manipulator_types import CameraType, DepthType

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    camera_type_str = str(context.perform_substitution(LaunchConfiguration('camera_type'))).upper()
    camera_type = CameraType[camera_type_str]

    depth_type_str = str(context.perform_substitution(LaunchConfiguration('depth_type')))
    depth_type = DepthType[depth_type_str]
    ess_model_width, ess_model_height = get_dnn_stereo_depth_resolution(depth_type)
    engine_file_path = str(
        context.perform_substitution(LaunchConfiguration('ess_engine_file_path')))
    # If the engine file path is not set use the defaults
    if engine_file_path == '':
        raise ValueError('ess_engine_file_path is not set.')

    threshold = LaunchConfiguration('ess_threshold')
    composable_node_descriptions = []

    left_image_raw = LaunchConfiguration('left_image_raw_topic')
    left_camera_info = LaunchConfiguration('left_camera_info_topic')
    right_image_raw = LaunchConfiguration('right_image_raw_topic')
    right_camera_info = LaunchConfiguration('right_camera_info_topic')
    depth_output = LaunchConfiguration('depth_output_topic')
    rgb_output = LaunchConfiguration('rgb_output_topic')
    rgb_camera_info_output = LaunchConfiguration('rgb_camera_info_output_topic')
    input_image_height = int(
        context.perform_substitution(LaunchConfiguration('input_image_height')))
    input_image_width = int(
        context.perform_substitution(LaunchConfiguration('input_image_width')))
    camera_name = LaunchConfiguration('camera_namespace')
    # Different processing based on camera type
    if camera_type == CameraType.REALSENSE:
        # RealSense images are already rectified, but need format conversion
        composable_node_descriptions.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_left',
            namespace=camera_name,
            parameters=[{
                'encoding_desired': 'rgb8',
                'input_qos': 'SENSOR_DATA',
                'output_qos': 'SENSOR_DATA'
            }],
            remappings=[
                ('image_raw', left_image_raw),
                ('image', 'left/image_rect')]
        ))

        composable_node_descriptions.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='image_format_node_right',
            namespace=camera_name,
            parameters=[{
                'encoding_desired': 'rgb8',
                'input_qos': 'SENSOR_DATA',
                'output_qos': 'SENSOR_DATA'
            }],
            remappings=[
                ('image_raw', right_image_raw),
                ('image', 'right/image_rect')]
        ))

    if camera_type == CameraType.REALSENSE:
        camera_info_left_for_disparity = left_camera_info
        camera_info_right_for_disparity = right_camera_info
        camera_info_left_for_resize = left_camera_info
    else:
        raise ValueError(f'Invalid camera type {camera_type}')

    composable_node_descriptions.append(ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        namespace=camera_name,
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_qos': 'SENSOR_DATA',
                     'input_layer_width': int(ess_model_width),
                     'input_layer_height': int(ess_model_height)}],
        remappings=[
            ('left/camera_info', camera_info_left_for_disparity),
            ('right/camera_info', camera_info_right_for_disparity),
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='DisparityToDepthNode',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        namespace=camera_name,
        remappings=[(
            'depth', depth_output
        )],
    ))

    composable_node_descriptions.append(ComposableNode(
        name='resize_left_ess_size',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=camera_name,
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'input_width': int(input_image_width),
            'input_height': int(input_image_height),
            'output_width': int(ess_model_width),
            'output_height': int(ess_model_height),
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False,
            'use_latest_camera_info': True,
            'drop_old_messages': False
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', camera_info_left_for_resize),
            ('resize/image', rgb_output),
            ('resize/camera_info', rgb_camera_info_output)
        ]
    ))

    load_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_node_descriptions,
    )

    final_launch = GroupAction(
        actions=[
            load_nodes,
        ],
    )

    return [final_launch]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'camera_type',
            default_value='REALSENSE',
            description='Type of camera (REALSENSE or ISAAC_SIM)'),
        DeclareLaunchArgument(
            'ess_engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),
        DeclareLaunchArgument(
            'ess_threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'depth_type',
            default_value=str(DepthType.ESS_FULL),
            choices=DepthType.names(),
            description=f'Depth estimation type. Choose between {", ".join(DepthType.names())}'),
        DeclareLaunchArgument(
            'left_image_raw_topic',
            default_value='left/image_raw_drop',
            description='Input topic for left camera raw image'),
        DeclareLaunchArgument(
            'left_camera_info_topic',
            default_value='left/camera_info_drop',
            description='Input topic for left camera info'),
        DeclareLaunchArgument(
            'right_image_raw_topic',
            default_value='right/image_raw_drop',
            description='Input topic for right camera raw image'),
        DeclareLaunchArgument(
            'right_camera_info_topic',
            default_value='right/camera_info_drop',
            description='Input topic for right camera info'),
        DeclareLaunchArgument(
            'depth_output_topic',
            default_value='depth_image',
            description='Output topic for depth image'),
        DeclareLaunchArgument(
            'rgb_output_topic',
            default_value='rgb/image_rect_color',
            description='Output topic for RGB image'),
        DeclareLaunchArgument(
            'rgb_camera_info_output_topic',
            default_value='rgb/camera_info',
            description='Output topic for RGB camera info'),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='camera_1',
            description='Namespace for the camera'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value=str(constants.ESS_INPUT_IMAGE_HEIGHT),
            description='Input image height'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value=str(constants.ESS_INPUT_IMAGE_WIDTH),
            description='Input image width')
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
