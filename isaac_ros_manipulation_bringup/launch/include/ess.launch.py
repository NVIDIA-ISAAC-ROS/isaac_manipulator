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

import os
import platform

import isaac_ros_manipulation_ros_python_utils.constants as constants
from isaac_ros_manipulation_ros_python_utils.launch_utils import get_dnn_stereo_depth_resolution
from isaac_ros_manipulation_ros_python_utils.manipulator_types import CameraType, DepthType

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
    if engine_file_path == '':
        raise ValueError('ess_engine_file_path is not set.')

    ess_plugin_file_path = str(
        context.perform_substitution(LaunchConfiguration('ess_plugin_file_path')))
    if ess_plugin_file_path == '':
        ess_plugin_file_path = os.path.join(
            os.path.dirname(engine_file_path), 'plugins', platform.machine(), 'ess_plugins.so')
    if not os.path.isfile(ess_plugin_file_path):
        raise FileNotFoundError(
            f'ESS TensorRT plugin library does not exist: {ess_plugin_file_path}')

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

    network_width = int(ess_model_width)
    network_height = int(ess_model_height)

    composable_node_descriptions.append(ComposableNode(
        name='left_format_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=camera_name,
        parameters=[{'encoding_desired': 'rgb8', 'input_qos': 'SENSOR_DATA'}],
        remappings=[('image_raw', 'left/image_rect'), ('image', 'left/image_rgb')]))

    composable_node_descriptions.append(ComposableNode(
        name='left_resize_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=camera_name,
        parameters=[{
            'output_width': network_width, 'output_height': network_height,
            'keep_aspect_ratio': False,
            'input_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('camera_info', camera_info_left_for_disparity),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]))

    composable_node_descriptions.append(ComposableNode(
        name='left_normalize_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        namespace=camera_name,
        parameters=[{'mean': [127.5, 127.5, 127.5], 'stddev': [127.5, 127.5, 127.5]}],
        remappings=[('image', 'left/image_resize'),
                    ('normalized_image', 'left/image_normalize')]))

    composable_node_descriptions.append(ComposableNode(
        name='left_tensor_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=camera_name,
        parameters=[{'scale': False, 'tensor_name': 'left_image'}],
        remappings=[('image', 'left/image_normalize'), ('tensor', 'left/tensor')]))

    composable_node_descriptions.append(ComposableNode(
        name='left_planar_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        namespace=camera_name,
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'left_image',
        }],
        remappings=[('interleaved_tensor', 'left/tensor'),
                    ('planar_tensor', 'left/tensor_planar')]))

    composable_node_descriptions.append(ComposableNode(
        name='left_reshape_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=camera_name,
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[('tensor', 'left/tensor_planar'),
                    ('reshaped_tensor', 'left/tensor_reshape')]))

    composable_node_descriptions.append(ComposableNode(
        name='right_format_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=camera_name,
        parameters=[{'encoding_desired': 'rgb8', 'input_qos': 'SENSOR_DATA'}],
        remappings=[('image_raw', 'right/image_rect'), ('image', 'right/image_rgb')]))

    composable_node_descriptions.append(ComposableNode(
        name='right_resize_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=camera_name,
        parameters=[{
            'output_width': network_width, 'output_height': network_height,
            'keep_aspect_ratio': False,
            'input_qos': 'SENSOR_DATA',
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('camera_info', camera_info_right_for_disparity),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]))

    composable_node_descriptions.append(ComposableNode(
        name='right_normalize_node', package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        namespace=camera_name,
        parameters=[{'mean': [127.5, 127.5, 127.5], 'stddev': [127.5, 127.5, 127.5]}],
        remappings=[('image', 'right/image_resize'),
                    ('normalized_image', 'right/image_normalize')]))

    composable_node_descriptions.append(ComposableNode(
        name='right_tensor_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=camera_name,
        parameters=[{'scale': False, 'tensor_name': 'right_image'}],
        remappings=[('image', 'right/image_normalize'), ('tensor', 'right/tensor')]))

    composable_node_descriptions.append(ComposableNode(
        name='right_planar_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        namespace=camera_name,
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'right_image',
        }],
        remappings=[('interleaved_tensor', 'right/tensor'),
                    ('planar_tensor', 'right/tensor_planar')]))

    composable_node_descriptions.append(ComposableNode(
        name='right_reshape_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=camera_name,
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[('tensor', 'right/tensor_planar'),
                    ('reshaped_tensor', 'right/tensor_reshape')]))

    composable_node_descriptions.append(ComposableNode(
        name='tensor_pair_sync_node', package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        namespace=camera_name,
        parameters=[{
            'input_tensor1_name': 'left_image', 'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'input_left', 'output_tensor2_name': 'input_right',
        }],
        remappings=[('tensor1', 'left/tensor_reshape'),
                    ('tensor2', 'right/tensor_reshape')]))

    composable_node_descriptions.append(ComposableNode(
        name='tensor_rt', package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=camera_name,
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['input_left', 'input_right'],
            'input_binding_names': ['input_left', 'input_right'],
            'output_tensor_names': ['output_left', 'output_conf'],
            'output_binding_names': ['output_left', 'output_conf'],
            'verbose': False,
            'force_engine_update': False,
            'custom_plugin_lib': ess_plugin_file_path,
        }]))

    composable_node_descriptions.append(ComposableNode(
        name='dnn_stereo_decoder', package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        namespace=camera_name,
        parameters=[{
            'disparity_tensor_name': 'output_left',
            'confidence_tensor_name': 'output_conf',
            'confidence_threshold': threshold,
            'cache_camera_info': True,
        }],
        remappings=[('right/camera_info', 'right/camera_info_resize')]))

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
            description='Absolute path to the ESS engine plan.'),
        DeclareLaunchArgument(
            'ess_plugin_file_path',
            default_value='',
            description='Absolute path to the ESS TensorRT plugin library. '
                        'Defaults to plugins/<arch>/ess_plugins.so next to the ESS engine.'),
        DeclareLaunchArgument(
            'ess_threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'depth_type',
            default_value=str(DepthType.ESS_FULL),
            choices=[str(DepthType.ESS_FULL), str(DepthType.ESS_LIGHT)],
            description='Depth estimation type. Choose between ESS_FULL, ESS_LIGHT'),
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
