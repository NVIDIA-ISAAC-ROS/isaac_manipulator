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

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.launch_utils import get_dnn_stereo_depth_resolution
from isaac_manipulator_ros_python_utils.manipulator_types import CameraType, DepthType

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode

NETWORK_BATCH_SIZE = 1
NETWORK_NUM_CHANNELS = 3


def launch_setup(context, *args, **kwargs):
    camera_type_str = str(context.perform_substitution(LaunchConfiguration('camera_type'))).upper()
    camera_type = CameraType[camera_type_str]

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

    model_width, model_height = get_dnn_stereo_depth_resolution(DepthType.FOUNDATION_STEREO)
    resize_width_int = int(
        max(int(model_width), input_image_width * int(model_height) / input_image_height))
    resize_height_int = int(
        max(int(model_height), input_image_height * int(model_width) / input_image_width))

    engine_file_path = str(
        context.perform_substitution(LaunchConfiguration('foundationstereo_engine_file_path')))
    # If the engine file path is not set use the defaults
    if engine_file_path == '':
        raise ValueError('foundationstereo_engine_file_path is not set.')

    composable_node_descriptions = []

    # RealSense images are already rectified, but need format conversion
    composable_node_descriptions.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='left_image_format_node',
            namespace=camera_name,
            parameters=[{
                'encoding_desired': 'rgb8',
                'input_qos': 'SENSOR_DATA',
                'output_qos': 'SENSOR_DATA',
                'input_qos_depth': 1,
                'output_qos_depth': 1,
            }],
            remappings=[
                ('image_raw', left_image_raw),
                ('image', 'left/image_rect')
            ]
        ))

    composable_node_descriptions.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='right_image_format_node',
            namespace=camera_name,
            parameters=[{
                'encoding_desired': 'rgb8',
                'input_qos': 'SENSOR_DATA',
                'output_qos': 'SENSOR_DATA',
                'input_qos_depth': 1,
                'output_qos_depth': 1,
            }],
            remappings=[
                ('image_raw', right_image_raw),
                ('image', 'right/image_rect')
            ]
        ))

    if camera_type == CameraType.REALSENSE:
        camera_info_left_for_resize = left_camera_info
        camera_info_right_for_resize = right_camera_info
    else:
        raise ValueError(f'Invalid camera type {camera_type}')

    composable_node_descriptions.append(ComposableNode(
        name='left_resize_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'input_width': int(input_image_width),
            'input_height': int(input_image_height),
            'output_width': resize_width_int,
            'output_height': resize_height_int,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': False,
            'use_latest_camera_info': True,
            'drop_old_messages': False,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', camera_info_left_for_resize),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='left_crop_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::CropNode',
        parameters=[{
            'input_width': resize_width_int,
            'input_height': resize_height_int,
            'crop_width': int(model_width),
            'crop_height': int(model_height),
            'crop_mode': 'CENTER',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'left/image_resize'),
            ('camera_info', 'left/camera_info_resize'),
            ('crop/image', rgb_output),
            ('crop/camera_info', rgb_camera_info_output),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='left_normalize_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', rgb_output),
            ('normalized_image', 'left/image_normalize')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='left_tensor_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'left_image',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'left/image_normalize'),
            ('tensor', 'left/tensor'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='left_planar_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [int(model_height), int(model_width), NETWORK_NUM_CHANNELS],
            'output_tensor_name': 'left_image',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('interleaved_tensor', 'left/tensor'),
            ('planar_tensor', 'left/tensor_planar')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='left_reshape_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [
                NETWORK_NUM_CHANNELS, int(model_height), int(model_width)],
            'output_tensor_shape': [
                NETWORK_BATCH_SIZE, NETWORK_NUM_CHANNELS, int(model_height), int(model_width)],
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_resize_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_qos': 'SENSOR_DATA',
            'input_width': int(input_image_width),
            'input_height': int(input_image_height),
            'output_width': resize_width_int,
            'output_height': resize_height_int,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': False,
            'use_latest_camera_info': True,
            'drop_old_messages': False,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'right/image_rect'),
            ('camera_info', camera_info_right_for_resize),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_crop_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::CropNode',
        parameters=[{
            'input_width': resize_width_int,
            'input_height': resize_height_int,
            'crop_width': int(model_width),
            'crop_height': int(model_height),
            'crop_mode': 'CENTER',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'right/image_resize'),
            ('camera_info', 'right/camera_info_resize'),
            ('crop/image', 'right/image_crop'),
            ('crop/camera_info', 'right/camera_info_crop'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_normalize_node',
        namespace=camera_name,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'right/image_crop'),
            ('normalized_image', 'right/image_normalize')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_tensor_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'right_image',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('image', 'right/image_normalize'),
            ('tensor', 'right/tensor'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_planar_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [int(model_height), int(model_width), NETWORK_NUM_CHANNELS],
            'output_tensor_name': 'right_image',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('interleaved_tensor', 'right/tensor'),
            ('planar_tensor', 'right/tensor_planar')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='right_reshape_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [
                NETWORK_NUM_CHANNELS, int(model_height), int(model_width)],
            'output_tensor_shape': [
                NETWORK_BATCH_SIZE, NETWORK_NUM_CHANNELS, int(model_height), int(model_width)],
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='tensor_pair_sync_node',
        namespace=camera_name,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image',
            'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'left_image',
            'output_tensor2_name': 'right_image',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('tensor1', 'left/tensor_reshape'),
            ('tensor2', 'right/tensor_reshape'),
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='foundationstereo_tensor_rt',
        namespace=camera_name,
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['left_image', 'right_image'],
            'input_binding_names': ['left_image', 'right_image'],
            'output_tensor_names': ['disparity'],
            'output_binding_names': ['disparity'],
            'verbose': False,
            'force_engine_update': False,
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='foundationstereo_decoder',
        namespace=camera_name,
        package='isaac_ros_foundationstereo',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'disparity',
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_crop')
        ]
    ))

    composable_node_descriptions.append(ComposableNode(
        name='disparity_to_depth_node',
        namespace=camera_name,
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        parameters=[{
            'input_qos_depth': 1,
            'output_qos_depth': 1,
        }],
        remappings=[(
            'depth', depth_output
        )],
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
            'foundationstereo_engine_file_path',
            default_value='',
            description='The absolute path to the FoundationStereo engine plan.'),
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
            default_value=str(constants.REALSENSE_IMAGE_HEIGHT),
            description='Input image height'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value=str(constants.REALSENSE_IMAGE_WIDTH),
            description='Input image width')
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
