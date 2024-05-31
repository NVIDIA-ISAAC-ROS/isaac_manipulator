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


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    ess_mode = str(context.perform_substitution(LaunchConfiguration('ess_mode')))

    engine_file_path = str(context.perform_substitution(
        LaunchConfiguration('ess_engine_file_path')))
    threshold = LaunchConfiguration('ess_threshold')

    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    ess_model_width = LaunchConfiguration('ess_model_width')
    ess_model_height = LaunchConfiguration('ess_model_height')

    if ess_mode == 'ess_light' and  engine_file_path == '':
        engine_file_path = '/tmp/dnn_stereo_disparity_v4.0.0/ess_light.engine'
    elif ess_mode == 'ess_full' and engine_file_path == '':
        engine_file_path = '/tmp/dnn_stereo_disparity_v4.0.0/ess_full.engine'

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': image_width,
            'output_height': image_height,
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'DEFAULT'
        }],
        remappings=[
            ('image_raw', 'left/image_raw_drop'),
            ('camera_info', 'left/camera_info_drop'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': image_width,
            'output_height': image_height,
            'input_qos': 'SENSOR_DATA',
            'output_qos': 'DEFAULT'
        }],
        remappings=[
            ('image_raw', 'right/image_raw_drop'),
            ('camera_info', 'right/camera_info_drop'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    disparity_node = ComposableNode(
        name='disparity',
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': engine_file_path,
                     'threshold': threshold,
                     'input_layer_width': ess_model_width,
                     'input_layer_height': ess_model_height}],
        remappings=[
            ('left/camera_info', 'left/camera_info_rect'),
            ('right/camera_info', 'right/camera_info_rect'),
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
        ]
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        remappings=[(
            'depth', 'depth_image'
        )],
    )

    resize_left_ess_size = ComposableNode(
        name='resize_left_ess_size',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': image_width,
            'input_height': image_height,
            'output_width': ess_model_width,
            'output_height': ess_model_height,
            'keep_aspect_ratio': False,
            'encoding_desired': 'rgb8',
            'disable_padding': False
        }],
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'rgb/image_rect_color'),
            ('resize/camera_info', 'rgb/camera_info')
        ]
    )

    load_nodes = LoadComposableNodes(
        target_container='manipulation_container',
        composable_node_descriptions=[
            left_rectify_node,
            right_rectify_node,
            disparity_node,
            disparity_to_depth_node,
            resize_left_ess_size
        ],
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
            'ess_engine_file_path',
            default_value='',
            description='The absolute path to the ESS engine plan.'),
        DeclareLaunchArgument(
            'ess_threshold',
            default_value='0.4',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),
        DeclareLaunchArgument(
            'ess_mode',
            default_value='ess_light',
            choices=['ess_light', 'ess_full'],
            description='ESS model type. Choose between ess_light and ess_full.'),
        DeclareLaunchArgument(
            'image_width',
            default_value='1920',
            description='Input image width.'),
        DeclareLaunchArgument(
            'image_height',
            default_value='1200',
            description='Input image height.'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
