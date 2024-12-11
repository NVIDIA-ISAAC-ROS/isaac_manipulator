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

from isaac_ros_launch_utils.all_types import ComposableNode, DeclareLaunchArgument, \
    LaunchConfiguration, LoadComposableNodes, LaunchDescription, GroupAction, OpaqueFunction

import isaac_manipulator_ros_python_utils.constants as constants


def launch_setup(context, *args, **kwargs):
    engine_file_path = LaunchConfiguration('rtdetr_engine_file_path')
    rtdetr_input_qos = LaunchConfiguration('rtdetr_input_qos')
    detections_2d_array_output_topic = LaunchConfiguration(
        'detections_2d_array_output_topic', default='detections_output')

    image_input_topic = LaunchConfiguration("image_input_topic")
    camera_info_input_topic = LaunchConfiguration("camera_info_input_topic")

    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')

    input_fps = LaunchConfiguration('input_fps')
    dropped_fps = LaunchConfiguration('dropped_fps')

    rt_detr_confidence_threshold = LaunchConfiguration('rt_detr_confidence_threshold')

    rtdetr_is_object_following = str(context.perform_substitution(LaunchConfiguration(
        'rtdetr_is_object_following', default='False')))

    drop_node_nodes = []

    dropped_image_topic_name = image_input_topic
    dropped_camera_info_topic_name = camera_info_input_topic

    if rtdetr_is_object_following == 'True':
        dropped_image_topic_name = '/rtdetr/image_dropped'
        dropped_camera_info_topic_name = '/rtdetr/camera_info_dropped'
        drop_node_nodes.append(ComposableNode(
            name='rtdetr_drop_node',
            package='isaac_ros_nitros_topic_tools',
            plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
            parameters=[{
                'input_qos': rtdetr_input_qos,
                'output_qos': rtdetr_input_qos,
                'X': dropped_fps,
                'Y': input_fps,
                'mode': 'mono',
                'sync_queue_size': 100
            }],
            remappings=[
                ('image_1', image_input_topic),
                ('camera_info_1', camera_info_input_topic),
                ('image_1_drop', dropped_image_topic_name),
                ('camera_info_1_drop', dropped_camera_info_topic_name)
            ]
        ))

    resize_node = ComposableNode(
        name='rtdetr_resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_qos': rtdetr_input_qos,
            'input_width': image_width,
            'input_height': image_height,
            'output_width': 640,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', dropped_image_topic_name),
            ('camera_info', dropped_camera_info_topic_name),
        ],
    )

    pad_node = ComposableNode(
        name='rtdetr_pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': 640,
            'output_image_height': 640,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'resize/image')])

    image_to_tensor_node = ComposableNode(
        name='rtdetr_image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'normalized_tensor'),
        ])

    interleave_to_planar_node = ComposableNode(
        name='rtdetr_interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [640, 640, 3]
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')])

    reshape_node = ComposableNode(
        name='rtdetr_reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, 640, 640],
            'output_tensor_shape': [1, 3, 640, 640]
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[('encoded_tensor', 'reshaped_tensor')])

    tensor_rt_node = ComposableNode(
        name='rtdetr_tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'verbose': False,
            'force_engine_update': False,
        }],
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{
            'confidence_threshold': rt_detr_confidence_threshold,
        }],
        remappings=[
            ('detections_output', detections_2d_array_output_topic),
        ],
    )
    final_nodes = [
        resize_node,
        pad_node,
        image_to_tensor_node,
        interleave_to_planar_node,
        reshape_node,
        rtdetr_preprocessor_node,
        tensor_rt_node,
        rtdetr_decoder_node,
    ]
    final_nodes += drop_node_nodes
    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=final_nodes)

    final_launch = GroupAction(
        actions=[
            load_composable_nodes,
        ],)

    return [final_launch]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'rtdetr_engine_file_path',
            default_value='/tmp/rtdetr.plan',
            description='The absolute path to the RTDETR engine plan.',
        ),
        DeclareLaunchArgument(
            'rt_detr_confidence_threshold',
            default_value='0.7',
            description='The minimum score for a bounding box to be published.',
        ),
        DeclareLaunchArgument(
            'input_fps',
            default_value='30',
            description='FPS for input message to the drop node'
        ),
        DeclareLaunchArgument(
            'dropped_fps',
            default_value='28',
            description='FPS that are dropped by the drop node'),
        DeclareLaunchArgument(
            'rtdetr_input_qos',
            default_value='SENSOR_DATA',
            description='QOS setting used for RT-DETR input'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
