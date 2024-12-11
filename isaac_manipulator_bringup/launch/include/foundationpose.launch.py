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
    ComposableNode,
    LaunchConfiguration,
    DeclareLaunchArgument,
    LoadComposableNodes,
    IfCondition,
    PythonExpression,
    GroupAction,
    OpaqueFunction,
    LaunchDescription
)

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.types import CameraType


def launch_setup(context, *args, **kwargs):
    camera_type_str = str(context.perform_substitution(LaunchConfiguration('camera_type')))
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    rgb_image_width = int(context.perform_substitution(LaunchConfiguration('rgb_image_width')))
    rgb_image_height = int(context.perform_substitution(LaunchConfiguration('rgb_image_height')))
    depth_image_width = LaunchConfiguration('depth_image_width')
    depth_image_height = LaunchConfiguration('depth_image_height')
    object_class_id = LaunchConfiguration('object_class_id')
    refine_iterations = int(context.perform_substitution(LaunchConfiguration('refine_iterations')))
    symmetry_planes = LaunchConfiguration('symmetry_planes')
    is_object_following = str(context.perform_substitution(LaunchConfiguration(
        'is_object_following', default='False')))

    camera_type = CameraType[camera_type_str]
    detection2_d_array_topic = LaunchConfiguration('detection2_d_array_topic')
    rgb_image_topic = LaunchConfiguration('rgb_image_topic')
    rgb_camera_info_topic = LaunchConfiguration('rgb_camera_info_topic')
    realsense_depth_image_topic = str(context.perform_substitution(
        LaunchConfiguration(
            'realsense_depth_image_topic', default='/camera_1/aligned_depth_to_color/image_raw')))
    foundation_pose_server_depth_topic_name = str(context.perform_substitution(
        LaunchConfiguration(
            'foundation_pose_server_depth_topic_name',
            default='/camera_1/aligned_depth_to_color/image_raw')))

    sensor_data_config = 'SENSOR_DATA'
    resize_input_qos = 'SENSOR_DATA'
    if camera_type is CameraType.hawk:
        if is_object_following == 'True':
            rgb_image_topic = '/rgb/image_rect_color'
            rgb_camera_info_topic = '/rgb/camera_info'
        if is_object_following == 'True':
            depth_image_topic = '/depth_image'
        else:
            depth_image_topic = foundation_pose_server_depth_topic_name
    elif camera_type is CameraType.realsense:
        if is_object_following == 'True':
            realsense_depth_image_topic = '/camera_1/aligned_depth_to_color/image_raw'
            rgb_image_topic = '/camera_1/color/image_raw'
            rgb_camera_info_topic = '/camera_1/color/camera_info'
        depth_image_topic = realsense_depth_image_topic + '_metric'
    elif camera_type is CameraType.isaac_sim:
        depth_image_topic = foundation_pose_server_depth_topic_name
        sensor_data_config = 'DEFAULT'
        resize_input_qos = 'DEFAULT'
    else:
        raise Exception(f'CameraType {camera_type} not implemented.')

    rt_detr_model_input_width = 640

    rgb_image_to_rt_detr_ratio = rgb_image_width / rt_detr_model_input_width

    if is_object_following == 'True':
        detection2_d_array_filter_node = ComposableNode(
            name='detection2_d_array_filter',
            package='isaac_ros_foundationpose',
            plugin='nvidia::isaac_ros::foundationpose::Detection2DArrayFilter',
            parameters=[{
                'desired_class_id': str(context.perform_substitution(object_class_id))}
            ],
            remappings=[('detection2_d_array', detection2_d_array_topic)]
        )

    detection2_d_to_mask_node_input_topic = detection2_d_array_topic
    resize_camera_info_topic = rgb_camera_info_topic

    if is_object_following == 'True':
        detection2_d_to_mask_node_input_topic = 'detection2_d'
        resize_camera_info_topic = 'resize/camera_info'
        resize_input_qos = 'DEFAULT'

    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': int(rgb_image_width/rgb_image_to_rt_detr_ratio),
            'mask_height': int(rgb_image_height/rgb_image_to_rt_detr_ratio)}],
        remappings=[('detection2_d', detection2_d_to_mask_node_input_topic),
                    ('segmentation', 'rt_detr_segmentation')]
    )

    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(rgb_image_width/rgb_image_to_rt_detr_ratio),
            'input_height': int(rgb_image_height/rgb_image_to_rt_detr_ratio),
            'output_width': depth_image_width,
            'output_height': depth_image_height,
            'keep_aspect_ratio': False,
            'disable_padding': False,
            'input_qos': resize_input_qos,
            'output_qos': 'DEFAULT',
        }],
        remappings=[
            ('image', 'rt_detr_segmentation'),
            ('camera_info', resize_camera_info_topic),
            ('resize/image', 'segmentation'),
            ('resize/camera_info', 'camera_info_segmentation')
        ]
    )

    # Realsense depth is in uint16 and millimeters. Convert to float32 and meters
    convert_metric_node = ComposableNode(
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        parameters=[{'input_qos': sensor_data_config}],
        remappings=[
            ('image_raw', realsense_depth_image_topic),
            ('image', depth_image_topic)
        ]
    )

    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'depth_qos': sensor_data_config,
            'color_qos': sensor_data_config,
            'color_info_qos': sensor_data_config,
            'segmentation_qos': 'DEFAULT',

            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
            'refine_iterations': refine_iterations,

            'score_model_file_path': score_model_file_path,
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
            'tf_frame_name': 'detected_object1',

            'symmetry_planes': symmetry_planes
        }],
        remappings=[
            ('pose_estimation/depth_image', depth_image_topic),
            ('pose_estimation/image', rgb_image_topic),
            ('pose_estimation/camera_info', 'camera_info_segmentation'),
            ('pose_estimation/segmentation', 'segmentation'),
            ('pose_estimation/output', 'pose_estimation/output')]
    )

    composable_node_descriptions = [
        detection2_d_to_mask_node,
        resize_mask_node,
        foundationpose_node,
    ]

    if is_object_following == 'True':
        composable_node_descriptions.append(detection2_d_array_filter_node)

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_node_descriptions,
    )

    load_convert_metric_node = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[convert_metric_node],
        condition=IfCondition(PythonExpression([f'"{camera_type}"', ' == ', '"realsense"']))
    )

    final_launch = GroupAction(
        actions=[
            load_composable_nodes, load_convert_metric_node
        ],
    )

    return [final_launch]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value='/tmp/textured_simple.obj',
            description='The absolute file path to the mesh file'),
        DeclareLaunchArgument(
            'texture_path',
            default_value='/tmp/texture_map.png',
            description='The absolute file path to the texture map'),
        DeclareLaunchArgument(
            'refine_model_file_path',
            default_value='/tmp/refine_model.onnx',
            description='The absolute file path to the refine model'),
        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value='/tmp/refine_trt_engine.plan',
            description='The absolute file path to the refine trt engine'),
        DeclareLaunchArgument(
            'score_model_file_path',
            default_value='/tmp/score_model.onnx',
            description='The absolute file path to the score model'),
        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value='/tmp/score_trt_engine.plan',
            description='The absolute file path to the score trt engine'),
        DeclareLaunchArgument(
            'object_class_id',
            default_value='22',
            description='The RT-DETR class ID of the object'),
        DeclareLaunchArgument(
            'refine_iterations',
            default_value='3',
            description='The number of pose refinement iterations to run'),
        DeclareLaunchArgument(
            'symmetry_planes',
            default_value='["x", "y", "z"]',
            description='The plane(s) that the object is symmetric about'),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
