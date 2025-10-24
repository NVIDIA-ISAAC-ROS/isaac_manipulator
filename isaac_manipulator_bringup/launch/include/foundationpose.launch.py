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
from isaac_manipulator_ros_python_utils.manipulator_types import CameraType

from isaac_ros_launch_utils.all_types import (
    ComposableNode, DeclareLaunchArgument, GroupAction, IfCondition, LaunchConfiguration,
    LaunchDescription, LoadComposableNodes, OpaqueFunction, PythonExpression
)


def launch_setup(context, *args, **kwargs):
    camera_type_str = str(context.perform_substitution(LaunchConfiguration('camera_type')))
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    tf_frame_name = context.perform_substitution(LaunchConfiguration('tf_frame_name'))
    refine_iterations = int(context.perform_substitution(LaunchConfiguration('refine_iterations')))
    symmetry_axes = LaunchConfiguration('symmetry_axes')
    segmentation_mask_camera_info_topic = str(context.perform_substitution(
        LaunchConfiguration('segmentation_mask_camera_info_topic')))
    segmentation_mask_topic = str(context.perform_substitution(
        LaunchConfiguration('segmentation_mask_topic')))
    output_pose_estimate_topic = str(context.perform_substitution(
        LaunchConfiguration('output_pose_estimate_topic')))
    discard_old_messages_str = str(context.perform_substitution(LaunchConfiguration(
        'discard_old_messages', default='True')))
    enable_dnn_depth_in_realsense = str(context.perform_substitution(LaunchConfiguration(
        'enable_dnn_depth_in_realsense', default='False')))
    realsense_depth_camera_info_topic = str(context.perform_substitution(
        LaunchConfiguration('realsense_depth_camera_info_topic')))
    discard_msg_older_than_ms_int = int(context.perform_substitution(
        LaunchConfiguration('discard_msg_older_than_ms')))
    discard_old_messages = True if discard_old_messages_str == 'True' else False
    rgb_image_width = int(
        context.perform_substitution(LaunchConfiguration('rgb_image_width')))
    rgb_image_height = int(
        context.perform_substitution(LaunchConfiguration('rgb_image_height')))
    depth_image_width = int(
        context.perform_substitution(LaunchConfiguration('depth_image_width')))
    depth_image_height = int(
        context.perform_substitution(LaunchConfiguration('depth_image_height')))

    camera_type = CameraType[camera_type_str]
    rgb_image_topic = LaunchConfiguration('rgb_image_topic')
    foundationpose_sensor_qos_config = str(context.perform_substitution(
                                LaunchConfiguration('foundationpose_sensor_qos_config')))

    realsense_depth_image_topic = str(context.perform_substitution(
        LaunchConfiguration(
            'realsense_depth_image_topic', default='/camera_1/aligned_depth_to_color/image_raw')))
    foundation_pose_server_depth_topic_name = str(context.perform_substitution(
        LaunchConfiguration(
            'foundation_pose_server_depth_topic_name',
            default='/camera_1/aligned_depth_to_color/image_raw')))

    # Realsense depth is in uint16 and millimeters. Convert to float32 and meters
    convert_metric_node = ComposableNode(
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        parameters=[{'input_qos': foundationpose_sensor_qos_config}],
        remappings=[
            ('image_raw',  realsense_depth_image_topic),
            ('image', foundation_pose_server_depth_topic_name)
        ]
    )

    composable_node_descriptions = []
    node_descriptions = []

    foundationpose_image_input = rgb_image_topic
    foundationpose_camera_info_input = segmentation_mask_camera_info_topic

    if enable_dnn_depth_in_realsense == 'True' and camera_type == CameraType.REALSENSE:
        # Also we need to resize the rgb image to the DNN depth model size.
        foundationpose_image_input = 'resized_rgb_image'
        foundationpose_camera_info_input = 'resized_rgb_camera_info'
        composable_node_descriptions.append(ComposableNode(
            name='resize_for_foundationpose_for_realsense_rgb_for_dnn_depth',
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ResizeNode',
            parameters=[{
                'input_qos': foundationpose_sensor_qos_config,
                'input_width': int(rgb_image_width),
                'input_height': int(rgb_image_height),
                'output_width': int(depth_image_width),
                'output_height': int(depth_image_height),
                'keep_aspect_ratio': False,
                'encoding_desired': 'rgb8',
                'disable_padding': False,
                'use_latest_camera_info': True,
                'drop_old_messages': False,
                'output_qos': foundationpose_sensor_qos_config,
                'input_qos': foundationpose_sensor_qos_config
            }],
            remappings=[
                ('image', rgb_image_topic),
                # This is actually output from foundation pose server node
                # (the camera info for the realsense oringal RGB)
                ('camera_info', segmentation_mask_camera_info_topic),
                ('resize/image', foundationpose_image_input),
                ('resize/camera_info', foundationpose_camera_info_input)
            ]
        ))

        foundation_pose_server_depth_topic_name = \
            'foundation_pose_aligned_dnn_depth_from_realsense_stereo'
        align_depth_to_color_node = ComposableNode(
            name='align_depth_to_color_node',
            package='isaac_ros_depth_image_proc',
            plugin='nvidia::isaac_ros::depth_image_proc::AlignDepthToColorNode',
            parameters=[{
                'input_qos': foundationpose_sensor_qos_config,
                'input_qos_depth': 1,
                'output_qos': foundationpose_sensor_qos_config,
                'output_qos_depth': 1,
                'use_cached_camera_info': True,
            }],
            remappings=[
                ('depth_image',  realsense_depth_image_topic),
                ('camera_info_depth', realsense_depth_camera_info_topic),
                ('camera_info_color', foundationpose_camera_info_input),
                ('aligned_depth', foundation_pose_server_depth_topic_name)
            ]
        )
        composable_node_descriptions.append(align_depth_to_color_node)

    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'depth_qos': foundationpose_sensor_qos_config,
            'color_qos': foundationpose_sensor_qos_config,
            'color_info_qos': foundationpose_sensor_qos_config,
            'segmentation_qos': foundationpose_sensor_qos_config,
            # This depth parameter is important since it reduces any buffering in ROS side
            # that makes foundation pose which is an expensive operation on older frames that we
            # no longer care for (especially in an online workflow such as object following)
            'depth_qos_depth': 1,
            'color_qos_depth': 1,
            'segmentation_qos_depth': 1,
            'color_info_qos_depth': 1,

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
            'tf_frame_name': tf_frame_name,

            'symmetry_axes': symmetry_axes,

            # We can only do this on real robot since GXF does not support Sim time clock
            # And we only do this if workflow is Object following.
            'discard_old_messages': discard_old_messages,
            'discard_msg_older_than_ms': discard_msg_older_than_ms_int,  # usually 1 second
            'pose_estimation_timeout_ms': 5000,  # 5 seconds
        }],
        remappings=[
            ('pose_estimation/depth_image', foundation_pose_server_depth_topic_name),
            ('pose_estimation/image',  foundationpose_image_input),
            ('pose_estimation/camera_info', foundationpose_camera_info_input),
            ('pose_estimation/segmentation', segmentation_mask_topic),
            ('pose_estimation/output', output_pose_estimate_topic)]
    )

    composable_node_descriptions.append(foundationpose_node)

    load_composable_nodes = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=composable_node_descriptions,
    )

    load_convert_metric_node = LoadComposableNodes(
        target_container=constants.MANIPULATOR_CONTAINER_NAME,
        composable_node_descriptions=[convert_metric_node],
        condition=IfCondition(PythonExpression([
            f'"{camera_type}"', ' == ', '"REALSENSE"', 'and',
            f'"{enable_dnn_depth_in_realsense}"', ' != ', '"True"'
        ]))
    )

    final_launch = GroupAction(
        actions=[
            load_composable_nodes, load_convert_metric_node
        ],
    )

    return [final_launch] + node_descriptions


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
            'tf_frame_name',
            default_value='detected_object1',
            description='TF name for detected object'),
        DeclareLaunchArgument(
            'refine_iterations',
            default_value='3',
            description='The number of pose refinement iterations to run'),
        DeclareLaunchArgument(
            'symmetry_axes',
            default_value='["x_180", "y_180", "z_180"]',
            description='The axes that the object is symmetric about'),
        DeclareLaunchArgument(
            'foundationpose_sensor_qos_config',
            default_value='SENSOR_DATA',
            description='The sensor data quality of service for FoundationPose nodes.'
                        'This applies to the resize mask node, mask node and pose estimation nodes'
                        'Also applies to the convert metric node'),
        DeclareLaunchArgument(
            'discard_old_messages',
            default_value='True',
            description='Whether to discard old messages or not. This is only applicable with'
                        ' the real robot since GXF does not support ROS sim time clock'),
        DeclareLaunchArgument(
            'segmentation_mask_camera_info_topic',
            default_value='/segment_anything/camera_info_dropped',
            description='The topic name of the camera info for the segmentation mask'
        ),
        DeclareLaunchArgument(
            'segmentation_mask_topic',
            default_value='/segment_anything/binary_segmentation_mask',
            description='The topic name of the segmentation mask'
        ),
        DeclareLaunchArgument(
            'output_pose_estimate_topic',
            default_value='pose_estimation/output',
            description='The topic name of the output pose estimate'
        ),
        DeclareLaunchArgument(
            'discard_msg_older_than_ms',
            default_value='1000',
            description='The time in milliseconds to discard old messages'
        ),
        DeclareLaunchArgument(
            'enable_dnn_depth_in_realsense',
            default_value='False',
            description='Whether to enable DNN depth in Realsense or not'
        ),
        DeclareLaunchArgument(
            'realsense_depth_camera_info_topic',
            description='The topic name of the depth camera info for the realsense'
        ),
        DeclareLaunchArgument(
            'rgb_image_width',
            description='The width of the RGB image'
        ),
        DeclareLaunchArgument(
            'rgb_image_height',
            description='The height of the RGB image'
        ),
        DeclareLaunchArgument(
            'depth_image_width',
            description='The width of the depth image'
        ),
        DeclareLaunchArgument(
            'depth_image_height',
            description='The height of the depth image'
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
