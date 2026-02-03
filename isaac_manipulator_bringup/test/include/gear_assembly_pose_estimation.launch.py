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

import os
from typing import List

from ament_index_python.packages import get_package_share_directory

import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.manipulator_types import CameraType, TrackingType
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    launch_dir = os.path.join(get_package_share_directory('isaac_manipulator_bringup'), 'launch')

    goal_frame = LaunchConfiguration('goal_frame')
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    frequency = LaunchConfiguration('frequency')
    calibration_name = LaunchConfiguration('calibration_name')

    nodes = []

    nodes.append(
        ComposableNodeContainer(
            name=constants.MANIPULATOR_CONTAINER_NAME,
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            output='both',
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_dir, '/include/realsense.launch.py']),
            launch_arguments={
                'camera_ids_config_name': calibration_name,
            }.items()
        )
    )

    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_dir, '/include/segment_anything.launch.py']),
            launch_arguments={
                'image_input_topic': '/camera_1/color/image_raw',
                'depth_topic_name': '/camera_1/aligned_depth_to_color/image_raw',
                'camera_info_input_topic': '/camera_1/color/camera_info',
                'segment_anything_input_points_topic': 'colored_segmentation_mask_mouse_left',
                'image_width': image_width,
                'image_height': image_height,
                'depth_image_width': image_width,
                'depth_image_height': image_height,
                'sam_model_repository_paths': sam_model_repository_paths,
                'segment_anything_is_point_triggered': 'True',
                'segment_anything_enable_debug_output': 'True',
                'input_qos': 'SENSOR_DATA',
                'output_qos': 'DEFAULT',
            }.items()
        )
    )

    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_dir, '/include/foundationpose.launch.py']),
            launch_arguments={
                'camera_type': str(CameraType.REALSENSE),
                'rgb_image_topic': '/segment_anything/image_dropped',
                'realsense_depth_image_topic': '/segment_anything/depth_dropped',
                'realsense_depth_camera_info_topic': '/segment_anything/camera_info_dropped',
                'foundation_pose_server_depth_topic_name': '/camera_1/depth/image',
                'mesh_file_path': mesh_file_path,
                'texture_path': texture_path,
                'refine_model_file_path': refine_model_file_path,
                'refine_engine_file_path': refine_engine_file_path,
                'score_model_file_path': score_model_file_path,
                'score_engine_file_path': score_engine_file_path,
                'tf_frame_name': 'gear_assembly_frame',
                'refine_iterations': '3',
                'symmetry_axes': '[""]',
                'discard_old_messages': 'False',
                'foundationpose_sensor_qos_config': 'DEFAULT',
                'rgb_image_width': image_width,
                'rgb_image_height': image_height,
                'depth_image_width': image_width,
                'depth_image_height': image_height,
            }.items()
        )
    )

    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_dir, '/include/static_transforms.launch.py']),
            launch_arguments={
                'camera_type': str(CameraType.REALSENSE),
                'tracking_type': str(TrackingType.GEAR_ASSEMBLY),
                'calibration_name': calibration_name,
            }.items(),
        )
    )

    nodes.append(
        Node(
            name='goal_pose_publisher_node',
            package='isaac_manipulator_ur_dnn_policy',
            executable='goal_pose_publisher_node.py',
            parameters=[{
                'world_frame': 'base',
                'goal_frame': goal_frame,
                'frequency': frequency,
            }],
            output='both',
            on_exit=Shutdown(),
        )
    )

    return nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'goal_frame',
            description='Frame of the goal pose to be published',
            choices=['gear_shaft_small', 'gear_shaft_medium', 'gear_shaft_large'],
        ),
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value='isaac_ros_assets/isaac_manipulator_ur_dnn_policy/'
                          'gear_base/gear_base.obj',
            description='File path to the mesh file',
        ),
        DeclareLaunchArgument(
            'texture_path',
            default_value='',
            description='File path to the texture map',
        ),
        DeclareLaunchArgument(
            'sam_model_repository_paths',
            default_value='["isaac_ros_assets/models"]',
            description='File path to the repository of SAM models',
        ),
        DeclareLaunchArgument(
            'refine_model_file_path',
            default_value='isaac_ros_assets/models/foundationpose/refine_model.onnx',
            description='File path to the FoundationPose refine model',
        ),
        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value='isaac_ros_assets/models/foundationpose/refine_trt_engine.plan',
            description='File path to the FoundationPose refine trt engine',
        ),
        DeclareLaunchArgument(
            'score_model_file_path',
            default_value='isaac_ros_assets/models/foundationpose/score_model.onnx',
            description='File path to the FoundationPose score model',
        ),
        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value='isaac_ros_assets/models/foundationpose/score_trt_engine.plan',
            description='File path to the FoundationPose score trt engine',
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='1280',
            description='Image width',
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='720',
            description='Image height',
        ),
        DeclareLaunchArgument(
            'frequency',
            default_value='30.0',
            description='Inference frequency',
        ),
        DeclareLaunchArgument(
            'calibration_name',
            description='Name of the calibration setup (eg: hubble_ur10e_test_bench)',
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
