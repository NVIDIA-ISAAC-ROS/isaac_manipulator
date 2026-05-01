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
from typing import List, Tuple

from ament_index_python.packages import get_package_share_directory

import isaac_ros_launch_utils as lu
from isaac_ros_launch_utils.all_types import Action, LaunchDescription
from isaac_ros_manipulation_ros_python_utils.constants import MANIPULATOR_CONTAINER_NAME
from isaac_ros_manipulation_ros_python_utils.manipulator_types import CameraType


ISAAC_ROS_WS = os.getenv('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise ValueError('ISAAC_ROS_WS env variable is not set')


def get_realsense_depth_topics(
    num_cameras: int, enable_dnn_depth_in_realsense: bool
) -> Tuple[str, str]:
    depth_image_topics = []
    depth_camera_infos = []
    for i in range(num_cameras):
        if not enable_dnn_depth_in_realsense:
            depth_image_topics.append(f'/camera_{i+1}/aligned_depth_to_color/image_raw')
            depth_camera_infos.append(f'/camera_{i+1}/aligned_depth_to_color/camera_info')
        else:
            depth_image_topics.append(f'/camera_{i+1}/depth_image')
            depth_camera_infos.append(f'/camera_{i+1}/rgb/camera_info')
    return depth_image_topics, depth_camera_infos


def get_isaac_sim_depth_topics() -> Tuple[str, str]:
    depth_image_topics: str = '["/front_stereo_camera/depth/ground_truth"]'
    depth_camera_infos: str = '["/front_stereo_camera/left/camera_info"]'
    return depth_image_topics, depth_camera_infos


def add_cumotion(args: lu.ArgumentContainer) -> List[Action]:
    camera_type = CameraType[args.camera_type]
    num_cameras = int(args.num_cameras)
    from_bag = lu.is_true(args.from_bag)
    no_robot_mode = lu.is_true(args.no_robot_mode)
    enable_object_attachment = lu.is_true(args.enable_object_attachment)
    enable_dnn_depth_in_realsense = lu.is_true(args.enable_dnn_depth_in_realsense)
    workspace_bounds_name = str(args.workspace_bounds_name)
    read_esdf_world = lu.is_true(args.read_esdf_world)
    nvblox_global_frame = str(args.nvblox_global_frame)
    actions = []

    # Get topics to subscribe
    if camera_type is CameraType.REALSENSE:
        depth_image_topics, depth_camera_infos = get_realsense_depth_topics(
            num_cameras, enable_dnn_depth_in_realsense)
    elif camera_type is CameraType.ISAAC_SIM:
        depth_image_topics, depth_camera_infos = get_isaac_sim_depth_topics()
    else:
        raise Exception(f'CameraType {camera_type} not implemented.')

    # Get topics to publish
    robot_mask_publish_topics = []
    world_depth_publish_topics = []
    for i in range(num_cameras):
        robot_mask_publish_topics.append(f'/cumotion/camera_{i+1}/robot_mask')
        world_depth_publish_topics.append(f'/cumotion/camera_{i+1}/world_depth')

    # Get the workspace.
    workspace_file_path = lu.get_path(
        'isaac_ros_manipulation_bringup',
        f'config/nvblox/workspace_bounds/{workspace_bounds_name}.yaml')
    if not os.path.exists(workspace_file_path):
        raise Exception(
            f'Workspace with name {workspace_bounds_name} does not exist. '
            'Launching cumotion without a valid workspace is not allowed.')
    actions.append(
        lu.log_info([
            'Loading the ', workspace_bounds_name,
            ' workspace. Ignoring the grid_center_m and grid_size_m parameters of cumotion.'
        ]))

    # Only enable cumotion when running live with a robot arm.
    if not from_bag and not no_robot_mode:
        # Launch cumotion planner as a separate process
        actions.append(
            lu.include(
                'isaac_ros_cumotion',
                'launch/isaac_ros_cumotion.launch.py',
                launch_arguments={
                    # Robot configuration
                    'cumotion_action_server.xrdf_file_path': args.xrdf_file_path,
                    'cumotion_action_server.urdf_file_path': args.urdf_file_path,
                    'cumotion_action_server.tool_frame': args.tool_frame,

                    # Planning parameters
                    'cumotion_action_server.time_dilation_factor': args.time_dilation_factor,

                    # ESDF/World configuration
                    'cumotion_action_server.read_esdf_world': args.read_esdf_world,
                    'cumotion_action_server.publish_cumotion_world_as_voxels':
                        args.publish_cumotion_world_as_voxels,

                    # ROS topics/services
                    'cumotion_action_server.joint_states_topic': args.joint_states_topic,

                    # CUDA MPS configuration
                    'cumotion_action_server.enable_cuda_mps': args.enable_cuda_mps,
                    'cumotion_action_server.cuda_mps_pipe_directory': args.cuda_mps_pipe_directory,
                    'cumotion_action_server.cuda_mps_client_priority':
                        args.cuda_mps_client_priority_planner,
                    'cumotion_action_server.cuda_mps_active_thread_percentage':
                        args.cuda_mps_active_thread_percentage_planner,

                    # Static planning scene
                    'cumotion_action_server.moveit_collision_objects_scene_file':
                        args.moveit_collision_objects_scene_file,
                }))

    if not no_robot_mode and read_esdf_world:
        actions.append(
            lu.include(
                'isaac_ros_cumotion_robot_segmenter',
                'launch/robot_segmenter.launch.py',
                launch_arguments={
                    'robot_segmenter.input_qos': args.qos_setting,
                    'robot_segmenter.output_qos': args.qos_setting,
                    'robot_segmenter.depth_image_topics': depth_image_topics,
                    'robot_segmenter.depth_camera_infos': depth_camera_infos,
                    'robot_segmenter.robot_mask_publish_topics': robot_mask_publish_topics,
                    'robot_segmenter.world_depth_publish_topics': world_depth_publish_topics,
                    'robot_segmenter.distance_threshold': args.distance_threshold,
                    'robot_segmenter.time_sync_slop': args.time_sync_slop,
                    'robot_segmenter.joint_states_topic': args.joint_states_topic,
                    'robot_segmenter.urdf_path': args.urdf_file_path,
                    'robot_segmenter.xrdf_path': args.xrdf_file_path,
                    'robot_segmenter.enable_cuda_mps': args.enable_cuda_mps,
                    'robot_segmenter.cuda_mps_pipe_directory': args.cuda_mps_pipe_directory,
                    'robot_segmenter.cuda_mps_client_priority':
                        args.cuda_mps_client_priority_robot_segmenter,
                    'robot_segmenter.cuda_mps_active_thread_percentage':
                        args.cuda_mps_active_thread_percentage_robot_segmenter,
                    'robot_segmenter.num_cameras': num_cameras,
                    'robot_segmenter.container_name': MANIPULATOR_CONTAINER_NAME,
                }))

    if not no_robot_mode and enable_object_attachment:
        actions.append(
            lu.include(
                'isaac_ros_cumotion_object_attachment',
                'launch/object_attachment.launch.py',
                launch_arguments={
                    # Enable/disable ESDF clearing (automatically enabled when nvblox is active)
                    'object_attachment.clear_esdf_on_attach':
                        'True' if read_esdf_world else 'False',
                    # Pass ESDF reference frame to object attachment
                    'object_attachment.esdf_reference_frame': nvblox_global_frame,

                    'object_attachment.container_name': MANIPULATOR_CONTAINER_NAME,
                }))

    return actions


def generate_launch_description() -> LaunchDescription:
    default_urdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'),
        'urdf',
        'ur5e_robotiq_2f_140.urdf',
    )

    default_xrdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'),
        'xrdf',
        'ur5e_robotiq_2f_140.xrdf',
    )

    args = lu.ArgumentContainer()
    args.add_arg('camera_type')
    args.add_arg('no_robot_mode', False)
    args.add_arg('enable_object_attachment', False)
    args.add_arg('enable_dnn_depth_in_realsense', False)
    args.add_arg('from_bag', False)
    args.add_arg('num_cameras', 1)
    args.add_arg('workspace_bounds_name', '')
    args.add_arg('use_sim_time', False)
    args.add_arg(
        'urdf_file_path',
        cli=True,
        default=default_urdf_file_path,
        description='The URDF for curobo to injest for planning - used by sim')
    args.add_arg(
        'xrdf_file_path',
        cli=True,
        default=default_xrdf_file_path,
        description='The XRDF for cuMotion to injest for planning - used by sim')
    args.add_arg(
        'time_dilation_factor',
        cli=True,
        default='0.25',
        description='Speed scaling factor for the planner')
    args.add_arg(
        'distance_threshold',
        cli=True,
        default='0.15',
        description='Maximum distance from a given collision sphere (in meters) at which'
                    'to mask points in the robot segmenter'
    )
    args.add_arg(
        'time_sync_slop',
        cli=True,
        default='0.1',
        description='Maximum allowed delay (in seconds) for which depth image and joint state '
                    'messages are considered synchronized in the robot segmenter'
    )
    args.add_arg(
        'joint_states_topic',
        cli=True,
        default='/joint_states',
        description='The joint states topic that receive robot position')
    args.add_arg(
        'tool_frame',
        cli=True,
        default='wrist_3_link',
        description='The tool frame of the robot')
    args.add_arg(
        'read_esdf_world',
        cli=True,
        default='True',
        description='When true, indicates that cuMotion should read a Euclidean signed distance'
                    'field (ESDF) as part of its world'
    )
    args.add_arg(
        'publish_cumotion_world_as_voxels',
        cli=True,
        default='True',
        description='When true, indicates that cuMotion should publish its world representation')
    args.add_arg(
        'qos_setting',
        cli=True,
        default='SENSOR_DATA',
        description='Indicates what QOS setting is used in the app')
    args.add_arg(
        'enable_cuda_mps',
        cli=True,
        default='False',
        description='Whether to enable MPS')
    args.add_arg(
        'cuda_mps_pipe_directory',
        cli=True,
        default=f'{ISAAC_ROS_WS}/mps_pipe_dir',
        description='The directory for the MPS pipe')
    args.add_arg(
        'cuda_mps_client_priority_robot_segmenter',
        cli=True,
        default='1',
        description='The client priority for the MPS for robot segmenter')
    args.add_arg(
        'cuda_mps_active_thread_percentage_robot_segmenter',
        cli=True,
        default='100',
        description='The active thread percentage for MPS for robot segmenter')
    args.add_arg(
        'cuda_mps_client_priority_planner',
        cli=True,
        default='0',
        description='The client priority for the MPS for cumotion planner')
    args.add_arg(
        'cuda_mps_active_thread_percentage_planner',
        cli=True,
        default='100',
        description='The active thread percentage for MPS for cumotion planner')
    args.add_arg(
        'moveit_collision_objects_scene_file',
        cli=True,
        default='',
        description='Path to Moveit .scene file with static collision objects to preload'
    )
    args.add_arg(
        'nvblox_global_frame',
        cli=True,
        default='base_link',
        description='Global frame for nvblox ESDF and object attachment'
    )

    args.add_opaque_function(add_cumotion)
    return LaunchDescription(args.get_launch_actions())
