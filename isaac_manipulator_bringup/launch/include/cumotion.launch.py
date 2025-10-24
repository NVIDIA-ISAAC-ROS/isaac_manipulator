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

from isaac_manipulator_ros_python_utils.manipulator_types import CameraType
import isaac_ros_launch_utils as lu
from isaac_ros_launch_utils.all_types import Action, LaunchDescription, Node


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
    actions = []

    # Get topics to subscribe
    if camera_type is CameraType.REALSENSE:
        depth_image_topics, depth_camera_infos = get_realsense_depth_topics(
            num_cameras, enable_dnn_depth_in_realsense)
    elif camera_type is CameraType.ISAAC_SIM:
        depth_image_topics, depth_camera_infos = get_isaac_sim_depth_topics()
    else:
        raise Exception(f'CameraType {camera_type} not implemented.')

    # Speckle filtering is disabled by default, but it's been found to improve depth when
    # a single RealSense camera is used with a shiny robot (e.g., UR e-Series) in view.
    # For that case, setting max_filtered_speckles_size to 1250 was found to work well
    # empirically.
    filter_speckles_in_robot_mask = False
    max_filtered_speckles_size = 0

    # Get topics to publish
    robot_mask_publish_topics = []
    world_depth_publish_topics = []
    for i in range(num_cameras):
        robot_mask_publish_topics.append(f'/cumotion/camera_{i+1}/robot_mask')
        world_depth_publish_topics.append(f'/cumotion/camera_{i+1}/world_depth')

    # Get the workspace.
    workspace_file_path = lu.get_path(
        'isaac_manipulator_bringup',
        f'config/nvblox/workspace_bounds/{workspace_bounds_name}.yaml')
    if not os.path.exists(workspace_file_path):
        raise Exception(
            f'Workspace with name {workspace_bounds_name} does not exist. '
            'Launching cumotion or esdf visualizer without valid workspace is not allowed.')
    actions.append(
        lu.log_info([
            'Loading the ', workspace_bounds_name,
            ' workspace. Ignoring the grid_center_m and grid_size_m parameters of '
            'cumotion and esdf visualizer.'
        ]))

    # We only enable cumotion if we run live including a robot arm,
    # otherwise we enable the esdf visualizer node.
    if not from_bag and not no_robot_mode:
        actions.append(
            lu.include(
                'isaac_ros_cumotion',
                'launch/isaac_ros_cumotion.launch.py',
                launch_arguments={
                    'cumotion_planner.robot': args.robot_file_name,
                    'cumotion_planner.workspace_file_path': str(workspace_file_path),
                    'cumotion_planner.grid_size_m': '[2.0, 2.0, 2.0]',
                    'cumotion_planner.grid_center_m': '[0.0, 0.0, 0.0]',
                    'cumotion_planner.time_dilation_factor': args.time_dilation_factor,
                    'cumotion_planner.tool_frame': args.tool_frame,
                    'cumotion_planner.read_esdf_world': args.read_esdf_world,
                    'cumotion_planner.publish_curobo_world_as_voxels':
                        args.publish_curobo_world_as_voxels,
                    'cumotion_planner.override_moveit_scaling_factors': 'True',
                    'cumotion_planner.joint_states_topic': args.joint_states_topic,
                    'cumotion_planner.voxel_size': '0.01',
                    'cumotion_planner.publish_voxel_size': '0.01',
                    'cumotion_planner.update_link_sphere_server':
                        args.update_link_sphere_server_planner,
                    'cumotion_planner.urdf_path': args.urdf_file_path,
                    'cumotion_planner.enable_cuda_mps': args.enable_cuda_mps,
                    'cumotion_planner.cuda_mps_pipe_directory': args.cuda_mps_pipe_directory,
                    'cumotion_planner.cuda_mps_client_priority':
                        args.cuda_mps_client_priority_planner,
                    'cumotion_planner.cuda_mps_active_thread_percentage':
                        args.cuda_mps_active_thread_percentage_planner,
                    'cumotion_planner.moveit_collision_objects_scene_file':
                        args.moveit_collision_objects_scene_file,
                },
            ))
    else:
        actions.append(
            Node(
                package='isaac_ros_esdf_visualizer',
                namespace='',
                executable='esdf_visualizer',
                name='esdf_visualizer',
                parameters=[{
                    'workspace_file_path': str(workspace_file_path),
                    # Currently the esdf visualizer skips a service call without replacement,
                    # when the response of the last call has not arrived.
                    'esdf_service_call_period_secs': 0.05,  # trying to reach 10 Hz calls
                }],
                output='screen',
            ))

    if not no_robot_mode and read_esdf_world:
        actions.append(
            lu.include(
                'isaac_ros_cumotion',
                'launch/robot_segmentation.launch.py',
                launch_arguments={
                    'robot_segmenter.robot': args.robot_file_name,
                    'robot_segmenter.depth_qos': args.qos_setting,
                    'robot_segmenter.depth_info_qos': args.qos_setting,
                    'robot_segmenter.mask_qos': args.qos_setting,
                    'robot_segmenter.world_depth_qos': args.qos_setting,
                    'robot_segmenter.depth_image_topics': depth_image_topics,
                    'robot_segmenter.depth_camera_infos': depth_camera_infos,
                    'robot_segmenter.robot_mask_publish_topics': robot_mask_publish_topics,
                    'robot_segmenter.world_depth_publish_topics': world_depth_publish_topics,
                    'robot_segmenter.filter_speckles_in_mask': filter_speckles_in_robot_mask,
                    'robot_segmenter.max_filtered_speckles_size': max_filtered_speckles_size,
                    'robot_segmenter.distance_threshold': args.distance_threshold,
                    'robot_segmenter.time_sync_slop': args.time_sync_slop,
                    'robot_segmenter.joint_states_topic': args.joint_states_topic,
                    'robot_segmenter.urdf_path': args.urdf_file_path,
                    'robot_segmenter.update_link_sphere_server':
                        args.update_link_sphere_server_segmenter,
                    'robot_segmenter.enable_cuda_mps': args.enable_cuda_mps,
                    'robot_segmenter.cuda_mps_pipe_directory': args.cuda_mps_pipe_directory,
                    'robot_segmenter.cuda_mps_client_priority':
                        args.cuda_mps_client_priority_robot_segmenter,
                    'robot_segmenter.cuda_mps_active_thread_percentage':
                        args.cuda_mps_active_thread_percentage_robot_segmenter,
                    'robot_segmenter.num_cameras': num_cameras,
                }))

    if not no_robot_mode and enable_object_attachment:
        if read_esdf_world:
            depth_image_topics_for_attachment = world_depth_publish_topics
            action_names = args.action_names
        else:
            if args.object_attachment_type == 'SPHERE':
                raise Exception(
                    'When nvblox is disabled, object_attachment_type must be '
                    'CUSTOM_MESH or CUBOID.')
            depth_image_topics_for_attachment = ['']
            action_names = [args.update_link_sphere_server_planner]

        actions.append(
            lu.include(
                'isaac_ros_cumotion_object_attachment',
                'launch/object_attachment.launch.py',
                launch_arguments={
                    'object_attachment.robot': args.robot_file_name,
                    'object_attachment.urdf_path': args.urdf_file_path,
                    'object_attachment.time_sync_slop': args.time_sync_slop,
                    'object_attachment.filter_depth_buffer_time': args.filter_depth_buffer_time,
                    'object_attachment.joint_states_topic': args.joint_states_topic,
                    'object_attachment.depth_image_topics': depth_image_topics_for_attachment,
                    'object_attachment.depth_camera_infos': depth_camera_infos,
                    'object_attachment.object_link_name': args.object_link_name,
                    'object_attachment.action_names': action_names,
                    'object_attachment.search_radius': args.search_radius,
                    'object_attachment.surface_sphere_radius': args.surface_sphere_radius,
                    'object_attachment.clustering_bypass_clustering': args.clustering_bypass,
                    'object_attachment.clustering_hdbscan_min_samples':
                        args.clustering_hdbscan_min_samples,
                    'object_attachment.clustering_hdbscan_min_cluster_size':
                        args.clustering_hdbscan_min_cluster_size,
                    'object_attachment.clustering_hdbscan_cluster_selection_epsilon':
                        args.clustering_hdbscan_cluster_selection_epsilon,
                    'object_attachment.clustering_num_top_clusters_to_select':
                        args.clustering_num_top_clusters_to_select,
                    'object_attachment.clustering_group_clusters': args.clustering_group_clusters,
                    'object_attachment.clustering_min_points': args.clustering_min_points,
                    'object_attachment.depth_qos': args.qos_setting,
                    'object_attachment.depth_info_qos': args.qos_setting,
                    'use_sim_time': args.use_sim_time,
                    'object_attachment.object_esdf_clearing_padding':
                        args.object_esdf_clearing_padding,
                    'object_attachment.trigger_aabb_object_clearing':
                        args.trigger_aabb_object_clearing if read_esdf_world else 'false'
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
        'robot_file_name',
        cli=True,
        default=default_xrdf_file_path,
        description='The file path that describes robot')
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
        'filter_depth_buffer_time',
        cli=True,
        default='0.1',
        description='Maximum allowed delay between the buffer of depth images with respect to '
                    'current time. Makes sure object attachment does not operate on older depth '
                    'images'
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
        'trigger_aabb_object_clearing',
        cli=True,
        default='False',
        description='When true, indicates that object attachment should instruct nvblox to clear '
                    'an axis-aligned bounding box (AABB) encompassing the object'
    )
    args.add_arg(
        'object_attachment_type',
        cli=True,
        default='CUSTOM_MESH',
        description='Object attachment type')
    args.add_arg(
        'object_link_name',
        cli=True,
        default='attached_object',
        description='Object link name for object attachment')
    args.add_arg(
        'search_radius',
        cli=True,
        default='0.1',
        description='Search radius for object attachment')
    args.add_arg(
        'update_link_sphere_server_segmenter',
        cli=True,
        default='segmenter_attach_object',
        description='Update link sphere server for robot segmenter')
    args.add_arg(
        'update_link_sphere_server_planner',
        cli=True,
        default='planner_attach_object',
        description='Update link sphere server for cumotion planner')
    args.add_arg(
        'clustering_bypass',
        cli=True,
        default='False',
        description='Whether to bypass clustering')
    args.add_arg(
        'action_names',
        cli=True,
        default="['segmenter_attach_object', 'planner_attach_object']",
        description='List of action names for the object attachment')
    args.add_arg(
        'clustering_hdbscan_min_samples',
        cli=True,
        default='20',
        description='HDBSCAN min samples for clustering')
    args.add_arg(
        'clustering_hdbscan_min_cluster_size',
        cli=True,
        default='30',
        description='HDBSCAN min cluster size for clustering')
    args.add_arg(
        'clustering_hdbscan_cluster_selection_epsilon',
        cli=True,
        default='0.5',
        description='HDBSCAN cluster selection epsilon')
    args.add_arg(
        'clustering_num_top_clusters_to_select',
        cli=True,
        default='3',
        description='Number of top clusters to select')
    args.add_arg(
        'clustering_group_clusters',
        cli=True,
        default='False',
        description='Whether to group clusters')
    args.add_arg(
        'clustering_min_points',
        cli=True,
        default='100',
        description='Minimum points for clustering')
    args.add_arg(
        'publish_curobo_world_as_voxels',
        cli=True,
        default='True',
        description='When true, indicates that cuMotion should publish its world representation')
    args.add_arg(
        'qos_setting',
        cli=True,
        default='SENSOR_DATA',
        description='Indicates what QOS setting is used in the app')
    args.add_arg(
        'surface_sphere_radius',
        cli=True,
        default='0.01',
        description='Radius for object surface collision spheres')
    args.add_arg(
        'object_esdf_clearing_padding',
        cli=True,
        default='[0.025, 0.025, 0.025]',
        description='Amount by which to pad each dimension of the AABB enclosing the object, in '
                    'meters, for the purpose of ESDF clearing',
    )
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

    args.add_opaque_function(add_cumotion)
    return LaunchDescription(args.get_launch_actions())
