# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import enum
from typing import Tuple, List, Dict
from dataclasses import dataclass

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch.actions import (
    IncludeLaunchDescription,
    OpaqueFunction,
    DeclareLaunchArgument,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration

from moveit_configs_utils import MoveItConfigsBuilder
import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.types import (
    CameraType, PoseEstimationType, IsaacSimCameraConfig, GripperType, ObjectAttachmentShape
)
from isaac_manipulator_ros_python_utils.launch_utils import (
    get_variable,
    get_robot_description_contents,
    get_gripper_collision_links
)


BASE_PATH = os.environ.get('ISAAC_ROS_WS')
WORKSPACE_BOUNDS_NAME = 'sim_test_bench'


class IsaacSimWorkflowType(enum.Enum):
    POSE_TO_POSE = 'pose_to_pose'
    PICK_AND_PLACE = 'pick_and_place'
    OBJECT_FOLLOWING = 'object_following'


@dataclass
class IsaacSimWorkflowConfig:
    '''Config that tracks all variables needed to perform workflows in Isaac Sim'''
    workflow_type: IsaacSimWorkflowType
    pose_estimator_type: PoseEstimationType
    grasps_file_path: str
    remapped_joint_states: Dict
    foundation_pose_mesh_file_path: str
    foundation_pose_texture_path: str
    foundation_pose_refine_engine_file_path: str
    foundation_pose_score_engine_file_path: str
    dope_model_file_path: str
    dope_engine_file_path: str
    rtdetr_engine_file_path: str
    object_class_id: str
    rt_detr_confidence_threshold: str
    rviz_config_file: str
    enable_pose_estimation: bool
    use_ground_truth_pose_in_sim: bool
    pick_and_place_planner_retries: int
    pick_and_place_retry_wait_time: float
    sim_gt_asset_frame_id: str
    object_attachment_type: ObjectAttachmentShape
    attach_object_mesh_file_path: str
    filter_depth_buffer_time: str
    time_sync_slop: str
    object_attachment_scale: LaunchConfiguration
    use_pose_from_rviz: LaunchConfiguration
    grasp_parent_frame: str = 'robotiq_base_link'
    srdf_asset_name: str = 'ur_robotiq_2f_140_gripper'
    gripper_type: str = 'robotiq_2f_140'
    asset_name: str = 'ur_robotiq_gripper'
    ur_type: str = 'ur10e'
    color_camera_topic_name: str = IsaacSimCameraConfig().color_camera_topic_name
    depth_camera_topic_name: str = IsaacSimCameraConfig().depth_camera_topic_name
    color_camera_info_topic_name: str = IsaacSimCameraConfig().color_camera_info_topic_name
    depth_camera_info_topic_name: str = IsaacSimCameraConfig().depth_camera_info_topic_name
    rgb_image_width: str = IsaacSimCameraConfig().rgb_image_width
    rgb_image_height: str = IsaacSimCameraConfig().rgb_image_height
    depth_image_width: str = IsaacSimCameraConfig().depth_image_width
    depth_image_height: str = IsaacSimCameraConfig().depth_image_height
    camera_type: CameraType = CameraType.isaac_sim
    use_sim_time: bool = True
    log_level: str = 'error'
    # Pick Place config
    enable_object_attachment: bool = True
    use_ground_truth_pose: bool = True
    use_pose_estimator_pose: bool = False
    dope_rotation_y_axis: str = '-90.0'
    dope_enable_tf_publishing: str = 'true'
    foundation_pose_detection2_d_array_topic: str = 'detections_output'


def get_moveit_group_node(workflow_config: IsaacSimWorkflowConfig) -> Node:
    '''Returns the move it group node for a particular configuration

    Args:
        use_sim_time (bool): Use sim time
        asset_name (str): Asset name (UR10e_robotiq, UR5e_robotiq etc)

    Returns:
        Node: The move group ROS node
    '''

    controller_file_name = 'moveit_sim_controllers.yaml'
    kinematics_file_name = 'kinematics_sim.yaml'
    joint_limits_file_name = 'joint_limits.yaml'

    srdf_file_name = f'{workflow_config.srdf_asset_name}.srdf.xacro'

    srdf_path = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'srdf',
        srdf_file_name,
    )
    kinematics_path = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'config',
        kinematics_file_name,
    )
    joint_limits = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'config',
        joint_limits_file_name,
    )
    moveit_controllers = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'config',
        controller_file_name,
    )
    robot_description_content = get_robot_description_contents(
        asset_name=workflow_config.asset_name,
        ur_type=workflow_config.ur_type,
        use_sim_time=workflow_config.use_sim_time,
        gripper_type=workflow_config.gripper_type,
        grasp_parent_frame=workflow_config.grasp_parent_frame,
        dump_to_file=False,
        output_file=None,
    )
    moveit_config = (
        MoveItConfigsBuilder('ur10e', package_name='isaac_manipulator_pick_and_place')
        .robot_description_semantic(file_path=srdf_path)
        .robot_description_kinematics(file_path=kinematics_path)
        .joint_limits(file_path=joint_limits)
        .trajectory_execution(file_path=moveit_controllers)
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    robot_description = {'robot_description': robot_description_content}
    # The mapping features of MoveItConfigsBuilder to pass the xacro parameters
    # did not work, hence we overide the robot description seperately
    moveit_config.robot_description = robot_description

    # Add cuMotion to list of planning pipelines.
    cumotion_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml',
    )
    with open(cumotion_config_file_path) as cumotion_config_file:
        cumotion_config = yaml.safe_load(cumotion_config_file)
    moveit_config.planning_pipelines['planning_pipelines'].insert(
        0, 'isaac_ros_cumotion'
    )
    moveit_config.planning_pipelines['planning_pipelines'].insert(1, 'ompl')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'
    if workflow_config.use_sim_time:
        moveit_config.trajectory_execution['trajectory_execution'][
            'allowed_start_tolerance'
        ] = 0.0
    moveit_config.moveit_cpp.update({'use_sim_time': workflow_config.use_sim_time})
    # Start the actual move_group node/action server
    move_it_dict = moveit_config.to_dict()

    move_it_dict['capabilities'] = 'move_group/ExecuteTaskSolutionCapability'
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[move_it_dict],
        arguments=['--ros-args', '--log-level', 'info'],
    )
    return move_group_node


def get_object_detection_servers(workflow_config: IsaacSimWorkflowConfig):
    '''Object detection servers

    Args:
        workflow_config (IsaacSimWorkflowConfig): Workflow config
    '''
    # Add objectinfo servers
    isaac_manipulator_servers_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_servers'), 'launch')

    object_detection_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/object_detection_server.launch.py']),
        launch_arguments={
            'obj_input_img_topic_name': workflow_config.color_camera_topic_name,
        }.items())

    foundation_pose_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/foundation_pose_server.launch.py']),
        launch_arguments={
            'fp_in_img_topic_name': workflow_config.color_camera_topic_name,
            'fp_in_depth_topic_name': workflow_config.depth_camera_topic_name
        }.items())

    objectinfo_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/object_info_server.launch.py']),)
    return [
        object_detection_server_launch,
        foundation_pose_server_launch,
        objectinfo_server_launch
    ]


def get_cumotion_node(
    camera_type: CameraType, asset_name: str, enable_object_attachment: bool,
    workflow_config: IsaacSimWorkflowConfig
) -> Node:
    '''

    Get cumotion node

    Args:
        camera_type (CameraType): The camera type to be used
        asset_name (str): Asset to be moved

    Returns:
        Node: The cumotion launch node
    '''
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )

    xrdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'), 'xrdf',
        f'{asset_name}.xrdf'
    )

    urdf_file_name = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'), 'urdf',
        f'{asset_name}.urdf'
    )

    tool_frame = 'gripper_frame' if \
        workflow_config.workflow_type == IsaacSimWorkflowType.PICK_AND_PLACE else 'wrist_3_link',
    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/cumotion.launch.py']
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'camera_type': str(camera_type),
            'robot_file_name': xrdf_file_path,
            'tool_frame': tool_frame,
            'joint_states_topic': '/isaac_joint_states',
            'time_sync_slop': workflow_config.time_sync_slop,
            'filter_depth_buffer_time': workflow_config.filter_depth_buffer_time,
            'distance_threshold': '0.05',
            'time_dilation_factor': '1.0',
            'urdf_file_path': urdf_file_name,
            'qos_setting': 'DEFAULT',
            'enable_object_attachment': 'true' if enable_object_attachment else 'false',
            'workspace_bounds_name': WORKSPACE_BOUNDS_NAME,
            'trigger_aabb_object_clearing': 'True'
        }.items(),
    )

    return cumotion_launch


def get_nvblox_node(camera_type: CameraType) -> Tuple[Node, Node]:
    '''

    Get nvblox node

    Args:
        camera_type (CameraType): The camera type to be used

    Returns:
        Node: The nvblox launch nodes
    '''

    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )

    nvblox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_files_include_dir, '/nvblox.launch.py']),
        launch_arguments={
            'camera_type': str(camera_type),
            'use_sim_time': 'true',
            'workspace_bounds_name': WORKSPACE_BOUNDS_NAME,
        }.items(),
    )

    return nvblox_launch


def get_ros2_control_nodes(
    use_sim_time: bool, remapped_joint_states: Dict
) -> List[Node]:
    '''Returns the controller manager, and controller nodes that need to be run to enable robotic
    control in Isaac sim/real.

    Real robot not yet supported

    Args:
        use_sim_time (bool): Use sim time

    Returns:
        List[Node]: List of ros2 control nodes
    '''
    if use_sim_time:
        controller_file_name = 'controllers_sim.yaml'
    else:
        raise NotImplementedError(
            'This file is not ready to be used on the real robot'
        )

    ros2_controllers_path = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'config',
        controller_file_name,
    )
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[ros2_controllers_path, {'use_sim_time': use_sim_time}],
        remappings=[
            (
                '/controller_manager/robot_description',
                remapped_joint_states['/controller_manager/robot_description'],
            )
        ],
        output='screen',
    )

    scaled_joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller', '-c', '/controller_manager'],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
        ],
    )

    gripper_controller = Node(
        package='isaac_manipulator_pick_and_place',
        executable='isaac_sim_gripper_action_server.py',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    return [
        ros2_control_node,
        scaled_joint_trajectory_controller_spawner,
        gripper_controller,
        joint_state_broadcaster_spawner,
    ]


def get_isaac_sim_workflow_config(context: LaunchContext) -> IsaacSimWorkflowConfig:
    '''Gets the Isaac sim workflow config that is used to propogate state to different nodes

    Raises:
        ValueError if some input is wrong
    Returns:
        Workflow config containing global variables used throughtout the launch file
    '''
    workflow_type = IsaacSimWorkflowType(get_variable(context, 'workflow_type'))
    grasps_file_path = get_variable(context, 'grasps_file_path')
    pose_estimation_user_input = get_variable(context, 'pose_estimation_type')
    foundation_pose_mesh_file_path = get_variable(context, 'foundation_pose_mesh_file_path')
    foundation_pose_texture_path = get_variable(context, 'foundation_pose_texture_path')
    foundation_pose_refine_engine_file_path = get_variable(
        context, 'foundation_pose_refine_engine_file_path')
    foundation_pose_score_engine_file_path = get_variable(
        context, 'foundation_pose_score_engine_file_path')
    foundation_pose_score_engine_file_path = get_variable(
        context, 'foundation_pose_score_engine_file_path')
    rtdetr_engine_file_path = get_variable(context, 'rtdetr_engine_file_path')
    rtdetr_object_class_id = get_variable(context, 'rtdetr_object_class_id')
    rt_detr_confidence_threshold = get_variable(context, 'rt_detr_confidence_threshold')
    dope_model_file_path = get_variable(context, 'dope_model_file_path')
    dope_engine_file_path = get_variable(context, 'dope_engine_file_path')
    pick_and_place_retry_wait_time = float(get_variable(context, 'pick_and_place_retry_wait_time'))
    use_ground_truth_pose_in_sim = get_variable(context, 'use_ground_truth_pose_in_sim')
    pick_and_place_planner_retries = int(get_variable(context, 'pick_and_place_planner_retries'))
    sim_gt_asset_frame_id = get_variable(context, 'sim_gt_asset_frame_id')
    object_attachment_scale = LaunchConfiguration('object_attachment_scale')
    attach_object_mesh_file_path = get_variable(context, 'attach_object_mesh_file_path')
    filter_depth_buffer_time = get_variable(context, 'filter_depth_buffer_time')
    time_sync_slop = get_variable(context, 'time_sync_slop')
    object_attachment_type = ObjectAttachmentShape(get_variable(context, 'object_attachment_type'))

    if pose_estimation_user_input == 'dope':
        pose_estimator_type = PoseEstimationType.dope
    elif pose_estimation_user_input == 'foundationpose':
        pose_estimator_type = PoseEstimationType.foundationpose
    else:
        raise ValueError(f'Pose estimation type {pose_estimation_user_input} not supported')

    if use_ground_truth_pose_in_sim == 'true':
        use_ground_truth_pose_in_sim = True
    else:
        use_ground_truth_pose_in_sim = False

    enable_pose_estimation = False
    if workflow_type == IsaacSimWorkflowType.OBJECT_FOLLOWING:
        enable_pose_estimation = True

    remapped_joint_states = {
        '/joint_states': '/isaac_parsed_joint_states',
        '/controller_manager/robot_description': '/robot_description',
    }

    rviz_config_file = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'),
        'rviz',
        'viewport.rviz',
    )
    log_level = get_variable(context, 'log_level')

    use_pose_from_rviz = LaunchConfiguration('use_pose_from_rviz')

    workflow_config = IsaacSimWorkflowConfig(
        pose_estimator_type=pose_estimator_type,
        workflow_type=workflow_type,
        grasps_file_path=grasps_file_path,
        foundation_pose_mesh_file_path=foundation_pose_mesh_file_path,
        foundation_pose_texture_path=foundation_pose_texture_path,
        foundation_pose_refine_engine_file_path=foundation_pose_refine_engine_file_path,
        foundation_pose_score_engine_file_path=foundation_pose_score_engine_file_path,
        rtdetr_engine_file_path=rtdetr_engine_file_path,
        rt_detr_confidence_threshold=rt_detr_confidence_threshold,
        object_class_id=rtdetr_object_class_id,
        dope_model_file_path=dope_model_file_path,
        dope_engine_file_path=dope_engine_file_path,
        enable_pose_estimation=enable_pose_estimation,
        remapped_joint_states=remapped_joint_states,
        rviz_config_file=rviz_config_file,
        pick_and_place_retry_wait_time=pick_and_place_retry_wait_time,
        use_ground_truth_pose_in_sim=use_ground_truth_pose_in_sim,
        pick_and_place_planner_retries=pick_and_place_planner_retries,
        sim_gt_asset_frame_id=sim_gt_asset_frame_id,
        log_level=log_level,
        attach_object_mesh_file_path=attach_object_mesh_file_path,
        object_attachment_type=object_attachment_type,
        object_attachment_scale=object_attachment_scale,
        filter_depth_buffer_time=filter_depth_buffer_time,
        time_sync_slop=time_sync_slop,
        use_pose_from_rviz=use_pose_from_rviz
    )

    return workflow_config


def get_robot_state_publisher(workflow_config: IsaacSimWorkflowConfig) -> Node:
    '''Returns the robot state publisher that publishes TF and robot model for the scene.

    Args:
        workflow_config (IsaacSimWorkflowConfig): Config for workflow

    Returns:
        Node: Robot state publisher node
    '''
    robot_description = get_robot_description_contents(
        asset_name=workflow_config.asset_name,
        ur_type=workflow_config.ur_type,
        use_sim_time=workflow_config.use_sim_time,
        gripper_type=workflow_config.gripper_type,
        grasp_parent_frame=workflow_config.grasp_parent_frame,
        dump_to_file=False,
        output_file=None,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description, 'use_sim_time': workflow_config.use_sim_time}
        ],
        remappings=[('/joint_states', workflow_config.remapped_joint_states['/joint_states'])],
    )

    return robot_state_publisher


def get_joint_state_publisher(workflow_config: IsaacSimWorkflowConfig) -> Node:
    '''Returns joint state publisher that publishes TF of joints of a robot arm

    Args:
        workflow_config (IsaacSimWorkflowConfig): Workflow config

    Returns:
        Node: Joint state publisher node
    '''
    return Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': workflow_config.use_sim_time},
        ],
    )


def get_visualization_node(rviz_config_file: str, use_sim_time: bool) -> Node:
    '''Returns the RVIz visualization node

    Args:
        rviz_config_file (str): Viewport file path

    Returns:
        Node: ROS node for RViz2
    '''
    return Node(
        name='rviz2',
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
    )


def get_isaac_sim_joint_parser_node(use_sim_time: bool) -> Node:
    '''Isaac Sim joint parser node which parses joint states coming from Isaac sim and sets every
    joint underneath finger_joint to be a mimic of it, such that it does not integere in Moveit
    planning because of URDF/USD structural differences

    Args:
        use_sim_time (bool): Use sim time

    Returns:
        Node: ROS Node
    '''
    return Node(
        package='isaac_manipulator_pick_and_place',
        executable='joint_parser_node.py',
        name='joint_parser',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )


def get_foundation_pose_nodes(workflow_config: IsaacSimWorkflowConfig) -> List[Node]:
    '''Return foundation pose nodes for RT-DETR and Foundation Pose nodes

    Args:
        workflow_config (IsaacSimWorkflowConfig): Workflow config
    Returns:
        List[Node]: RTDETR and Foundation Pose
    '''
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )

    rtdetr_detections_topic = 'detections_output'
    foundation_pose_rgb_image_topic = workflow_config.color_camera_topic_name
    foundation_pose_rgb_camera_info = workflow_config.color_camera_info_topic_name
    foundation_pose_depth_image_topic = workflow_config.depth_camera_topic_name
    foundation_pose_detections_topic = workflow_config.foundation_pose_detection2_d_array_topic
    is_object_following = 'True'
    if workflow_config.workflow_type == IsaacSimWorkflowType.PICK_AND_PLACE:
        foundation_pose_depth_image_topic = \
            '/foundation_pose_server/camera_1/aligned_depth_to_color/image_raw'
        foundation_pose_rgb_image_topic = '/foundation_pose_server/camera_1/color/image_raw'
        foundation_pose_rgb_camera_info = '/foundation_pose_server/resize/camera_info'
        foundation_pose_detections_topic = '/foundation_pose_server/bbox'
        is_object_following = 'False'
        rtdetr_detections_topic = '/detections'

    pose_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/foundationpose.launch.py']
        ),
        launch_arguments={
            'camera_type': str(workflow_config.camera_type),
            'rgb_image_width': workflow_config.rgb_image_width,
            'rgb_image_height': workflow_config.rgb_image_height,
            'depth_image_width': workflow_config.depth_image_width,
            'rgb_image_topic': foundation_pose_rgb_image_topic,
            'rgb_camera_info_topic': foundation_pose_rgb_camera_info,
            'foundation_pose_server_depth_topic_name': foundation_pose_depth_image_topic,
            'depth_image_height': workflow_config.depth_image_height,
            'object_class_id': workflow_config.object_class_id,  # for soup can
            'mesh_file_path': workflow_config.foundation_pose_mesh_file_path,
            'texture_path': workflow_config.foundation_pose_texture_path,
            'refine_engine_file_path': workflow_config.foundation_pose_refine_engine_file_path,
            'score_engine_file_path': workflow_config.foundation_pose_score_engine_file_path,
            'detection2_d_array_topic': foundation_pose_detections_topic,
            'is_object_following': is_object_following
        }.items(),
    )
    rtdetr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/rtdetr.launch.py']
        ),
        launch_arguments={
            'image_width': workflow_config.rgb_image_width,
            'image_height': workflow_config.rgb_image_height,
            'image_input_topic': workflow_config.color_camera_topic_name,
            'camera_info_input_topic': workflow_config.color_camera_info_topic_name,
            'rtdetr_engine_file_path': workflow_config.rtdetr_engine_file_path,
            'rt_detr_confidence_threshold': workflow_config.rt_detr_confidence_threshold,
            'detections_2d_array_output_topic': rtdetr_detections_topic,
            'rtdetr_is_object_following': 'True',
            # These are numbers found on the test machine, this might vary for your machine
            'input_fps': '15',
            'dropped_fps': '10',
            'rtdetr_input_qos': 'DEFAULT'
        }.items(),
    )

    return [pose_estimator, rtdetr_launch]


def get_dope_nodes(workflow_config: IsaacSimWorkflowConfig) -> List[Node]:
    '''Get DOPE nodes

    Returns:
        List[Node]: Node
    '''
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )
    pose_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/dope.launch.py']
        ),
        launch_arguments={
            'image_input_topic': workflow_config.color_camera_topic_name,
            'camera_info_input_topic': workflow_config.color_camera_info_topic_name,
            'input_image_width': workflow_config.rgb_image_width,
            'input_image_height': workflow_config.rgb_image_height,
            'dope_network_image_width': workflow_config.rgb_image_width,
            'dope_network_image_height': workflow_config.rgb_image_height,
            'dope_model_file_path': workflow_config.dope_model_file_path,
            'dope_engine_file_path': workflow_config.dope_engine_file_path,
            'dope_enable_tf_publishing': workflow_config.dope_enable_tf_publishing,
            'rotation_y_axis': workflow_config.dope_rotation_y_axis,
            'input_fps': '15',
            'dropped_fps': '10',
            'dope_input_qos': 'DEFAULT',
        }.items(),
    )
    return [pose_estimator]


def get_pose_to_pose() -> List[Node]:
    '''Get pose to pose nodes

    Args:
        workflow_config (IsaacSimWorkflowConfig): Workflow config

    Returns:
        List[Node]: List of nodes
    '''
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )
    pose_to_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/pose_to_pose.launch.py']
        ),
    )
    return [pose_to_pose_launch]


def get_object_following() -> List[Node]:
    '''
    Object Following. Add offset over detected pose so that the robot gripper does not
    collide with it when following the object

    Returns:
        List[Node]: List of nodes for object following
    '''
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include'
    )
    return [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.043', '0.359', '0.065', '0.553',
                       '0.475', '-0.454', '0.513', 'detected_object1', 'goal_frame'],
            ), IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [launch_files_include_dir, '/goal.launch.py']
                ),
                launch_arguments={
                    'grasp_frame': 'goal_frame',
                }.items()
            )
    ]


def get_manipulation_container(workflow_config: IsaacSimWorkflowConfig) -> Node:
    '''Returns manipulation container that allows multiple nodes to run in a single process

    Args:
        workflow_config (IsaacSimWorkflowConfig): Config for workflow

    Returns:
        Node: Manipulation container
    '''
    manipulation_container = Node(
        name=constants.MANIPULATOR_CONTAINER_NAME,
        package='rclcpp_components',
        executable='component_container_mt',
        arguments=['--ros-args', '--log-level', workflow_config.log_level],
        parameters=[{'use_sim_time': workflow_config.use_sim_time}],
    )

    return manipulation_container


def get_pick_and_place_orchestrator(workflow_config: IsaacSimWorkflowConfig) -> List[Node]:
    '''
    Gets pick and place orchestrator

    Parameters
    ----------
    workflow_config : IsaacSimWorkflowConfig
        Config that dictates isaac sim workflows

    Returns
    -------
    Node
        ROS node for Pick and place
    '''
    if workflow_config.use_ground_truth_pose_in_sim:
        object_frame_name = workflow_config.sim_gt_asset_frame_id
    else:
        object_frame_name = 'detected_object1'

    # We only support 2F-140 in Isaac Sim for this release
    gripper_collision_links = get_gripper_collision_links(GripperType('robotiq_2f_140'))
    mesh_uri = f'package://isaac_manipulator_pick_and_place/'\
               f'meshes/{workflow_config.gripper_type}.obj'

    pick_and_place_orchestrator_node = Node(
        package='isaac_manipulator_pick_and_place',
        executable='pick_and_place_orchestrator.py',
        name='pick_and_place_orchestrator',
        parameters=[{
            'attach_object_fallback_radius': 0.055,
            'grasp_file_path': workflow_config.grasps_file_path,
            'isaac_sim': workflow_config.use_sim_time,
            'use_sim_time': workflow_config.use_sim_time,
            'time_dilation_factor': 0.8,
            'retract_offset_distance': [0.0, 0.0, 0.30],
            'object_frame_name': object_frame_name,
            'use_ground_truth_pose_from_sim': workflow_config.use_ground_truth_pose_in_sim,
            'sleep_time_before_planner_tries_sec': workflow_config.pick_and_place_retry_wait_time,
            'num_planner_tries': workflow_config.pick_and_place_planner_retries,
            'publish_grasp_frame': True,
            'gripper_collision_links': gripper_collision_links,
            'attach_object_shape': str(workflow_config.object_attachment_type.value),
            'attach_object_scale': workflow_config.object_attachment_scale,
            'attach_object_mesh_file_path': workflow_config.attach_object_mesh_file_path,
            'grasp_approach_in_world_frame': False,
            'retract_in_world_frame': True,
            'use_pose_from_rviz': workflow_config.use_pose_from_rviz,
            'end_effector_mesh_resource_uri': mesh_uri,
            'joint_states_topic': '/isaac_joint_states'
        }],
        output='screen',
    )
    return [pick_and_place_orchestrator_node]


def launch_setup(context, *args, **kwargs):
    workflow_config = get_isaac_sim_workflow_config(context)

    manipulator_init_nodes = []

    manipulator_init_nodes.append(get_manipulation_container(workflow_config))

    manipulator_init_nodes.append(get_robot_state_publisher(workflow_config))

    manipulator_init_nodes.append(get_joint_state_publisher(workflow_config))

    manipulator_init_nodes.append(get_cumotion_node(
        camera_type=workflow_config.camera_type, asset_name='ur10e_robotiq_2f_140',
        enable_object_attachment=workflow_config.enable_object_attachment,
        workflow_config=workflow_config))

    manipulator_init_nodes.append(get_nvblox_node(camera_type=workflow_config.camera_type))

    manipulator_init_nodes.append(get_moveit_group_node(workflow_config=workflow_config))
    manipulator_init_nodes.append(get_visualization_node(
        rviz_config_file=workflow_config.rviz_config_file,
        use_sim_time=workflow_config.use_sim_time))

    if workflow_config.use_sim_time:
        manipulator_init_nodes.append(
            get_isaac_sim_joint_parser_node(workflow_config.use_sim_time))

    if workflow_config.use_sim_time:
        ros2_control_nodes = get_ros2_control_nodes(
            use_sim_time=workflow_config.use_sim_time,
            remapped_joint_states=workflow_config.remapped_joint_states
        )
    else:
        raise NotImplementedError('This launch file cannot currently be run on real robot')

    pose_estimation_nodes = []
    if workflow_config.enable_pose_estimation and \
            workflow_config.pose_estimator_type == PoseEstimationType.foundationpose:
        pose_estimation_nodes = get_foundation_pose_nodes(workflow_config)
    elif workflow_config.enable_pose_estimation and \
            workflow_config.pose_estimator_type == PoseEstimationType.dope:
        pose_estimation_nodes = get_dope_nodes(workflow_config)

    workflow_nodes = []
    if workflow_config.workflow_type == IsaacSimWorkflowType.POSE_TO_POSE:
        workflow_nodes = get_pose_to_pose()
    elif workflow_config.workflow_type == IsaacSimWorkflowType.PICK_AND_PLACE:
        workflow_nodes = get_pick_and_place_orchestrator(workflow_config=workflow_config)
        if not workflow_config.use_ground_truth_pose_in_sim:
            workflow_nodes += get_object_detection_servers(workflow_config=workflow_config)
            workflow_nodes += get_foundation_pose_nodes(workflow_config)
    elif workflow_config.workflow_type == IsaacSimWorkflowType.OBJECT_FOLLOWING:
        workflow_nodes = get_object_following()
    use_sim_time_param = [SetParameter(name='use_sim_time', value=True)]
    sub_qos = [SetParameter(name='sub_qos', value='DEFAULT')]
    pub_qos = [SetParameter(name='pub_qos', value='DEFAULT')]
    return (
        use_sim_time_param + sub_qos + pub_qos +
        manipulator_init_nodes
        + ros2_control_nodes
        + workflow_nodes
        + pose_estimation_nodes
    )


def generate_launch_description():
    sim_grasps_path = os.path.join(
        get_package_share_directory('isaac_manipulator_pick_and_place'), 'config',
        'ur_robotiq_grasps_sim.yaml'
    )
    launch_args = [
        DeclareLaunchArgument(
            'workflow_type',
            default_value='pose_to_pose',
            choices=['pose_to_pose', 'pick_and_place', 'object_following'],
            description='Type of workflow to run the sim based manipulator pipeline on',
        ),
        DeclareLaunchArgument(
            'pose_estimation_type',
            default_value='foundationpose',
            choices=['dope', 'foundationpose'],
            description='Type of pose estimation to use for Object Following',
        ),
        DeclareLaunchArgument(
            'grasps_file_path',
            default_value=f'{str(sim_grasps_path)}',
            description='File name in the config folder for grasps file path',
        ),
        DeclareLaunchArgument(
            'foundation_pose_mesh_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/isaac_ros_foundationpose'
                          '/soup_can/soup_can.obj',
            description='Mesh file path in foundation pose object',
        ),
        DeclareLaunchArgument(
            'foundation_pose_refine_engine_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/models/'
                          'foundationpose/refine_trt_engine.plan',
            description='Texture path for foundation pose object',
        ),
        DeclareLaunchArgument(
            'foundation_pose_texture_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/'
                          'isaac_ros_foundationpose/soup_can/baked_mesh_tex0.png',
            description='Texture path for foundation pose object',
        ),
        DeclareLaunchArgument(
            'foundation_pose_score_engine_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/models/'
                          'foundationpose/score_trt_engine.plan',
            description='Texture path for foundation pose object',
        ),
        DeclareLaunchArgument(
            'rtdetr_engine_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan',
            description='Texture path for foundation pose object',
        ),
        DeclareLaunchArgument(
            'rtdetr_object_class_id',
            default_value='3',
            description='Class ID of the object to be detected. The default corresponds to the '
                        'soup can if the SyntheticaDETR v1.0.0 model file is used. Refer to the '
                        'SyntheticaDETR model documentation for additional supported objects and '
                        'their class IDs.',
        ),
        DeclareLaunchArgument(
            'rt_detr_confidence_threshold',
            default_value='0.5',
            description='Confidence threshold for RT-DETR',
        ),
        DeclareLaunchArgument(
            'dope_model_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/models/dope/soup_can.onnx',
            description='DOPE model file path',
        ),
        DeclareLaunchArgument(
            'dope_engine_file_path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/models/dope/soup_can.engine',
            description='DOPE engine file path',
        ),
        DeclareLaunchArgument(
            'use_ground_truth_pose_in_sim',
            default_value='false',
            description='Whether to use ground truth pose in sim (true/false)',
        ),
        DeclareLaunchArgument(
            'pick_and_place_planner_retries',
            default_value='5',
            description='Num retries for num planner retries',
        ),
        DeclareLaunchArgument(
            'pick_and_place_retry_wait_time',
            default_value='5.0',
            description='Number of seconds to wait before retrying planner',
        ),
        DeclareLaunchArgument(
            'sim_gt_asset_frame_id',
            default_value='soup_can',
            description='The TF exposed from Isaac sim for object asset for running with ground'
                        'truth pose',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='error',
            description='Log level of the container.',
            choices=['debug', 'info', 'warn', 'error']
        ),
        DeclareLaunchArgument(
            'object_attachment_type',
            description='Object attachment type',
            choices=[object_attachment.value for object_attachment in ObjectAttachmentShape],
            default_value=ObjectAttachmentShape.SPHERE.value,
        ),
        DeclareLaunchArgument(
            'object_attachment_scale',
            description='Object attachment scale of object / cube',
            default_value='[0.05, 0.05, 0.1]',
        ),
        DeclareLaunchArgument(
            'attach_object_mesh_file_path',
            description='Object attachment mesh file path',
            default_value=f'{BASE_PATH}/isaac_ros_assets/isaac_ros_foundationpose'
                          '/soup_can/soup_can.obj',
        ),
        DeclareLaunchArgument(
            'filter_depth_buffer_time',
            description='Filter Depth buffers for object attachment. This informs how many seconds'
                        ' in the past object attachment looks at to get depth image input for '
                        'object detection',
            default_value='0.05'
        ),
        DeclareLaunchArgument(
            'time_sync_slop',
            description='The time in seconds nodes keep as sync threshold to sync images and '
                        ' joint states. If one has a slower machine, tweaking this variable is'
                        ' useful to get syncs but at the cost of accuracy. If the slop parameter'
                        'is too high, the robot will sync with older images or joint states '
                        'leading to incorrect depth segmentation and object attachment.',
            default_value='0.5'
        ),
        DeclareLaunchArgument(
            'use_pose_from_rviz',
            description='When enabled, the end effector interactive marker is used to set the '
                        'place pose through RViz',
            default_value='False'
        )
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
