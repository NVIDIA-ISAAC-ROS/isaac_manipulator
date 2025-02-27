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

from ament_index_python.packages import get_package_share_directory

import os
import yaml

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

import isaac_ros_launch_utils as lu

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils import MoveItConfigsBuilder
from isaac_manipulator_ros_python_utils.launch_utils import (
    get_gripper_collision_links
)
from isaac_manipulator_ros_python_utils.types import (
        CameraType, DepthType, TrackingType, GripperType, ObjectAttachmentShape
)
import isaac_manipulator_ros_python_utils.constants as constants


def launch_setup(context, *args, **kwargs):
    # Initialize Arguments
    ur_type = LaunchConfiguration('ur_type')
    gripper_type = LaunchConfiguration('gripper_type')
    object_attachment_type = LaunchConfiguration('object_attachment_type')
    attach_object_mesh_file_path = LaunchConfiguration('attach_object_mesh_file_path')
    attach_object_scale = LaunchConfiguration('object_attachment_scale')
    robot_ip = LaunchConfiguration('robot_ip')
    voxel_size = LaunchConfiguration('voxel_size')
    runtime_config_package = LaunchConfiguration('runtime_config_package')
    controller_spawner_timeout = LaunchConfiguration('controller_spawner_timeout')
    initial_joint_controller = LaunchConfiguration('initial_joint_controller')
    camera_type = str(context.perform_substitution(LaunchConfiguration('camera_type')))
    num_cameras = LaunchConfiguration('num_cameras')
    hawk_depth_mode = str(context.perform_substitution(LaunchConfiguration('hawk_depth_mode')))
    time_sync_slop = str(context.perform_substitution(LaunchConfiguration('time_sync_slop')))
    use_pose_from_rviz = LaunchConfiguration('use_pose_from_rviz')
    rtdetr_object_class_id = str(context.perform_substitution(
        LaunchConfiguration('rtdetr_object_class_id')))
    filter_depth_buffer_time = str(context.perform_substitution(
        LaunchConfiguration('filter_depth_buffer_time')))
    script_filename = PathJoinSubstitution(
        [FindPackageShare('ur_client_library'), 'resources', 'external_control.urscript']
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare('ur_robot_driver'), 'resources', 'rtde_input_recipe.txt']
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare('ur_robot_driver'), 'resources', 'rtde_output_recipe.txt']
    )
    grasp_parent_frame = ''
    if gripper_type.perform(context) == 'robotiq_2f_140':
        grasp_parent_frame = 'robotiq_base_link'
    elif gripper_type.perform(context) == 'robotiq_2f_85':
        grasp_parent_frame = 'robotiq_85_base_link'
    else:
        raise NotImplementedError('Gripper type is not supported')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution([FindPackageShare('isaac_manipulator_pick_and_place'),
                                 'urdf', 'ur_robotiq_gripper.urdf.xacro']),
            ' ',
            'robot_ip:=',
            robot_ip,
            ' ',
            'name:=',
            ur_type,
            ' ',
            'script_filename:=',
            script_filename,
            ' ',
            'input_recipe_filename:=',
            input_recipe_filename,
            ' ',
            'output_recipe_filename:=',
            output_recipe_filename,
            ' ',
            'ur_type:=',
            ur_type,
            ' ',
            'gripper_type:=',
            gripper_type,
            ' ',
            'grasp_parent_frame:=',
            grasp_parent_frame,
            ' '
        ]
    )

    setup = LaunchConfiguration('setup')
    robot_description = {'robot_description': robot_description_content}

    initial_joint_controllers = PathJoinSubstitution(
        [FindPackageShare('isaac_manipulator_pick_and_place'), 'config', 'controllers.yaml']
    )

    robotiq_gripper_controllers = PathJoinSubstitution(
        [FindPackageShare('isaac_manipulator_pick_and_place'), 'config',
         f'{gripper_type.perform(context)}_controllers.yaml']
    )

    update_rate_config_file = PathJoinSubstitution(
        [
            FindPackageShare(runtime_config_package),
            'config', ur_type.perform(context) + '_update_rate.yaml',
        ]
    )

    ur_control_node = Node(
        package='ur_robot_driver',
        executable='ur_ros2_control_node',
        parameters=[
            robot_description,
            update_rate_config_file,
            ParameterFile(initial_joint_controllers, allow_substs=True),
            ParameterFile(robotiq_gripper_controllers, allow_substs=True),
        ],
        output='screen'
    )

    urscript_interface = Node(
        package='ur_robot_driver',
        executable='urscript_interface',
        parameters=[{'robot_ip': robot_ip}],
        output='screen',
    )

    controller_stopper_node = Node(
        package='ur_robot_driver',
        executable='controller_stopper_node',
        name='controller_stopper',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'consistent_controllers': [
                'io_and_status_controller',
                'force_torque_sensor_broadcaster',
                'joint_state_broadcaster',
                'speed_scaling_state_broadcaster',]
            }],
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    # Spawn controllers
    def controller_spawner(controllers, active=True):
        inactive_flags = ['--inactive'] if not active else []
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                '--controller-manager',
                '/controller_manager',
                '--controller-manager-timeout',
                controller_spawner_timeout,
            ]
            + inactive_flags
            + controllers,
        )

    controllers_active = [
        'joint_state_broadcaster',
        'io_and_status_controller',
        'speed_scaling_state_broadcaster',
        'force_torque_sensor_broadcaster',
        'robotiq_gripper_controller',
        'robotiq_activation_controller'
    ]

    controller_spawners = [controller_spawner(controllers_active)]

    initial_joint_controller_spawner_started = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            initial_joint_controller,
            '-c',
            '/controller_manager',
            '--controller-manager-timeout',
            controller_spawner_timeout,
        ],
    )

    moveit_config = (
        MoveItConfigsBuilder('ur_with_gripper', package_name='isaac_manipulator_pick_and_place')
        .robot_description_semantic(file_path='srdf/ur_' + gripper_type.perform(context) +
                                    '_gripper.srdf.xacro')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    # The mapping features of MoveItConfigsBuilder to pass the xacro parameters
    # did not work, hence we overide the robot description seperately
    moveit_config.robot_description = robot_description

    # Add cuMotion to list of planning pipelines.
    cumotion_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml'
    )
    with open(cumotion_config_file_path) as cumotion_config_file:
        cumotion_config = yaml.safe_load(cumotion_config_file)
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'
    # Start the actual move_group node/action server
    move_it_dict = moveit_config.to_dict()

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[move_it_dict],
    )

    if camera_type == str(CameraType.hawk):
        # Image Resolution
        if hawk_depth_mode == str(DepthType.ess_light):
            depth_image_width = 480
            depth_image_height = 288
        else:
            depth_image_width = 960
            depth_image_height = 576
        rgb_image_width = 1920
        rgb_image_height = 1200

        # CuMotion topics
        cumotion_depth_image_topics = '["/depth_image"]'
        cumotion_depth_camera_infos = '["/rgb/camera_info"]'

        # Nvblox topics
        nvblox_rgb_image_topic = '/rgb/image_rect_color'
        nvblox_rgb_camera_info = '/rgb/camera_info'
        nvblox_depth_camera_info = '/rgb/camera_info'

        # object detection server and RT-DETR topics
        obj_input_img_topic_name = '/rgb/image_rect_color'
        rtdetr_rgb_image_topic = '/object_detection_server/image_rect'
        rtdetr_rgb_camera_info = '/rgb/camera_info'
        rtdetr_detections_topic = '/detections'

        # foundation pose server and foundationpose topics
        fp_in_img_topic_name = '/rgb/image_rect_color'
        fp_in_camera_info_topic_name = '/resize/camera_info'
        fp_in_depth_topic_name = '/depth_image'

        foundation_pose_rgb_image_topic = 'foundation_pose_server/rgb/image_rect_color'
        foundation_pose_rgb_camera_info = 'foundation_pose_server/resize/camera_info'
        foundation_pose_depth_image_topic = 'foundation_pose_server/depth_image'
        foundation_pose_detections_topic = 'foundation_pose_server/bbox'

    elif camera_type == str(CameraType.realsense):
        # Image Resolution
        depth_image_width = 1280
        depth_image_height = 720
        rgb_image_width = 1280
        rgb_image_height = 720

        # CuMotion topics
        cumotion_depth_image_topics = '[\'/camera_1/aligned_depth_to_color/image_raw\']'
        cumotion_depth_camera_infos = '[\'/camera_1/aligned_depth_to_color/camera_info\']'

        # Nvblox topics
        nvblox_rgb_image_topic = '/camera_1/color/image_raw'
        nvblox_rgb_camera_info = '/camera_1/color/camera_info'
        nvblox_depth_camera_info = '/camera_1/aligned_depth_to_color/camera_info'

        # object detection server and RT-DETR topics
        obj_input_img_topic_name = '/camera_1/color/image_raw'
        rtdetr_rgb_image_topic = '/object_detection_server/image_rect'
        rtdetr_rgb_camera_info = '/camera_1/color/camera_info'
        rtdetr_detections_topic = '/detections'

        # foundation pose server and foundationpose topics
        fp_in_img_topic_name = '/camera_1/color/image_raw'
        fp_in_camera_info_topic_name = '/resize/camera_info'
        fp_in_depth_topic_name = '/camera_1/aligned_depth_to_color/image_raw'

        foundation_pose_rgb_image_topic = '/foundation_pose_server/camera_1/color/image_raw'
        foundation_pose_rgb_camera_info = '/foundation_pose_server/resize/camera_info'
        foundation_pose_depth_image_topic = \
            '/foundation_pose_server/camera_1/aligned_depth_to_color/image_raw'
        foundation_pose_detections_topic = '/foundation_pose_server/bbox'

    else:
        print('Error received unexpected camera type!')
        exit(-1)

    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include')

    static_transform_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/static_transforms.launch.py']
        ),
        launch_arguments={
            'broadcast_world_base_link': 'False',
            'camera_type': camera_type,
            'tracking_type': str(TrackingType.none),
            'calibration_name': setup,
        }.items(),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_files_include_dir, '/realsense.launch.py']),
        launch_arguments={
            'num_cameras': num_cameras,
            'camera_ids_config_name': setup,
        }.items(),
        condition=IfCondition(PythonExpression(
            [f'"{camera_type}"', ' == ', f'"{str(CameraType.realsense)}"'])),
    )

    hawk_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_files_include_dir, '/hawk.launch.py']),
        condition=IfCondition(PythonExpression(
            [f'"{camera_type}"', ' == ', f'"{str(CameraType.hawk)}"'])),
    )

    ess_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_files_include_dir, '/ess.launch.py']),
        launch_arguments={
            'ess_mode': hawk_depth_mode,
        }.items(),
        condition=IfCondition(PythonExpression(
            [f'"{camera_type}"', ' == ', f'"{str(CameraType.hawk)}"'])),
    )

    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_bringup'), 'launch', 'include')
    # We want nvblox to consume the depth map that segments out the robot
    enable_nvblox = context.perform_substitution(LaunchConfiguration('enable_nvblox'))
    nvblox_depth_image_topic = '/cumotion/camera_1/world_depth'
    nvblox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/nvblox.launch.py']
        ),
        launch_arguments={
            'rgb_image_topic': nvblox_rgb_image_topic,
            'rgb_camera_info': nvblox_rgb_camera_info,
            'depth_image_topic': nvblox_depth_image_topic,
            'depth_camera_info': nvblox_depth_camera_info,
            'voxel_size': voxel_size,
            'camera_type': camera_type,
            'num_cameras': num_cameras,
            'no_robot_mode': 'False',
            'workspace_bounds_name': setup,
        }.items(),
        condition=IfCondition(PythonExpression([f'\'{enable_nvblox}\'', ' == ', '\'True\'']))
    )

    asset_name = ur_type.perform(context) + '_' + gripper_type.perform(context)
    xrdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'), 'xrdf',
        f'{asset_name}.xrdf'
    )
    urdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'),
        'urdf',
        f'{asset_name}.urdf',
    )

    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/cumotion.launch.py']
        ),
        launch_arguments={
            'depth_image_topics': cumotion_depth_image_topics,
            'depth_camera_infos': cumotion_depth_camera_infos,
            'robot_file_name': xrdf_file_path,
            'voxel_size': voxel_size,
            'read_esdf_world': str(enable_nvblox),
            'publish_curobo_world_as_voxels': str(enable_nvblox),
            'distance_threshold': '0.05',
            'joint_states_topic': '/joint_states',
            'time_sync_slop': time_sync_slop,
            'filter_depth_buffer_time': filter_depth_buffer_time,
            'camera_type': camera_type,
            'num_cameras': num_cameras,
            'no_robot_mode': 'False',
            'from_bag': 'False',
            'workspace_bounds_name': setup,
            'tool_frame': 'gripper_frame',
            'urdf_file_path': urdf_file_path,
            'enable_object_attachment': 'True',
            'trigger_aabb_object_clearing': 'True'
        }.items(),
    )

    isaac_ros_ws_path = lu.get_isaac_ros_ws_path()

    rtdetr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/rtdetr.launch.py']
        ),
        launch_arguments={
            'camera_type': '',
            'image_width': str(rgb_image_width),
            'image_height': str(rgb_image_height),
            'image_input_topic': rtdetr_rgb_image_topic,
            'camera_info_input_topic': rtdetr_rgb_camera_info,
            'detections_2d_array_output_topic': rtdetr_detections_topic,
            'rtdetr_is_object_following': 'False',
            'rtdetr_engine_file_path':
                isaac_ros_ws_path + '/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan'
        }.items()
    )

    foundationpose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/foundationpose.launch.py']
        ),
        launch_arguments={
            'camera_type': camera_type,
            'rgb_image_width': str(rgb_image_width),
            'rgb_image_height': str(rgb_image_height),
            'depth_image_width': str(depth_image_width),
            'depth_image_height': str(depth_image_height),
            'rgb_image_topic': foundation_pose_rgb_image_topic,
            'rgb_camera_info_topic': foundation_pose_rgb_camera_info,
            'foundation_pose_server_depth_topic_name': foundation_pose_depth_image_topic,
            'realsense_depth_image_topic': foundation_pose_depth_image_topic,
            'detection2_d_array_topic': foundation_pose_detections_topic,
            'mesh_file_path': isaac_ros_ws_path + '/isaac_ros_assets/isaac_ros_foundationpose'
                                                  '/Mac_and_cheese_0_1/Mac_and_cheese_0_1.obj',
            'texture_path': isaac_ros_ws_path + '/isaac_ros_assets/isaac_ros_foundationpose/'
                                                'Mac_and_cheese_0_1/materials/textures/'
                                                'baked_mesh_tex0.png',
            'refine_model_file_path': isaac_ros_ws_path + '/isaac_ros_assets/models'
                                                          '/foundationpose/refine_model.onnx',
            'refine_engine_file_path': isaac_ros_ws_path + '/isaac_ros_assets/models'
                                                           '/foundationpose/'
                                                           'refine_trt_engine.plan',
            'score_model_file_path': isaac_ros_ws_path + '/isaac_ros_assets'
                                                         '/models/foundationpose/score_model.onnx',
            'score_engine_file_path': isaac_ros_ws_path + '/isaac_ros_assets/models/'
                                                          'foundationpose/score_trt_engine.plan',
            'object_class_id': rtdetr_object_class_id
        }.items()
    )

    # Add objectinfo servers
    isaac_manipulator_servers_include_dir = os.path.join(
        get_package_share_directory('isaac_manipulator_servers'), 'launch')

    object_detection_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/object_detection_server.launch.py']),
        launch_arguments={
            'obj_input_img_topic_name': obj_input_img_topic_name,
            'obj_output_img_topic_name': rtdetr_rgb_image_topic,
        }.items(),
    )

    foundation_pose_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/foundation_pose_server.launch.py']),
        launch_arguments={
            'fp_in_img_topic_name': fp_in_img_topic_name,
            'fp_out_img_topic_name': foundation_pose_rgb_image_topic,
            'fp_in_camera_info_topic_name': fp_in_camera_info_topic_name,
            'fp_out_camera_info_topic_name': foundation_pose_rgb_camera_info,
            'fp_in_depth_topic_name': fp_in_depth_topic_name,
            'fp_out_depth_topic_name': foundation_pose_depth_image_topic,
            'fp_out_bbox_topic_name': foundation_pose_detections_topic,
        }.items(),
    )

    objectinfo_server_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [isaac_manipulator_servers_include_dir, '/object_info_server.launch.py']),)

    grasp_config_file = (
        get_package_share_directory('isaac_manipulator_pick_and_place') +
        '/config/' + gripper_type.perform(context) + '_grasps_mac_and_cheese.yaml'
    )
    gripper_collision_links = get_gripper_collision_links(GripperType(
        gripper_type.perform(context)))

    mesh_uri = f'package://isaac_manipulator_pick_and_place/'\
               f'meshes/{gripper_type.perform(context)}.obj'
    pick_and_place_orchestrator_node = Node(
        package='isaac_manipulator_pick_and_place',
        executable='pick_and_place_orchestrator.py',
        name='pick_and_place_orchestrator',
        parameters=[{'attach_object_fallback_radius': 0.075,
                     'grasp_file_path': grasp_config_file,
                     'publish_grasp_frame': True,
                     'time_dilation_factor': 0.3,
                     'gripper_collision_links': gripper_collision_links,
                     'attach_object_shape': object_attachment_type,
                     'attach_object_scale': attach_object_scale,
                     'attach_object_mesh_file_path': attach_object_mesh_file_path,
                     'grasp_approach_in_world_frame': False,
                     'retract_in_world_frame': True,
                     'use_pose_from_rviz': use_pose_from_rviz,
                     'end_effector_mesh_resource_uri': mesh_uri,
                     'joint_states_topic': '/joint_states'
                     }],
        output='screen',
    )

    manipulation_container = ComposableNodeContainer(
        name=constants.MANIPULATOR_CONTAINER_NAME,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[],
        arguments=['--ros-args', '--log-level', 'nvblox_node:=error'],
        output='screen'
    )

    # RViz
    rviz_config_file = (
        get_package_share_directory('isaac_manipulator_pick_and_place') +
        '/rviz/pick_and_place.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )

    nodes_to_start = [
        manipulation_container,
        nvblox_launch,
        cumotion_launch,
        realsense_launch,
        hawk_launch,
        ess_launch,
        static_transform_launch,
        rtdetr_launch,
        foundationpose_launch,
        rviz_node,
        ur_control_node,
        controller_stopper_node,
        urscript_interface,
        robot_state_publisher_node,
        initial_joint_controller_spawner_started,
        move_group_node,
        object_detection_server_launch,
        foundation_pose_server_launch,
        objectinfo_server_launch,
        pick_and_place_orchestrator_node,
    ] + controller_spawners

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            'ur_type',
            description='Type/series of used UR robot.',
            choices=['ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e', 'ur16e', 'ur20', 'ur30'],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'object_attachment_type',
            description='Object attachment type',
            choices=[object_attachment.value for object_attachment in ObjectAttachmentShape],
            default_value=ObjectAttachmentShape.SPHERE.value,
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'object_attachment_scale',
            description='Object attachment scale of object / cube',
            default_value='[0.09, 0.185, 0.035]',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'attach_object_mesh_file_path',
            description='Object attachment mesh file path',
            default_value='',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'gripper_type',
            description='Type of gripper to use with UR robot',
            choices=['robotiq_2f_85', 'robotiq_2f_140'],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'robot_ip', description='IP address by which the robot can be reached.'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'voxel_size', default_value='0.01',
            description='Resolution of 3D voxels for nvblox and curobo in meters.'
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'enable_nvblox', default_value='True',
            description='Enable Nvblox for cumotion.'
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            'runtime_config_package',
            default_value='ur_robot_driver',
            description='Package with the controller\'s configuration in \'config\' folder. '
            'Usually the argument is not set, it enables use of a custom setup.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'tf_prefix',
            default_value='',
            description='tf_prefix of the joint names, useful for '
            'multi-robot setup. If changed, also joint names in the controllers\' configuration '
            'have to be updated.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'controller_spawner_timeout',
            default_value='10',
            description='Timeout used when spawning controllers.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'initial_joint_controller',
            default_value='scaled_joint_trajectory_controller',
            description='Initially loaded robot controller.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'camera_type',
            default_value=str(CameraType.realsense),
            choices=[str(CameraType.hawk), str(CameraType.realsense)],
            description='Camera sensor to use'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'hawk_depth_mode',
            default_value=str(DepthType.ess_full),
            choices=[str(DepthType.ess_full), str(DepthType.ess_light)],
            description='Depth mode for Hawk camera'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'num_cameras',
            default_value='1',
            choices=['1', '2'],
            description='Number of cameras to run for 3d reconstruction',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'setup',
            default_value='hubble_test_bench',
            description='The name of the setup you are running on '
                        '(specifying calibration, workspace bounds and camera ids)',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'filter_depth_buffer_time',
            description='Filter Depth buffers for object attachment. This informs how many seconds'
                        ' in the past object attachment looks at to get depth image input for '
                        'object detection',
            default_value='0.7'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'time_sync_slop',
            description='The time in seconds nodes keep as sync threshold to sync images and '
                        ' joint states. If one has a slower machine, tweaking this variable is'
                        ' useful to get syncs but at the cost of accuracy. If the slop parameter'
                        'is too high, the robot will sync with older images or joint states '
                        'leading to incorrect depth segmentation and object attachment.',
            default_value='0.5'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_pose_from_rviz',
            description='When enabled, the end effector interactive marker is used to set the '
                        'place pose through RViz',
            default_value='False'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'rtdetr_object_class_id',
            default_value='22',
            description='Class ID of the object to be detected. The default corresponds to the '
                        'Mac and Cheese box if the SyntheticaDETR v1.0.0 model file is used. '
                        'Refer to the SyntheticaDETR model documentation for additional supported '
                        'objects and their class IDs.',
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
