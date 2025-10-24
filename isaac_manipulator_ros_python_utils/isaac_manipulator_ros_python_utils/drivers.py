# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
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
from typing import Dict, List, Tuple

from ament_index_python.packages import get_package_share_directory

# flake8: noqa: F403,F405
from isaac_manipulator_ros_python_utils.launch_utils import (
    get_bool_variable, get_str_variable, get_workflow_type
)
from isaac_manipulator_ros_python_utils.config import CoreConfig, UrRobotiqDriverConfig
from isaac_manipulator_ros_python_utils.robot_description_utils import (
    get_robot_description_contents_for_real_robot, get_robot_description_contents_for_sim
)
from isaac_manipulator_ros_python_utils.manipulator_types import WorkflowType

from isaac_ros_launch_utils.all_types import (
    GroupAction, TimerAction
)

from launch.actions import (
    IncludeLaunchDescription, Shutdown
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

import yaml


def get_visualization_node(core_config: CoreConfig, moveit_config: MoveItConfigsBuilder) -> Node:
    """
    Return the RViz visualization node.

    Args
    ----
        rviz_config_file (str): Viewport file path

    Returns
    -------
        Node: ROS node for RViz2

    """
    if not core_config.use_sim_time:
        if core_config.workflow_config.workflow_type in (
            WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY
        ):
            return Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='log',
                arguments=['-d', core_config.rviz_config_file],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                ],
            )
        else:
            launch_files_include_dir = os.path.join(
                get_package_share_directory('isaac_manipulator_bringup'),
                'launch', 'visualization'
            )
            return IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [launch_files_include_dir, '/visualization.launch.py']
                ),
                launch_arguments={
                    'camera_type': str(core_config.camera_config.camera_type),
                    'run_rviz': 'True' if core_config.enable_rviz_visualization else 'False',
                    'run_foxglove': 'True' if core_config.enable_foxglove_visualization else 'False'
                }.items(),
            )

    return Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', core_config.rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': core_config.use_sim_time}
        ],
    )


def start_tool_communication(robot_ip: str):
    """
    Create and return a ROS 2 Node that runs the tool communication node from the UR driver.

    This node is equivalent to running the following command:
      ros2 run ur_robot_driver tool_communication.py --ros-args -p robot_ip:=<robot_ip>

    Args
    ----
        robot_ip (str): The IP address of the UR robot to communicate with.

    Returns
    -------
        Node: A ROS 2 Node configured to execute tool communication.

    """
    return Node(
        package='ur_robot_driver',
        executable='tool_communication.py',
        name='tool_communication',
        output='screen',
        parameters=[{'robot_ip': robot_ip}],
        on_exit=Shutdown(),
    )


def get_ur_drivers(driver_config: UrRobotiqDriverConfig, core_config: CoreConfig) -> List[Node]:
    """
    Get only UR-specific drivers, not assuming that user has hooked up a gripper to the UR.

    Args
    ----
        driver_config (UrRobotiqDriverConfig): Driver config.
        core_config (CoreConfig): Core config.

    Returns
    -------
        List[Node]: List of nodes

    """
    ur_workflow_bringup = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'), 'launch')

    ur_robot_driver_bringup = os.path.join(
        get_package_share_directory('ur_robot_driver'), 'launch')

    kinematics_params_file = driver_config.ur_calibration_file_path

    driver_parameters = {
        'ur_type': driver_config.ur_type,
        'robot_ip': driver_config.robot_ip,
        'launch_rviz': 'false',
        'controllers_file': driver_config.ros2_controllers_file_path,
        'activate_joint_controller': 'true',
        'initial_joint_controller': 'scaled_joint_trajectory_controller'
    }

    if kinematics_params_file is not None and kinematics_params_file != '':
        driver_parameters['kinematics_params_file'] = kinematics_params_file

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [ur_robot_driver_bringup,
                    '/ur_control.launch.py']
            ),
            launch_arguments=driver_parameters.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [ur_workflow_bringup,
                    '/ur.launch.py']
            ),
            launch_arguments={
                'ur_type': driver_config.ur_type,
                'robot_ip': driver_config.robot_ip,
                'launch_rviz': 'true' if core_config.enable_rviz_visualization else 'false',
                'warehouse_sqlite_path': '/tmp/warehouse.sqlite',
                'publish_robot_description_semantic': 'false'
            }.items())
    ]


def get_moveit_group_node(driver_config: UrRobotiqDriverConfig) -> Tuple[Node, Dict]:
    """
    Return the MoveIt group node for a particular configuration.

    Args
    ----
        use_sim_time (bool): Use sim time
        asset_name (str): Asset name (UR10e_robotiq, UR5e_robotiq etc)

    Returns
    -------
        Node: The move group ROS node

    """
    if driver_config.use_sim_time:
        robot_description_content = get_robot_description_contents_for_sim(
            urdf_xacro_file=driver_config.urdf_path,
            ur_type=driver_config.ur_type,
            use_sim_time=driver_config.use_sim_time,
            gripper_type=driver_config.gripper_type,
            grasp_parent_frame=driver_config.grasp_parent_frame,
            robot_ip=driver_config.robot_ip,
            dump_to_file=False,
            output_file=None,
        )
    else:
        robot_description_content = get_robot_description_contents_for_real_robot(driver_config)

    moveit_config = (
        MoveItConfigsBuilder('ur_with_gripper', package_name='isaac_manipulator_robot_description')
        .robot_description_semantic(file_path=driver_config.srdf_path)
        .robot_description_kinematics(file_path=driver_config.kinematics_file_path)
        .joint_limits(file_path=driver_config.joint_limits_file_path)
        .trajectory_execution(file_path=driver_config.moveit_controllers_file_path)
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    def cumotion_params():
        """Load cuMotion planning parameters."""
        config_file_path = os.path.join(
            get_package_share_directory('isaac_ros_cumotion_moveit'),
            'config',
            'isaac_ros_cumotion_planning.yaml'
        )
        with open(config_file_path) as config_file:
            config = yaml.safe_load(config_file)

        return config

    # Add cuMotion to planning pipelines
    cumotion_config = cumotion_params()
    moveit_config.planning_pipelines['planning_pipelines'].insert(0, 'isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'

    robot_description = {'robot_description': robot_description_content}

    # The mapping features of MoveItConfigsBuilder to pass the xacro parameters
    # did not work, hence we overide the robot description seperately
    moveit_config.robot_description = robot_description

    # Start the actual move_group node/action server
    move_it_dict = moveit_config.to_dict()

    # The structure of MoveIt dict looks like this:
    """
    {
        isaac_ros_cumotion:
            num_steps: 32
            planning_plugins:
            - isaac_ros_cumotion_moveit/CumotionPlanner
            ...
        planning_pipelines:
            pipeline_names:
            - isaac_ros_cumotion
            ...
    }
    """

    move_it_dict['planning_pipelines'] = {}
    move_it_dict['planning_pipelines']['pipeline_names'] = ['isaac_ros_cumotion']

    move_it_dict['capabilities'] = 'move_group/ExecuteTaskSolutionCapability'
    if driver_config.use_sim_time:
        joint_state_topic = '/isaac_parsed_joint_states'
    else:
        joint_state_topic = '/joint_states'

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[move_it_dict, {'use_sim_time': driver_config.use_sim_time}],
        arguments=['--ros-args', '--log-level', 'info'],
        remappings=[
            ('joint_states', joint_state_topic)
        ],
        on_exit=Shutdown(),
    )
    return move_group_node, moveit_config


def get_robot_state_publisher(workflow_config: UrRobotiqDriverConfig) -> Node:
    """
    Return the robot state publisher that publishes TF and robot model for the scene.

    Args
    ----
        workflow_config (WorkflowConfig): Config for workflow

    Returns
    -------
        Node: Robot state publisher node

    """
    if workflow_config.use_sim_time:
        robot_description_contents = get_robot_description_contents_for_sim(
            urdf_xacro_file=workflow_config.urdf_path,
            ur_type=workflow_config.ur_type,
            use_sim_time=workflow_config.use_sim_time,
            gripper_type=workflow_config.gripper_type,
            grasp_parent_frame=workflow_config.grasp_parent_frame,
            robot_ip=workflow_config.robot_ip,
            dump_to_file=False,
            output_file=None,
        )
    else:
        robot_description_contents = get_robot_description_contents_for_real_robot(workflow_config)

    remappings = []
    if workflow_config.use_sim_time:
        remappings = [('/joint_states', workflow_config.remapped_joint_states['/joint_states'])]

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description_contents,
             'use_sim_time': workflow_config.use_sim_time}
        ],
        remappings=remappings,
        on_exit=Shutdown(),
    )

    return robot_state_publisher


def get_ros2_control_nodes_for_sim(driver_config: UrRobotiqDriverConfig) -> List[Node]:
    """
    Return the controller nodes that need to be run to enable robotic control in Isaac sim/real.

    Real robot not yet supported

    Args
    ----
        use_sim_time (bool): Use sim time

    Returns
    -------
        List[Node]: List of ros2 control nodes

    """
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            ParameterFile(driver_config.ros2_controllers_file_path, allow_substs=True),
            {'use_sim_time': driver_config.use_sim_time}
        ],
        remappings=[
            (
                '/controller_manager/robot_description',
                driver_config.remapped_joint_states['/controller_manager/robot_description'],
            )
        ],
        arguments=['--ros-args', '--log-level', 'error'],
        output='screen',
        on_exit=Shutdown(),
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
        package='isaac_manipulator_isaac_sim_utils',
        executable='isaac_sim_gripper_driver.py',
        parameters=[{
            'use_sim_time': driver_config.use_sim_time
        }],
        on_exit=Shutdown(),
    )

    return [
        ros2_control_node,
        scaled_joint_trajectory_controller_spawner,
        gripper_controller,
        joint_state_broadcaster_spawner,
    ]


def get_ros2_control_nodes_for_real(driver_config: UrRobotiqDriverConfig) -> List[Node]:
    """
    Return the controller nodes for real robot.

    Args
    ----
        use_sim_time (bool): Use sim time

    Returns
    -------
        List[Node]: List of ros2 control nodes

    """
    robot_description_contents = get_robot_description_contents_for_real_robot(driver_config)
    robot_description = {'robot_description': robot_description_contents}

    runtime_config_package = LaunchConfiguration('runtime_config_package')
    update_rate_config_file = PathJoinSubstitution(
        [
            FindPackageShare(runtime_config_package),
            'config', str(driver_config.ur_type) + '_update_rate.yaml',
        ]
    )

    initial_joint_controllers = PathJoinSubstitution(
        [FindPackageShare('isaac_manipulator_robot_description'),
         'config', 'ros2_control_controllers.yaml']
    )

    robotiq_gripper_controllers = PathJoinSubstitution(
        [FindPackageShare('isaac_manipulator_robot_description'), 'config',
         f'{driver_config.gripper_type}_controllers.yaml']
    )

    ur_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            update_rate_config_file,
            ParameterFile(initial_joint_controllers, allow_substs=True),
            ParameterFile(robotiq_gripper_controllers, allow_substs=True),
        ],
        arguments=['--ros-args', '--log-level', 'error'],
        output='screen',
        on_exit=Shutdown(),
    )

    urscript_interface = Node(
        package='ur_robot_driver',
        executable='urscript_interface',
        parameters=[{'robot_ip': driver_config.robot_ip}],
        output='screen',
        on_exit=Shutdown(),
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
                'speed_scaling_state_broadcaster',
                'tcp_pose_broadcaster',
                'ur_configuration_controller'
            ]
        }],
        on_exit=Shutdown(),
    )

    # Spawn controllers
    def spawn_controllers(controllers, active=True):
        inactive_flags = ['--inactive'] if not active else []
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                '--controller-manager',
                '/controller_manager',
                '--controller-manager-timeout',
                driver_config.controller_spawner_timeout,
            ]
            + inactive_flags
            + controllers,
            # Do not hook into exit signal as these nodes are ephemeral
        )

    controllers_active = [
        'joint_state_broadcaster',
        'io_and_status_controller',
        'speed_scaling_state_broadcaster',
        'force_torque_sensor_broadcaster',
        'robotiq_gripper_controller',
        'robotiq_activation_controller',
        'tcp_pose_broadcaster',
        'ur_configuration_controller',
    ]

    controllers_inactive = [
        'impedance_controller'
    ]

    initial_joint_controller = LaunchConfiguration('initial_joint_controller')
    initial_joint_controller_spawner_started = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            initial_joint_controller,
            '-c',
            '/controller_manager',
            '--controller-manager-timeout',
            driver_config.controller_spawner_timeout,
        ],
        # Do not hook into exit signal as these nodes are ephemeral
    )

    # Add timer action to wait for 1 second before launching all below nodes
    nodes = [urscript_interface, ur_control_node, controller_stopper_node,
             initial_joint_controller_spawner_started,
             spawn_controllers(controllers_active),
             spawn_controllers(controllers_inactive, active=False)]

    group_action = GroupAction(actions=nodes)
    timer_action = TimerAction(
        period=1.0,
        actions=[group_action]
    )
    return [timer_action]


def get_robot_control_nodes(driver_config: UrRobotiqDriverConfig) -> List[Node]:
    """
    For real robot workflows, we want to use the UR driver nodes.

    Returns
    -------
        List[Node]: List of Nodes

    """
    if driver_config.use_sim_time:
        return get_ros2_control_nodes_for_sim(driver_config)
    else:
        return get_ros2_control_nodes_for_real(driver_config)


def get_isaac_sim_joint_parser_node(use_sim_time: bool) -> Node:
    """
    Return Isaac Sim joint parser node.

    This parses joint states coming from Isaac Sim and sets every joint underneath finger_joint
    to be a mimic of it, such that it does not interfere with MoveIt planning because of
    URDF/USD structural differences

    Args
    ----
        use_sim_time (bool): Use sim time

    Returns
    -------
        Node: ROS Node

    """
    return Node(
        package='isaac_manipulator_isaac_sim_utils',
        executable='isaac_sim_joint_parser_node.py',
        name='joint_parser',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        on_exit=Shutdown(),
    )
