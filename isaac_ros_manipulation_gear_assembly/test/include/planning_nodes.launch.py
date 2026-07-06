# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from isaac_ros_launch_utils import GroupAction

from isaac_ros_manipulation_ros_python_utils import (
    CoreConfig, DepthType, get_cumotion_node,
    get_manipulation_container,
)
from isaac_ros_manipulation_ur_driver_utils import URDriverUtils
from isaac_ros_manipulation_ur_driver_utils.config import UrRobotiqDriverConfig

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)


def launch_setup(context, *args, **kwargs):

    core_config = CoreConfig(context)
    driver_config = UrRobotiqDriverConfig(context)

    manipulator_init_nodes = []
    manipulator_init_nodes.append(get_manipulation_container(core_config))
    manipulator_init_nodes.append(get_cumotion_node(
        camera_type=core_config.camera_config.camera_type,
        xrdf_file_path=core_config.cumotion_config.cumotion_xrdf_file_path,
        urdf_file_path=core_config.cumotion_config.cumotion_urdf_file_path,
        distance_threshold=core_config.cumotion_config.distance_threshold,
        num_cameras=core_config.camera_config.num_cameras,
        time_sync_slop=core_config.time_sync_slop,
        use_sim_time=core_config.use_sim_time,
        setup=core_config.setup,
        workflow_type=core_config.workflow_config.workflow_type,
        read_esdf_world=core_config.enable_nvblox,
        core_config=core_config
    ))
    manipulator_init_nodes.append(URDriverUtils(driver_config).get_robot_state_publisher())
    return manipulator_init_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'gripper_type',
            description='Type of gripper to use with UR robot',
            choices=['robotiq_2f_85', 'robotiq_2f_140'],
        ),
        DeclareLaunchArgument(
            'camera_type',
            choices=['HAWK', 'REALSENSE', 'ISAAC_SIM'],
            description='Camera sensor to use'
        ),
        DeclareLaunchArgument(
            'depth_type',
            choices=DepthType.names(),
            description=f'Depth estimation type. Choose between {", ".join(DepthType.names())}'
        ),
        DeclareLaunchArgument(
            'num_cameras',
            choices=['1', '2',],
            description='Num cameras'
        ),
        DeclareLaunchArgument(
            'setup',
            description='Setup'
        ),
        DeclareLaunchArgument(
            'cumotion_urdf_file_path',
            description='URDF for cumotion planner, not the same as Moveit planner'
        ),
        DeclareLaunchArgument(
            'cumotion_xrdf_file_path',
            description='XRDF for cumotion planner, not the same as Moveit planner'
        ),
        DeclareLaunchArgument(
            'distance_threshold',
            description='Maximum distance from a given collision sphere (in meters) at which'
                        'to mask points in the robot segmenter'
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_qos',
            description='QoS input profile for pose estimation input',
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_fps',
            description='FPS for input into pose estimation pipeline'
        ),
        DeclareLaunchArgument(
            'pose_estimation_dropped_fps',
            description='Number of frames to drop before input into pose estimation pipeline'
        ),
        DeclareLaunchArgument('ur_type', description='UR robot type'),
        DeclareLaunchArgument('robot_ip', description='Robot IP address'),
        DeclareLaunchArgument('ur_calibration_file_path', default_value='',
                              description='UR calibration file path'),
        DeclareLaunchArgument('urdf_path', description='URDF xacro path'),
        DeclareLaunchArgument('srdf_path', description='SRDF xacro path'),
        DeclareLaunchArgument('joint_limits_file_path',
                              description='Joint limits file path'),
        DeclareLaunchArgument('kinematics_file_path',
                              description='Kinematics file path'),
        DeclareLaunchArgument('moveit_controllers_file_path',
                              description='MoveIt controllers file path'),
        DeclareLaunchArgument('ros2_controllers_file_path',
                              description='ROS2 controllers file path'),
        DeclareLaunchArgument('log_level', default_value='error',
                              description='Log level'),
        DeclareLaunchArgument('controller_spawner_timeout', default_value='10',
                              description='Controller spawner timeout'),
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )
    manipulator_init_nodes = []

    return LaunchDescription(launch_args + manipulator_init_nodes + [group_action])
