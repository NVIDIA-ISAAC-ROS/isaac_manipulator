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
from typing import Dict
import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    OpaqueFunction,
    DeclareLaunchArgument,
)

from isaac_manipulator_ros_python_utils.launch_utils import (
    get_variable,
    get_robot_description_contents
)


def read_grasp_data(file_path: str) -> Dict | None:
    """
    Reads grasp data from a YAML file.
    Args:
        file_path (str): Path to the YAML file.
    Returns:
        Dict -> A dictionary with grasp information.
    """
    try:
        with open(file_path, "r") as file:
            grasp_data = yaml.safe_load(file)
        return grasp_data
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None


def read_urdf_file(file_path: str) -> str:
    """
    Reads the contents of a URDF file and returns it as a string.
    Args
        file_path: The absolute path to the URDF file.

    Returns:
        The URDF file contents as a string.
    """
    try:
        with open(file_path, "r") as urdf_file:
            urdf_content = urdf_file.read()
        return urdf_content
    except Exception as e:
        print(f"Error reading URDF file: {e}")
        return ""


def invert_transformation(quaternion_dict, translation):
    """
    Inverts the transformation defined by a quaternion (in dictionary format) and translation.

    Args:
        quaternion_dict (dict): The original quaternion with keys "x", "y", "z", "w".
        translation (list or np.array): The original translation [tx, ty, tz].

    Returns:
        tuple: A tuple containing the inverted quaternion (in dictionary format) and translation.
    """
    # Convert the dictionary quaternion to a list format
    quaternion = [
        quaternion_dict["x"],
        quaternion_dict["y"],
        quaternion_dict["z"],
        quaternion_dict["w"]
    ]

    # Convert Quaternion to Rotation Matrix
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    # Combine Rotation Matrix and Translation into a Transformation Matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    # Find the Inverse of the Transformation Matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Extract the Rotation Matrix and Translation from the Inverted Matrix
    inverse_rotation_matrix = inverse_transformation_matrix[:3, :3]
    inverse_translation = inverse_transformation_matrix[:3, 3]

    # Convert the Rotation Matrix Back to a Quaternion
    inverse_rotation = R.from_matrix(inverse_rotation_matrix)
    inverse_quaternion = inverse_rotation.as_quat()

    # Convert the inverted quaternion back to dictionary format
    inverse_quaternion_dict = {
        "x": inverse_quaternion[0],
        "y": inverse_quaternion[1],
        "z": inverse_quaternion[2],
        "w": inverse_quaternion[3]
    }

    return inverse_quaternion_dict, inverse_translation


def launch_setup(context, *args, **kwargs):
    # Init global variables for config tracking
    use_sim_time = False
    asset_name = "ur_robotiq_gripper"
    ur_type = "ur10e"
    grasps_file_path = get_variable(context, "grasps_file_path")

    rviz_config_file = os.path.join(
        get_package_share_directory("isaac_manipulator_pick_and_place"),
        "rviz",
        "grasp_visualizer.rviz",
    )
    path_to_urdf = os.path.join(
        get_package_share_directory("isaac_manipulator_pick_and_place"),
        "urdf",
        "soup_can.urdf",
    )

    object_contents = read_urdf_file(path_to_urdf)
    robot_description_contents = get_robot_description_contents(
        asset_name=asset_name, ur_type=ur_type, use_sim_time=use_sim_time,
        gripper_type="robotiq_2f_140", grasp_parent_frame="robotiq_base_link"
    )

    manipulation_container = Node(
        name="manipulation_container",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=["--ros-args", "--log-level", "error"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    object_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="object_publisher",
        namespace="robot1",
        output="screen",
        parameters=[{"robot_description": object_contents}],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace="robot2",
        output="screen",
        parameters=[{"robot_description": robot_description_contents}],
    )

    joint_state_publisher_1 = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        namespace="robot1",
    )

    joint_state_publisher_2 = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        namespace="robot2",
    )

    rviz2_node = Node(
        name="rviz2",
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_file],
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
    )

    grasp_data = read_grasp_data(grasps_file_path)
    if not grasp_data:
        raise ValueError("Grasp file is empty !")

    # Extract data for the first grasp
    grasp = grasp_data["grasps"]["grasp_0"]
    position = grasp["position"]
    orientation = grasp["orientation"]

    # Convert quaternion to euler (roll, pitch, yaw) if necessary
    quat = {
        "x": orientation["xyz"][0],
        "y": orientation["xyz"][1],
        "z": orientation["xyz"][2],
        "w": orientation["w"],
    }
    # Need to invert transformation as we need gripper_pose_object but grasp format stores file in
    # object_pose_gripper
    new_quat, new_position = invert_transformation(quat, position)

    static_transform_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            str(new_position[0]),
            str(new_position[1]),
            str(new_position[2]),
            str(new_quat["x"]),
            str(new_quat["y"]),
            str(new_quat["z"]),
            str(new_quat["w"]),
            # Here robotiq_base_link is the parent/origin
            "robotiq_base_link",
            "can_base_link",
        ],
    )

    return [
        manipulation_container,
        object_publisher,
        robot_state_publisher,
        joint_state_publisher_1,
        joint_state_publisher_2,
        static_transform_publisher,
        rviz2_node,
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "grasps_file_path",
            default_value="/tmp/grasps_sim.yaml",
            description="File name in the config folder for grasps file path",
        )
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
