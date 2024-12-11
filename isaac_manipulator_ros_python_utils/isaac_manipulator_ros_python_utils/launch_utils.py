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

import xacro

from isaac_ros_launch_utils.all_types import Substitution
import isaac_ros_launch_utils as lu

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Vector3
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration


import isaac_manipulator_ros_python_utils.constants as constants
from isaac_manipulator_ros_python_utils.types import CameraType, DepthType, GripperType


def get_hawk_depth_resolution(ess_mode: Substitution) -> Tuple[Substitution, Substitution]:
    """Get the hawk depth resolution depending the ess type.
    Returns:
        Substitution, Substitution: width, height
    """
    # Check if either a string or enum is passed
    is_ess_light = lu.is_equal(ess_mode, str(DepthType.ess_light))
    is_ess_light = is_ess_light or lu.is_equal(ess_mode, DepthType.ess_light)
    depth_image_width = lu.if_else_substitution(is_ess_light,
                                                str(constants.ESS_LIGHT_INPUT_IMAGE_WIDTH),
                                                str(constants.ESS_INPUT_IMAGE_WIDTH))
    depth_image_height = lu.if_else_substitution(is_ess_light,
                                                 str(constants.ESS_LIGHT_INPUT_IMAGE_HEIGHT),
                                                 str(constants.ESS_INPUT_IMAGE_HEIGHT))
    return depth_image_width, depth_image_height


def get_depth_resolution(camera_type: Substitution,
                         ess_mode: Substitution) -> Tuple[Substitution, Substitution]:
    """Get the depth resolution depending on the camera and ess type.
    - hawk            -> hawk resolution (depending on ess_mode)
    - realsense       -> realsense resolution
    Returns:
        Substitution, Substitution: width, height
    """
    is_hawk_camera = lu.is_equal(camera_type, str(CameraType.hawk))
    depth_image_width = lu.if_else_substitution(is_hawk_camera,
                                                get_hawk_depth_resolution(ess_mode)[0],
                                                str(constants.REALSENSE_IMAGE_WIDTH))
    depth_image_height = lu.if_else_substitution(is_hawk_camera,
                                                 get_hawk_depth_resolution(ess_mode)[1],
                                                 str(constants.REALSENSE_IMAGE_HEIGHT))
    return depth_image_width, depth_image_height


def get_rgb_resolution(camera_type: Substitution) -> Tuple[Substitution, Substitution]:
    """Get the rgb resolution depending on the camera type.
    Returns:
        Substitution, Substitution: width, height
    """
    is_hawk_camera = lu.is_equal(camera_type, str(CameraType.hawk))
    rgb_image_width = lu.if_else_substitution(is_hawk_camera, str(constants.HAWK_IMAGE_WIDTH),
                                              str(constants.REALSENSE_IMAGE_WIDTH))
    rgb_image_height = lu.if_else_substitution(is_hawk_camera, str(constants.HAWK_IMAGE_HEIGHT),
                                               str(constants.REALSENSE_IMAGE_HEIGHT))
    return rgb_image_width, rgb_image_height


def get_variable(context: LaunchContext, variable_name: str) -> str:
    """Returns a string from a launch variable

    Args:
        context (_type_): Launch context
        variable_name (str): Name of variable

    Returns:
        str: Returns string representation
    """
    return str(
        context.perform_substitution(LaunchConfiguration(variable_name))
    )


def get_robot_description_contents(
    asset_name: str,
    ur_type: str,
    use_sim_time: bool,
    gripper_type: str,
    grasp_parent_frame: str,
    dump_to_file: bool = False,
    output_file: str = None,
) -> str:
    """Get robot description contents and optionally dump content to file.

    Args:
        asset_name (str): The asset name for robot description
        ur_type (str): UR Type
        use_sim_time (bool): Use sim time for isaac sim platform
        dump_to_file (bool, optional): Dumps xml to file. Defaults to False.
        output_file (str, optional): Output file path if dumps output is True. Defaults to None.

    Returns:
        str: XML contents of robot model
    """
    urdf_file_name = f"{asset_name}.urdf.xacro"
    # Update the file extension and path as needed
    urdf_xacro_file = os.path.join(
        get_package_share_directory("isaac_manipulator_pick_and_place"),
        "urdf",
        urdf_file_name,
    )

    # Process the .xacro file to convert it to a URDF string
    xacro_processed = xacro.process_file(
        urdf_xacro_file,
        mappings={
            "ur_type": ur_type,
            "name": f"{ur_type}_robot",
            "sim_isaac": "true" if use_sim_time else "false",
            "use_fake_hardware": "true" if use_sim_time else "false",
            "generate_ros2_control_tag": "false" if use_sim_time else "true",
            "gripper_type": gripper_type,
            "grasp_parent_frame": grasp_parent_frame
        },
    )
    robot_description = xacro_processed.toxml()

    if dump_to_file and output_file:
        with open(output_file, "w") as file:
            file.write(robot_description)

    return robot_description


def get_gripper_collision_links(gripper_name: GripperType = GripperType.ROBOTIQ_2F_140
                                ) -> List[str]:
    """Gets gripper collision linkes to disable during retract and approach object phase.

    Args:
        gripper_name (GripperType, optional): _description_. Defaults to
            GripperType.ROBOTIQ_2F_140.

    Raises:
        NotImplementedError: If gripper name is not supported

    Returns:
        List[str]: List of collision links to ignore while planning the retract and approach phase
    """
    if gripper_name == GripperType.ROBOTIQ_2F_140:
        return [
            'left_outer_knuckle',
            'left_inner_knuckle',
            'left_outer_finger',
            'left_inner_finger',
            'left_inner_finger_pad',
            'right_outer_knuckle',
            'right_inner_knuckle',
            'right_outer_finger',
            'right_inner_finger',
            'right_inner_finger_pad',
            'robotiq_base_link',
            'tool0',
            'wrist_3_link',
        ]
    elif gripper_name == GripperType.ROBOTIQ_2F_85:
        return [
            'robotiq_85_base_link',
            'robotiq_85_left_finger_link',
            'robotiq_85_left_finger_tip_link',
            'robotiq_85_left_inner_knuckle_link',
            'robotiq_85_left_knuckle_link',
            'robotiq_85_right_finger_link',
            'robotiq_85_right_finger_tip_link',
            'robotiq_85_right_inner_knuckle_link',
            'robotiq_85_right_knuckle_link',
            'robotiq_85_base_link',
            'tool0',
            'wrist_3_link',
        ]
    else:
        raise NotImplementedError(f"Gripper type {gripper_name} not supported")


def extract_pose_from_parameter(pose_list: list) -> Pose:
    """Extract pose from a launch param

    Args:
        pose_dict (dict): The pose dict

    Raises:
        ValueError: Value Error if key not found or missing key
        ValueError: Value error if invalid value that is not float

    Returns:
        Pose: _description_
    """
    try:
        pose = Pose()
        x, y, z, rot_x, rot_y, rot_z, rot_w = pose_list
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)

        pose.orientation.x = float(rot_x)
        pose.orientation.y = float(rot_y)
        pose.orientation.z = float(rot_z)
        pose.orientation.w = float(rot_w)
        return pose
    except ValueError as e:
        raise ValueError(f"Invalid value in pose: {e}")


def extract_vector3_from_parameter(scale_dict: list) -> Vector3:
    """Extract a Vector3 from a launch parameter.

    Args:
        scale_dict (dict): The scale dictionary.

    Raises:
        ValueError: If a value cannot be converted to a float.

    Returns:
        Vector3: The extracted Vector3 message.
    """
    try:
        scale = Vector3()

        scale.x = float(scale_dict[0])
        scale.y = float(scale_dict[1])
        scale.z = float(scale_dict[2])

        return scale
    except ValueError as e:
        raise ValueError(f"Invalid value in scale: {e}")
