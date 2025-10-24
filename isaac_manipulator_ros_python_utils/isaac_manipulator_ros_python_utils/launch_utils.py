# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple

from geometry_msgs.msg import Pose, Vector3
import isaac_manipulator_ros_python_utils.constants as constants
# flake8: noqa: I100
from isaac_manipulator_ros_python_utils.manipulator_types import (
    CameraType,
    DepthType,
    GripperType,
    ObjectAttachmentShape,
    ObjectDetectionType,
    ObjectSelectionType,
    PoseEstimationType,
    SegmentationType,
    WorkflowType,
)
from isaac_ros_launch_utils.all_types import Substitution
import isaac_ros_launch_utils as lu
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration


def get_dnn_stereo_depth_resolution(depth_type: Substitution) -> Tuple[str, str]:
    """
    Get the DNN stereo depth resolution depending on the depth type.

    Returns
    -------
        str, str: width, height

    """
    # Check if either a string or enum is passed
    is_ess_light = lu.is_equal(depth_type, str(DepthType.ESS_LIGHT))
    is_ess_light = is_ess_light or lu.is_equal(depth_type, DepthType.ESS_LIGHT)
    is_foundation_stereo = lu.is_equal(depth_type, str(DepthType.FOUNDATION_STEREO))
    is_foundation_stereo = is_foundation_stereo or lu.is_equal(
        depth_type, DepthType.FOUNDATION_STEREO)

    depth_image_width = lu.if_else_substitution(
        is_foundation_stereo,
        str(constants.FOUNDATION_STEREO_INPUT_IMAGE_WIDTH),
        lu.if_else_substitution(
            is_ess_light,
            str(constants.ESS_LIGHT_INPUT_IMAGE_WIDTH),
            str(constants.ESS_INPUT_IMAGE_WIDTH)
        )
    )
    depth_image_height = lu.if_else_substitution(
        is_foundation_stereo,
        str(constants.FOUNDATION_STEREO_INPUT_IMAGE_HEIGHT),
        lu.if_else_substitution(
            is_ess_light,
            str(constants.ESS_LIGHT_INPUT_IMAGE_HEIGHT),
            str(constants.ESS_INPUT_IMAGE_HEIGHT)
        )
    )
    return depth_image_width, depth_image_height


def get_str_variable(context: LaunchContext, variable_name: str) -> str:
    """
    Return a string from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        str: Returns string representation

    """
    return str(
        context.perform_substitution(LaunchConfiguration(variable_name))
    )


def get_float_variable(context: LaunchContext, variable_name: str) -> float:
    """
    Return a float from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        float: Returns float representation

    """
    return float(
        context.perform_substitution(LaunchConfiguration(variable_name))
    )


def get_bool_variable(context: LaunchContext, variable_name: str) -> bool:
    """
    Return a boolean from a launch variable.

    Args
    ----
        context (LaunchContext): Launch context
        variable_name (str): Name of variable

    Returns
    -------
        bool: Returns boolean representation

    """
    str_var = str(context.perform_substitution(LaunchConfiguration(variable_name)))
    return True if str_var == 'true' else False


def get_camera_type(camera_type_str: str) -> CameraType:
    """
    Get the pythonic camera type.

    Args
    ----
        camera_type_str (str): Str from user

    Raises
    ------
        NotImplementedError: If str is not supported

    Returns
    -------
        CameraType: Camera type variable

    """
    if camera_type_str == str(CameraType.ISAAC_SIM):
        return CameraType.ISAAC_SIM
    elif camera_type_str == str(CameraType.REALSENSE):
        return CameraType.REALSENSE
    else:
        raise NotImplementedError(f'Camera type {camera_type_str} not supported')


def get_object_attachment_type(object_attachment_type: str) -> ObjectAttachmentShape:
    """
    Get pythonic object attachment type.

    Args
    ----
        object_attachment_type (str): Object attachment types

    Returns
    -------
        ObjectAttachmentShape: Return value

    """
    if object_attachment_type == ObjectAttachmentShape.CUBOID.value:
        return ObjectAttachmentShape.CUBOID
    elif object_attachment_type == ObjectAttachmentShape.CUSTOM_MESH.value:
        return ObjectAttachmentShape.CUSTOM_MESH
    elif object_attachment_type == ObjectAttachmentShape.SPHERE.value:
        return ObjectAttachmentShape.SPHERE
    else:
        raise NotImplementedError(f'Object attachment type {object_attachment_type} not supported')


def get_workflow_type(workflow_type_str: str) -> WorkflowType:
    """
    Get workflow type object.

    Args
    ----
        workflow_type_str (str): Workflow type str

    Returns
    -------
        WorkflowType: Workflow type object

    """
    if workflow_type_str == WorkflowType.OBJECT_FOLLOWING.value:
        return WorkflowType.OBJECT_FOLLOWING
    elif workflow_type_str == WorkflowType.PICK_AND_PLACE.value:
        return WorkflowType.PICK_AND_PLACE
    elif workflow_type_str == WorkflowType.POSE_TO_POSE.value:
        return WorkflowType.POSE_TO_POSE
    elif workflow_type_str == WorkflowType.GEAR_ASSEMBLY.value:
        return WorkflowType.GEAR_ASSEMBLY
    else:
        raise NotImplementedError(f'Workflow type {workflow_type_str} not supported !')


def get_pose_estimation_type(pose_estimation_str: str) -> PoseEstimationType:
    """
    Pose estimation type of the context string.

    Args
    ----
        pose_estimation_str (str): Pose estimator type

    Returns
    -------
        PoseEstimationType: Pose estimation type object

    """
    if pose_estimation_str == str(PoseEstimationType.DOPE):
        return PoseEstimationType.DOPE
    elif pose_estimation_str == str(PoseEstimationType.FOUNDATION_POSE):
        return PoseEstimationType.FOUNDATION_POSE
    else:
        raise NotImplementedError(f'Pose estimation type {pose_estimation_str} not supported')


def extract_pose_from_parameter(pose_list: list) -> Pose:
    """
    Extract pose from a launch param.

    Args
    ----
        pose_dict (dict): The pose dict

    Raises
    ------
        ValueError: Value Error if key not found or missing key
        ValueError: Value error if invalid value that is not float

    Returns
    -------
        Pose: The extracted pose

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
        raise ValueError(f'Invalid value in pose: {e}')


def extract_vector3_from_parameter(scale_dict: list) -> Vector3:
    """
    Extract a Vector3 from a launch parameter.

    Args
    ----
        scale_dict (dict): The scale dictionary.

    Raises
    ------
        ValueError: If a value cannot be converted to a float.

    Returns
    -------
        Vector3: The extracted Vector3 message.

    """
    try:
        scale = Vector3()

        scale.x = float(scale_dict[0])
        scale.y = float(scale_dict[1])
        scale.z = float(scale_dict[2])

        return scale
    except ValueError as e:
        raise ValueError(f'Invalid value in scale: {e}')


def get_depth_type(depth_type_str: str) -> DepthType:
    """
    Get the pythonic depth type.

    Args
    ----
        depth_type_str (str): Depth type str

    Returns
    -------
        DepthType: Depth type object

    """
    if depth_type_str == str(DepthType.ESS_FULL):
        return DepthType.ESS_FULL
    elif depth_type_str == str(DepthType.ESS_LIGHT):
        return DepthType.ESS_LIGHT
    elif depth_type_str == str(DepthType.FOUNDATION_STEREO):
        return DepthType.FOUNDATION_STEREO
    elif depth_type_str == str(DepthType.REALSENSE):
        return DepthType.REALSENSE
    elif depth_type_str == str(DepthType.ISAAC_SIM):
        return DepthType.ISAAC_SIM
    else:
        raise NotImplementedError(f'Depth type {depth_type_str} not supported')


def get_object_detection_namespace(object_detection_type: ObjectDetectionType) -> str:
    """
    Get the namespace for the object detection type.

    Args
    ----
        object_detection_type (ObjectDetectionType): Object detection type

    Returns
    -------
        str: Namespace for the object detection type

    """
    if object_detection_type == ObjectDetectionType.GROUNDING_DINO:
        return 'grounding_dino'
    elif object_detection_type == ObjectDetectionType.RTDETR:
        return 'rtdetr'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING:
        return 'segment_anything'
    elif object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2:
        return 'segment_anything2'
    else:
        raise NotImplementedError(f'Object detection type {object_detection_type} not supported')


def get_object_detection_type(object_detection_type_str: str) -> ObjectDetectionType:
    """
    Get the pythonic object detection type.

    Args
    ----
        object_detection_type_str (str): Object detection type str

    Returns
    -------
        ObjectDetectionType: Object detection type object

    """
    if object_detection_type_str == str(ObjectDetectionType.DOPE):
        return ObjectDetectionType.DOPE
    elif object_detection_type_str == str(ObjectDetectionType.GROUNDING_DINO):
        return ObjectDetectionType.GROUNDING_DINO
    elif object_detection_type_str == str(ObjectDetectionType.RTDETR):
        return ObjectDetectionType.RTDETR
    elif object_detection_type_str == str(ObjectDetectionType.SEGMENT_ANYTHING):
        return ObjectDetectionType.SEGMENT_ANYTHING
    elif object_detection_type_str == str(ObjectDetectionType.SEGMENT_ANYTHING2):
        return ObjectDetectionType.SEGMENT_ANYTHING2
    else:
        raise NotImplementedError(
            f'Object detection type {object_detection_type_str} not supported')


def get_object_selection_type(object_selection_type_str: str) -> ObjectSelectionType:
    """
    Get the pythonic object selection type.

    Args
    ----
        object_selection_type_str (str): Object selection type str

    Returns
    -------
        ObjectSelectionType: Object selection type object

    """
    if object_selection_type_str == str(ObjectSelectionType.FIRST):
        return ObjectSelectionType.FIRST
    elif object_selection_type_str == str(ObjectSelectionType.RANDOM):
        return ObjectSelectionType.RANDOM
    elif object_selection_type_str == str(ObjectSelectionType.HIGHEST_SCORE):
        return ObjectSelectionType.HIGHEST_SCORE
    else:
        raise NotImplementedError(
            f'Object selection type {object_selection_type_str} not supported')


def get_segmentation_type(segmentation_type_str: str) -> SegmentationType:
    """
    Get the pythonic segmentation type.

    Args
    ----
        segmentation_type_str (str): Segmentation type str

    Returns
    -------
        SegmentationType: Segmentation type object

    """
    if segmentation_type_str == str(SegmentationType.SEGMENT_ANYTHING):
        return SegmentationType.SEGMENT_ANYTHING
    elif segmentation_type_str == str(SegmentationType.SEGMENT_ANYTHING2):
        return SegmentationType.SEGMENT_ANYTHING2
    elif segmentation_type_str == str(SegmentationType.NONE):
        return SegmentationType.NONE
    else:
        raise NotImplementedError(
            f'Segmentation type {segmentation_type_str} not supported')
