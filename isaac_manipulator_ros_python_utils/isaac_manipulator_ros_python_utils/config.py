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

from dataclasses import dataclass
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

from ament_index_python.packages import get_package_share_directory
from isaac_manipulator_ros_python_utils.launch_utils import (
    constants, get_bool_variable, get_camera_type, get_depth_type, get_dnn_stereo_depth_resolution,
    get_float_variable, get_object_attachment_type, get_object_detection_namespace,
    get_object_detection_type, get_object_selection_type, get_pose_estimation_type,
    get_segmentation_type, get_str_variable, get_workflow_type
)
from isaac_manipulator_ros_python_utils.manipulator_types import (
    CameraType,
    DepthType,
    GripperType,
    ObjectAttachmentShape,
    ObjectDetectionType,
    ObjectSelectionType,
    PoseEstimationType,
    SegmentationType,
    WorkflowType
)
from isaac_ros_launch_utils.all_types import LaunchConfiguration
from launch import LaunchContext
from sensor_msgs.msg import JointState
import yaml


def load_yaml_params(
    params_file_path: str,
    package_name: str = 'isaac_manipulator_bringup'
) -> Dict:
    """
    Load the YAML parameters for manipulator workflow config.

    Args
    ----
        params_file_path (str): The file path for the manipulator workflow config
        package_name (str): The name of the package to load the parameters from

    Returns
    -------
        Dict: Return dict with values in the file.

    Raises
    ------
        FileNotFoundError: If the parameter file is not found.

    """
    # If path is just a file name, then we need to get the full path
    if not os.path.exists(params_file_path):
        params_file_path = os.path.join(
            get_package_share_directory(package_name),
            'params',
            params_file_path
        )

    if not os.path.exists(params_file_path):
        raise FileNotFoundError(f'Parameter file not found: {params_file_path}')

    # Load the YAML file as text
    with open(params_file_path, 'r') as file:
        yaml_content = file.read()

    # Simple function to replace command expressions with their output
    def process_commands(text):
        pattern = r'\$\(([^)]+)\)'

        def replace_cmd(match):
            cmd = match.group(1)
            return subprocess.getoutput(cmd)

        return re.sub(pattern, replace_cmd, text)

    def process_commands_env_variables(text):
        """
        Replace environment variables in the text.

        Supports both ${VAR} and $VAR syntax.
        Variables that are not set will be left as-is.
        """
        # First, handle ${VAR} format (easier to match unambiguously)
        def replace_braced_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        text = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace_braced_var, text)

        # Then handle $VAR format (but not $(commands) which have already been processed)
        # Match $VAR where VAR is alphanumeric/underscore and NOT followed by (
        def replace_unbraced_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        text = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)(?![(\{])', replace_unbraced_var, text)

        return text

    # Process the YAML content
    processed_yaml = process_commands(yaml_content)

    # Make sure to replace all the env variables in the processed YAML
    processed_yaml = process_commands_env_variables(processed_yaml)

    # Parse the processed YAML
    return yaml.safe_load(processed_yaml)


class Config:
    """Config that tracks Configuration variables."""

    use_sim_time: bool

    def __init__(self, context: LaunchContext):
        self.use_sim_time = get_bool_variable(context, 'use_sim_time')


class DriverConfig(Config):
    """Config that tracks all variables needed to run drivers for the robot/Isaac Sim."""

    urdf_path: str
    srdf_path: str
    joint_limits_file_path: str
    kinematics_file_path: str
    moveit_controllers_file_path: str
    ros2_controllers_file_path: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.urdf_path = get_str_variable(context, 'urdf_path')
        self.srdf_path = get_str_variable(context, 'srdf_path')
        self.joint_limits_file_path = get_str_variable(context, 'joint_limits_file_path')
        self.kinematics_file_path = get_str_variable(context, 'kinematics_file_path')
        self.moveit_controllers_file_path = get_str_variable(
            context, 'moveit_controllers_file_path')
        self.ros2_controllers_file_path = get_str_variable(context, 'ros2_controllers_file_path')


class UrRobotiqDriverConfig(DriverConfig):
    """Config that tracks all variables needed to perform UR and Robotiq workflows."""

    controller_spawner_timeout: LaunchConfiguration
    ur_type: str
    robot_ip: str
    gripper_type: str
    grasp_parent_frame: str
    log_level: str
    remapped_joint_states: Dict
    workflow_type: WorkflowType
    # The calibration file generated by the UR ros2 driver to get calibration values for joints
    ur_calibration_file_path: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.gripper_type = get_str_variable(context, 'gripper_type')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.ur_type = get_str_variable(context, 'ur_type')
        self.robot_ip = get_str_variable(context, 'robot_ip')
        if self.use_sim_time and self.gripper_type == 'robotiq_2f_85':
            raise ValueError(f'Gripper type {self.gripper_type} not supported for Isaac sim')

        self.log_level = get_str_variable(context, 'log_level')
        self.ur_calibration_file_path = get_str_variable(context, 'ur_calibration_file_path')
        self.controller_spawner_timeout = LaunchConfiguration('controller_spawner_timeout')

        if self.gripper_type == 'robotiq_2f_140':
            self.grasp_parent_frame = 'robotiq_base_link'
        elif self.gripper_type == 'robotiq_2f_85':
            self.grasp_parent_frame = 'robotiq_85_base_link'

        if self.use_sim_time:
            self.remapped_joint_states = {
                '/joint_states': '/isaac_parsed_joint_states',
                '/controller_manager/robot_description': '/robot_description',
            }
        else:
            self.remapped_joint_states = {}


class CameraConfig(Config):
    """Config that tracks all variables needed to run cameras for the robot/Isaac Sim."""

    num_cameras: str
    depth_type: DepthType
    camera_type: CameraType

    # These are overridden by the child classes
    color_camera_topic_name: str = 'N/A'
    depth_camera_topic_name: str = 'N/A'
    color_camera_info_topic_name: str = 'N/A'
    depth_camera_info_topic_name: str = 'N/A'
    rgb_image_width: str = 'N/A'
    rgb_image_height: str = 'N/A'
    depth_image_width: str = 'N/A'
    depth_image_height: str = 'N/A'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.num_cameras = get_str_variable(context, 'num_cameras')
        self.camera_type = get_camera_type(get_str_variable(context, 'camera_type'))
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')
        if self.camera_type == CameraType.ISAAC_SIM:
            self.depth_type = DepthType.ISAAC_SIM
        elif self.camera_type == CameraType.REALSENSE:
            if self.enable_dnn_depth_in_realsense:
                self.depth_type = get_depth_type(get_str_variable(context, 'depth_type'))
            else:
                self.depth_type = DepthType.REALSENSE
        else:
            raise ValueError(f'Camera type {self.camera_type} not supported')


class IsaacSimCameraConfig(CameraConfig):
    """Contains image topic names coming from Isaac Sim."""

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = CameraType.ISAAC_SIM
        self.color_camera_topic_name = '/front_stereo_camera/left/image_raw'
        self.depth_camera_topic_name = '/front_stereo_camera/depth/ground_truth'
        self.color_camera_info_topic_name = '/front_stereo_camera/left/camera_info'
        self.depth_camera_info_topic_name = '/front_stereo_camera/left/camera_info'
        self.rgb_image_width = str(constants.HAWK_IMAGE_WIDTH)
        self.rgb_image_height = str(constants.HAWK_IMAGE_HEIGHT)
        self.depth_image_width = str(constants.HAWK_IMAGE_WIDTH)
        self.depth_image_height = str(constants.HAWK_IMAGE_HEIGHT)
        self.num_cameras = '1'
        self.use_sim_time = True


class RealsenseCameraConfig(CameraConfig):
    """Contains image topic names coming from RealSense camera."""

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = CameraType.REALSENSE
        self.color_camera_topic_name = '/camera_1/color/image_raw'
        self.depth_camera_topic_name = '/camera_1/aligned_depth_to_color/image_raw'
        self.color_camera_info_topic_name = '/camera_1/color/camera_info'
        self.depth_camera_info_topic_name = '/camera_1/aligned_depth_to_color/camera_info'
        self.rgb_image_width = str(constants.REALSENSE_IMAGE_WIDTH)
        self.rgb_image_height = str(constants.REALSENSE_IMAGE_HEIGHT)
        self.depth_image_width = str(constants.REALSENSE_IMAGE_WIDTH)
        self.depth_image_height = str(constants.REALSENSE_IMAGE_HEIGHT)

        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')
        if self.enable_dnn_depth_in_realsense:
            self.ess_depth_camera_topic_name = '/camera_1/depth_image'
            self.ess_depth_camera_info_topic_name = '/camera_1/rgb/camera_info'

            # Also align the images from realsense to be this resized one.
            self.depth_camera_topic_name = '/camera_1/depth_image'
            self.depth_camera_info_topic_name = '/camera_1/rgb/camera_info'

            depth_image_width, depth_image_height = \
                get_dnn_stereo_depth_resolution(self.depth_type)
            self.depth_image_width = depth_image_width
            self.depth_image_height = depth_image_height


def get_camera_config(context: LaunchContext, camera_type: CameraType) -> CameraConfig:
    """
    Get camera config.

    Args
    ----
        camera_type (CameraType): Camera type

    Returns
    -------
        CameraConfig: Camera config

    """
    if camera_type == CameraType.ISAAC_SIM:
        return IsaacSimCameraConfig(context)
    elif camera_type == CameraType.REALSENSE:
        return RealsenseCameraConfig(context)
    else:
        raise NotImplementedError(f'Camera type is not supported {camera_type}')


class SensorConfig(Config):
    """Config tracks variables needed to perform workflows in Isaac Sim and on the real robot."""

    camera_type: CameraType
    gripper_type: GripperType
    num_cameras: str
    setup: str
    depth_type: str
    workflow_type: WorkflowType
    enable_dnn_depth_in_realsense: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.gripper_type = get_str_variable(context, 'gripper_type')
        self.camera_type = get_camera_type(get_str_variable(context, 'camera_type'))
        self.num_cameras = get_str_variable(context, 'num_cameras')
        self.setup = get_str_variable(context, 'setup')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.depth_type = get_str_variable(context, 'depth_type')
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')

        if self.use_sim_time and self.gripper_type == 'robotiq_2f_85':
            raise ValueError(f'Gripper type {self.gripper_type} not supported for Isaac sim')


class PoseEstimationConfig(Config):
    """Config that tracks all variables needed to perform pose estimation."""

    pose_estimation_type: PoseEstimationType
    object_detection_type: ObjectDetectionType
    segmentation_type: SegmentationType

    def __init__(self, context: LaunchContext):
        super().__init__(context)

        self.pose_estimation_type = get_pose_estimation_type(
            get_str_variable(context, 'pose_estimation_type'))
        self.object_detection_type = get_object_detection_type(
            get_str_variable(context, 'object_detection_type'))
        self.segmentation_type = get_segmentation_type(
            get_str_variable(context, 'segmentation_type'))

    def update_server_topic_names(self, camera_config: CameraConfig):
        """
        Update the topic names for the servers.

        Args
        ----
            camera_config (CameraConfig): Camera config

        Raises
        ------
            NotImplementedError: Pose estimation config does not implement this method

        """
        raise NotImplementedError('Pose estimation config does not implement this method')


class ObjectDetectionConfig(Config):
    """Config that tracks all variables needed to perform object detection."""

    detections_2d_array_output_topic: str
    is_sam_detection_backend: bool
    object_detection_type: ObjectDetectionType
    name_space_prefix: str
    workflow_type: WorkflowType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.object_detection_type = get_object_detection_type(
            get_str_variable(context, 'object_detection_type'))
        self.name_space_prefix = get_object_detection_namespace(self.object_detection_type)
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))

        self.is_sam_detection_backend = (
            self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING or
            self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2
        )

        if self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'


class ObjectSelectionConfig(Config):
    """Config that tracks all variables needed to perform object selection."""

    selection_policy: ObjectSelectionType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.selection_policy = get_object_selection_type(
            get_str_variable(context, 'selection_policy'))


class SegmentAnythingConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Segment Anything object detection."""

    # This is used when action servers are not being used; manual inputs to this topic are needed.
    # This will be picked up by the SegmentAnythingTrigger node.
    segment_anything_input_points_topic: str
    segment_anything_input_detections_topic: str
    sam_model_repository_paths: LaunchConfiguration
    segment_anything_enable_debug_output: str

    # Topic names for SAM (Segment Anything) server
    sam_in_img_topic_name: str
    sam_out_img_topic_name: str

    sam_in_camera_info_topic_name: str
    sam_out_camera_info_topic_name: str

    # The servers will publish detections 2d array on this topic.
    # When servers are on, SegmentAnythingTrigger node is not operational, the servers will output
    # time synced messages onto image, camera info and detection 2d queues which are injested by
    # SegmentAnythingEncoder node.
    sam_out_detections_topic_name: str

    # Output of SAM, the detections 2d array of the mask generation and the mask piped into the
    # servers for caching purposes. These messages are send via action responses to ObjectInfo
    # Server.
    sam_in_detections_topic_name: str
    sam_in_mask_topic_name: str

    # True when using object following workflow and the object detection backend is SAM.
    is_point_triggered: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.segment_anything_input_points_topic = get_str_variable(
            context, 'segment_anything_input_points_topic')
        self.segment_anything_input_detections_topic = get_str_variable(
            context, 'segment_anything_input_detections_topic')
        self.sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')
        self.segment_anything_enable_debug_output = get_str_variable(
            context, 'segment_anything_enable_debug_output')

        # Topic names for SAM (Segment Anything) server
        self.sam_in_img_topic_name = 'UPDATED_LATER'
        self.sam_out_img_topic_name = '/segment_anything_server/image'
        self.sam_in_camera_info_topic_name = 'UPDATED_LATER'
        self.sam_out_camera_info_topic_name = '/segment_anything_server/camera_info'
        self.sam_out_detections_topic_name = \
            '/segment_anything_server/detections_initial_guess'
        self.sam_in_mask_topic_name = '/segment_anything/binary_segmentation_mask'
        self.sam_in_detections_topic_name = '/segment_anything/detections'

        # Main launch file input topic names
        self.sam_main_launch_file_image_topic = 'UPDATED_LATER'
        self.sam_main_launch_file_camera_info_topic = 'UPDATED_LATER'

        self.is_point_triggered = (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            self.is_sam_detection_backend
        )

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.sam_in_img_topic_name = camera_config.color_camera_topic_name
        self.sam_in_camera_info_topic_name = camera_config.color_camera_info_topic_name

        self.sam_main_launch_file_image_topic = camera_config.color_camera_topic_name
        self.sam_main_launch_file_camera_info_topic = camera_config.color_camera_info_topic_name
        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            self.sam_main_launch_file_image_topic = self.sam_out_img_topic_name
            self.sam_main_launch_file_camera_info_topic = self.sam_out_camera_info_topic_name

        # If using object following workflow and using an object detection model as the detection
        # backend, pass the output 2D detections to the SAM graph.
        if (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            not self.is_sam_detection_backend
        ):
            self.sam_out_detections_topic_name = \
                f'/{self.detections_2d_array_output_topic}'
            self.sam_main_launch_file_image_topic = f'/{self.name_space_prefix}/image_dropped'
            self.sam_main_launch_file_camera_info_topic = \
                f'/{self.name_space_prefix}/camera_info_dropped'


class SegmentAnything2Config(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Segment Anything 2 object detection."""

    # This is used when action servers are not being used; manual inputs to this topic are needed.
    # This will be picked up by the SegmentAnything2Trigger node.
    segment_anything_input_points_topic: str
    sam_model_repository_paths: LaunchConfiguration

    # Topic names for SAM2 (Segment Anything 2) server
    sam2_in_img_topic_name: str
    sam2_out_img_topic_name: str

    sam2_in_camera_info_topic_name: str
    sam2_out_camera_info_topic_name: str

    # Output of SAM2, the detections 2d array of the mask generation and the mask piped into the
    # servers for caching purposes. These messages are send via action responses to ObjectInfo
    # Server.
    sam2_in_detections_topic_name: str
    sam2_in_mask_topic_name: str

    # True when using object following workflow and the object detection backend is SAM.
    is_point_triggered: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.segment_anything_input_points_topic = get_str_variable(
            context, 'segment_anything_input_points_topic')
        self.sam_model_repository_paths = LaunchConfiguration('sam_model_repository_paths')

        # Topic names for SAM2 (Segment Anything 2) server
        self.sam2_in_img_topic_name = 'UPDATED_LATER'
        self.sam2_out_img_topic_name = '/segment_anything_server/image'
        self.sam2_in_camera_info_topic_name = 'UPDATED_LATER'
        self.sam2_out_camera_info_topic_name = '/segment_anything_server/camera_info'
        self.sam2_in_mask_topic_name = '/segment_anything2/binary_segmentation_mask'
        self.sam2_in_detections_topic_name = '/segment_anything2/detections'

        # Main launch file input topic names
        self.sam2_main_launch_file_image_topic = 'UPDATED_LATER'
        self.sam2_main_launch_file_camera_info_topic = 'UPDATED_LATER'

        self.is_point_triggered = (
            self.workflow_type == WorkflowType.OBJECT_FOLLOWING and
            self.is_sam_detection_backend
        )

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.sam2_in_img_topic_name = camera_config.color_camera_topic_name
        self.sam2_in_camera_info_topic_name = camera_config.color_camera_info_topic_name

        self.sam2_main_launch_file_image_topic = camera_config.color_camera_topic_name
        self.sam2_main_launch_file_camera_info_topic = camera_config.color_camera_info_topic_name
        if self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.sam2_main_launch_file_image_topic = self.sam2_out_img_topic_name
            self.sam2_main_launch_file_camera_info_topic = self.sam2_out_camera_info_topic_name


class DopeConfig(PoseEstimationConfig):
    """Config that tracks all variables needed to perform DOPE detection and pose estimation."""

    dope_model_file_path: str
    dope_engine_file_path: str
    input_fps: str
    dropped_fps: str
    dope_input_qos: str
    dope_rotation_y_axis: str = '-90.0'
    dope_enable_tf_publishing: str = 'true'

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.dope_engine_file_path = get_str_variable(context, 'dope_engine_file_path')
        self.dope_model_file_path = get_str_variable(context, 'dope_model_file_path')
        self.input_fps = get_str_variable(context, 'pose_estimation_input_fps')
        self.dropped_fps = get_str_variable(context, 'pose_estimation_dropped_fps')
        self.dope_input_qos = get_str_variable(context, 'pose_estimation_input_qos')


class RtDetrConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform RT-DETR object detection."""

    rt_detr_confidence_threshold: str
    rtdetr_engine_file_path: str
    object_class_id: str

    # Object Detection Server for RT-DETR
    rtdetr_in_img_topic_name: str
    rtdetr_out_img_topic_name: str

    rtdetr_in_camera_info_topic_name: str
    rtdetr_out_camera_info_topic_name: str

    # Not sure why we need this:
    rtdetr_out_detections_topic_name: str

    # Output of RT-DETR
    rtdetr_in_detections_topic_name: str

    # Input into RTDEtr launcg file.
    rtdetr_camera_input: str
    rtdetr_camera_info_input: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.rt_detr_confidence_threshold = get_str_variable(
            context, 'rt_detr_confidence_threshold'
        )
        self.rtdetr_engine_file_path = get_str_variable(
            context, 'rtdetr_engine_file_path'
        )
        self.object_class_id = get_str_variable(context, 'object_class_id')

        self.rtdetr_in_img_topic_name = 'UPDATED_LATER'
        self.rtdetr_out_img_topic_name = '/object_detection_server/image_rect'
        # Figure this out, pipe into rtdetr as well not hardcoded.
        self.rtdetr_in_detections_topic_name = '/detections'
        self.rtdetr_out_detections_topic_name = '/object_detection_server/detections_output'
        self.rtdetr_in_camera_info_topic_name = 'UPDATED_LATER'
        self.rtdetr_out_camera_info_topic_name = '/object_detection_server/camera_info'

        # Input into the main rtdetr.launch.py file
        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'

        self.rtdetr_camera_input = 'UPDATED_LATER'
        self.rtdetr_camera_info_input = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.rtdetr_in_img_topic_name = camera_config.color_camera_topic_name
        self.rtdetr_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            self.rtdetr_camera_input = self.rtdetr_out_img_topic_name
            self.rtdetr_camera_info_input = self.rtdetr_out_camera_info_topic_name
        else:
            self.rtdetr_camera_input = camera_config.color_camera_topic_name
            self.rtdetr_camera_info_input = camera_config.color_camera_info_topic_name


class GroundingDinoConfig(ObjectDetectionConfig):
    """Config that tracks all variables needed to perform Grounding DINO object detection."""

    grounding_dino_confidence_threshold: str
    grounding_dino_engine_file_path: str
    grounding_dino_model_file_path: str
    grounding_dino_default_prompt: str
    grounding_dino_in_camera_info_topic_name: str
    grounding_dino_in_detections_topic_name: str
    grounding_dino_in_img_topic_name: str
    grounding_dino_out_camera_info_topic_name: str
    grounding_dino_out_detections_topic_name: str
    grounding_dino_out_img_topic_name: str
    network_image_height: str
    network_image_width: str

    # Input into grounding dino launch file
    grounding_dino_camera_input: str
    grounding_dino_camera_info_input: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.grounding_dino_confidence_threshold = get_str_variable(
            context, 'grounding_dino_confidence_threshold'
        )
        self.grounding_dino_engine_file_path = get_str_variable(
            context, 'grounding_dino_engine_file_path'
        )
        self.grounding_dino_model_file_path = get_str_variable(
            context, 'grounding_dino_model_file_path'
        )
        self.grounding_dino_default_prompt = get_str_variable(
            context, 'grounding_dino_default_prompt'
        )
        self.network_image_width = get_str_variable(
            context, 'grounding_dino_network_image_width'
        )
        self.network_image_height = get_str_variable(
            context, 'grounding_dino_network_image_height'
        )

        # Inputs into the object detection server.
        self.grounding_dino_in_camera_info_topic_name = 'UPDATED_LATER'
        self.grounding_dino_in_img_topic_name = 'UPDATED_LATER'
        # Inputs into the main grounding_dino.launch.py file
        self.grounding_dino_in_detections_topic_name = '/detections'
        self.grounding_dino_out_camera_info_topic_name = '/object_detection_server/camera_info'
        self.grounding_dino_out_detections_topic_name = \
            '/object_detection_server/detections_output'
        self.grounding_dino_out_img_topic_name = '/object_detection_server/image_rect'

        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            self.detections_2d_array_output_topic = '/detections'
        else:
            self.detections_2d_array_output_topic = 'detections_output'

        self.grounding_dino_camera_input = 'UPDATED_LATER'
        self.grounding_dino_camera_info_input = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.grounding_dino_in_img_topic_name = camera_config.color_camera_topic_name
        self.grounding_dino_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            self.grounding_dino_camera_input = self.grounding_dino_out_img_topic_name
            self.grounding_dino_camera_info_input = self.grounding_dino_out_camera_info_topic_name
        else:
            self.grounding_dino_camera_input = camera_config.color_camera_topic_name
            self.grounding_dino_camera_info_input = camera_config.color_camera_info_topic_name


class FoundationPoseConfig(PoseEstimationConfig):
    """Config that tracks all variables needed to perform FoundationPose pose estimation."""

    foundation_pose_mesh_file_path: str
    foundation_pose_texture_path: str
    foundation_pose_refine_engine_file_path: str
    foundation_pose_score_engine_file_path: str
    foundation_pose_segmentation_mask_topic: str
    object_detection_config: Union[
        GroundingDinoConfig, RtDetrConfig, SegmentAnythingConfig, SegmentAnything2Config]
    segmentation_config: Union[SegmentAnythingConfig, SegmentAnything2Config, None]
    input_fps: str
    dropped_fps: str
    input_qos: str
    foundation_pose_detection2_d_topic: str = 'detections_output'
    workflow_type: WorkflowType
    discard_messages: str = 'True'
    discard_msg_older_than_ms: str = '1000'

    # Topic names to push perception input into foundation pose launch file
    foundation_pose_rgb_image_topic: str
    foundation_pose_depth_image_topic: str
    foundation_pose_rgb_camera_info: str

    # Topic names for foundation pose servers.
    fp_in_img_topic_name: str  # Input image topic name
    fp_out_img_topic_name: str  # Ouput from servers routed into perception pipeline

    fp_in_camera_info_topic_name: str  # Input camera info topic name
    fp_out_camera_info_topic_name: str  # Output topic name routed into perception pipeline

    fp_in_depth_topic_name: str  # Input depth topic name
    fp_out_depth_topic_name: str  # Output depth topic name routed into perception pipeline

    fp_out_detections_topic_name: str  # Output detections topic name routed into perception
    # Output segmented mask topic name routed into perception, only when SEGMENT ANYTHING
    # backend for segmentaiton is used. For RTDETR seg mask is generated per pose estimation call.
    fp_out_segmented_mask_topic_name: str
    # Output pose estimate topic name, just outputs the computed pose in this topic
    # TODO(kchahal): Do we need this ?
    fp_out_pose_estimate_topic_name: str

    enable_dnn_depth_in_realsense: bool
    depth_camera_info_for_ess: str
    depth_type: DepthType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.foundation_pose_mesh_file_path = get_str_variable(
            context, 'foundation_pose_mesh_file_path')
        self.foundation_pose_texture_path = get_str_variable(
            context, 'foundation_pose_texture_path')
        self.foundation_pose_score_engine_file_path = get_str_variable(
            context, 'foundation_pose_score_engine_file_path')
        self.foundation_pose_refine_engine_file_path = get_str_variable(
            context, 'foundation_pose_refine_engine_file_path'
        )
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))
        self.depth_type = get_depth_type(get_str_variable(context, 'depth_type'))

        if self.workflow_type == WorkflowType.OBJECT_FOLLOWING:
            # We don't allow FoundationPose to run on ESS/FoundationStereo depth for
            # object following. This is because the depth, color and segmentation need exact
            # timestamps for foundationpose and realsense emits infra1/infra2 at different
            # timestamps than rgb stream which requires a timestamp matching node which is not
            # currently setup. We get around this for PICK and PLACE / GEAR ASSEMBLY by manually
            # overwriting timestamps of the messages in Manipulator servers to be the same before
            # publishing.
            self.enable_dnn_depth_in_realsense = False

        if self.depth_type == DepthType.FOUNDATION_STEREO:
            # FoundationStereo can take at worst 3 seconds to process a frame.
            # If this is not added, robot segmentor will stop publishing updates for
            # nvblox to injest.
            # This might not be needed as FoundationStereo gets faster with time.
            self.time_sync_slop = '3.0'

        self.pose_estimation_type = get_pose_estimation_type(
            get_str_variable(context, 'pose_estimation_type'))
        self.dropped_fps = get_str_variable(context, 'pose_estimation_dropped_fps')
        self.input_fps = get_str_variable(context, 'pose_estimation_input_fps')
        # In Isaac Sim, we use DEFAULT QoS as we can apply backpressure to the input topic in
        # and make sim tick slower so that it can injest all the data. For real robot, we use
        # SENSOR_DATA so that we do not slow the pipeline down in the pursuit of no message drops.
        self.input_qos = get_str_variable(context, 'pose_estimation_input_qos')

        if self.object_detection_type == ObjectDetectionType.GROUNDING_DINO:
            self.object_detection_config = GroundingDinoConfig(context)
            self.foundation_pose_segmentation_mask_topic = 'segmentation'
        elif self.object_detection_type == ObjectDetectionType.RTDETR:
            self.object_detection_config = RtDetrConfig(context)
            self.foundation_pose_segmentation_mask_topic = 'segmentation'
        elif self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING:
            self.object_detection_config = SegmentAnythingConfig(context)
            self.foundation_pose_segmentation_mask_topic = \
                '/segment_anything/binary_segmentation_mask'
        elif self.object_detection_type == ObjectDetectionType.SEGMENT_ANYTHING2:
            self.object_detection_config = SegmentAnything2Config(context)
            self.foundation_pose_segmentation_mask_topic = \
                '/segment_anything2/binary_segmentation_mask'
        else:
            raise NotImplementedError(
                f'Object detection type {self.object_detection_type} not supported')

        self.segmentation_config = None
        if self.segmentation_type == SegmentationType.SEGMENT_ANYTHING:
            self.segmentation_config = SegmentAnythingConfig(context)
        elif self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
            self.segmentation_config = SegmentAnything2Config(context)

        self.foundation_pose_detection2_d_topic = 'detections_output'

        # Topic names for foundation pose servers.
        self.fp_in_img_topic_name = 'UPDATED_LATER'
        self.fp_out_img_topic_name = '/foundation_pose_server/image'
        self.fp_in_camera_info_topic_name = 'UPDATED_LATER'  # Verify this for SAM case
        self.fp_out_camera_info_topic_name = '/foundation_pose_server/camera_info'
        self.fp_in_depth_topic_name = 'UPDATED_LATER'
        self.fp_out_depth_topic_name = '/foundation_pose_server/depth'
        # This is correct for foundation pose.
        self.fp_in_pose_estimate_topic_name = '/pose_estimation/output'
        self.fp_out_segmented_mask_topic_name = '/foundation_pose_server/segmented_mask'
        self.fp_out_pose_estimate_topic_name = '/foundation_pose_server/pose_estimation/output'
        self.fp_out_detections_topic_name = '/foundation_pose_server/bbox'

        self.discard_old_messages = 'True'
        if (
            self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY)
            or self.use_sim_time
        ):
            # Do not discard messages in FoundationPoseSynchronizer as we only send messages per
            # client request, we do not want to drop that data.
            self.discard_old_messages = 'False'

        if self.workflow_type in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            if (
                self.segmentation_type == SegmentationType.SEGMENT_ANYTHING or
                self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2
            ):
                self.segmentation_mask_topic = self.fp_out_segmented_mask_topic_name
            else:
                self.segmentation_mask_topic = 'segmentation'
        else:
            if self.segmentation_type == SegmentationType.SEGMENT_ANYTHING:
                self.segmentation_mask_topic = self.segmentation_config.sam_in_mask_topic_name
            elif self.segmentation_type == SegmentationType.SEGMENT_ANYTHING2:
                self.segmentation_mask_topic = self.segmentation_config.sam2_in_mask_topic_name
            else:
                self.segmentation_mask_topic = 'segmentation'

        self.name_space_prefix = get_object_detection_namespace(self.object_detection_type)

        self.depth_camera_info_for_ess = 'UPDATED_LATER'

    def update_server_topic_names(self, camera_config: CameraConfig):
        self.fp_in_img_topic_name = camera_config.color_camera_topic_name
        self.fp_in_camera_info_topic_name = camera_config.color_camera_info_topic_name
        self.fp_in_depth_topic_name = camera_config.depth_camera_topic_name

        if self.enable_dnn_depth_in_realsense:
            self.fp_in_depth_topic_name = camera_config.ess_depth_camera_topic_name

        self.foundation_pose_rgb_image_topic = camera_config.color_camera_topic_name
        self.foundation_pose_rgb_camera_info = camera_config.color_camera_info_topic_name
        self.foundation_pose_depth_image_topic = camera_config.depth_camera_topic_name
        self.foundation_pose_detections_topic = self.foundation_pose_detection2_d_topic
        self.realsense_depth_image_topic = self.foundation_pose_depth_image_topic
        if self.workflow_type not in (WorkflowType.PICK_AND_PLACE, WorkflowType.GEAR_ASSEMBLY):
            # A new node is added that drops input to acceptable input hz into perception pipelines
            # It has more hz than the server based approach but not at the same rate as camera
            # streams. This is done to manage compute on the Orin AGX system.
            if camera_config.camera_type is CameraType.REALSENSE:
                self.realsense_depth_image_topic = f'/{self.name_space_prefix}/depth_dropped'
                self.foundation_pose_rgb_image_topic = f'/{self.name_space_prefix}/image_dropped'
                self.foundation_pose_rgb_camera_info = \
                    f'/{self.name_space_prefix}/camera_info_dropped'
            elif camera_config.camera_type is CameraType.ISAAC_SIM:
                self.foundation_pose_rgb_image_topic = f'/{self.name_space_prefix}/image_dropped'
                self.foundation_pose_depth_image_topic = f'/{self.name_space_prefix}/depth_dropped'
                self.foundation_pose_rgb_camera_info = \
                    f'/{self.name_space_prefix}/camera_info_dropped'
            else:
                raise Exception(f'CameraType {camera_config.camera_type} not implemented.')
        else:
            self.foundation_pose_depth_image_topic = self.fp_out_depth_topic_name
            self.foundation_pose_rgb_camera_info = self.fp_out_camera_info_topic_name
            self.foundation_pose_rgb_image_topic = self.fp_out_img_topic_name
            self.foundation_pose_detections_topic = self.fp_out_detections_topic_name
            self.realsense_depth_image_topic = self.fp_out_depth_topic_name

        if camera_config.camera_type is CameraType.REALSENSE:
            self.foundation_pose_depth_image_topic = self.realsense_depth_image_topic + '_metric'

        if self.enable_dnn_depth_in_realsense:
            self.depth_camera_info_for_ess = camera_config.ess_depth_camera_info_topic_name
        else:
            self.depth_camera_info_for_ess = camera_config.depth_camera_info_topic_name


class OrchestrationConfig(Config):
    """Config that tracks all variables needed for orchestration script."""

    behavior_tree_config_file: str
    blackboard_config_file: str
    print_ascii_tree: str
    manual_mode: str
    log_level: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.behavior_tree_config_file = get_str_variable(context, 'behavior_tree_config_file')
        self.blackboard_config_file = get_str_variable(context, 'blackboard_config_file')
        self.print_ascii_tree = get_str_variable(context, 'print_ascii_tree')
        self.manual_mode = get_str_variable(context, 'manual_mode')
        self.log_level = get_str_variable(context, 'log_level')


class WorkflowConfigParams(Config):
    """Config that tracks all variables needed to perform workflows."""

    workflow_type: WorkflowType

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))


class ObjectFollowingConfig(WorkflowConfigParams):
    """Config that tracks all variables needed to perform object following."""

    time_dilation_factor: float

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')


class PoseToPoseConfig(WorkflowConfigParams):
    """Config that tracks all variables needed to perform pose to pose workflows."""

    time_dilation_factor: float

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')


class PickAndPlaceConfig(WorkflowConfigParams):
    """Config that tracks all variables needed to perform pick and place workflows."""

    use_ground_truth_pose_in_sim: bool
    pick_and_place_planner_retries: int
    pick_and_place_retry_wait_time: float
    sim_gt_asset_frame_id: str
    grasps_file_path: str
    object_attachment_type: ObjectAttachmentShape
    object_attachment_scale: LaunchConfiguration
    use_pose_from_rviz: LaunchConfiguration
    move_to_home_pose_after_place: LaunchConfiguration
    home_pose: LaunchConfiguration
    time_dilation_factor: str
    attach_object_mesh_file_path: str
    filter_depth_buffer_time: str
    end_effector_mesh_resource_uri: str
    gripper_type: GripperType
    orchestration_config: OrchestrationConfig

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.filter_depth_buffer_time = get_str_variable(context, 'filter_depth_buffer_time')
        self.use_ground_truth_pose_in_sim = get_bool_variable(
            context, 'use_ground_truth_pose_in_sim'
        )
        self.pick_and_place_planner_retries = int(get_str_variable(
            context, 'pick_and_place_planner_retries'
        ))
        self.pick_and_place_retry_wait_time = get_float_variable(
            context, 'pick_and_place_retry_wait_time'
        )
        self.sim_gt_asset_frame_id = get_str_variable(
            context, 'sim_gt_asset_frame_id'
        )
        self.grasps_file_path = get_str_variable(
            context, 'grasps_file_path'
        )
        self.object_attachment_type = get_object_attachment_type(get_str_variable(
            context, 'object_attachment_type'
        ))
        self.attach_object_mesh_file_path = get_str_variable(
            context, 'attach_object_mesh_file_path'
        )
        self.end_effector_mesh_resource_uri = get_str_variable(
            context, 'end_effector_mesh_resource_uri'
        )
        self.time_dilation_factor = get_float_variable(context, 'time_dilation_factor')
        self.gripper_type = GripperType.get_gripper_type(get_str_variable(context, 'gripper_type'))
        self.object_attachment_scale = LaunchConfiguration('object_attachment_scale')
        self.use_pose_from_rviz = LaunchConfiguration('use_pose_from_rviz')
        self.move_to_home_pose_after_place = LaunchConfiguration('move_to_home_pose_after_place')
        self.home_pose = LaunchConfiguration('home_pose')
        self.orchestration_config = OrchestrationConfig(context)


class GearAssemblyConfig(PickAndPlaceConfig):
    """Config that tracks all variables needed to perform gear assembly."""

    model_path: str
    model_file_name: str
    policy_alpha: float
    observation_topic: str
    insertion_request_topic: str
    goal_pose_topic: str
    insertion_status_topic: str
    joint_state_topic: str
    target_joint_state_topic: str
    target_tcp_state_topic: str
    ros_bag_folder_path: str
    enable_recording: bool
    seed_state_for_ik_solver_for_joint_space_planner: List[float]
    use_joint_space_planner: bool
    peg_stand_mesh_file_path: str
    gear_large_mesh_file_path: str
    gear_small_mesh_file_path: str
    gear_medium_mesh_file_path: str
    gear_assembly_use_ground_truth_pose_in_sim: bool
    verify_pose_estimation_accuracy: bool
    run_rl_inference: bool
    output_dir: str
    offset_for_place_pose: float
    offset_for_insertion_pose: float
    timeout_for_insertion_action_call: float
    model_frequency: float

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.model_path = get_str_variable(context, 'gear_assembly_model_path')
        self.model_file_name = get_str_variable(context, 'gear_assembly_model_file_name')
        self.policy_alpha = get_float_variable(context, 'gear_assembly_policy_alpha')

        # Publishers and subscriper topic names
        self.observation_topic = get_str_variable(context,
                                                  'gear_assembly_observation_topic')
        self.joint_state_topic = get_str_variable(context,
                                                  'gear_assembly_joint_state_topic')
        self.insertion_request_topic = get_str_variable(
            context, 'gear_assembly_gear_insertion_request_topic')
        self.insertion_status_topic = get_str_variable(
            context, 'gear_assembly_gear_insertion_status_topic')
        self.target_joint_state_topic = get_str_variable(
            context, 'gear_assembly_target_joint_state_topic')
        self.target_tcp_state_topic = get_str_variable(
            context, 'gear_assembly_target_tcp_state_topic')
        self.ros_bag_folder_path = get_str_variable(
            context, 'gear_assembly_ros_bag_folder_path')
        self.goal_pose_topic = get_str_variable(
            context, 'gear_assembly_goal_pose_topic')
        self.enable_recording = get_bool_variable(context, 'gear_assembly_enable_recording')

        # Variables for gear assembly orchestration
        self.use_joint_space_planner = get_bool_variable(
            context, 'gear_assembly_use_joint_state_planner')
        self.peg_stand_mesh_file_path = get_str_variable(
            context, 'gear_assembly_peg_stand_mesh_file_path')
        self.gear_large_mesh_file_path = get_str_variable(
            context, 'gear_assembly_gear_large_mesh_file_path')
        self.gear_small_mesh_file_path = get_str_variable(
            context, 'gear_assembly_gear_small_mesh_file_path')
        self.gear_medium_mesh_file_path = get_str_variable(
            context, 'gear_assembly_gear_medium_mesh_file_path')
        self.gear_assembly_use_ground_truth_pose_in_sim = get_bool_variable(
            context, 'gear_assembly_use_ground_truth_pose_in_sim')
        self.verify_pose_estimation_accuracy = get_bool_variable(
            context, 'gear_assembly_verify_pose_estimation_accuracy')
        self.run_rl_inference = get_bool_variable(
            context, 'gear_assembly_run_rl_inference')
        self.output_dir = get_str_variable(context, 'gear_assembly_output_dir')
        self.offset_for_place_pose = get_float_variable(
            context, 'gear_assembly_offset_for_place_pose')
        self.offset_for_insertion_pose = get_float_variable(
            context, 'gear_assembly_offset_for_insertion_pose')
        self.timeout_for_insertion_action_call = get_float_variable(
            context, 'gear_assembly_timeout_for_insertion_action_call')
        self.model_frequency = get_float_variable(context, 'gear_assembly_model_frequency')

        # Get the home target position from the policy.
        file_path_for_env_yaml = os.path.join(
            self.model_path, 'params', 'env.yaml')
        # Load env file without safe loading.
        with open(file_path_for_env_yaml, 'r') as file:
            env_yaml = yaml.load(file, Loader=yaml.UnsafeLoader)

        # Get the home target position from the policy.
        if 'initial_joint_pos' in env_yaml:
            home_target_position = env_yaml['initial_joint_pos']
        else:
            raise ValueError(f'initial_joint_pos not found in {file_path_for_env_yaml}')

        added_size = 0
        if self.use_sim_time:
            added_size = 1

        joint_state = JointState()

        joint_state.position = home_target_position
        joint_state.velocity = [0.0] * (len(home_target_position) + added_size)
        joint_state.effort = [0.0] * (len(home_target_position) + added_size)

        # Set joint names based on the number of joints (assuming UR10e joint names)
        joint_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        if self.use_sim_time:
            joint_state.position.append(0.0)
            joint_state.name.append('finger_joint')

        self.home_joint_state = joint_state

        self.seed_state_for_ik_solver_for_joint_space_planner = home_target_position


class DepthEstimationConfig(Config):
    """Config that tracks all variables needed to perform depth estimation."""

    depth_type: DepthType
    enable_dnn_depth_in_realsense: bool

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.depth_type = get_str_variable(context, 'depth_type')
        self.enable_dnn_depth_in_realsense = get_bool_variable(
            context, 'enable_dnn_depth_in_realsense')


class CuMotionConfig(Config):
    """Config that tracks all variables needed to configure cuMotion."""

    cumotion_urdf_file_path: str
    cumotion_xrdf_file_path: str
    distance_threshold: float
    num_cameras: int
    moveit_collision_objects_scene_file: str

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.num_cameras = int(get_str_variable(context, 'num_cameras'))
        self.cumotion_urdf_file_path = get_str_variable(context, 'cumotion_urdf_file_path')
        self.cumotion_xrdf_file_path = get_str_variable(context, 'cumotion_xrdf_file_path')
        self.distance_threshold = get_str_variable(context, 'distance_threshold')
        self.moveit_collision_objects_scene_file = (
            get_str_variable(context, 'moveit_collision_objects_scene_file')
        )


class CoreConfig(Config):
    """Config that tracks all variables needed to perform core workflows."""

    camera_config: CameraConfig
    rviz_config_file: str
    object_selection_config: ObjectSelectionConfig
    pose_estimation_config: PoseEstimationConfig
    workflow_config: WorkflowConfigParams
    cumotion_config: CuMotionConfig
    # cuMotion and object attachment params
    time_sync_slop: str
    filter_depth_buffer_time: str
    # nvblox configs
    setup: str
    enable_nvblox: bool
    # Depth estimation params
    depth_estimation_config: DepthEstimationConfig
    # Visualization
    enable_rviz_visualization: bool
    enable_foxglove_visualization: bool
    # Nsight Profiling settings
    delay_to_start_nsight: str
    nsight_profile_duration: str
    nsight_profile_output_file_path: str
    enable_nsight_profiling: bool
    enable_system_wide_profiling: bool
    # MPS parameters
    enable_cuda_mps: bool
    cuda_mps_pipe_directory: str
    cuda_mps_active_thread_percentage_container: str
    cuda_mps_client_priority_container: str
    cuda_mps_client_priority_robot_segmenter: str
    cuda_mps_active_thread_percentage_robot_segmenter: str
    # Logging level
    log_level: str = 'error'
    trigger_aabb_object_clearing: bool = True

    def __init__(self, context: LaunchContext):
        super().__init__(context)
        self.camera_type = get_camera_type(get_str_variable(context, 'camera_type'))
        self.rviz_config_file = get_str_variable(context, 'rviz_config_file')
        self.time_sync_slop = get_str_variable(context, 'time_sync_slop')
        self.filter_depth_buffer_time = get_str_variable(context, 'filter_depth_buffer_time')
        self.depth_type = get_str_variable(context, 'depth_type')
        self.use_sim_time = get_bool_variable(context, 'use_sim_time')
        self.num_cameras = int(get_str_variable(context, 'num_cameras'))
        self.setup = get_str_variable(context, 'setup')
        self.enable_nvblox = get_bool_variable(context, 'enable_nvblox')
        self.pose_estimation_type = get_pose_estimation_type(
            get_str_variable(context, 'pose_estimation_type'))
        self.delay_to_start_nsight = get_str_variable(context, 'delay_to_start_nsight')
        self.nsight_profile_duration = get_str_variable(context, 'nsight_profile_duration')
        self.nsight_profile_output_file_path = get_str_variable(
            context, 'nsight_profile_output_file_path')
        self.enable_nsight_profiling = get_bool_variable(
            context, 'enable_nsight_profiling')
        self.enable_system_wide_profiling = get_bool_variable(
            context, 'enable_system_wide_profiling')

        self.workflow_type = get_workflow_type(get_str_variable(context, 'workflow_type'))

        self.trigger_aabb_object_clearing = get_bool_variable(
            context, 'trigger_aabb_object_clearing'
        )
        self.enable_rviz_visualization = get_bool_variable(
            context, 'enable_rviz_visualization'
        )
        self.enable_foxglove_visualization = get_bool_variable(
            context, 'enable_foxglove_visualization'
        )
        # MPS parameters
        self.enable_cuda_mps = get_bool_variable(context, 'enable_cuda_mps')
        self.cuda_mps_pipe_directory = get_str_variable(context, 'cuda_mps_pipe_directory')
        self.cuda_mps_active_thread_percentage_container = get_str_variable(
            context, 'cuda_mps_active_thread_percentage_container'
        )
        self.cuda_mps_client_priority_container = get_str_variable(
            context, 'cuda_mps_client_priority_container'
        )
        self.cuda_mps_client_priority_robot_segmenter = get_str_variable(
            context, 'cuda_mps_client_priority_robot_segmenter'
        )
        self.cuda_mps_active_thread_percentage_robot_segmenter = get_str_variable(
            context, 'cuda_mps_active_thread_percentage_robot_segmenter'
        )
        self.cuda_mps_client_priority_planner = get_str_variable(
            context, 'cuda_mps_client_priority_planner'
        )
        self.cuda_mps_active_thread_percentage_planner = get_str_variable(
            context, 'cuda_mps_active_thread_percentage_planner'
        )

        self.depth_estimation_config = DepthEstimationConfig(context)
        if self.camera_type == CameraType.ISAAC_SIM and self.num_cameras != 1:
            raise NotImplementedError(f'Camera type of isaac sim only support 1 camera'
                                      f', not {self.num_cameras}')
        self.camera_config = get_camera_config(context, self.camera_type)

        self.object_selection_config = ObjectSelectionConfig(context)

        if self.pose_estimation_type == PoseEstimationType.DOPE:
            if self.workflow_type in (
                WorkflowType.PICK_AND_PLACE,
                WorkflowType.GEAR_ASSEMBLY
            ):
                raise NotImplementedError(
                    'Dope is not supported for pick and place or gear assembly')
            self.pose_estimation_config = DopeConfig(context)
        elif self.pose_estimation_type == PoseEstimationType.FOUNDATION_POSE:
            self.pose_estimation_config = FoundationPoseConfig(context)
            # Now update the topic names for segmentation and object detection configs.
            self.pose_estimation_config.object_detection_config.update_server_topic_names(
                self.camera_config)
            if self.pose_estimation_config.segmentation_type != SegmentationType.NONE:
                self.pose_estimation_config.segmentation_config.update_server_topic_names(
                    self.camera_config)
            self.pose_estimation_config.update_server_topic_names(self.camera_config)
        else:
            raise NotImplementedError(f'Pose estimation type {self.pose_estimation_type} '
                                      f'not supported')

        if self.workflow_type == WorkflowType.POSE_TO_POSE:
            self.workflow_config = PoseToPoseConfig(context)
        elif self.workflow_type == WorkflowType.OBJECT_FOLLOWING:
            self.workflow_config = ObjectFollowingConfig(context)
        elif self.workflow_type == WorkflowType.PICK_AND_PLACE:
            self.workflow_config = PickAndPlaceConfig(context)
        elif self.workflow_type == WorkflowType.GEAR_ASSEMBLY:
            self.workflow_config = GearAssemblyConfig(context)
        else:
            raise NotImplementedError(f'Workflow type not supported {self.workflow_type} !')

        object_detection_type = self.pose_estimation_config.object_detection_type

        if self.workflow_type == WorkflowType.GEAR_ASSEMBLY:
            if object_detection_type != ObjectDetectionType.SEGMENT_ANYTHING:
                raise ValueError(
                    'Object detection type != SEGMENT_ANYTHING is not supported for gear assembly')

        self.cumotion_config = CuMotionConfig(context)


@dataclass
class SupportedObject:
    """Configuration for individual supported object."""

    class_id: str
    grasp_file_path: str
    mesh_file_path: str
    object_name: str

    @classmethod
    def from_dict(cls, object_name: str, params: Dict[str, Any]) -> 'SupportedObject':
        return cls(
            class_id=params.get('class_id', ''),
            grasp_file_path=params.get('grasp_file_path', ''),
            mesh_file_path=params.get('mesh_file_path', ''),
            object_name=object_name
        )


@dataclass
class SupportedObjectsConfig:
    """Configuration for supported objects containing multiple object types."""

    supported_objects: Dict[str, SupportedObject]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'SupportedObjectsConfig':
        supported_objects = {}

        for object_name, object_data in params.items():
            if not isinstance(object_data, dict):
                continue

            supported_object = SupportedObject.from_dict(object_name, object_data)
            supported_objects[supported_object.class_id] = supported_object

        return cls(
            supported_objects=supported_objects,
        )

    def get_object_by_class_id(self, class_id: str) -> Optional[SupportedObject]:
        """Get supported object configuration by class ID."""
        return self.supported_objects.get(class_id, None)
