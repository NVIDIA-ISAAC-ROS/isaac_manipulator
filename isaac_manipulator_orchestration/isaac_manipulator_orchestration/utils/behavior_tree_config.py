# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_orchestration.utils.params_loader import (
    AssignObjectNameConfig,
    CloseGripperConfig,
    DetectObjectConfig,
    ExecuteTrajectoryConfig,
    InteractiveMarkerConfig,
    MeshAssignerConfig,
    MultiObjPickPlaceConfig,
    ObjectAttachDetachConfig,
    ObjectSelectorConfig,
    OpenGripperConfig,
    PlanToGraspConfig,
    PlanToPoseConfig,
    PoseEstimationConfig,
    PublishStaticPlanningSceneConfig,
    ReadGraspPosesConfig,
    RetryConfig,
    ServerTimeoutConfig,
    StaleDetectionConfig,
    SwitchControllersConfig,
    WorkspaceBoundsConfig,
)


class BehaviorTreeConfigInitializer:
    """Initialize behavior tree configurations from configuration files."""

    def __init__(
        self,
        behavior_tree_params_file: str,
        package_name: str
    ):
        """
        Initialize the behavior tree configuration initializer.

        Args:
        ----
        behavior_tree_params_file (str): Path to the behavior tree parameters YAML file.
            Expected to contain all necessary behavior configuration parameters.
        package_name (str): Name of the package to load parameters from.

        """
        self.config = MultiObjPickPlaceConfig(behavior_tree_params_file, package_name)

    def get_plan_to_grasp_config(self) -> PlanToGraspConfig:
        """Get plan_to_grasp configuration."""
        return self.config.plan_to_grasp

    def get_plan_to_pose_config(self) -> PlanToPoseConfig:
        """Get plan_to_pose configuration."""
        return self.config.plan_to_pose

    def get_read_grasp_poses_config(self) -> ReadGraspPosesConfig:
        """Get read_grasp_poses configuration."""
        return self.config.read_grasp_poses

    def get_close_gripper_config(self) -> CloseGripperConfig:
        """Get close_gripper configuration."""
        return self.config.close_gripper

    def get_open_gripper_config(self) -> OpenGripperConfig:
        """Get open_gripper configuration."""
        return self.config.open_gripper

    def get_attach_object_config(self) -> ObjectAttachDetachConfig:
        """Get attach_object configuration."""
        return self.config.attach_object

    def get_detach_object_config(self) -> ObjectAttachDetachConfig:
        """Get detach_object configuration."""
        return self.config.detach_object

    def get_execute_trajectory_config(self) -> ExecuteTrajectoryConfig:
        """Get execute_trajectory configuration."""
        return self.config.execute_trajectory

    def get_detect_object_config(self) -> DetectObjectConfig:
        """Get detect_object configuration."""
        return self.config.detect_object

    def get_stale_detection_config(self) -> StaleDetectionConfig:
        """Get stale_detection configuration."""
        return self.config.stale_detection

    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration."""
        return self.config.retry_config

    def get_mesh_assigner_config(self) -> MeshAssignerConfig:
        """Get mesh_assigner configuration."""
        return self.config.mesh_assigner

    def get_assign_object_name_config(self) -> AssignObjectNameConfig:
        """Get assign_object_name configuration."""
        return self.config.assign_object_name

    def get_publish_static_planning_scene_config(self) -> PublishStaticPlanningSceneConfig:
        """Get publish_static_planning_scene configuration."""
        return self.config.publish_static_planning_scene

    def get_object_selector_config(self) -> ObjectSelectorConfig:
        """Get object_selector configuration."""
        return self.config.object_selector

    def get_pose_estimation_config(self) -> PoseEstimationConfig:
        """Get pose_estimation configuration."""
        return self.config.pose_estimation

    def get_arm_controllers_config(self) -> SwitchControllersConfig:
        """Get arm controllers configuration."""
        return self.config.arm_controllers

    def get_tool_controllers_config(self) -> SwitchControllersConfig:
        """Get tool controllers configuration."""
        return self.config.tool_controllers

    def get_interactive_marker_config(self) -> InteractiveMarkerConfig:
        """Get interactive_marker configuration."""
        return self.config.interactive_marker

    def get_server_timeout_config(self) -> ServerTimeoutConfig:
        """Get server_timeout configuration."""
        return self.config.server_timeout_config

    def get_workspace_bounds_config(self) -> WorkspaceBoundsConfig:
        """
        Get workspace bounds configuration for detection filtering.

        Gets workspace bounds from pose_estimation config.
        """
        return self.config.pose_estimation.workspace_bounds
