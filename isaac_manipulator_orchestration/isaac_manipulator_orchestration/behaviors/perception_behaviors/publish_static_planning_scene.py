# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from isaac_manipulator_orchestration.behaviors.base_service import BaseServiceBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene
import py_trees


class PublishStaticPlanningSceneBehavior(BaseServiceBehavior):
    """
    Publish static planning scene using the publish static planning scene service.

    This behavior calls the publish static planning scene service to publish
    collision objects to the planning scene. It is used within a one-shot decorator to
    ensure it only executes once.

    Parameters
    ----------
    name : str
        Name of the behavior
    service_name : str
        Name of the service to connect to
    scene_file_path : str
        Path to the MoveIt collision objects scene file.

    """

    def __init__(self, name: str, service_name: str, scene_file_path: str):
        super().__init__(
            name=name,
            service_type=PublishStaticPlanningScene,
            service_name=service_name
        )
        self.scene_file_path = scene_file_path

    def update(self):
        """
        Update the behavior state and handle service calls.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when static scene published successfully or no scene file provided,
            FAILURE when service call failed or server unavailable,
            RUNNING while service call is in progress

        """
        # First check if the server is available
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Handle the state machine for this specific behavior
        if self.get_service_state() == BehaviorStatus.IDLE:
            # Start the publish static planning scene process
            return self._trigger_publish_static_planning_scene()

        elif self.get_service_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the service call to complete
            return py_trees.common.Status.RUNNING

        elif self.get_service_state() == BehaviorStatus.SUCCEEDED:
            # Process the successful response
            return self._process_response()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_service_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_publish_static_planning_scene(self):
        """
        Prepare and trigger the service call to publish static planning scene.

        Returns
        -------
        py_trees.common.Status
            RUNNING if the service call was initiated successfully,
            FAILURE otherwise

        """
        # Create the service request
        request = PublishStaticPlanningScene.Request()
        request.scene_file_path = self.scene_file_path if self.scene_file_path else ''

        # Call the service using the base class method
        if self.call_service(request):
            if self.scene_file_path:
                scene_path_msg = f' with scene file: {self.scene_file_path}'
            else:
                scene_path_msg = (' with moveit_collision_objects_scene_file from '
                                  'workflow parameters file')
            self.node.get_logger().info(
                f'Requesting to publish static planning scene via '
                f'{self._service_name}{scene_path_msg}')
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

    def _process_response(self):
        """
        Process the service response.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if static planning scene was published successfully or no scene file provided,
            FAILURE otherwise

        """
        service_response = self.get_service_response()
        if not service_response.success:
            if service_response.status == 1:
                self.node.get_logger().warning(
                    'No static planning scene file provided - Absence of user-defined'
                    'workspace boundaries could potential lead to collision risk with'
                    'objects outside camera field of view.')
                return py_trees.common.Status.SUCCESS

            self.node.get_logger().error(
                f'Static planning scene service failed: {service_response.message}')
            return py_trees.common.Status.FAILURE

        self.node.get_logger().info(
            f'Successfully published static planning scene: {service_response.message}')
        return py_trees.common.Status.SUCCESS
