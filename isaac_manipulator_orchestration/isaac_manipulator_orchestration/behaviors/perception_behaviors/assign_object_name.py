# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_interfaces.srv import AssignNameToObject
from isaac_manipulator_orchestration.behaviors.base_service import BaseServiceBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
import py_trees


class AssignObjectName(BaseServiceBehavior):
    """
    Assign a name to the selected object for pose estimation.

    This behavior updates the object frame name that will be used by Foundation Pose
    for TF frame publishing. It assigns a name to the selected object by calling the
    AssignNameToObject service on the ObjectInfoServer. The assigned name is cached
    in the object info and later retrieved during pose estimation to set the correct
    object_frame_name parameter for Foundation Pose, ensuring poses are published
    in the appropriate coordinate frame.

    Parameters
    ----------
    name : str
        Name of the behavior
    service_name : str
        Name of the service to connect to.

    """

    def __init__(self, name: str, service_name: str):
        blackboard_keys = {
            'selected_object_id': py_trees.common.Access.READ,
            'object_info_cache': py_trees.common.Access.READ
        }
        super().__init__(
            name=name,
            service_type=AssignNameToObject,
            service_name=service_name,
            blackboard_keys=blackboard_keys
        )

    def update(self):
        """
        Drive the frame name update behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when frame name is updated,
            FAILURE when update fails,
            RUNNING if waiting for service response

        """
        # First, check for server availability and service call failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_service_state() == BehaviorStatus.IDLE:
            # Start the object name assignment process
            return self._trigger_assign_object_name()

        elif self.get_service_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the service call to complete
            return py_trees.common.Status.RUNNING

        elif self.get_service_state() == BehaviorStatus.SUCCEEDED:
            # Process the service response
            return self._process_response()

        # This should not happen since we're handling all states
        self.logger.warning(
            f'Unexpected state in {self.name}: {self.get_service_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_assign_object_name(self):
        """
        Prepare and trigger the service call to assign the object name.

        Returns
        -------
        py_trees.common.Status
            RUNNING if the service call was initiated successfully,
            FAILURE otherwise

        """
        # Check if we have a selected object ID
        if self.blackboard.selected_object_id is None:
            self.node.get_logger().error(
                f'[{self.name}] No selected object ID found in blackboard')
            return py_trees.common.Status.FAILURE

        # Get the selected object ID and its info
        obj_id = self.blackboard.selected_object_id

        if (self.blackboard.object_info_cache is None or
                obj_id not in self.blackboard.object_info_cache):
            self.node.get_logger().error(
                f'[{self.name}] Selected object ID {obj_id} not found in object_info_cache')
            return py_trees.common.Status.FAILURE

        # Get object frame name from object info
        obj_info = self.blackboard.object_info_cache[obj_id]
        frame_name = obj_info.get('object_frame_name')
        if not frame_name:
            self.node.get_logger().error(f'[{self.name}] No frame name found for object {obj_id}')
            return py_trees.common.Status.FAILURE

        # Create the service request
        request = AssignNameToObject.Request()
        request.object_id = obj_id
        request.name = frame_name

        # Call the service using the base class method
        if self.call_service(request):
            self.node.get_logger().info(
                f'[{self.name}] Requesting to assign name "{frame_name}" to object {obj_id}')
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

    def _process_response(self):
        """
        Process the service response.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the object name was assigned successfully,
            FAILURE otherwise

        """
        service_response = self.get_service_response()
        if not service_response.result:
            self.node.get_logger().error(
                f'[{self.name}] Failed to assign object name')
            return py_trees.common.Status.FAILURE

        # Get the object ID and frame name from the blackboard again to log the result
        obj_id = self.blackboard.selected_object_id
        frame_name = self.blackboard.object_info_cache[obj_id].get('object_frame_name')
        self.node.get_logger().info(
            f'[{self.name}] Successfully assigned name "{frame_name}" to object {obj_id}')

        return py_trees.common.Status.SUCCESS
