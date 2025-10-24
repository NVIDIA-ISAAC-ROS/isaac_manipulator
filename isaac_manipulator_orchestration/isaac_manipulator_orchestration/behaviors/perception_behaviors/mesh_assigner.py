# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from isaac_manipulator_interfaces.srv import AddMeshToObject
from isaac_manipulator_orchestration.behaviors.base_service import BaseServiceBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
import py_trees


class MeshAssigner(BaseServiceBehavior):
    """
    Assign mesh files to selected objects based on their class IDs.

    This behavior looks up the mesh file path for a selected object using
    its class ID from the mesh_file_paths blackboard variable, then makes a service call
    to assign the mesh to the object.

    Parameters
    ----------
    name : str
        Name of the behavior
    service_name : str
        Name of the service to connect to

    """

    def __init__(self, name: str, service_name: str):
        blackboard_keys = {
            'selected_object_id': py_trees.common.Access.READ,
            'object_info_cache': py_trees.common.Access.READ,
            'mesh_file_paths': py_trees.common.Access.READ,
        }
        super().__init__(
            name=name,
            service_type=AddMeshToObject,
            service_name=service_name,
            blackboard_keys=blackboard_keys
        )

    def update(self):
        """
        Drive the mesh assignment behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when mesh is assigned successfully,
            FAILURE when assignment fails,
            RUNNING while the service call is in progress

        """
        # First, check for server availability and service call failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_service_state() == BehaviorStatus.IDLE:
            # Start the mesh assignment process
            return self._trigger_add_mesh()

        elif self.get_service_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the service call to complete
            return py_trees.common.Status.RUNNING

        elif self.get_service_state() == BehaviorStatus.SUCCEEDED:
            # Process the service response
            return self._process_response()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_service_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_add_mesh(self):
        """
        Trigger the add mesh service call.

        Returns
        -------
        py_trees.common.Status
            RUNNING if the service call was initiated successfully,
            FAILURE otherwise

        """
        # Check if we have a selected object ID
        if self.blackboard.selected_object_id is None:
            self.node.get_logger().error('No selected object ID found in blackboard')
            return py_trees.common.Status.FAILURE

        # Get the selected object ID and its info
        obj_id = self.blackboard.selected_object_id

        if (self.blackboard.object_info_cache is None or
                obj_id not in self.blackboard.object_info_cache):
            self.node.get_logger().error(
                f'Selected object ID {obj_id} not found in object_info_cache')
            return py_trees.common.Status.FAILURE

        # Get class ID from object info
        obj_info = self.blackboard.object_info_cache[obj_id]

        class_id = obj_info.get('class_id')

        if not class_id:
            self.node.get_logger().error(
                f'No valid class ID found for object {obj_id}')
            return py_trees.common.Status.FAILURE

        mesh_path = self.blackboard.mesh_file_paths.get(class_id)
        if mesh_path is None:
            self.node.get_logger().error(
                f'No mesh path found for class ID {class_id}')
            return py_trees.common.Status.FAILURE

        # Create the service request
        request = AddMeshToObject.Request()
        request.object_ids = [obj_id]
        request.mesh_file_paths = [mesh_path]

        # Call the service using the base class method
        if self.call_service(request):
            self.node.get_logger().info(
                f'Requesting to add mesh {mesh_path} to object {obj_id}')
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

    def _process_response(self):
        """
        Process the service response.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the mesh was assigned successfully,
            FAILURE otherwise

        """
        service_response = self.get_service_response()
        if not service_response.success:
            self.node.get_logger().error(
                f'[{self.name}] Failed to assign mesh to object: {service_response.message}')
            return py_trees.common.Status.FAILURE

        obj_id = self.blackboard.selected_object_id
        self.node.get_logger().info(
            f'[{self.name}] Successfully assigned mesh to object {obj_id}')
        return py_trees.common.Status.SUCCESS
