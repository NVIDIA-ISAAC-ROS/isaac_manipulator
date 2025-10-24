# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from controller_manager_msgs.srv import SwitchController
from isaac_manipulator_orchestration.behaviors.base_service import BaseServiceBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
import py_trees


class SwitchControllers(BaseServiceBehavior):
    """
    Switch controllers by calling the switch_controller service.

    It activates the controllers in the controllers_to_activate list and
    deactivates the controllers in the controllers_to_deactivate list.

    Parameters
    ----------
    name : str
        Name of the behavior
    controllers_to_activate : list[str]
        List of controller names to activate
    controllers_to_deactivate : list[str]
        List of controller names to deactivate
    strictness : int, optional
        Strictness level for controller switching. Defaults to STRICT (2).
        Available values:
        - BEST_EFFORT (1): Try best effort switching
        - STRICT (2): Fail if anything goes wrong
        - AUTO (3): Automatically resolve controller chain (ROS2 Jazzy onwards)
        - FORCE_AUTO (4): Auto-deactivate conflicting controllers (ROS2 Jazzy onwards)

    """

    def __init__(self,
                 name: str,
                 controllers_to_activate: list,
                 controllers_to_deactivate: list,
                 strictness: int = SwitchController.Request.FORCE_AUTO):
        super().__init__(
            name=name,
            service_type=SwitchController,
            service_name='/controller_manager/switch_controller'
        )
        self.controllers_to_activate = controllers_to_activate
        self.controllers_to_deactivate = controllers_to_deactivate
        self.strictness = strictness

    def update(self):
        """Update the behavior."""
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_service_state() == BehaviorStatus.IDLE:
            # Start the switch controllers process
            return self._trigger_switch_controllers()

        elif self.get_service_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the controllers to switch
            return py_trees.common.Status.RUNNING

        elif self.get_service_state() == BehaviorStatus.SUCCEEDED:
            # Process switch controllers results
            return self._process_switch_controllers_results()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_service_state()}')

        return py_trees.common.Status.FAILURE

    def _trigger_switch_controllers(self):
        """Trigger the switch controllers service."""
        # Validate controllers configuration
        if len(self.controllers_to_activate) == 0 and len(self.controllers_to_deactivate) == 0:
            self.node.get_logger().warning(
                'No controllers to activate or deactivate. '
                'Update the YAML file to specify the controllers.')
            return py_trees.common.Status.SUCCESS

        self.node.get_logger().info(
            f'Activating controllers [{self.controllers_to_activate}]'
            f' and deactivating controllers [{self.controllers_to_deactivate}]'
            f' with strictness {self.strictness}')

        # Create a request message for the switch controllers service
        self.request = SwitchController.Request()
        self.request.activate_controllers = self.controllers_to_activate
        self.request.deactivate_controllers = self.controllers_to_deactivate
        self.request.strictness = self.strictness
        self.request.activate_asap = True
        self.call_service(self.request)
        return py_trees.common.Status.RUNNING

    def _process_switch_controllers_results(self):
        """Process the switch controllers results."""
        result = self.get_service_response()
        if result is None:
            self.node.get_logger().error(
                f'[{self.name}] No response from switch controllers service')
            return py_trees.common.Status.FAILURE

        if result.ok:
            self.node.get_logger().info(
                f'[{self.name}] Successfully switched controllers: {result.message}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] Failed to switch controllers: {result.message}')
            return py_trees.common.Status.FAILURE
