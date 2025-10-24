# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_interfaces.action import GetSelectedObject
from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Point2D,
    Pose2D
)


class ObjectSelector(BaseActionBehavior):
    """
    Select objects from object_info_cache that have NOT_READY status.

    This behavior creates a detection array from objects in the object_info_cache
    that have NOT_READY status and sends them to a selection service. The service
    returns the selected object, which is then marked as SELECTED in the cache.
    This allows for object selection for the detected objects.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to

    """

    def __init__(self, name: str, action_server_name: str):
        blackboard_keys = {
            'object_info_cache': py_trees.common.Access.WRITE,
            'selected_object_id': py_trees.common.Access.WRITE
        }
        super().__init__(
            name=name,
            action_type=GetSelectedObject,
            action_server_name=action_server_name,
            blackboard_keys=blackboard_keys
        )

    def update(self):
        """
        Drive the object selection behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when objects are selected successfully,
            FAILURE when selection fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Check if we have valid object information
        if self.blackboard.object_info_cache is None:
            self.node.get_logger().error('Object info cache is empty')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the object selection process
            return self._trigger_select_object()

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the selection to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the selection result
            return self._process_selection_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _has_selectable_objects(self):
        """
        Check if there are any objects available for selection.

        Returns
        -------
        bool
            True if there are objects with NOT_READY status, False otherwise

        """
        for obj_info in self.blackboard.object_info_cache.values():
            if obj_info['status'] == ObjectStatus.NOT_READY.value:
                return True
        return False

    def _trigger_select_object(self):
        """
        Trigger the get selected object action with comprehensive Detection2D data.

        Returns
        -------
        py_trees.common.Status
            RUNNING if no objects are available or if the action was successfully triggered,
            allowing the behavior to continue trying

        """
        # Early check: are there any objects to select from?
        if not self._has_selectable_objects():
            return py_trees.common.Status.RUNNING

        get_selected_object_goal = GetSelectedObject.Goal()

        # Convert object_info_cache to Detection2DArray for the goal
        detection_array = Detection2DArray()

        # Set header for Detection2DArray with current timestamp
        current_time = self.node.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera'  # or appropriate frame
        detection_array.header.stamp = current_time

        detections_list = []

        for obj_id, obj_info in self.blackboard.object_info_cache.items():
            if obj_info['status'] == ObjectStatus.NOT_READY.value:
                # Create Detection2D message with comprehensive information
                detection_2d = Detection2D()

                # Set header for Detection2D with same timestamp
                detection_2d.header.frame_id = obj_info.get(
                    'object_frame_name', 'camera')
                detection_2d.header.stamp = current_time

                # Set the id field to the object_id
                detection_2d.id = str(obj_id)

                # Create comprehensive BoundingBox2D from stored bbox data
                bbox = BoundingBox2D()
                if 'bbox' in obj_info and obj_info['bbox'] is not None:
                    x_min, y_min, x_max, y_max = obj_info['bbox']

                    # Calculate center and size
                    center = Pose2D()
                    center.position = Point2D()
                    center.position.x = (x_min + x_max) / 2.0
                    center.position.y = (y_min + y_max) / 2.0
                    center.theta = 0.0  # 2D rotation, typically 0 for object detection

                    bbox.center = center
                    bbox.size_x = x_max - x_min
                    bbox.size_y = y_max - y_min

                    detection_2d.bbox = bbox

                # Create comprehensive ObjectHypothesisWithPose with classification information
                results_list = []

                # Always create at least one hypothesis entry
                hypothesis = ObjectHypothesisWithPose()

                # Set classification information if available
                if ('class_id' in obj_info and obj_info['class_id'] is not None and
                        'class_id_confidence' in obj_info and
                        obj_info['class_id_confidence'] is not None):
                    hypothesis.hypothesis = ObjectHypothesis()
                    hypothesis.hypothesis.class_id = obj_info['class_id']
                    hypothesis.hypothesis.score = obj_info['class_id_confidence']
                else:
                    # Create a default hypothesis if no classification data
                    hypothesis.hypothesis = ObjectHypothesis()
                    hypothesis.hypothesis.class_id = 'unknown'
                    hypothesis.hypothesis.score = 0.0

                results_list.append(hypothesis)
                detection_2d.results = results_list

                detections_list.append(detection_2d)

        detection_array.detections = detections_list
        get_selected_object_goal.detections = detection_array

        self.node.get_logger().info(
            f'[{self.name}] Sending {len(detections_list)} objects to selection server')
        self.send_goal(get_selected_object_goal)
        # Return RUNNING to indicate the action was triggered and behavior should continue
        return py_trees.common.Status.RUNNING

    def _process_selection_result(self):
        """
        Process the object selection results.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if objects were selected and updated,
            FAILURE otherwise

        """
        selected_detection = self.get_action_result().selected_detection

        # Check if selected_detection is None
        if selected_detection is None:
            self.node.get_logger().error(
                'No object was selected - selected_detection is None')
            return py_trees.common.Status.FAILURE

        # Extract the object_id from the detection's id field
        try:
            selected_object_id = int(selected_detection.id)
        except (ValueError, AttributeError):
            self.node.get_logger().error(
                'Cannot extract object ID from selected detection id field')
            return py_trees.common.Status.FAILURE

        # Verify the selected ID is valid
        if (self.blackboard.object_info_cache is None or
                selected_object_id not in self.blackboard.object_info_cache):
            self.node.get_logger().error(
                f'Selected object ID {selected_object_id} not in cache')
            return py_trees.common.Status.FAILURE

        self.blackboard.selected_object_id = selected_object_id
        obj_status = ObjectStatus.SELECTED.value
        self.blackboard.object_info_cache[selected_object_id]['status'] = obj_status

        self.node.get_logger().info(f'[{self.name}] Selected object_id={selected_object_id}')

        return py_trees.common.Status.SUCCESS
