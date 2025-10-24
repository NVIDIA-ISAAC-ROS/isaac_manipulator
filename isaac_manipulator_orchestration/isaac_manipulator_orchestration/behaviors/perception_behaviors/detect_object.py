# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_interfaces.action import GetObjects
from isaac_manipulator_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_manipulator_orchestration.utils.status_types import BehaviorStatus
from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


class DetectObject(BaseActionBehavior):
    """
    Detect objects using an object detection action client.

    This behavior calls an object detection service to identify objects in
    the scene. It filters detections based on confidence thresholds and
    saves valid detection results to the blackboard as object_info_cache.

    Parameters
    ----------
    name : str
        Name of the behavior
    action_server_name : str
        Name of the action server to connect to
    detection_confidence_threshold : float
        Minimum confidence score required for a detection to be considered valid
        (typically between 0.0 and 1.0)

    """

    def __init__(self,
                 name: str,
                 action_server_name: str,
                 detection_confidence_threshold: float):
        blackboard_keys = {
            'object_info_cache': py_trees.common.Access.WRITE
        }
        super().__init__(
            name=name,
            action_type=GetObjects,
            action_server_name=action_server_name,
            blackboard_keys=blackboard_keys
        )
        self.detection_confidence_threshold = detection_confidence_threshold

    def setup(self, **kwargs):
        """
        Set up the behavior by creating the action client.

        This is called once when the behavior tree is constructed.
        """
        super().setup(**kwargs)

        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e

    def update(self):
        """
        Drive the object detection behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when objects are detected successfully,
            FAILURE when detection fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the object detection process
            self._trigger_get_objects()
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the detection to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process detection results
            return self._process_detection_results()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_get_objects(self):
        """Trigger the get objects action."""
        get_objects_goal = GetObjects.Goal()
        self.send_goal(get_objects_goal)

    def _process_detection_results(self):
        """
        Process the detection results and save to blackboard.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if objects were found and saved,
            FAILURE otherwise

        """
        object_info = {}

        for obj in self.get_action_result().objects:
            # Only process objects that have valid detection_2d data with bounding box
            if obj.detection_2d and obj.detection_2d.bbox:
                bbox = obj.detection_2d.bbox
                # Calculate bounding box corners for better readability
                x_min = bbox.center.position.x - bbox.size_x/2
                y_min = bbox.center.position.y - bbox.size_y/2
                x_max = bbox.center.position.x + bbox.size_x/2
                y_max = bbox.center.position.y + bbox.size_y/2

                # Find the class ID with the highest score
                best_class_id = None
                best_score = -1.0

                if hasattr(obj.detection_2d, 'results') and obj.detection_2d.results:
                    for result in obj.detection_2d.results:
                        if hasattr(result, 'hypothesis') and result.hypothesis:
                            if result.hypothesis.score > best_score:
                                best_score = result.hypothesis.score
                                best_class_id = result.hypothesis.class_id

                if (best_score >= self.detection_confidence_threshold and
                        best_class_id is not None):
                    object_info[obj.object_id] = {
                        'bbox': [x_min, y_min, x_max, y_max],
                        'estimated_pose': None,  # Populated by pose_estimation_behavior.py
                        'goal_drop_pose': None,  # Populated by update_drop_pose.py
                        'status': ObjectStatus.NOT_READY.value,
                        'class_id': best_class_id,
                        'class_id_confidence': best_score if best_score > -1.0 else None,
                        'object_frame_name': f'object_{obj.object_id}'
                    }
                else:
                    self.node.get_logger().warning(
                        f'Skipping object ID {obj.object_id} due to low confidence '
                        f'score {best_score}')
            else:
                # Log when objects are skipped due to missing detection data
                self.node.get_logger().warning(
                    f'Skipping object ID {obj.object_id} due to missing detection_2d data')

        # Check if we have any valid objects to work with
        if not object_info:
            self.node.get_logger().error('No valid objects with detection data found')
            return py_trees.common.Status.FAILURE

        self.blackboard.object_info_cache = object_info

        # Log detected objects
        for obj_id, obj_data in object_info.items():
            self.node.get_logger().info(
                f'[{self.name}] Detected object_id={obj_id} '
                f'(class_id={obj_data["class_id"]}) with '
                f'frame_name={obj_data["object_frame_name"]}')

        return py_trees.common.Status.SUCCESS
