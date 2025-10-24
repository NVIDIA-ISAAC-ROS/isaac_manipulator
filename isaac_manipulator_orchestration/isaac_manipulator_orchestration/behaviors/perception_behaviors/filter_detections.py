# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import py_trees


class FilterDetections(py_trees.behaviour.Behaviour):
    """
    Filter detected objects based on supported class IDs.

    This behavior reads the object_info_cache from the blackboard and filters
    out any detections whose class_ids are not in the list of supported objects.
    The supported class IDs are read from the supported_class_ids blackboard variable,
    which contains the set of class IDs for all objects defined in the supported_objects
    configuration.

    The behavior ensures only objects that the system knows how to handle
    (i.e., those with defined configurations) are kept in the object_info_cache.

    Parameters
    ----------
    name : str
        Name of the behavior

    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()

        # Register blackboard keys
        self.blackboard.register_key(
            key='object_info_cache',
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key='supported_class_ids',
            access=py_trees.common.Access.READ
        )

    def setup(self, **kwargs):
        """
        Set up the behavior by storing the ROS node reference.

        This is called once when the behavior tree is constructed.
        """
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e

        return True

    def _check_blackboard_data(self):
        """
        Check that required blackboard data is available for filtering.

        Returns
        -------
        py_trees.common.Status or None
            Status to return if validation failed, None if validation passed

        """
        # Check if object_info_cache exists and has data
        if self.blackboard.object_info_cache is None:
            self.node.get_logger().error(
                f'[{self.name}] object_info_cache not available on blackboard')
            return py_trees.common.Status.FAILURE

        # Check if supported_class_ids exists
        if (not self.blackboard.exists('supported_class_ids') or
                self.blackboard.supported_class_ids is None):
            self.node.get_logger().error(
                f'[{self.name}] supported_class_ids not available on blackboard')
            return py_trees.common.Status.FAILURE

        # Early return if cache is empty - nothing to filter
        if not self.blackboard.object_info_cache:
            self.node.get_logger().info(
                f'[{self.name}] object_info_cache is empty - nothing to filter')
            return py_trees.common.Status.SUCCESS

        # Check if supported_class_ids is empty
        if not self.blackboard.supported_class_ids:
            self.node.get_logger().error(
                f'[{self.name}] No supported class IDs found in configuration')
            return py_trees.common.Status.FAILURE

        return None

    def _filter_objects(self, supported_class_ids):
        """
        Filter objects based on supported class IDs.

        Parameters
        ----------
        supported_class_ids : set
            Set of supported class IDs

        Returns
        -------
        list
            List of (obj_id, reason) tuples for objects that were removed

        """
        removed_objects = []

        for obj_id in list(self.blackboard.object_info_cache.keys()):
            obj_info = self.blackboard.object_info_cache[obj_id]
            class_id = obj_info.get('class_id')

            if class_id is None:
                removed_objects.append((obj_id, 'no class_id'))
                del self.blackboard.object_info_cache[obj_id]
            elif str(class_id) in supported_class_ids:
                self.node.get_logger().debug(
                    f'[{self.name}] Keeping object {obj_id} with supported '
                    f'class_id {str(class_id)}')
            else:
                removed_objects.append((obj_id, f'unsupported class_id {str(class_id)}'))
                del self.blackboard.object_info_cache[obj_id]

        return removed_objects

    def _log_filtering_results(self, removed_objects, original_count,
                               filtered_count, supported_class_ids):
        """
        Log the results of the filtering operation.

        Parameters
        ----------
        removed_objects : list
            List of (obj_id, reason) tuples for removed objects
        original_count : int
            Number of objects before filtering
        filtered_count : int
            Number of objects after filtering
        supported_class_ids : set
            Set of supported class IDs

        """
        if removed_objects:
            self.node.get_logger().debug(
                f'[{self.name}] Filtered out {len(removed_objects)} unsupported objects:')
            for obj_id, reason in removed_objects:
                self.node.get_logger().debug(
                    f'[{self.name}]   - Object {obj_id}: {reason}')

        self.node.get_logger().info(
            f'[{self.name}] Filtered detections: {original_count} -> {filtered_count} objects')

    def update(self):
        """
        Filter the detected objects based on supported class IDs.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if filtering was successful (even if no objects remain),
            FAILURE if there was an error accessing required data

        """
        # Check blackboard data
        validation_status = self._check_blackboard_data()
        if validation_status is not None:
            return validation_status

        # Get supported class IDs
        supported_class_ids = self.blackboard.supported_class_ids
        original_count = len(self.blackboard.object_info_cache)

        # Filter objects (modifies object_info_cache in place)
        removed_objects = self._filter_objects(supported_class_ids)

        # Get filtered count after in-place filtering
        filtered_count = len(self.blackboard.object_info_cache)

        # Log results
        self._log_filtering_results(removed_objects, original_count,
                                    filtered_count, supported_class_ids)

        # Return SUCCESS even if no objects remain after filtering
        # This allows the behavior tree to continue and potentially re-detect
        return py_trees.common.Status.SUCCESS
