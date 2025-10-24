# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Blackboard setup utilities for behavior testing.

This module provides utilities for setting up test blackboards and creating
mock test data for behavior tree testing.
"""

from typing import Any, Dict, Optional

from isaac_manipulator_ros_python_utils.manipulator_types import ObjectStatus
import py_trees


def setup_test_blackboard(**blackboard_data) -> py_trees.blackboard.Client:
    """
    Set up a minimal blackboard for behavior testing.

    Args:
        **blackboard_data: Arbitrary blackboard keys and values to set

    Returns
    -------
    py_trees.blackboard.Client
        Configured blackboard client with provided keys registered

    """
    blackboard = py_trees.blackboard.Client(name='TestClient')

    # Set any keys provided
    for key, value in blackboard_data.items():
        blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
        setattr(blackboard, key, value)

    return blackboard


def create_test_object(obj_id: int, class_id: str = '22',
                       status: str = ObjectStatus.NOT_READY.value,
                       bbox: Optional[list] = None,
                       **additional_fields) -> Dict[str, Any]:
    """
    Create a single test object for object_info_cache.

    Args:
        obj_id: Object ID
        class_id: Object class ID
        status: Object status
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        **additional_fields: Additional object fields

    Returns
    -------
    Dict[str, Any]
        Dictionary representing a single object in object_info_cache

    """
    if bbox is None:
        bbox = [100, 50, 200, 150]

    obj = {
        'class_id': class_id,
        'status': status,
        'bbox': bbox,
        'estimated_pose': None,
        'goal_drop_pose': None,
        'class_id_confidence': 0.9,
        'object_frame_name': f'object_{obj_id}',
        **additional_fields
    }

    return obj
