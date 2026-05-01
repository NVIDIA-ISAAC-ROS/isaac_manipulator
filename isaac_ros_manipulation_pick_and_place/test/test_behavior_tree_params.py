# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Default-value tests for the multi-object pick-and-place behavior tree parameters."""

import os

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml


PARAMS_FILE_NAME = 'multi_object_pick_and_place_behavior_tree_params.yaml'

# Minimum clearing region size (meters) per axis that will reliably clear
# nvblox voxel noise around a planning target. Typical nvblox voxel sizes are
# 2-5 cm, so a clearing region below 5 cm per axis often leaves ghost voxels
# that cause INVERSE_KINEMATICS_FAILURE. Users may raise these defaults for
# specific objects, but reducing below the minimum has been shown to
# reintroduce the bug.
MIN_CLEARING_SHAPE_SCALE_M = 0.05
MIN_CLEARING_PADDING_M = 0.02


@pytest.fixture(scope='module')
def pick_and_place_params():
    """Load the behavior tree parameters shipped with the package."""
    params_path = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_pick_and_place'),
        'params',
        PARAMS_FILE_NAME,
    )
    with open(params_path) as f:
        data = yaml.safe_load(f)
    return data['behavior_tree_params']['multi_object_pick_and_place']


def assert_aabb_clearing_block_is_valid(motion_plan_params, motion_plan_type):
    """Verify that a motion-plan request has a complete AABB-clearing configuration."""
    assert motion_plan_params['enable_aabb_clearing'] is True, (
        f'{motion_plan_type}.enable_aabb_clearing must default to True so cuMotion '
        f'clears ESDF voxels around the target before IK. When it is False, '
        f'ghost voxels from nvblox can make the target pose appear occupied '
        f'and IK will repeatedly report INVERSE_KINEMATICS_FAILURE.'
    )

    assert motion_plan_params['aabb_clearing_shape'] in ('SPHERE', 'CUBOID', 'CUSTOM_MESH'), (
        f'{motion_plan_type}.aabb_clearing_shape must be one of '
        f'SPHERE, CUBOID, or CUSTOM_MESH'
    )

    shape_scale = motion_plan_params['aabb_clearing_shape_scale']
    assert isinstance(shape_scale, list) and len(shape_scale) == 3, (
        f'{motion_plan_type}.aabb_clearing_shape_scale must be a 3-element list'
    )
    assert all(isinstance(v, (int, float)) and v > 0.0 for v in shape_scale), (
        f'{motion_plan_type}.aabb_clearing_shape_scale entries must be positive numbers'
    )

    padding = motion_plan_params['esdf_clearing_padding']
    assert isinstance(padding, list) and len(padding) == 3, (
        f'{motion_plan_type}.esdf_clearing_padding must be a 3-element list'
    )
    assert all(isinstance(v, (int, float)) and v >= 0.0 for v in padding), (
        f'{motion_plan_type}.esdf_clearing_padding entries must be non-negative numbers'
    )


def test_plan_to_grasp_has_aabb_clearing_enabled(pick_and_place_params):
    """Grasp planning must clear AABBs around the object before IK."""
    assert_aabb_clearing_block_is_valid(
        pick_and_place_params['plan_to_grasp'], 'plan_to_grasp')


def test_plan_to_pose_has_aabb_clearing_enabled(pick_and_place_params):
    """
    Verify plan_to_pose defaults enable AABB clearing around the target pose.

    Pose planning (used for the drop phase) must also clear AABBs around the
    target pose. Without this, nvblox voxels at the drop location survive
    into the ESDF that cuMotion uses for collision checking and IK fails.
    """
    assert_aabb_clearing_block_is_valid(
        pick_and_place_params['plan_to_pose'], 'plan_to_pose')


def assert_clearing_region_covers_voxel_noise(motion_plan_params, motion_plan_type):
    """Require shape_scale and padding defaults to cover typical voxel noise."""
    for value in motion_plan_params['aabb_clearing_shape_scale']:
        assert value >= MIN_CLEARING_SHAPE_SCALE_M, (
            f'{motion_plan_type}.aabb_clearing_shape_scale entry {value} is below '
            f'{MIN_CLEARING_SHAPE_SCALE_M} m. A region that small often '
            f'leaves ghost nvblox voxels at the target pose and IK fails '
            f'even though enable_aabb_clearing is True.'
        )
    for value in motion_plan_params['esdf_clearing_padding']:
        assert value >= MIN_CLEARING_PADDING_M, (
            f'{motion_plan_type}.esdf_clearing_padding entry {value} is below '
            f'{MIN_CLEARING_PADDING_M} m. Some padding is needed so the '
            f'cleared region extends past exact voxel boundaries.'
        )


def test_plan_to_grasp_clearing_region_covers_voxel_noise(pick_and_place_params):
    """Shape scale and padding defaults for plan_to_grasp must cover voxel noise."""
    assert_clearing_region_covers_voxel_noise(
        pick_and_place_params['plan_to_grasp'], 'plan_to_grasp')


def test_plan_to_pose_clearing_region_covers_voxel_noise(pick_and_place_params):
    """Shape scale and padding defaults for plan_to_pose must cover voxel noise."""
    assert_clearing_region_covers_voxel_noise(
        pick_and_place_params['plan_to_pose'], 'plan_to_pose')
