# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict, List, Tuple

import numpy as np
from sensor_msgs.msg import JointState


def find_closest_joint_state(target_joint_position: np.ndarray,
                             candidate_joint_state: np.ndarray) -> float:
    """
    Find the closest joint state to target positions.

    Args
    ----
        target_joint_position [dof]: Target joint position
        candidate_joint_state [dof]: Candidate joint state

    Returns
    -------
        Distance between the target and candidate joint states

    """
    assert len(target_joint_position) == len(candidate_joint_state), \
        f'Target joint position: {target_joint_position} and candidate joint' \
        f' state: {candidate_joint_state}have different lengths'

    target_position = np.array(target_joint_position)
    dist = np.linalg.norm(target_position - candidate_joint_state)

    return dist


def get_sorted_indexes_of_closest_joint_states(ik_possible_joint_states: List[JointState],
                                               target_joint_state: JointState,
                                               joint_limits: Dict[str, Tuple[float, float]]
                                               ) -> List[int]:
    """
    Get sorted indexes in order (closest to furthest) from the target joint state.

    Args
    ----
        ik_possible_joint_states: List of possible joint states
        target_joint_state: Target joint state
        joint_limits: Joint limits, the dict is key and the value is a tuple of (min, max)

    Returns
    -------
        List of indexes of the closest joint states

    """
    distances_and_indexes = []
    target_positions = np.array(target_joint_state.position)

    # Make sure joint limits have all the same keys as names and have a tuple of floats.
    assert set(joint_limits.keys()) == set(target_joint_state.name[:-1]), \
        f'Joint limits: {joint_limits} and target joint state: {target_joint_state}' \
        f'have different keys'

    for joint_name, joint_limit in joint_limits.items():
        assert isinstance(joint_limit, tuple) and len(joint_limit) == 2, \
            f'Joint limits for joint name: {joint_name} are not a tuple of length 2'

    for i, joint_state in enumerate(ik_possible_joint_states):
        # As cuMotion gives us an extra finger joint
        current_positions = np.array(joint_state.position)
        assert len(current_positions) == len(target_joint_state.position), \
            f'Current positions: {current_positions} and target positions' \
            f': {target_joint_state}' \
            f'have different lengths'

        distance = find_closest_joint_state(
            target_positions[:-1], current_positions[:-1])

        distances_and_indexes.append((distance, i))

    sorted_indexes = [index for _, index in sorted(distances_and_indexes)]
    return sorted_indexes
