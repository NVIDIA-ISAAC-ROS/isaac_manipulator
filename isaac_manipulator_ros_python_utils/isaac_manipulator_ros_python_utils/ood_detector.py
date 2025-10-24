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

from geometry_msgs.msg import Vector3


class OutOfDistributionDetector():
    """Class for detecting if policy inputs are out of distribution from training."""

    def __init__(self,
                 target_position_min: Vector3,
                 target_position_max: Vector3,
                 target_rotation_min: Vector3,
                 target_rotation_max: Vector3):
        self.target_position_min = target_position_min
        self.target_position_max = target_position_max
        self.target_rotation_min = target_rotation_min
        self.target_rotation_max = target_rotation_max

    def target_position_in_distribution(self, target_position: Vector3) -> bool:
        """
        Check if target position is in distribution.

        Args
        ----
            target_position (Vector3): Target position

        Returns
        -------
            bool: True if the target position is in distribution, False otherwise

        """
        if target_position.x < self.target_position_min.x:
            return False
        elif target_position.x > self.target_position_max.x:
            return False
        elif target_position.y < self.target_position_min.y:
            return False
        elif target_position.y > self.target_position_max.y:
            return False
        elif target_position.z < self.target_position_min.z:
            return False
        elif target_position.z > self.target_position_max.z:
            return False
        else:
            return True

    def target_rotation_in_distribution(self, target_rotation: Vector3) -> bool:
        """
        Check if target rotation is in distribution.

        Args
        ----
            target_rotation (Vector3): Target rotation

        Returns
        -------
            bool: True if the target rotation is in distribution, False otherwise

        """
        if target_rotation.x < self.target_rotation_min.x:
            return False
        elif target_rotation.x > self.target_rotation_max.x:
            return False
        elif target_rotation.y < self.target_rotation_min.y:
            return False
        elif target_rotation.y > self.target_rotation_max.y:
            return False
        elif target_rotation.z < self.target_rotation_min.z:
            return False
        elif target_rotation.z > self.target_rotation_max.z:
            return False
        else:
            return True
