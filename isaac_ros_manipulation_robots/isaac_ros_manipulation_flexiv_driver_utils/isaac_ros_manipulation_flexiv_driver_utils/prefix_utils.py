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

"""
Utility to apply a joint/link name prefix to cuMotion URDF and XRDF files.

The Flexiv third-party URDF prefixes all joint and link names with
``robot_sn + '_'`` (e.g. ``Rizon4s-062839_joint1``).  cuMotion's URDF and XRDF
template files use unprefixed names.  This module bridges the gap by reading a
template and producing a copy with every known name prefixed, so cuMotion plans
with the same names that MoveIt / ros2_control expect.
"""

import os
import re
from typing import Optional

# Every joint/link name that appears in the Flexiv Rizon + Grav cuMotion
# URDF/XRDF templates.  ``world`` is deliberately excluded because the
# Flexiv URDF never prefixes it.
#
# Sorted longest-first so that longer names are matched before shorter
# substrings (e.g. ``grav_base_link`` before ``base_link``).  The regex
# word-boundary approach (``\b``) already prevents cross-matching because
# ``_`` is a word character, but the ordering provides an extra safety net
# if the replacement strategy ever changes.
FLEXIV_RIZON_NAMES = sorted([
    'closed_fingers_tcp',
    'right_finger_mount', 'left_finger_mount',
    'right_finger_tip', 'left_finger_tip',
    'right_finger_tcp', 'left_finger_tcp',
    'finger_width_joint',
    'right_inner_bar', 'right_outer_bar',
    'left_inner_bar', 'left_outer_bar',
    'grav_base_link', 'grav_tcp',
    'gripper_frame', 'grasp_frame', 'insertion_frame',
    'base_joint', 'base_link',
    'flange',
    'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
], key=len, reverse=True)


def apply_joint_prefix(
    template_path: str,
    prefix: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Read *template_path*, prefix joint/link names, write result.

    Parameters
    ----------
    template_path : str
        Absolute path to the source URDF or XRDF file.
    prefix : str
        The string to prepend to every joint/link name
        (e.g. ``"Rizon4s-062839_"``).
    output_path : str, optional
        Where to write the result.  Defaults to
        ``/tmp/<original_stem>_prefixed<ext>``.

    Returns
    -------
    str
        Absolute path to the written file.

    """
    with open(template_path, 'r') as f:
        content = f.read()

    for name in FLEXIV_RIZON_NAMES:
        pattern = r'\b' + re.escape(name) + r'\b'
        content = re.sub(pattern, prefix + name, content)

    if output_path is None:
        stem, ext = os.path.splitext(os.path.basename(template_path))
        output_path = os.path.join('/tmp', f'{stem}_prefixed{ext}')

    with open(output_path, 'w') as f:
        f.write(content)

    return output_path
