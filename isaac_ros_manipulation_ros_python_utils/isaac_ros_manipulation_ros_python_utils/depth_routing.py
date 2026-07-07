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

"""Depth routing helpers for RealSense + DNN depth workflows."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FoundationPoseDepthRouting:
    """Depth input selected for the FoundationPose pose-estimation path."""

    input_depth_topic: str
    input_depth_image_width: str
    input_depth_image_height: str
    camera_info_topic: str
    uses_dnn_depth: bool
    uses_native_realsense_depth: bool


def _value_name(value: Any) -> str:
    return getattr(value, 'name', str(value))


def select_foundation_pose_depth_routing(
    *,
    camera_config: Any,
    workflow_type: Any,
    enable_dnn_depth_in_realsense: bool,
) -> FoundationPoseDepthRouting:
    """
    Select the depth stream consumed by FoundationPose.

    RealSense object-following keeps native aligned RealSense depth on the live
    pose-estimation path so RGB, depth, and segmentation remain timestamp-compatible.
    Other RealSense workflows can use the requested DNN depth stream.
    """
    depth_topic = camera_config.depth_camera_topic_name
    depth_image_width = camera_config.depth_image_width
    depth_image_height = camera_config.depth_image_height
    depth_camera_info_topic = camera_config.depth_camera_info_topic_name
    dnn_depth_topic = getattr(camera_config, 'ess_depth_camera_topic_name', depth_topic)
    dnn_depth_camera_info_topic = getattr(
        camera_config, 'ess_depth_camera_info_topic_name', depth_camera_info_topic
    )
    realsense_depth_topic = getattr(
        camera_config, 'realsense_depth_camera_topic_name', depth_topic
    )
    realsense_depth_image_width = getattr(
        camera_config, 'realsense_depth_image_width', depth_image_width
    )
    realsense_depth_image_height = getattr(
        camera_config, 'realsense_depth_image_height', depth_image_height
    )
    realsense_depth_camera_info_topic = getattr(
        camera_config, 'realsense_depth_camera_info_topic_name', depth_camera_info_topic
    )

    camera_type = camera_config.camera_type
    is_realsense = _value_name(camera_type) == 'REALSENSE'
    is_object_following = _value_name(workflow_type) == 'OBJECT_FOLLOWING'
    use_native_realsense_depth = is_realsense and is_object_following
    use_dnn_depth = (
        is_realsense and enable_dnn_depth_in_realsense and not use_native_realsense_depth
    )

    if use_native_realsense_depth:
        return FoundationPoseDepthRouting(
            input_depth_topic=realsense_depth_topic,
            input_depth_image_width=realsense_depth_image_width,
            input_depth_image_height=realsense_depth_image_height,
            camera_info_topic=realsense_depth_camera_info_topic,
            uses_dnn_depth=False,
            uses_native_realsense_depth=True,
        )

    if use_dnn_depth:
        return FoundationPoseDepthRouting(
            input_depth_topic=dnn_depth_topic,
            input_depth_image_width=depth_image_width,
            input_depth_image_height=depth_image_height,
            camera_info_topic=dnn_depth_camera_info_topic,
            uses_dnn_depth=True,
            uses_native_realsense_depth=False,
        )

    return FoundationPoseDepthRouting(
        input_depth_topic=depth_topic,
        input_depth_image_width=depth_image_width,
        input_depth_image_height=depth_image_height,
        camera_info_topic=depth_camera_info_topic,
        uses_dnn_depth=False,
        uses_native_realsense_depth=False,
    )
