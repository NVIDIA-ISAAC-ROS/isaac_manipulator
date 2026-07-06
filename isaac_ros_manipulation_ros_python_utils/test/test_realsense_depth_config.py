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

"""Tests for RealSense depth routing across manipulation workflows."""

from isaac_ros_manipulation_ros_python_utils.depth_routing import (
    select_foundation_pose_depth_routing,
)

import pytest


OBJECT_FOLLOWING = 'OBJECT_FOLLOWING'
PICK_AND_PLACE = 'PICK_AND_PLACE'
REALSENSE = 'REALSENSE'
ISAAC_SIM = 'ISAAC_SIM'
RAW_DEPTH_TOPIC = '/camera_1/aligned_depth_to_color/image_raw'
RAW_DEPTH_CAMERA_INFO_TOPIC = '/camera_1/aligned_depth_to_color/camera_info'
RAW_DEPTH_IMAGE_WIDTH = '1280'
RAW_DEPTH_IMAGE_HEIGHT = '720'
DNN_DEPTH_TOPIC = '/camera_1/depth_image'
DNN_DEPTH_CAMERA_INFO_TOPIC = '/camera_1/rgb/camera_info'
SIM_DEPTH_TOPIC = '/front_stereo_camera/depth/ground_truth'
SIM_CAMERA_INFO_TOPIC = '/front_stereo_camera/left/camera_info'


class _RealsenseCameraConfig:
    """Minimal RealsenseCameraConfig fixture for depth routing tests."""

    camera_type = REALSENSE
    color_camera_topic_name = '/camera_1/color/image_raw'
    color_camera_info_topic_name = '/camera_1/color/camera_info'
    realsense_depth_camera_topic_name = RAW_DEPTH_TOPIC
    realsense_depth_camera_info_topic_name = RAW_DEPTH_CAMERA_INFO_TOPIC
    realsense_depth_image_width = RAW_DEPTH_IMAGE_WIDTH
    realsense_depth_image_height = RAW_DEPTH_IMAGE_HEIGHT

    def __init__(
        self,
        *,
        depth_type='ESS_FULL',
        enable_dnn_depth_in_realsense=True,
        depth_image_width='960',
        depth_image_height='576',
    ):
        self.depth_type = depth_type
        self.enable_dnn_depth_in_realsense = enable_dnn_depth_in_realsense
        self.ess_depth_camera_topic_name = DNN_DEPTH_TOPIC
        self.ess_depth_camera_info_topic_name = DNN_DEPTH_CAMERA_INFO_TOPIC
        self.depth_camera_topic_name = RAW_DEPTH_TOPIC
        self.depth_camera_info_topic_name = RAW_DEPTH_CAMERA_INFO_TOPIC
        self.depth_image_width = RAW_DEPTH_IMAGE_WIDTH
        self.depth_image_height = RAW_DEPTH_IMAGE_HEIGHT

        if enable_dnn_depth_in_realsense:
            self.depth_camera_topic_name = DNN_DEPTH_TOPIC
            self.depth_camera_info_topic_name = DNN_DEPTH_CAMERA_INFO_TOPIC
            self.depth_image_width = depth_image_width
            self.depth_image_height = depth_image_height


class _IsaacSimCameraConfig:
    """Minimal IsaacSimCameraConfig fixture for depth routing tests."""

    camera_type = ISAAC_SIM
    color_camera_topic_name = '/front_stereo_camera/left/image_raw'
    color_camera_info_topic_name = SIM_CAMERA_INFO_TOPIC
    depth_camera_topic_name = SIM_DEPTH_TOPIC
    depth_camera_info_topic_name = SIM_CAMERA_INFO_TOPIC
    depth_image_width = '1920'
    depth_image_height = '1200'
    enable_dnn_depth_in_realsense = True


def _select_routing(camera_config, workflow_type):
    enable_dnn_depth = camera_config.enable_dnn_depth_in_realsense

    return select_foundation_pose_depth_routing(
        camera_config=camera_config,
        workflow_type=workflow_type,
        enable_dnn_depth_in_realsense=enable_dnn_depth,
    )


@pytest.mark.parametrize(
    ('depth_type', 'dnn_width', 'dnn_height'),
    [
        ('ESS_FULL', '960', '576'),
        ('ESS_LIGHT', '480', '288'),
        ('FOUNDATION_STEREO_LOW_RES', '736', '320'),
        ('FOUNDATION_STEREO_HIGH_RES', '960', '576'),
    ],
)
def test_object_following_realsense_dnn_uses_native_depth(
    depth_type,
    dnn_width,
    dnn_height,
):
    """Verify object following uses native RealSense depth."""
    camera_config = _RealsenseCameraConfig(
        depth_type=depth_type,
        enable_dnn_depth_in_realsense=True,
        depth_image_width=dnn_width,
        depth_image_height=dnn_height,
    )

    routing = _select_routing(
        camera_config=camera_config,
        workflow_type=OBJECT_FOLLOWING,
    )

    assert camera_config.depth_type == depth_type
    assert camera_config.depth_camera_topic_name == DNN_DEPTH_TOPIC
    assert (
        camera_config.depth_camera_info_topic_name ==
        DNN_DEPTH_CAMERA_INFO_TOPIC
    )
    assert camera_config.depth_image_width == dnn_width
    assert camera_config.depth_image_height == dnn_height
    assert camera_config.realsense_depth_camera_topic_name == RAW_DEPTH_TOPIC
    assert (
        camera_config.realsense_depth_camera_info_topic_name ==
        RAW_DEPTH_CAMERA_INFO_TOPIC
    )
    assert camera_config.realsense_depth_image_width == RAW_DEPTH_IMAGE_WIDTH
    assert camera_config.realsense_depth_image_height == RAW_DEPTH_IMAGE_HEIGHT
    assert routing.input_depth_topic == RAW_DEPTH_TOPIC
    assert routing.input_depth_image_width == RAW_DEPTH_IMAGE_WIDTH
    assert routing.input_depth_image_height == RAW_DEPTH_IMAGE_HEIGHT
    assert routing.camera_info_topic == RAW_DEPTH_CAMERA_INFO_TOPIC
    assert not routing.uses_dnn_depth
    assert routing.uses_native_realsense_depth


@pytest.mark.parametrize(
    ('depth_type', 'dnn_width', 'dnn_height'),
    [
        ('ESS_FULL', '960', '576'),
        ('ESS_LIGHT', '480', '288'),
        ('FOUNDATION_STEREO_LOW_RES', '736', '320'),
        ('FOUNDATION_STEREO_HIGH_RES', '960', '576'),
    ],
)
def test_pick_and_place_realsense_dnn_request_uses_dnn_depth(
    depth_type,
    dnn_width,
    dnn_height,
):
    """Verify pick and place keeps RealSense DNN depth routing."""
    camera_config = _RealsenseCameraConfig(
        depth_type=depth_type,
        enable_dnn_depth_in_realsense=True,
        depth_image_width=dnn_width,
        depth_image_height=dnn_height,
    )

    routing = _select_routing(
        camera_config=camera_config,
        workflow_type=PICK_AND_PLACE,
    )

    assert camera_config.depth_type == depth_type
    assert camera_config.depth_camera_topic_name == DNN_DEPTH_TOPIC
    assert (
        camera_config.depth_camera_info_topic_name ==
        DNN_DEPTH_CAMERA_INFO_TOPIC
    )
    assert routing.input_depth_topic == DNN_DEPTH_TOPIC
    assert routing.input_depth_image_width == dnn_width
    assert routing.input_depth_image_height == dnn_height
    assert routing.camera_info_topic == DNN_DEPTH_CAMERA_INFO_TOPIC
    assert routing.uses_dnn_depth
    assert not routing.uses_native_realsense_depth


def test_realsense_without_dnn_uses_configured_native_depth():
    """Verify native RealSense depth is used when DNN depth is disabled."""
    camera_config = _RealsenseCameraConfig(
        enable_dnn_depth_in_realsense=False,
    )

    routing = _select_routing(
        camera_config=camera_config,
        workflow_type=PICK_AND_PLACE,
    )

    assert camera_config.depth_camera_topic_name == RAW_DEPTH_TOPIC
    assert (
        camera_config.depth_camera_info_topic_name ==
        RAW_DEPTH_CAMERA_INFO_TOPIC
    )
    assert routing.input_depth_topic == RAW_DEPTH_TOPIC
    assert routing.input_depth_image_width == RAW_DEPTH_IMAGE_WIDTH
    assert routing.input_depth_image_height == RAW_DEPTH_IMAGE_HEIGHT
    assert routing.camera_info_topic == RAW_DEPTH_CAMERA_INFO_TOPIC
    assert not routing.uses_dnn_depth
    assert not routing.uses_native_realsense_depth


def test_isaac_sim_uses_configured_depth():
    """Verify Isaac Sim uses its configured depth routing."""
    camera_config = _IsaacSimCameraConfig()

    routing = _select_routing(
        camera_config=camera_config,
        workflow_type=OBJECT_FOLLOWING,
    )

    assert routing.input_depth_topic == SIM_DEPTH_TOPIC
    assert routing.input_depth_image_width == '1920'
    assert routing.input_depth_image_height == '1200'
    assert routing.camera_info_topic == SIM_CAMERA_INFO_TOPIC
    assert not routing.uses_dnn_depth
    assert not routing.uses_native_realsense_depth
