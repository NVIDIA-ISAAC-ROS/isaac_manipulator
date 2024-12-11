# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MANIPULATOR_CONTAINER_NAME = 'manipulator_container'

HAWK_IMAGE_WIDTH = 1920
HAWK_IMAGE_HEIGHT = 1200

REALSENSE_IMAGE_WIDTH = 1280
REALSENSE_IMAGE_HEIGHT = 720

ESS_INPUT_IMAGE_WIDTH = 960
ESS_INPUT_IMAGE_HEIGHT = 576

ESS_LIGHT_INPUT_IMAGE_WIDTH = 480
ESS_LIGHT_INPUT_IMAGE_HEIGHT = 288

ROBOT_SEGMENTER_OUTPUT_DEPTH = "['/cumotion/camera_1/world_depth']"