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

from dataclasses import dataclass
import enum


@dataclass
class IsaacSimCameraConfig:
    """Contains image topic names coming from isaac sim or real"""

    color_camera_topic_name: str = "/front_stereo_camera/left/image_raw"
    depth_camera_topic_name: str = "/front_stereo_camera/depth/ground_truth"
    color_camera_info_topic_name: str = "/front_stereo_camera/left/camera_info"
    depth_camera_info_topic_name: str = "/front_stereo_camera/depth/camera_info"
    rgb_image_width: str = "1920"
    rgb_image_height: str = "1200"
    depth_image_width: str = "1920"
    depth_image_height: str = "1200"


class GripperType(enum.Enum):
    ROBOTIQ_2F_140 = 'robotiq_2f_140'
    ROBOTIQ_2F_85 = 'robotiq_2f_85'


class EnumMeta(enum.EnumMeta):
    """Enum metaclass to add meaningful error messages for KeyErrors and AttributeErrors."""

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError as error:
            raise KeyError(f'The key {name} is not part of the {self.__name__} enum '
                           f'(valid options are {self.names()}). KeyError: {error}')

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            raise AttributeError(f'The attribute {name} is not part of the {self.__name__} enum. '
                                 f'AttributeError: {error}')


class ManipulatorEnum(enum.Enum, metaclass=EnumMeta):
    """Base class for manipulator enums defining common functions."""

    @classmethod
    def names(cls):
        return [mode.name for mode in cls]

    def __str__(self):
        return self.name


class CameraType(ManipulatorEnum):
    """Enum defining the camera that manipulator should be run with."""

    realsense = 1
    hawk = 2
    isaac_sim = 3


class DepthType(ManipulatorEnum):
    """Enum defining the ESS model to be used if CameraType is hawk."""

    ess_full = 1
    ess_light = 2


class TrackingType(ManipulatorEnum):
    """Enum defining the tracking type that manipulator should be run in."""

    pose_to_pose = 1
    follow_object = 2
    none = 3


class PoseEstimationType(ManipulatorEnum):
    """Enum defining the pose estimation model to be used if TrackingType is follow_object."""

    foundationpose = 1
    dope = 2


class ObjectAttachmentShape(enum.Enum):
    """Object attachment types"""
    SPHERE = 'sphere'
    CUBOID = 'cuboid'
    CUSTOM_MESH = 'custom_mesh'


class Mode(enum.Enum):
    """
    Enumeration for operation modes.

    Values:
        ONCE: Execute the action once.
        CYCLE: Continuously cycle through attach and detach actions.
    """

    ONCE = 'once'
    CYCLE = 'cycle'


class AttachState(enum.Enum):
    """
    Enumeration for attachment states.

    Values:
        ATTACH: Attach the object.
        DETACH: Detach the object.
    """

    ATTACH = True
    DETACH = False
