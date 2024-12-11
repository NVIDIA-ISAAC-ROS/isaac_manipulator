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

from typing import List

from isaac_ros_launch_utils.all_types import Action, LaunchDescription
import isaac_ros_launch_utils as lu

from isaac_manipulator_ros_python_utils.types import CameraType, TrackingType

# Dictionary containing the calibration of various camera setups.
# Every item of the dictionary represents the calibration of a single setup
calibrations_dict = {
    'hubble_test_bench': {
        'world_to_hawk': {
            'parent_frame': 'world',
            'child_frame': 'hawk',
            'translation': [-1.75433, -0.0887958, 0.419998],
            'rotation': [-0.00447052, 0.138631, -0.0101076, 0.990282],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-1.51812, 0.321693, 0.567912],
            'rotation': [0.0271786, 0.171242, -0.36313, 0.915464],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-1.47782, -1.23458, 0.533205],
            'rotation': [-0.0636737, 0.206499, 0.343712, 0.913874],  # [qx, qy ,qz, qw]
        },
        'world_to_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_to_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'zurich_test_bench': {
        'world_to_hawk': {
            'parent_frame': 'world',
            'child_frame': 'hawk',
            'translation': [-0.646121, 0.634906, 0.657998],
            'rotation': [0.0647987, 0.0853649, -0.5974046, 0.794747],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [2.131679, 0.563435, 0.775389],
            'rotation': [-0.324828, 0.001799, 0.945766, -0.002907],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [-0.250322, 0.598947, 0.864349],
            'rotation': [0.039936, 0.349054, -0.058430, 0.934426],  # [qx, qy ,qz, qw]
        },
        'world_to_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'object_to_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
    'hubble_ur5e_test_bench': {
        'world_to_base_link': {
            'parent_frame': 'world',
            'child_frame': 'base_link',
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],  # [qx, qy ,qz, qw]
        },
        'world_to_hawk': {
            'parent_frame': 'world',
            'child_frame': 'hawk',
            'translation': [-0.646121, 0.634906, 0.657998],
            'rotation': [0.0647987, 0.0853649, -0.597404, 0.794747],  # [qx, qy ,qz, qw]
        },
        'world_to_hawk_2': {
            'parent_frame': 'world',
            'child_frame': 'hawk_2',
            'translation': [0.00717077, 0.73, 0.524834],
            'rotation': [-0.09738, -0.0599994, 0.870336, -0.478991],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_2': {
            'parent_frame': 'world',
            'child_frame': 'camera_2_link',
            'translation': [0.0442943, 0.821461, 0.521577],
            'rotation': [-0.103339, -0.0650548, 0.881767, -0.455604],  # [qx, qy ,qz, qw]
        },
        'world_to_realsense_1': {
            'parent_frame': 'world',
            'child_frame': 'camera_1_link',
            'translation': [-1.3285, 0.563134, 0.383402],
            'rotation': [0.0157158, 0.01767, -0.308705, 0.950864],  # [qx, qy ,qz, qw]
        },
        'object_to_grasp_frame': {
            'parent_frame': 'detected_object1',
            'child_frame': 'goal_frame',
            'translation': [0.043, 0.359, 0.065],
            'rotation': [0.553, 0.475, -0.454, 0.513],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_1': {
            'parent_frame': 'world',
            'child_frame': 'target1_frame',
            'translation': [-0.7, 0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
        'world_to_target_frame_2': {
            'parent_frame': 'world',
            'child_frame': 'target2_frame',
            'translation': [-0.7, -0.3, 0.4],
            'rotation': [1.0, 0.0, 0.0, 0.0],  # [qx, qy ,qz, qw]
        },
    },
}


def static_transform_from_dict(transform_dict):
    return lu.static_transform(
        parent=transform_dict['parent_frame'],
        child=transform_dict['child_frame'],
        translation=transform_dict['translation'],
        orientation_quaternion=transform_dict['rotation'])


def add_static_transforms(args: lu.ArgumentContainer) -> List[Action]:
    camera_type = CameraType[args.camera_type]
    tracking_type = TrackingType[args.tracking_type]
    num_cameras = int(args.num_cameras)
    broadcast_world_base_link = bool(args.broadcast_world_base_link)

    # Get the calibration dict
    if args.calibration_name not in calibrations_dict:
        return [
            lu.log_info([
                "Calibration with name '",
                str(args.calibration_name), "' does not exits. Not loading static transforms."
            ])
        ]
    transforms = calibrations_dict[args.calibration_name]

    actions = []
    # Get world transform
    if broadcast_world_base_link:
        actions.append(static_transform_from_dict(transforms['world_to_base_link']))

    # Get camera transforms
    if camera_type is CameraType.hawk:
        actions.append(static_transform_from_dict(transforms['world_to_hawk']))
    elif camera_type is CameraType.realsense:
        actions.append(static_transform_from_dict(transforms['world_to_realsense_1']))
        if num_cameras > 1:
            actions.append(static_transform_from_dict(transforms['world_to_realsense_2']))
            assert num_cameras <= 2, 'Running more than 2 cameras not allowed.'
    else:
        raise Exception(f'CameraType {camera_type} not implemented.')

    # Get target and grasp frames
    if tracking_type is TrackingType.follow_object:
        actions.append(static_transform_from_dict(transforms['object_to_grasp_frame']))
    elif tracking_type is TrackingType.pose_to_pose:
        actions.append(static_transform_from_dict(transforms['world_to_target_frame_1']))
        actions.append(static_transform_from_dict(transforms['world_to_target_frame_2']))
    elif tracking_type is not TrackingType.none:
        raise Exception(f'TrackingType {tracking_type} not implemented.')

    actions.append(
        lu.log_info([
            "Successfully loaded the static transforms of the '",
            str(args.calibration_name),
            "' calibration.",
        ]))
    return actions


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('num_cameras', 1)
    args.add_arg('broadcast_world_base_link', False)
    args.add_arg('camera_type')
    args.add_arg('tracking_type')
    args.add_arg('calibration_name', '')

    args.add_opaque_function(add_static_transforms)
    return LaunchDescription(args.get_launch_actions())
