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

import isaac_ros_launch_utils as lu
from isaac_ros_launch_utils.all_types import IfCondition, LaunchDescription, Node, TimerAction


def get_realsense_topics(camera_id: int):
    return [
        f'/camera_{camera_id}/color/camera_info', f'/camera_{camera_id}/color/image_raw',
        f'/camera_{camera_id}/depth/image_rect_raw', f'/camera_{camera_id}/depth/camera_info',
        f'/camera_{camera_id}/aligned_depth_to_color/image_raw',
        f'/camera_{camera_id}/aligned_depth_to_color/camera_info',
        f'/cumotion/camera_{camera_id}/world_depth'
    ]


def generate_launch_description() -> LaunchDescription:
    args = lu.ArgumentContainer()
    args.add_arg('run_realsense', True, cli=True)
    args.add_arg('num_cameras', 1, cli=True)
    args.add_arg('run_rqt', True, cli=True)
    args.add_arg('output', 'None', cli=True)

    actions = args.get_launch_actions()

    # Launch realsense
    actions.append(
        lu.include(
            'isaac_manipulator_bringup',
            'launch/include/realsense.launch.py',
            launch_arguments={
                'run_standalone': 'True',
                'num_cameras': args.num_cameras
            },
            condition=IfCondition(args.run_realsense)))

    # Rqt
    actions.append(
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            condition=IfCondition(args.run_rqt)))

    recording_started_msg =\
        """\n\n\n
        -----------------------------------------------------
                    BAG RECORDING IS STARTING NOW

                 (make sure the realsense node is up)
        -----------------------------------------------------
        \n\n\n"""

    topics_to_record = []
    topics_to_record.extend(get_realsense_topics(1))
    topics_to_record.extend(get_realsense_topics(2))
    topics_to_record.extend([
        '/tf_static',
        '/joint_states',
    ])

    record_action = lu.record_rosbag(topics=' '.join(topics_to_record), bag_path=args.output)
    actions.append(
        TimerAction(period=10.0, actions=[record_action,
                                          lu.log_info(recording_started_msg)]))

    return LaunchDescription(actions)
