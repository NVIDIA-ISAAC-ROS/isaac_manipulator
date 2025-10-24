#!/usr/bin/env python3

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

from argparse import ArgumentParser

from geometry_msgs.msg import PoseStamped
import isaac_manipulator_ros_python_utils as utils
from isaac_manipulator_ur_dnn_policy.msg import Inference
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


def plot(rosbag: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=rosbag, storage_id='mcap'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
    )

    policy = {
        't': [],
        'obs': [],
        'action': [],
    }

    joint_state = {
        't': [],
        'position': {
            'shoulder_pan_joint': [],
            'shoulder_lift_joint': [],
            'elbow_joint': [],
            'wrist_1_joint': [],
            'wrist_2_joint': [],
            'wrist_3_joint': [],
        },
        'velocity': {
            'shoulder_pan_joint': [],
            'shoulder_lift_joint': [],
            'elbow_joint': [],
            'wrist_1_joint': [],
            'wrist_2_joint': [],
            'wrist_3_joint': [],
        },
        'effort': {
            'shoulder_pan_joint': [],
            'shoulder_lift_joint': [],
            'elbow_joint': [],
            'wrist_1_joint': [],
            'wrist_2_joint': [],
            'wrist_3_joint': [],
        },
    }

    tcp_state = {
        't': [],
        'x': [],
        'y': [],
        'z': [],
        'rx': [],
        'ry': [],
        'rz': [],
    }

    tcp_error = {
        't': [],
        'x': [],
        'y': [],
        'z': [],
        'rx': [],
        'ry': [],
        'rz': [],
    }

    start = None
    target_tcp_pose = None
    control_start = None

    transforms = {}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if not start:
            start = timestamp

        if topic == '/goal_pose':
            msg = deserialize_message(data, PoseStamped)
            target_tcp_pose = msg.pose
            if not control_start:
                control_start = timestamp

        elif topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            for transform in msg.transforms:
                T = utils.transform_to_matrix(transform.transform)
                transforms[f'{transform.header.frame_id}.{transform.child_frame_id}'] = T

        elif topic == '/tf':
            msg = deserialize_message(data, TFMessage)

            for transform in msg.transforms:
                T = utils.transform_to_matrix(transform.transform)
                transforms[f'{transform.header.frame_id}.{transform.child_frame_id}'] = T

            if 'base_link.base' in transforms and len(msg.transforms) > 1:
                transform = utils.matrix_to_transform(
                    np.linalg.inv(transforms['base_link.base']) @
                    transforms['base_link.base_link_inertia'] @
                    transforms['base_link_inertia.shoulder_link'] @
                    transforms['shoulder_link.upper_arm_link'] @
                    transforms['upper_arm_link.forearm_link'] @
                    transforms['forearm_link.wrist_1_link'] @
                    transforms['wrist_1_link.wrist_2_link'] @
                    transforms['wrist_2_link.wrist_3_link'] @
                    transforms['wrist_3_link.flange'] @
                    transforms['flange.tool0'])

                tcp_state['t'].append(utils.ns_to_s(timestamp - start))

                tcp_state['x'].append(transform.translation.x)
                tcp_state['y'].append(transform.translation.y)
                tcp_state['z'].append(transform.translation.z)

                rotation = utils.quaternion_to_rpy(transform.rotation)
                tcp_state['rx'].append(utils.rad_to_deg(rotation.x))
                tcp_state['ry'].append(utils.rad_to_deg(rotation.y))
                tcp_state['rz'].append(utils.rad_to_deg(rotation.z))

                if target_tcp_pose:
                    tcp_error['t'].append(utils.ns_to_s(timestamp - control_start))

                    tcp_error['x'].append(target_tcp_pose.position.x - transform.translation.x)
                    tcp_error['y'].append(target_tcp_pose.position.y - transform.translation.y)
                    tcp_error['z'].append(target_tcp_pose.position.z - transform.translation.z)

                    quat_diff = utils.quaternion_difference(
                        target_tcp_pose.orientation, transform.rotation)
                    rotation = utils.quaternion_to_rpy(quat_diff)

                    tcp_error['rx'].append(utils.rad_to_deg(rotation.x))
                    tcp_error['ry'].append(utils.rad_to_deg(rotation.y))
                    tcp_error['rz'].append(utils.rad_to_deg(rotation.z))

        elif topic == '/observation':
            msg = deserialize_message(data, Inference)
            policy['t'].append(utils.ns_to_s(timestamp - start))
            policy['obs'].append(list(msg.data))

        elif topic == '/action':
            msg = deserialize_message(data, Inference)
            policy['action'].append(list(msg.data))

        elif topic == '/joint_states':
            msg = deserialize_message(data, JointState)

            joint_state['t'].append(utils.ns_to_s(timestamp - start))

            for name, position in zip(msg.name, msg.position):
                if name in joint_state['position']:
                    joint_state['position'][name].append(utils.rad_to_deg(position))

            for name, velocity in zip(msg.name, msg.velocity):
                if name in joint_state['velocity']:
                    joint_state['velocity'][name].append(utils.rad_to_deg(velocity))

            for name, effort in zip(msg.name, msg.effort):
                if name in joint_state['effort']:
                    joint_state['effort'][name].append(utils.rad_to_deg(effort))

    # plot TCP error
    plt.figure(num='tcp_error')
    plt.suptitle('TCP Error')

    plt.subplot(2, 1, 1)
    plt.ylabel('Position Error (m)')
    plt.plot(tcp_error['t'], tcp_error['x'], label='X', color='tab:red')
    plt.plot(tcp_error['t'], tcp_error['y'], label='Y', color='tab:green')
    plt.plot(tcp_error['t'], tcp_error['z'], label='Z', color='tab:blue')
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.ylabel('Rotation Error (deg)')
    plt.xlabel('Time (s)')
    plt.plot(tcp_error['t'], tcp_error['rx'], label='RX', color='tab:red')
    plt.plot(tcp_error['t'], tcp_error['ry'], label='RY', color='tab:green')
    plt.plot(tcp_error['t'], tcp_error['rz'], label='RZ', color='tab:blue')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # plot policy
    plt.figure(num='policy')
    plt.suptitle('Policy')

    while len(policy['t']) > len(policy['action']):
        policy['t'].pop()
        policy['obs'].pop()

    while len(policy['action']) > len(policy['t']):
        policy['action'].pop()

    t = np.array(policy['t'])
    obs = np.array(policy['obs'])
    action = np.array(policy['action'])

    plt.subplot(2, 1, 1)
    plt.ylabel('Observation')
    for i in range(obs.shape[1]):
        plt.plot(t, obs[:, i], label=f'[{i}]')
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.ylabel('Action')
    plt.xlabel('Time (s)')
    for i in range(action.shape[1]):
        plt.plot(t, action[:, i], label=f'[{i}]')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # plot joint state
    plt.figure(num='joint_state')
    plt.suptitle('Joint State')

    plt.subplot(2, 1, 1)
    plt.ylabel('Position (deg)')
    plt.plot(joint_state['t'], joint_state['position']['shoulder_pan_joint'], label='Base')
    plt.plot(joint_state['t'], joint_state['position']['shoulder_lift_joint'], label='Shoulder')
    plt.plot(joint_state['t'], joint_state['position']['elbow_joint'], label='Elbow')
    plt.plot(joint_state['t'], joint_state['position']['wrist_1_joint'], label='Wrist 1')
    plt.plot(joint_state['t'], joint_state['position']['wrist_2_joint'], label='Wrist 2')
    plt.plot(joint_state['t'], joint_state['position']['wrist_3_joint'], label='Wrist 3')
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.ylabel('Velocity (deg/s)')
    plt.xlabel('Time (s)')
    plt.plot(joint_state['t'], joint_state['velocity']['shoulder_pan_joint'], label='Base')
    plt.plot(joint_state['t'], joint_state['velocity']['shoulder_lift_joint'], label='Shoulder')
    plt.plot(joint_state['t'], joint_state['velocity']['elbow_joint'], label='Elbow')
    plt.plot(joint_state['t'], joint_state['velocity']['wrist_1_joint'], label='Wrist 1')
    plt.plot(joint_state['t'], joint_state['velocity']['wrist_2_joint'], label='Wrist 2')
    plt.plot(joint_state['t'], joint_state['velocity']['wrist_3_joint'], label='Wrist 3')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # plot TCP state
    plt.figure(num='tcp_state')
    plt.suptitle('TCP State')

    plt.subplot(2, 1, 1)
    plt.ylabel('Position (m)')
    plt.plot(tcp_state['t'], tcp_state['x'], label='X', color='tab:red')
    plt.plot(tcp_state['t'], tcp_state['y'], label='Y', color='tab:green')
    plt.plot(tcp_state['t'], tcp_state['z'], label='Z', color='tab:blue')
    plt.legend(loc='upper left')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.ylabel('Rotation (deg)')
    plt.plot(tcp_state['t'], tcp_state['rx'], label='RX', color='tab:red')
    plt.plot(tcp_state['t'], tcp_state['ry'], label='RY', color='tab:green')
    plt.plot(tcp_state['t'], tcp_state['rz'], label='RZ', color='tab:blue')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = ArgumentParser(prog='plot')
    parser.add_argument(
        'rosbag',
        help='rosbag path',
        type=str,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot(args.rosbag)


if __name__ == '__main__':
    main()
