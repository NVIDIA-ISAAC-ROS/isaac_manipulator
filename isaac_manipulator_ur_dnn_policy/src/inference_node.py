#!/usr/bin/env python3

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

import os

from isaac_manipulator_ur_dnn_policy.msg import Inference
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent   # noqa: F401
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent   # noqa: F401
import torch
import yaml


def load_agent_and_env_cfg(checkpoint_filepath):
    # Load configs from yaml files saved during training
    cfg_yaml_path = os.path.join(os.path.dirname(checkpoint_filepath), 'params', 'agent.yaml')
    with open(cfg_yaml_path, 'r') as f:
        agent_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg_yaml_path = os.path.join(os.path.dirname(checkpoint_filepath), 'params', 'env.yaml')
    with open(cfg_yaml_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    return agent_cfg, env_cfg


def load_agent_rsl_rl(checkpoint_filepath):

    agent_cfg, env_cfg = load_agent_and_env_cfg(checkpoint_filepath)

    # Get the policy class
    policy_class = eval(agent_cfg['policy'].pop('class_name'))
    num_obs = env_cfg['observation_space']
    num_states = env_cfg['state_space']
    num_actions = env_cfg['action_space']
    obs_groups = agent_cfg['obs_groups']

    # Create mock obs dictionary and groups for rsl_rl ActorCritic API
    obs_dict = {
        'policy': torch.zeros((1, num_obs), dtype=torch.float32),
        'critic': torch.zeros((1, num_states), dtype=torch.float32),
    }

    policy = policy_class(obs_dict, obs_groups, num_actions, **agent_cfg['policy'])

    # Load model weights
    loaded_dict = torch.load(checkpoint_filepath, weights_only=False)
    policy.load_state_dict(loaded_dict['model_state_dict'])

    policy_nn = policy

    # Get inference policy
    policy_nn.eval()
    policy = policy_nn.act_inference

    return policy, agent_cfg, env_cfg


class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        # set default torch device
        torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('checkpoint', rclpy.Parameter.Type.STRING)
        checkpoint = self.get_parameter('checkpoint')
        self.checkpoint = checkpoint.get_parameter_value().string_value

        self.declare_parameter('alpha', 1.0)
        alpha = self.get_parameter('alpha')
        self.alpha = alpha.get_parameter_value().double_value

        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("'alpha' must be in the range [0.0, 1.0]")

        # load agent
        self.policy, agent_cfg, env_cfg = load_agent_rsl_rl(self.checkpoint)

        if 'fixed_asset_init_pos_center' in env_cfg:
            target_pos_centre = env_cfg['fixed_asset_init_pos_center']
        elif 'target_pos_centre' in env_cfg:
            target_pos_centre = env_cfg['target_pos_centre']
        else:
            raise ValueError('Environment configuration missing '
                             'fixed_asset_init_pos_center or target_pos_centre')

        if 'fixed_asset_init_pos_range' in env_cfg:
            target_pos_range = env_cfg['fixed_asset_init_pos_range']
        elif 'target_pos_range' in env_cfg:
            target_pos_range = env_cfg['target_pos_range']
        else:
            raise ValueError('Environment configuration missing '
                             'fixed_asset_init_pos_range or target_pos_range')

        if 'fixed_asset_init_orn_deg' in env_cfg:
            target_rot_centre = env_cfg['fixed_asset_init_orn_deg']
        elif 'target_rot_centre' in env_cfg:
            target_rot_centre = env_cfg['target_rot_centre']
        else:
            raise ValueError('Environment configuration missing '
                             'fixed_asset_init_orn_deg or target_rot_centre')

        if 'fixed_asset_init_orn_deg_range' in env_cfg:
            target_rot_range = env_cfg['fixed_asset_init_orn_deg_range']
        elif 'target_rot_range' in env_cfg:
            target_rot_range = env_cfg['target_rot_range']
        else:
            raise ValueError('Environment configuration missing '
                             'fixed_asset_init_orn_deg_range or target_rot_range')

        self.declare_parameter(
            'obs_order', env_cfg['obs_order'], ignore_override=True)
        self.declare_parameter(
            'arm_joint_names', env_cfg['arm_joint_names'], ignore_override=True)
        self.declare_parameter(
            'policy_action_space', env_cfg['policy_action_space'], ignore_override=True)
        self.declare_parameter(
            'action_scale_joint_space', env_cfg['action_scale_joint_space'], ignore_override=True)
        self.declare_parameter(
            'target_pos_centre', target_pos_centre, ignore_override=True)
        self.declare_parameter(
            'target_pos_range', target_pos_range, ignore_override=True)
        self.declare_parameter(
            'target_rot_centre', target_rot_centre, ignore_override=True)
        self.declare_parameter(
            'target_rot_range', target_rot_range, ignore_override=True)

        self.create_subscription(
            Inference, 'observation', self.callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.publisher = self.create_publisher(
            Inference, 'action',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10))

        self.prev_action = None

    def callback(self, msg: Inference):
        with torch.inference_mode():
            obs = {'policy': torch.tensor(msg.data).unsqueeze(0)}
            action = self.policy(obs).clamp(-1.0, 1.0)

            # apply exponential moving average
            if self.prev_action is not None:
                action = self.alpha * action + (1.0 - self.alpha) * self.prev_action
            self.prev_action = action.clone()

            self.publisher.publish(
                Inference(
                    header=msg.header,
                    joint_state=msg.joint_state,
                    data=action.squeeze(0).tolist(),
                )
            )


def main():
    rclpy.init()
    rclpy.spin(InferenceNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
