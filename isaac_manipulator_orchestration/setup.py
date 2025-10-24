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

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'isaac_manipulator_orchestration'

setup(
    name=package_name,
    version='3.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))
        ),
        (
            os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*'))
        ),
        (
            os.path.join('share', package_name, 'test', 'include'),
            glob(os.path.join('test', 'include', '*launch.[pxy][yma]*'))
        )
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='Reference orchestration package for Isaac Manipulator',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest'
        ]
    },
    entry_points={
        'console_scripts': [
            'robot_tf_broadcaster = '
            'isaac_manipulator_orchestration.mock_servers.robot_tf_broadcaster:main',
            'pose_est_server = '
            'isaac_manipulator_orchestration.mock_servers.pose_est_server:main',
            'object_detection_server = '
            'isaac_manipulator_orchestration.mock_servers.object_detection_server:main',
            'object_selector_server = '
            'isaac_manipulator_orchestration.mock_servers.object_selector_server:main',
            'add_mesh_server = '
            'isaac_manipulator_orchestration.mock_servers.add_mesh_server:main',
            'gripper_command_server = '
            'isaac_manipulator_orchestration.mock_servers.gripper_command_server:main',
            'attach_object_server = '
            'isaac_manipulator_orchestration.mock_servers.attach_object_server:main',
            'assign_name_server = '
            'isaac_manipulator_orchestration.mock_servers.assign_name_server:main',
            'motion_plan_server = '
            'isaac_manipulator_orchestration.mock_servers.motion_plan_server:main',
            'execute_trajectory_server = '
            'isaac_manipulator_orchestration.mock_servers.execute_trajectory_server:main',
            'controller_manager_server = '
            'isaac_manipulator_orchestration.mock_servers.controller_manager_server:main',
            'publish_static_planning_scene_server = '
            'isaac_manipulator_orchestration.mock_servers.'
            'publish_static_planning_scene_server:main',
        ],
    },
)
