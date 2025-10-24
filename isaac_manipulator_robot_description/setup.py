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

from glob import glob
import os
from typing import List

from setuptools import find_packages, setup

package_name = 'isaac_manipulator_robot_description'


def get_all_files_in_directory(directory: str) -> List[str]:
    """
    List of paths for all files in a directory.

    Args
    ----
        directory (str): Directory that needs to be parsed

    Returns
    -------
        List[str]: List of paths

    """
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths


setup(
    name=package_name,
    version='3.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'srdf'),
            glob(os.path.join('srdf', '*')),
        ),
        (
            os.path.join('share', package_name, 'urdf'),
            glob(os.path.join('urdf', '*')),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*')),
        ),
        (
            os.path.join('share', package_name, 'meshes'),
            get_all_files_in_directory('meshes'),
        ),
        (
            os.path.join('share', package_name, 'meshes', 'soup_can'),
            glob(os.path.join('meshes', 'soup_can', '*')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='Package containing URDF, meshes, SRDFs '
                'for various robots and grippers for Isaac Manipulator',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest'
        ]
    },
)
