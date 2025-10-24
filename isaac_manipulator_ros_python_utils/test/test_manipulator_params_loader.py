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
import tempfile
import unittest

from isaac_manipulator_ros_python_utils.config import load_yaml_params


ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise ValueError('ISAAC_ROS_WS is not set')


class TestLoadYamlParams(unittest.TestCase):
    """Test cases for the load_yaml_params function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        # get the full path to the test directory
        self.test_file_path = os.path.join(ISAAC_ROS_WS,
                                           'src',
                                           'isaac_manipulator',
                                           'isaac_manipulator_ros_python_utils',
                                           'test',
                                           'config',
                                           'test_manipulator_config_file.yaml')

        # create a test dir in tmp folder
        self.test_dir = tempfile.mkdtemp()

        # create the test directory if it doesn't exist
        if not os.path.exists(self.test_file_path):
            raise ValueError('Test file not found at path: ' + self.test_file_path)

    def test_load_basic_yaml(self):
        """Test loading a basic YAML file without command substitutions."""
        # Load the YAML file
        params = load_yaml_params(self.test_file_path)

        # Verify the content
        self.assertEqual(params['camera_type'], 'REALSENSE')
        self.assertEqual(params['num_cameras'], 1)
        self.assertEqual(params['workflow_type'], 'GEAR_ASSEMBLY')
        self.assertEqual(params['ur_type'], 'ur10e')
        self.assertEqual(params['gripper_type'], 'robotiq_2f_85')
        self.assertEqual(params['use_sim_time'], 'false')

    def test_load_yaml_with_command_substitution(self):
        """Test loading a YAML file with command substitutions."""
        # Load the YAML file
        params = load_yaml_params(self.test_file_path)

        # Verify the content
        self.assertEqual(params['object_detection_type'], 'SEGMENT_ANYTHING')
        self.assertEqual(params['segmentation_type'], 'SEGMENT_ANYTHING')

        self.assertEqual(
            params['ess_engine_file_path'],
            f'{ISAAC_ROS_WS}/isaac_ros_assets/models'
            '/dnn_stereo_disparity/dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine')

        # also test env variables are applied inside of lists.
        self.assertEqual(len(params['sam_model_repository_paths']), 1)
        self.assertEqual(
            params['sam_model_repository_paths'][0],
            f'{ISAAC_ROS_WS}/isaac_ros_assets/models')

        # also check the scene objects file path
        self.assertEqual(
            params['moveit_collision_objects_scene_file'],
            f'{ISAAC_ROS_WS}/src/isaac_manipulator'
            '/isaac_manipulator_bringup/config/'
            'scene_objects/rosie_ur10e_test_bench/rosie_ur10e_test_scene_objects.scene')

        # boolean check, dont turn to bool, keep it as string.
        self.assertEqual(
            params['segment_anything_enable_debug_output'],
            'false'
        )


if __name__ == '__main__':
    unittest.main()
